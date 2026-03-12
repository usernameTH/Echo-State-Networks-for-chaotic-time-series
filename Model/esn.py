import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import eigs
from sklearn.linear_model import Ridge

class EchoStateNetwork:
    """
    Echo State Network (ESN) for time-series prediction.
    """
    def __init__(
        self, 
        input_size: int, 
        reservoir_size: int = 1000, 
        spectral_radius: float = 0.9, 
        sparsity: float = 0.1, 
        leak_rate: float = 1.0, 
        ridge_alpha: float = 1e-4
    ):
        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.leak_rate = leak_rate
        
        # Scikit-learn's Ridge Regression for calculating W_out
        self.readout_model = Ridge(alpha=ridge_alpha)
        
        # Initialize W_in and W
        self._initialize_weights()

    def _initialize_weights(self):
        """Creates the input weight matrix and the sparse reservoir weight matrix."""
        # W_in: Dense matrix mapping inputs to the reservoir
        self.W_in = np.random.rand(self.reservoir_size, self.input_size) - 0.5
        
        # W: Sparse matrix representing the reservoir connections
        W = sparse.random(self.reservoir_size, self.reservoir_size, density=self.sparsity)
        
        # Scale W by its spectral radius (largest absolute eigenvalue)
        eigenvalues = eigs(W, k=1, return_eigenvectors=False)
        max_eigenvalue = np.abs(eigenvalues[0])
        self.W = (W / max_eigenvalue) * self.spectral_radius

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Drives the reservoir with training data and fits the output weights.
        """
        num_samples = X_train.shape[0]
        states = np.zeros((num_samples, self.reservoir_size))
        
        # Initial state set to zero
        current_state = np.zeros(self.reservoir_size)
        
        # Run data through the reservoir
        for t in range(num_samples):
            u_t = X_train[t]
            # Update equation: x(t) = (1-a)*x(t-1) + a*tanh(W_in * u(t) + W * x(t-1))
            update = np.tanh(self.W_in @ u_t + self.W @ current_state)
            current_state = (1 - self.leak_rate) * current_state + self.leak_rate * update
            states[t] = current_state
            
        # Fit the readout weights to target 
        self.readout_model.fit(states, y_train)

    def predict(self, initial_input: np.ndarray, num_steps: int) -> np.ndarray:
        """
        Generates autonomous predictions by feeding outputs back in as inputs.
        """
        predictions = np.zeros((num_steps, self.input_size))
        current_input = initial_input
        current_state = np.zeros(self.reservoir_size) # Or the last state from training
        
        for t in range(num_steps):
            update = np.tanh(self.W_in @ current_input + self.W @ current_state)
            current_state = (1 - self.leak_rate) * current_state + self.leak_rate * update
            
            # Predict the next step
            next_step = self.readout_model.predict(current_state.reshape(1, -1))[0]
            predictions[t] = next_step
            
            # Feed prediction back in for the next iteration (closed-loop)
            current_input = next_step
            
        return predictions