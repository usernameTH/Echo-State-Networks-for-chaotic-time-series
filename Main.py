import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


from Data.Lorenz import generate_lorenz_data
from Model.esn import EchoStateNetwork

def main():
    print("1. Generating Lorenz system data...")
    # Generate 5000 steps of chaotic data
    data = generate_lorenz_data(num_steps=5000)
    
    print("2. Preprocessing data for time-series forecasting...")
    # To predict the future, X is the current state, and Y is the next state (shifted by 1)
    X = data[:-1]
    Y = data[1:]
    
    # Train/Test Split (80% train, 20% test)
    train_size = int(len(X) * 0.8)
    X_train, Y_train = X[:train_size], Y[:train_size]
    X_test, Y_test = X[train_size:], Y[train_size:]
    
    print(f"   Training samples: {len(X_train)}")
    print(f"   Testing samples: {len(X_test)}")

    print("3. Initializing and training the Echo State Network...")
    # Initialize hyperparameters
    model = EchoStateNetwork(
        input_size=3, 
        reservoir_size=1000,  
        spectral_radius=0.9, 
        sparsity=0.1,
        ridge_alpha=1e-4
    )
    
    # Train the model (this fits the W_out readout weights)
    model.fit(X_train, Y_train)
    print("   Training complete!")

    print("4. Generating autonomous predictions...")
    # Feed model the first point of the test set, and it predicts the rest autonomously
    initial_test_input = X_test[0]
    num_prediction_steps = len(X_test)
    
    predictions = model.predict(initial_input=initial_test_input, num_steps=num_prediction_steps)
    
    # Calculate Mean Squared Error
    mse = mean_squared_error(Y_test, predictions)
    print(f"   Prediction Mean Squared Error: {mse:.4f}")

    print("5. Plotting results...")
    plot_results(Y_test, predictions)


def plot_results(true_data: np.ndarray, predicted_data: np.ndarray):
    """Generates 2D and 3D plots comparing true vs predicted trajectories."""
    
    # --- Plot 1: 2D Time Series for X, Y, Z ---
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig.suptitle("Lorenz System: True vs Predicted (Over Time)", fontsize=16)
    
    labels = ['X', 'Y', 'Z']
    time_steps = np.arange(len(true_data))
    
    for i in range(3):
        axes[i].plot(time_steps, true_data[:, i], label="True", color='black', linewidth=1.5)
        axes[i].plot(time_steps, predicted_data[:, i], label="Predicted", color='red', linestyle='dashed', linewidth=1.5)
        axes[i].set_ylabel(labels[i])
        axes[i].legend(loc="upper right")
        axes[i].grid(True, alpha=0.3)
        
    axes[2].set_xlabel("Time Steps")
    plt.tight_layout()
    plt.show()

    # --- Plot 2: 3D Phase Space Attractor ---
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Lorenz Attractor in 3D Phase Space", fontsize=16)
    
    # Plot true trajectory
    ax.plot(true_data[:, 0], true_data[:, 1], true_data[:, 2], 
            label="True Trajectory", color='black', alpha=0.6, linewidth=1)
    
    # Plot predicted trajectory
    ax.plot(predicted_data[:, 0], predicted_data[:, 1], predicted_data[:, 2], 
            label="Predicted Trajectory", color='red', alpha=0.8, linewidth=1, linestyle='dashed')
    
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.show()

if __name__ == "__main__":
    main()