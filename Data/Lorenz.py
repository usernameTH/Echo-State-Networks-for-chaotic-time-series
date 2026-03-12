import numpy as np
from scipy.integrate import solve_ivp
from typing import Tuple, List

def generate_lorenz_data(
    sigma: float = 10.0, 
    rho: float = 28.0, 
    beta: float = 8.0 / 3.0,
    initial_state: List[float] = [1.0, 1.0, 1.0],
    t_span: Tuple[float, float] = (0.0, 50.0),
    num_steps: int = 5000
) -> np.ndarray:
    """
    Generates time-series data from the chaotic Lorenz system.
    
    Args:
        sigma (float): System parameter, proportional to Prandtl number.
        rho (float): System parameter, proportional to Rayleigh number.
        beta (float): System parameter, related to physical dimensions.
        initial_state (List[float]): Starting coordinates [x, y, z].
        t_span (Tuple[float, float]): Start and end time for the simulation.
        num_steps (int): Number of time steps to generate.
        
    Returns:
        np.ndarray: An array of shape (num_steps, 3) representing the trajectory.
    """
    def lorenz_deriv(t, state):
        x, y, z = state
        dxdt = sigma * (y - x)
        dydt = x * (rho - z) - y
        dzdt = x * y - beta * z
        return [dxdt, dydt, dzdt]

    t_eval = np.linspace(t_span[0], t_span[1], num_steps)
    
    # Solve the Initial Value Problem (IVP)
    solution = solve_ivp(lorenz_deriv, t_span, initial_state, t_eval=t_eval)
    
    # Transpose so the shape is (time_steps, features)
    return solution.y.T