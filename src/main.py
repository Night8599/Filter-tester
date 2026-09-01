# Test linear system

from systems.linear import LinearSystem

import numpy as np

# Define a configuration for the linear system
config = {
    'linear_matrix': lambda t: np.array([[1, t], [0, 1]]),  # Example linear matrix as a function of time
    'standard_deviation': 0.1
}

# Create an instance of the LinearSystem
linear_system = LinearSystem(name="TestLinearSystem", config=config)

# Simulate the system for a range of time steps
time_steps = np.linspace(0, 10, num=100)  # Simulate
for t in time_steps:
    linear_system.iterate(t)
    print(f"Time: {t}, State: {linear_system.state_data['state']}")
    