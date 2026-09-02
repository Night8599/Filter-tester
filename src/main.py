# Test linear system

from systems.linear import LinearSystem

import numpy as np
import matplotlib.pyplot as plt

# Define a configuration for the linear system
config = {
    'linear_matrix': lambda t: np.array([[t], [2*t]]),  # Example linear matrix as a function of time
    'standard_deviation': 1.0
}

# Create an instance of the LinearSystem
linear_system = LinearSystem(name="TestLinearSystem", config=config)

# Plot the linear system's response over time
time_steps = np.linspace(0, 10, 100)

for t in time_steps:
    linear_system.iterate(t)
    state = linear_system.state_data['state']
    plt.plot(t, state[0], 'bo')
    plt.plot(t, state[1], 'ro')

plt.xlabel("Time")
plt.ylabel("State")
plt.title("Linear System Response")
plt.show()
