'''
A class representing a linear system.
'''

from systems.system import System

import numpy as np

class LinearSystem(System):
    def __init__(self, name=str, config=dict):
        super().__init__(name, config)
        self.linear_matrix = config.get('linear_matrix',)
        self.state_data['standard_deviation'] = config.get('standard_deviation', 0.0)

    def check_config(self, config):
        # Implement configuration validation logic specific to linear systems
        if config is None:
            raise ValueError("Configuration cannot be None.")
        if not isinstance(config, dict):
            raise TypeError("Configuration must be a dictionary.")
        
        required_keys = ['linear_matrix', 'standard_deviation']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required configuration key: {key}")

        # Check types of the matrices
        if not callable(config['linear_matrix']):  # Linear matrix should be a function that returns a numpy ndarray
            raise TypeError("linear_matrix must be a function that returns a numpy ndarray.")
        if not isinstance(config['standard_deviation'], (int, float)):
            raise TypeError("standard_deviation must be a number.")

    def iterate(self, t):
        # Update the state of the linear system based on the linear matrix and noise vector
        self.state_data['noiseless_state'] = self.linear_matrix(t)
        self.state_data['state'] = self.state_data['noiseless_state'] + np.random.normal(0, self.state_data['standard_deviation'], size=self.state_data['noiseless_state'].shape)
