'''
Kalman Filter implementation.
'''

from filters.filter import Filter

import numpy as np

class KalmanFilter(Filter):
    def __init__(self, name="KalmanFilter", config=None):
        super().__init__(name, config)
        # Initialize Kalman filter parameters here
        self.state_matrix = config.get('initial_state_matrix')
        self.covariance_matrix = config.get('initial_covariance_matrix')
        self.process_noise_covariance = config.get('process_noise_covariance')


    def check_config(self, config):
        # Implement configuration validation logic specific to Kalman filter
        required_keys = ['initial_state_matrix', 'initial_covariance_matrix', 'process_noise_covariance']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required configuration key: {key}")

        # Check types of the matrices
        if not isinstance(config['initial_state_matrix'], np.ndarray):
            raise TypeError("initial_state_matrix must be a numpy ndarray.")
        if not isinstance(config['initial_covariance_matrix'], np.ndarray):
            raise TypeError("initial_covariance_matrix must be a numpy ndarray.")
        if not isinstance(config['process_noise_covariance'], np.ndarray):
            raise TypeError("process_noise_covariance must be a numpy ndarray.")

    def iterate(self, data=None):

        if data is None:
            raise ValueError("Data is required for the Kalman filter iteration.")

        # Possible key checks for data structure can be added here
        if 'dt' not in data:
            raise ValueError("Data must contain 'dt' for time step.")
        
        # Check if control input is provided; if not, use a default value
        if 'control_input' not in data:
            self.predict(data['dt'])
        else:
            self.predict(data['dt'], data['control_input'])

        # If measurement data is provided, perform the update step
        if 'measurement' in data and 'measurement_matrix' in data and 'measurement_noise_covariance' in data:
            self.update(data['measurement'], data['measurement_matrix'], data['measurement_noise_covariance'])

        self.state_estimate = self.state_matrix  # Update the state estimate after iteration

    def predict(self, dt, control_input=None):
        # Implement the prediction step of the Kalman filter
        if control_input is None:
            control_input = np.zeros((2, 2))

        # Function lambda for A and B matrices
        A = lambda dt: np.array([[1, 0, dt, 0],
                                  [0, 1, 0, dt],
                                  [0, 0, 1, 0],
                                  [0, 0, 0, 1]])
        B = lambda dt: np.array([[0.5 * dt**2, 0],
                                  [0, 0.5 * dt**2],
                                  [0, 0],
                                  [0, 0]])
        self.state_matrix = A(dt) @ self.state_matrix + B(dt) @ control_input
        self.covariance_matrix = A(dt) @ self.covariance_matrix @ A(dt).T + self.process_noise_covariance

    def update(self, measurement, measurement_matrix, measurement_noise_covariance):
        if measurement is None:
            raise ValueError("Measurement data is required for the update step.")
        if measurement_matrix is None:
            raise ValueError("Measurement matrix is required for the update step.")
        if measurement_noise_covariance is None:
            raise ValueError("Measurement noise covariance is required for the update step.")
        
        # Implement the update step of the Kalman filter
        y = measurement - measurement_matrix @ self.state_matrix  # Measurement residual
        S = measurement_matrix @ self.covariance_matrix @ measurement_matrix.T + measurement_noise_covariance  # Residual covariance
        K = self.covariance_matrix @ measurement_matrix.T @ np.linalg.inv(S)  # Kalman gain

        self.state_matrix = self.state_matrix + K @ y  # Update state estimate
        I = np.eye(self.covariance_matrix.shape[0])  # Identity matrix
        self.covariance_matrix = (I - K @ measurement_matrix) @ self.covariance_matrix @ (I - K @ measurement_matrix).T  + K @ measurement_noise_covariance @ K.T  # Update covariance estimate (Joseph form)
