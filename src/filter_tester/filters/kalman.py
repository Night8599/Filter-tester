"""Linear Kalman filter."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from numpy.typing import ArrayLike

from ..models.linear import LinearMeasurementModel, LinearProcessModel
from .base import BayesFilter


class KalmanFilter(BayesFilter):
    """Linear Kalman filter.

    ::

        kf = KalmanFilter(
            process=LinearProcessModel(A=lambda dt: ..., Q=lambda dt: ...),
            measurements={"gps": LinearMeasurementModel(H=[[1, 0]], R=[[0.25]])},
            x0=[0.0, 0.0],
            P0=np.eye(2),
        )

    ``measurements`` maps sensor names to their models; measurements from
    other sensors are ignored. Steps without measurements only predict.

    The covariance update uses the Joseph form, which keeps ``P`` symmetric
    and positive semi-definite despite rounding errors.
    """

    def __init__(
        self,
        process: LinearProcessModel,
        measurements: Mapping[str, LinearMeasurementModel],
        x0: ArrayLike,
        P0: ArrayLike,
        name: str | None = None,
    ):
        self.process = process
        self.measurement_models = dict(measurements)
        super().__init__(x0, P0, name)
        n = self.x0.size
        for sensor, model in self.measurement_models.items():
            if model.H.shape[1] != n:
                raise ValueError(f"H of {sensor!r} has {model.H.shape[1]} columns, the state has {n}")

    def knows_sensor(self, sensor: str) -> bool:
        return sensor in self.measurement_models

    def predict(self, dt: float) -> None:
        n = self.x.size
        A, Q = self.process.A(dt), self.process.Q(dt)
        if A.shape != (n, n) or Q.shape != (n, n):
            raise ValueError(f"A and Q must be {n}x{n}, got {A.shape} and {Q.shape}")
        self.x = A @ self.x
        self.P = _symmetric(A @ self.P @ A.T + Q)

    def update(self, sensor: str, z: np.ndarray):
        model = self.measurement_models[sensor]
        H, R = model.H, model.R
        if z.size != model.dim:
            raise ValueError(f"measurement from {sensor!r} has size {z.size}, its model expects {model.dim}")
        y = z - H @ self.x
        S = _symmetric(H @ self.P @ H.T + R)
        K = np.linalg.solve(S, H @ self.P).T  # = P H^T S^-1, without forming the inverse
        self.x = self.x + K @ y
        I_KH = np.eye(self.x.size) - K @ H
        self.P = _symmetric(I_KH @ self.P @ I_KH.T + K @ R @ K.T)
        return y, S


def _symmetric(M: np.ndarray) -> np.ndarray:
    return 0.5 * (M + M.T)
