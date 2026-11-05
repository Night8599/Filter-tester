"""Extended Kalman filter."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from numpy.typing import ArrayLike

from ..models.nonlinear import AnyMeasurementModel, AnyProcessModel, as_measurement_model, as_process_model
from .base import BayesFilter
from .kalman import _symmetric


class ExtendedKalmanFilter(BayesFilter):
    """Extended Kalman filter: a Kalman filter linearized around the current estimate.

    - **Predict**: ``x = f(x)``, ``P = F P F^T + Q`` with ``F = ∂f/∂x`` at ``x``.
    - **Update**: ``y = z - h(x)``, ``S = H P H^T + R`` with ``H = ∂h/∂x``,
      then the usual Kalman gain and a Joseph-form covariance update.

    Jacobians come from the models, or from finite differences when the
    models don't provide them. Linear models are accepted too (the EKF then
    equals the linear Kalman filter).
    """

    def __init__(
        self,
        process: AnyProcessModel,
        measurements: Mapping[str, AnyMeasurementModel],
        x0: ArrayLike,
        P0: ArrayLike,
        name: str | None = None,
    ):
        self.process = as_process_model(process)
        self.measurement_models = {s: as_measurement_model(m) for s, m in measurements.items()}
        super().__init__(x0, P0, name)

    def knows_sensor(self, sensor: str) -> bool:
        return sensor in self.measurement_models

    def predict(self, dt: float) -> None:
        F = self.process.jacobian(self.x, dt)
        n = self.x.size
        Q = self.process.Q(dt)
        if F.shape != (n, n) or Q.shape != (n, n):
            raise ValueError(f"F and Q must be {n}x{n}, got {F.shape} and {Q.shape}")
        self.x = self.process.f(self.x, dt)
        self.P = _symmetric(F @ self.P @ F.T + Q)

    def update(self, sensor: str, z: np.ndarray):
        model = self.measurement_models[sensor]
        if z.size != model.dim:
            raise ValueError(f"measurement from {sensor!r} has size {z.size}, its model expects {model.dim}")
        H = model.jacobian(self.x)
        y = z - model.h(self.x)
        S = _symmetric(H @ self.P @ H.T + model.R)
        K = np.linalg.solve(S, H @ self.P).T
        self.x = self.x + K @ y
        I_KH = np.eye(self.x.size) - K @ H
        self.P = _symmetric(I_KH @ self.P @ I_KH.T + K @ model.R @ K.T)
        return y, S
