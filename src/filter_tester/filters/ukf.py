"""Unscented Kalman filter."""

from __future__ import annotations

from typing import Mapping

import numpy as np
from numpy.typing import ArrayLike

from ..models.nonlinear import AnyMeasurementModel, AnyProcessModel, as_measurement_model, as_process_model
from .base import BayesFilter
from .kalman import _symmetric


class UnscentedKalmanFilter(BayesFilter):
    """Unscented Kalman filter with scaled sigma points (Van der Merwe).

    ``2n + 1`` sigma points with the current mean and covariance are pushed
    through the nonlinear ``f`` and ``h``; the predicted mean, covariance and
    cross-covariance are their weighted statistics. No Jacobians are needed.

    - ``alpha`` (0 < alpha <= 1) spreads the points around the mean;
    - ``beta`` encodes prior knowledge of the distribution (2 is optimal for Gaussians);
    - ``kappa`` is a secondary scaling parameter (usually 0).

    The defaults ``alpha=1, beta=2, kappa=0`` keep all weights well behaved.
    Small ``alpha`` (e.g. 1e-3) concentrates the points near the mean but
    gives a large negative central weight, which can make ``P`` lose
    positive-definiteness.
    """

    def __init__(
        self,
        process: AnyProcessModel,
        measurements: Mapping[str, AnyMeasurementModel],
        x0: ArrayLike,
        P0: ArrayLike,
        alpha: float = 1.0,
        beta: float = 2.0,
        kappa: float = 0.0,
        name: str | None = None,
    ):
        self.process = as_process_model(process)
        self.measurement_models = {s: as_measurement_model(m) for s, m in measurements.items()}
        n = np.asarray(x0).size
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        lam = alpha**2 * (n + kappa) - n
        if n + lam <= 0:
            raise ValueError("n + lambda must be positive; increase alpha or kappa")
        self.alpha, self.beta, self.kappa = float(alpha), float(beta), float(kappa)
        self._scale = n + lam
        self.Wm = np.full(2 * n + 1, 1.0 / (2 * self._scale))
        self.Wc = self.Wm.copy()
        self.Wm[0] = lam / self._scale
        self.Wc[0] = lam / self._scale + (1 - alpha**2 + beta)
        super().__init__(x0, P0, name)

    def knows_sensor(self, sensor: str) -> bool:
        return sensor in self.measurement_models

    def sigma_points(self, x: np.ndarray, P: np.ndarray) -> np.ndarray:
        """``(2n + 1, n)`` sigma points of ``N(x, P)``."""
        L = _sqrt_psd(self._scale * P)
        return np.vstack([x, x + L.T, x - L.T])

    def predict(self, dt: float) -> None:
        X = self.sigma_points(self.x, self.P)
        Y = self.process.propagate(X, dt)
        x = self.Wm @ Y
        D = Y - x
        self.x = x
        self.P = _symmetric((self.Wc[:, None] * D).T @ D + self.process.Q(dt))

    def update(self, sensor: str, z: np.ndarray):
        model = self.measurement_models[sensor]
        if z.size != model.dim:
            raise ValueError(f"measurement from {sensor!r} has size {z.size}, its model expects {model.dim}")
        X = self.sigma_points(self.x, self.P)
        Z = model.measure_batch(X)
        z_hat = self.Wm @ Z
        dZ = Z - z_hat
        dX = X - self.x
        S = _symmetric((self.Wc[:, None] * dZ).T @ dZ + model.R)
        Pxz = (self.Wc[:, None] * dX).T @ dZ
        K = np.linalg.solve(S, Pxz.T).T
        y = z - z_hat
        self.x = self.x + K @ y
        self.P = _symmetric(self.P - K @ S @ K.T)
        return y, S


def _sqrt_psd(M: np.ndarray) -> np.ndarray:
    """A matrix L with L @ L.T == M, for a positive semi-definite M."""
    try:
        return np.linalg.cholesky(M)
    except np.linalg.LinAlgError:
        # Fall back to the eigen decomposition, clipping tiny negative eigenvalues from rounding.
        w, V = np.linalg.eigh(0.5 * (M + M.T))
        return V @ np.diag(np.sqrt(np.clip(w, 0.0, None)))
