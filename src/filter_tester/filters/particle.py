"""Particle filter (sequential importance resampling)."""

from __future__ import annotations

from typing import Mapping

import numpy as np
from numpy.typing import ArrayLike

from ..models.nonlinear import AnyMeasurementModel, AnyProcessModel, as_measurement_model, as_process_model
from .base import BayesFilter
from .kalman import _symmetric

RESAMPLERS = ("systematic", "multinomial")


class ParticleFilter(BayesFilter):
    """Bootstrap particle filter with adaptive resampling.

    The belief is a cloud of ``n_particles`` weighted samples, so it can take
    any shape. Each step:

    1. **predict**: every particle goes through ``f`` plus a sample of the
       process noise ``N(0, Q)``;
    2. **update**: weights are multiplied by the likelihood ``N(z; h(x), R)``
       of the measurement (computed in log space to avoid underflow);
    3. **resample** when the effective sample size ``1 / sum(w^2)`` drops
       below ``resample_threshold * n_particles`` (systematic resampling by default).

    Resampling duplicates good particles; with little process noise the copies
    stay identical and the cloud collapses (*sample impoverishment*), which
    can make the filter diverge. ``regularization > 0`` adds a small Gaussian
    jitter after each resampling (regularized particle filter), with covariance
    ``(regularization * h)^2 P``, where ``P`` is the cloud covariance and ``h``
    the optimal kernel bandwidth ``(4 / (N (n + 2)))^(1 / (n + 4))``.
    ``0`` gives the plain bootstrap filter; ``1`` uses the optimal bandwidth.

    The returned estimate is the weighted mean and covariance of the cloud.
    For NIS, the innovation is ``z - E[h(x)]`` with covariance ``Cov[h(x)] + R``
    over the predicted particles.

    The filter is random: ``seed`` makes it reproducible, and every
    :meth:`reset` restarts the same random sequence.
    """

    def __init__(
        self,
        process: AnyProcessModel,
        measurements: Mapping[str, AnyMeasurementModel],
        x0: ArrayLike,
        P0: ArrayLike,
        n_particles: int = 500,
        resample_threshold: float = 0.5,
        resampling: str = "systematic",
        regularization: float = 0.0,
        seed: int | None = 0,
        name: str | None = None,
    ):
        if int(n_particles) < 2:
            raise ValueError("n_particles must be at least 2")
        if not 0 <= resample_threshold <= 1:
            raise ValueError("resample_threshold must be between 0 and 1")
        if resampling not in RESAMPLERS:
            raise ValueError(f"resampling must be one of {RESAMPLERS}")
        if regularization < 0:
            raise ValueError("regularization must be >= 0")
        self.regularization = float(regularization)
        self.process = as_process_model(process)
        self.measurement_models = {s: as_measurement_model(m) for s, m in measurements.items()}
        self.n_particles = int(n_particles)
        self.resample_threshold = float(resample_threshold)
        self.resampling = resampling
        self.seed = seed
        self.resample_count = 0
        super().__init__(x0, P0, name)

    def reset(self) -> None:
        super().reset()
        self.rng = np.random.default_rng(self.seed)
        self.particles = self.rng.multivariate_normal(self.x0, self.P0, size=self.n_particles)
        self.log_weights = np.full(self.n_particles, -np.log(self.n_particles))
        self.resample_count = 0
        self._summarize()

    def knows_sensor(self, sensor: str) -> bool:
        return sensor in self.measurement_models

    @property
    def weights(self) -> np.ndarray:
        return np.exp(self.log_weights)

    @property
    def effective_sample_size(self) -> float:
        return float(1.0 / np.sum(self.weights**2))

    def predict(self, dt: float) -> None:
        X = self.process.propagate(self.particles, dt)
        Q = self.process.Q(dt)
        if np.any(Q):
            X = X + self.rng.multivariate_normal(np.zeros(X.shape[1]), Q, size=X.shape[0], method="eigh")
        self.particles = X
        self._summarize()

    def update(self, sensor: str, z: np.ndarray):
        model = self.measurement_models[sensor]
        if z.size != model.dim:
            raise ValueError(f"measurement from {sensor!r} has size {z.size}, its model expects {model.dim}")
        Z = model.measure_batch(self.particles)
        w = self.weights
        z_hat = w @ Z
        dZ = Z - z_hat
        S = _symmetric((w[:, None] * dZ).T @ dZ + model.R)
        # Gaussian log-likelihood of z for every particle (constants cancel on normalization).
        r = z - Z
        R_inv_r = np.linalg.solve(model.R, r.T).T
        log_lik = -0.5 * np.einsum("ij,ij->i", r, R_inv_r)
        self.log_weights = _normalize(self.log_weights + log_lik)
        if self.effective_sample_size < self.resample_threshold * self.n_particles:
            self._resample()
        self._summarize()
        return z - z_hat, S

    def _resample(self) -> None:
        w = self.weights
        N = self.n_particles
        if self.resampling == "systematic":
            positions = (self.rng.random() + np.arange(N)) / N
            idx = np.searchsorted(np.cumsum(w), positions)
        else:
            idx = self.rng.choice(N, size=N, p=w)
        cloud_cov = self.P
        self.particles = self.particles[np.minimum(idx, N - 1)]
        self.log_weights = np.full(N, -np.log(N))
        self.resample_count += 1
        if self.regularization > 0:
            d = self.particles.shape[1]
            h = self.regularization * (4.0 / (N * (d + 2))) ** (1.0 / (d + 4))
            self.particles = self.particles + self.rng.multivariate_normal(
                np.zeros(d), h**2 * cloud_cov, size=N, method="eigh"
            )

    def _summarize(self) -> None:
        w = self.weights
        self.x = w @ self.particles
        D = self.particles - self.x
        P = (w[:, None] * D).T @ D
        # A tiny floor keeps P invertible if the cloud collapses (e.g. no process noise).
        jitter = 1e-12 * max(1.0, float(np.trace(P)) / P.shape[0])
        self.P = _symmetric(P) + jitter * np.eye(P.shape[0])


def _normalize(log_w: np.ndarray) -> np.ndarray:
    m = np.max(log_w)
    if not np.isfinite(m):  # every particle has zero likelihood: fall back to uniform weights
        return np.full(log_w.size, -np.log(log_w.size))
    return log_w - (m + np.log(np.sum(np.exp(log_w - m))))
