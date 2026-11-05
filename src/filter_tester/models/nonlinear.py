"""Nonlinear models used by filters (EKF, UKF, particle filter).

The models have additive Gaussian noise:

- process: ``x_next = f(x, dt) + w``, ``w ~ N(0, Q(dt))``;
- measurement: ``z = h(x) + v``, ``v ~ N(0, R)``.

Jacobians are optional: when not given they are computed by central finite
differences. Models marked ``vectorized=True`` accept a batch of states of
shape ``(N, n)`` (as well as a single ``(n,)`` state), which makes the UKF and
especially the particle filter much faster.

Linear models (:class:`LinearProcessModel`, :class:`LinearMeasurementModel`)
can be used wherever a nonlinear model is expected; :func:`as_process_model`
and :func:`as_measurement_model` convert them.
"""

from __future__ import annotations

from typing import Callable, Union

import numpy as np
from numpy.typing import ArrayLike

from .linear import LinearMeasurementModel, LinearProcessModel, MatrixOrFn, _square


class ProcessModel:
    """``x_next = f(x, dt) + w`` with ``w ~ N(0, Q(dt))``.

    - ``f(x, dt)``: noise-free transition.
    - ``Q``: a fixed matrix or a function of ``dt``.
    - ``F(x, dt)``: optional Jacobian ``∂f/∂x``; finite differences otherwise.
    - ``vectorized``: ``f`` also accepts a batch ``(N, n)`` and returns ``(N, n)``.
    """

    def __init__(
        self,
        f: Callable[[np.ndarray, float], ArrayLike],
        Q: MatrixOrFn,
        F: Callable[[np.ndarray, float], ArrayLike] | None = None,
        vectorized: bool = False,
    ):
        if not callable(f):
            raise TypeError("f must be callable as f(x, dt)")
        self._f = f
        self._Q = Q
        self._F = F
        self.vectorized = bool(vectorized)

    def f(self, x: np.ndarray, dt: float) -> np.ndarray:
        """Transition of one state ``(n,)``."""
        if self.vectorized:
            return np.asarray(self._f(x[None, :], dt), dtype=float).reshape(-1)
        return np.asarray(self._f(x, dt), dtype=float).reshape(-1)

    def propagate(self, X: np.ndarray, dt: float) -> np.ndarray:
        """Transition of a batch of states ``(N, n)``."""
        if self.vectorized:
            return np.asarray(self._f(X, dt), dtype=float).reshape(X.shape)
        return np.array([self.f(x, dt) for x in X])

    def Q(self, dt: float) -> np.ndarray:
        return _square(self._Q(dt) if callable(self._Q) else self._Q, "Q")

    def jacobian(self, x: np.ndarray, dt: float) -> np.ndarray:
        if self._F is not None:
            return np.atleast_2d(np.asarray(self._F(x, dt), dtype=float))
        if self.vectorized:  # all perturbed states in one call
            return _batch_jacobian(lambda X: self.propagate(X, dt), x)
        return numerical_jacobian(lambda s: self.f(s, dt), x)


class MeasurementModel:
    """``z = h(x) + v`` with ``v ~ N(0, R)``.

    - ``h(x)``: noise-free measurement.
    - ``H(x)``: optional Jacobian ``∂h/∂x``; finite differences otherwise.
    - ``vectorized``: ``h`` also accepts a batch ``(N, n)`` and returns ``(N, m)``.
    """

    def __init__(
        self,
        h: Callable[[np.ndarray], ArrayLike],
        R: ArrayLike,
        H: Callable[[np.ndarray], ArrayLike] | None = None,
        vectorized: bool = False,
    ):
        if not callable(h):
            raise TypeError("h must be callable as h(x)")
        self._h = h
        self.R = _square(R, "R")
        self._H = H
        self.vectorized = bool(vectorized)

    @property
    def dim(self) -> int:
        return self.R.shape[0]

    def h(self, x: np.ndarray) -> np.ndarray:
        if self.vectorized:
            z = np.asarray(self._h(x[None, :]), dtype=float).reshape(-1)
        else:
            z = np.atleast_1d(np.asarray(self._h(x), dtype=float)).reshape(-1)
        if z.size != self.dim:
            raise ValueError(f"h returned {z.size} values but R is {self.dim}x{self.dim}")
        return z

    def measure_batch(self, X: np.ndarray) -> np.ndarray:
        """Noise-free measurements of a batch of states, shape ``(N, m)``."""
        if self.vectorized:
            return np.asarray(self._h(X), dtype=float).reshape(X.shape[0], self.dim)
        return np.array([self.h(x) for x in X])

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        if self._H is not None:
            return np.atleast_2d(np.asarray(self._H(x), dtype=float))
        if self.vectorized:
            return _batch_jacobian(self.measure_batch, x)
        return numerical_jacobian(self.h, x)


AnyProcessModel = Union[ProcessModel, LinearProcessModel]
AnyMeasurementModel = Union[MeasurementModel, LinearMeasurementModel]


def as_process_model(model: AnyProcessModel) -> ProcessModel:
    """A :class:`ProcessModel` (linear models are wrapped, with their exact Jacobian)."""
    if isinstance(model, ProcessModel):
        return model
    if isinstance(model, LinearProcessModel):
        return ProcessModel(
            f=lambda X, dt: X @ model.A(dt).T,
            Q=model.Q,
            F=lambda x, dt: model.A(dt),
            vectorized=True,
        )
    raise TypeError(f"expected a process model, got {type(model).__name__}")


def as_measurement_model(model: AnyMeasurementModel) -> MeasurementModel:
    """A :class:`MeasurementModel` (linear models are wrapped, with their exact Jacobian)."""
    if isinstance(model, MeasurementModel):
        return model
    if isinstance(model, LinearMeasurementModel):
        H = model.H
        return MeasurementModel(h=lambda X: X @ H.T, R=model.R, H=lambda x: H, vectorized=True)
    raise TypeError(f"expected a measurement model, got {type(model).__name__}")


def numerical_jacobian(fn: Callable[[np.ndarray], np.ndarray], x: np.ndarray) -> np.ndarray:
    """Jacobian of ``fn`` at ``x`` by central finite differences."""
    x = np.asarray(x, dtype=float)
    y0 = np.asarray(fn(x), dtype=float).reshape(-1)
    J = np.empty((y0.size, x.size))
    steps = _steps(x)
    for i in range(x.size):
        h = steps[i]
        e = np.zeros_like(x)
        e[i] = h
        J[:, i] = (np.asarray(fn(x + e), dtype=float).reshape(-1) - np.asarray(fn(x - e), dtype=float).reshape(-1)) / (2 * h)
    return J


def _steps(x: np.ndarray) -> np.ndarray:
    return 1e-6 * np.maximum(1.0, np.abs(x))


def _batch_jacobian(fn_batch: Callable[[np.ndarray], np.ndarray], x: np.ndarray) -> np.ndarray:
    """Central-difference Jacobian evaluating all ``2n`` perturbed states in one batch call."""
    x = np.asarray(x, dtype=float)
    h = _steps(x)
    E = np.diag(h)
    Y = np.asarray(fn_batch(np.vstack([x + E, x - E])), dtype=float)
    n = x.size
    return ((Y[:n] - Y[n:]) / (2 * h[:, None])).T


def rk4(deriv: Callable[[np.ndarray], np.ndarray], max_step: float = 0.01):
    """Build a transition ``f(x, dt)`` that integrates ``dx/dt = deriv(x)`` with RK4.

    ``deriv`` works on single states or batches; the interval is split into
    equal sub-steps of at most ``max_step``. Useful for continuous-time models.
    """

    def f(x, dt):
        x = np.asarray(x, dtype=float)
        n = max(1, int(np.ceil(dt / max_step - 1e-12)))
        h = dt / n
        for _ in range(n):
            k1 = deriv(x)
            k2 = deriv(x + 0.5 * h * k1)
            k3 = deriv(x + 0.5 * h * k2)
            k4 = deriv(x + h * k3)
            x = x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return x

    return f
