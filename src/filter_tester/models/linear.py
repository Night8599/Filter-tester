"""Linear models used by filters (what the filter believes, not the true system)."""

from __future__ import annotations

from typing import Callable, Union

import numpy as np
from numpy.typing import ArrayLike

MatrixOrFn = Union[ArrayLike, Callable[[float], ArrayLike]]


class LinearProcessModel:
    """``x_next = A x + w``, ``w ~ N(0, Q)``.

    ``A`` and ``Q`` are either fixed matrices or functions of the time step
    ``dt``, e.g. a constant-velocity model valid for any ``dt``::

        LinearProcessModel(
            A=lambda dt: np.array([[1, dt], [0, 1]]),
            Q=lambda dt: q * np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]]),
        )

    Fixed matrices are used as given whatever ``dt`` is, so they suit a
    filter that runs on a fixed time step.
    """

    def __init__(self, A: MatrixOrFn, Q: MatrixOrFn):
        self._A = A
        self._Q = Q

    def A(self, dt: float) -> np.ndarray:
        return _square(self._A(dt) if callable(self._A) else self._A, "A")

    def Q(self, dt: float) -> np.ndarray:
        return _square(self._Q(dt) if callable(self._Q) else self._Q, "Q")


class LinearMeasurementModel:
    """``z = H x + v``, ``v ~ N(0, R)``, with fixed ``H`` and ``R``."""

    def __init__(self, H: ArrayLike, R: ArrayLike):
        H = np.atleast_2d(np.asarray(H, dtype=float))
        R = _square(R, "R")
        if R.shape[0] != H.shape[0]:
            raise ValueError(f"R must be {H.shape[0]}x{H.shape[0]} to match H, got {R.shape}")
        self.H = H
        self.R = R

    @property
    def dim(self) -> int:
        """Dimension of a measurement."""
        return self.H.shape[0]


def _square(M: ArrayLike, name: str) -> np.ndarray:
    M = np.atleast_2d(np.asarray(M, dtype=float))
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f"{name} must be a square matrix, got shape {M.shape}")
    return M
