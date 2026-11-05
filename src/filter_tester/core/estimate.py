"""The output of a filter at one time instant."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Estimate:
    """A filter's estimate at time ``t``.

    - ``mean``: estimated state, shape ``(n,)``.
    - ``covariance``: the filter's uncertainty about ``mean``, shape ``(n, n)``.
    - ``extras``: optional filter-specific information. Kalman-type filters
      store the innovation of each sensor used at ``t`` under
      ``extras["innovations"][sensor] = (y, S)`` (residual and its covariance),
      which metrics such as NIS can use.
    """

    t: float
    mean: np.ndarray
    covariance: np.ndarray
    extras: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        mean = np.asarray(self.mean, dtype=float).reshape(-1)
        cov = np.atleast_2d(np.asarray(self.covariance, dtype=float))
        if cov.shape != (mean.size, mean.size):
            raise ValueError(f"covariance must have shape ({mean.size}, {mean.size}), got {cov.shape}")
        object.__setattr__(self, "t", float(self.t))
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "covariance", cov)

    @property
    def std(self) -> np.ndarray:
        """Standard deviation of each state component (sqrt of the covariance diagonal)."""
        return np.sqrt(np.clip(np.diag(self.covariance), 0.0, None))
