"""Noise distributions used by the true systems.

A noise source is anything that can draw a sample from a random generator.
Systems accept either a :class:`Noise` instance or a plain callable
``noise(rng) -> np.ndarray``, so custom distributions need no subclassing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Union

import numpy as np
from numpy.typing import ArrayLike

Sampler = Callable[[np.random.Generator], np.ndarray]


class Noise(ABC):
    """Base class for noise distributions."""

    @abstractmethod
    def sample(self, rng: np.random.Generator) -> np.ndarray:
        """Draw one sample."""

    def __call__(self, rng: np.random.Generator) -> np.ndarray:
        return self.sample(rng)


class GaussianNoise(Noise):
    """Multivariate normal noise ``N(mean, cov)``."""

    def __init__(self, cov: ArrayLike, mean: ArrayLike | None = None):
        cov = np.atleast_2d(np.asarray(cov, dtype=float))
        if cov.shape[0] != cov.shape[1]:
            raise ValueError(f"cov must be square, got shape {cov.shape}")
        if not np.allclose(cov, cov.T):
            raise ValueError("cov must be symmetric")
        dim = cov.shape[0]
        mean = np.zeros(dim) if mean is None else np.asarray(mean, dtype=float).reshape(-1)
        if mean.shape != (dim,):
            raise ValueError(f"mean must have shape ({dim},), got {mean.shape}")
        self.cov = cov
        self.mean = mean
        self.dim = dim

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        return rng.multivariate_normal(self.mean, self.cov)


class UniformNoise(Noise):
    """Independent uniform noise on ``[low, high]`` per component."""

    def __init__(self, low: ArrayLike, high: ArrayLike):
        low = np.asarray(low, dtype=float).reshape(-1)
        high = np.asarray(high, dtype=float).reshape(-1)
        if low.shape != high.shape:
            raise ValueError("low and high must have the same shape")
        if np.any(high < low):
            raise ValueError("high must be >= low")
        self.low = low
        self.high = high
        self.dim = low.size

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(self.low, self.high)


NoiseLike = Union[Noise, Sampler, None]


def as_sampler(noise: NoiseLike) -> Sampler | None:
    """Normalise a noise argument to a callable (or ``None`` for no noise)."""
    if noise is None:
        return None
    if callable(noise):
        return noise
    raise TypeError("noise must be a Noise instance, a callable rng -> array, or None")
