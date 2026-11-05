"""Noise distributions used by the true systems.

A noise source is anything that can draw a sample from a random generator.
Systems accept either a :class:`Noise` instance or a plain callable
``noise(rng) -> np.ndarray``, so custom distributions need no subclassing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

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
        """One sample of ``N(mean, cov)``."""
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
        """One sample, uniform on ``[low, high]`` per component."""
        return rng.uniform(self.low, self.high)


NoiseLike = Noise | Sampler | None


def as_sampler(noise: NoiseLike) -> Sampler | None:
    """Normalise a noise argument to a callable (or ``None`` for no noise)."""
    if noise is None:
        return None
    if callable(noise):
        return noise
    raise TypeError("noise must be a Noise instance, a callable rng -> array, or None")


class NoiseTerm:
    """A noise source that can be switched off.

    Used by systems and sensors. :meth:`draw` returns:

    - ``None`` if no noise was given at all;
    - a fresh sample if noise is enabled;
    - a zero vector of the sample's shape if noise is disabled, so model
      functions such as ``A @ x + w`` keep working and produce the
      noiseless (nominal) value.
    """

    def __init__(self, noise: NoiseLike):
        self.sampler = as_sampler(noise)
        self._zero: np.ndarray | None = None

    def draw(self, rng: np.random.Generator | None, enabled: bool = True) -> np.ndarray | None:
        """A noise sample, zeros of the same shape if disabled, or ``None`` if there is no noise."""
        if self.sampler is None:
            return None
        if enabled:
            if rng is None:
                raise ValueError("a random generator is required to draw noise")
            return np.asarray(self.sampler(rng), dtype=float)
        return self.zero()

    def zero(self) -> np.ndarray:
        """Zero vector with the shape of one noise sample."""
        if self._zero is None:
            dim = getattr(self.sampler, "dim", None)
            if dim is not None:
                self._zero = np.zeros(dim)
            else:
                # Learn the shape from a throwaway generator, leaving the run's rng untouched.
                self._zero = np.zeros_like(np.asarray(self.sampler(np.random.default_rng(0)), dtype=float))
        return self._zero.copy()
