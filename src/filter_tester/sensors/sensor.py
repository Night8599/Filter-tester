"""Sensors: turn the true state into (noisy) measurements."""

from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np
from numpy.typing import ArrayLike

from ..noise import NoiseLike, NoiseTerm

MeasurementFn = Callable[[np.ndarray, "np.ndarray | None", float], ArrayLike]

_TIME_TOL = 1e-9  # relative tolerance when matching times on a periodic grid


class Sensor:
    """A sensor producing measurements ``z = h(x, v, t)``.

    ``name`` is required and identifies the sensor's measurements in each
    step, so it must be unique within a data source.

    ``v`` is a fresh noise sample per measurement (``None`` if no noise is
    given, zeros if noise is disabled), and may enter ``h`` in any way.
    Additive Gaussian noise on the position of ``[position, velocity]``::

        Sensor("gps", h=lambda x, v, t: x[:1] + v, noise=GaussianNoise([[0.25]]), period=0.1)

    When the sensor measures is defined in exactly one of two ways:

    - ``period``: measurements at ``offset + k * period`` for every integer
      ``k`` (an absolute grid, independent of when a run starts);
    - ``times``: an explicit list of measurement times.

    Noise can be switched off with ``sensor.noise_enabled = False`` (or
    ``noise_enabled=False`` in the constructor).
    """

    def __init__(
        self,
        name: str,
        h: MeasurementFn,
        noise: NoiseLike = None,
        period: float | None = None,
        times: ArrayLike | None = None,
        offset: float = 0.0,
        measurement_names: Sequence[str] | None = None,
        noise_enabled: bool = True,
    ):
        if not isinstance(name, str) or not name.strip():
            raise ValueError("name must be a non-empty string")
        if not callable(h):
            raise TypeError("h must be callable as h(x, v, t)")
        if (period is None) == (times is None):
            raise ValueError("give exactly one of period or times")
        if period is not None:
            if not period > 0:
                raise ValueError("period must be positive")
            period = float(period)
        if times is not None:
            times = np.asarray(times, dtype=float).reshape(-1)
            if np.any(np.diff(times) <= 0):
                raise ValueError("times must be strictly increasing")
            if offset != 0.0:
                raise ValueError("offset only applies to period")
        self.h = h
        self.period = period
        self.times = times
        self.offset = float(offset)
        self.name = name
        self.measurement_names = None if measurement_names is None else [str(n) for n in measurement_names]
        self.noise_enabled = bool(noise_enabled)
        self._noise = NoiseTerm(noise)

    def measurement_times(self, t_start: float, t_end: float) -> np.ndarray:
        """All measurement times in the closed interval ``[t_start, t_end]``."""
        if t_end < t_start:
            raise ValueError("t_end must be >= t_start")
        if self.times is not None:
            return self.times[(self.times >= t_start) & (self.times <= t_end)].copy()
        p, o = self.period, self.offset
        tol = _TIME_TOL * max(1.0, abs(t_start), abs(t_end)) / p
        k_first = math.ceil((t_start - o) / p - tol)
        k_last = math.floor((t_end - o) / p + tol)
        if k_last < k_first:
            return np.empty(0)
        # Multiplying (instead of accumulating period) avoids drift on long runs.
        return o + p * np.arange(k_first, k_last + 1, dtype=float)

    def measure(self, x: ArrayLike, t: float, rng: np.random.Generator | None = None) -> np.ndarray:
        """Return the measurement of true state ``x`` at time ``t``.

        ``rng`` is needed only when noise is drawn (noise given and enabled).
        """
        x = np.asarray(x, dtype=float).reshape(-1)
        v = self._noise.draw(rng, self.noise_enabled)
        z = np.atleast_1d(np.asarray(self.h(x, v, float(t)), dtype=float)).reshape(-1)
        if self.measurement_names is not None and len(self.measurement_names) != z.size:
            raise ValueError(
                f"{len(self.measurement_names)} measurement_names given for a measurement of dimension {z.size}"
            )
        return z
