"""Base classes for filters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

import numpy as np

from ..core.estimate import Estimate


class Filter(ABC):
    """Any filter: noisy measurements in, state estimate out.

    A filter is fed every step of a data source, in time order::

        filter.reset()
        for step in source.steps():
            estimate = filter.step(step.t, step.measurements)

    It only receives the time and the measurements, never the true state.
    ``measurements`` maps sensor names to measurement vectors and may be
    empty (a step without measurements).
    """

    def __init__(self, name: str | None = None):
        self.name = name or type(self).__name__

    @abstractmethod
    def reset(self) -> None:
        """Forget everything and start again from the initial estimate."""

    @abstractmethod
    def step(self, t: float, measurements: Mapping[str, np.ndarray]) -> Estimate:
        """Process one step and return the estimate at ``t``."""


class BayesFilter(Filter):
    """A filter with a predict / update cycle and an initial estimate.

    :meth:`step` works as follows:

    - **first step**: the estimate starts from ``x0``, ``P0`` *at that step's
      time* (no prediction);
    - **later steps**: :meth:`predict` to the new time (skipped if the time
      did not change);
    - then :meth:`update` once per measurement, in the order of the
      ``measurements`` mapping. Measurements from sensors the filter has no
      model for are ignored.

    Subclasses implement :meth:`predict`, :meth:`update` and
    :meth:`knows_sensor`, using ``self.x`` and ``self.P``.
    """

    def __init__(self, x0, P0, name: str | None = None):
        super().__init__(name)
        x0 = np.asarray(x0, dtype=float).reshape(-1)
        P0 = np.atleast_2d(np.asarray(P0, dtype=float))
        if P0.shape != (x0.size, x0.size):
            raise ValueError(f"P0 must have shape ({x0.size}, {x0.size}), got {P0.shape}")
        if not np.allclose(P0, P0.T):
            raise ValueError("P0 must be symmetric")
        self.x0 = x0
        self.P0 = P0
        self.reset()

    def reset(self) -> None:
        """Go back to the initial estimate ``x0``, ``P0``; the next step starts a new run."""
        self.x = self.x0.copy()
        self.P = self.P0.copy()
        self.t: float | None = None

    def step(self, t: float, measurements: Mapping[str, np.ndarray]) -> Estimate:
        """Predict to ``t``, update with each known measurement, return the estimate.

        The first step after :meth:`reset` does not predict: ``x0``, ``P0`` hold
        at that step's time. Measurements from unknown sensors are ignored.
        """
        t = float(t)
        if self.t is not None:
            if t < self.t:
                raise ValueError(f"time must not decrease (filter at t={self.t}, step at t={t})")
            if t > self.t:
                self.predict(t - self.t)
        self.t = t
        extras: dict[str, Any] = {"innovations": {}}
        for sensor, z in measurements.items():
            if self.knows_sensor(sensor):
                innovation = self.update(sensor, np.asarray(z, dtype=float).reshape(-1))
                if innovation is not None:
                    extras["innovations"][sensor] = innovation
        return Estimate(t=t, mean=self.x.copy(), covariance=self.P.copy(), extras=extras)

    @abstractmethod
    def knows_sensor(self, sensor: str) -> bool:
        """Whether the filter has a measurement model for ``sensor``."""

    @abstractmethod
    def predict(self, dt: float) -> None:
        """Propagate ``self.x`` and ``self.P`` forward by ``dt > 0``."""

    @abstractmethod
    def update(self, sensor: str, z: np.ndarray):
        """Correct ``self.x`` and ``self.P`` with measurement ``z`` from ``sensor``.

        May return ``(innovation, innovation_covariance)``, stored in the
        estimate's ``extras["innovations"][sensor]``.
        """
