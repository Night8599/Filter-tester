"""The unit of data that flows from a data source to the filters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class Step:
    """Everything known at one time instant.

    - ``t``: time.
    - ``measurements``: measurements taken at ``t``, keyed by sensor name.
      Empty when no sensor measured at ``t`` (e.g. a point of the truth grid).
      Several sensors measuring at the same instant share one step.
    - ``truth``: the true state at ``t``, or ``None`` when unknown (real data).
      Filters never see it; it is only used for metrics and plots.
    """

    t: float
    measurements: Mapping[str, np.ndarray] = field(default_factory=dict)
    truth: np.ndarray | None = None

    @property
    def has_measurements(self) -> bool:
        return bool(self.measurements)


def truth_series(steps: Sequence[Step]) -> tuple[np.ndarray, np.ndarray]:
    """``(times, states)`` of every step that has a true state."""
    known = [s for s in steps if s.truth is not None]
    if not known:
        return np.empty(0), np.empty((0, 0))
    return np.array([s.t for s in known]), np.array([s.truth for s in known])


def measurement_series(steps: Sequence[Step], sensor: str) -> tuple[np.ndarray, np.ndarray]:
    """``(times, measurements)`` of one sensor across all steps."""
    taken = [s for s in steps if sensor in s.measurements]
    if not taken:
        return np.empty(0), np.empty((0, 0))
    return np.array([s.t for s in taken]), np.array([s.measurements[sensor] for s in taken])
