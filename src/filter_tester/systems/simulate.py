"""Helper to run a true system over a time grid."""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import ArrayLike

from .base import RngLike, System


def simulate(
    system: System,
    times: ArrayLike,
    rng: RngLike = None,
    on_step: Callable[[float, np.ndarray], None] | None = None,
) -> np.ndarray:
    """Reset ``system`` at ``times[0]`` and return its states at every time.

    ``times`` must be non-decreasing. Returns an array of shape
    ``(len(times), n)`` whose first row is the initial state.

    If given, ``on_step(t, x)`` is called after every time point, including
    the initial one (used e.g. by live plots).
    """
    times = np.asarray(times, dtype=float).reshape(-1)
    if times.size == 0:
        raise ValueError("times must not be empty")
    if np.any(np.diff(times) < 0):
        raise ValueError("times must be non-decreasing")
    first = system.reset(times[0], rng)
    states = np.empty((times.size, first.size))
    states[0] = first
    if on_step is not None:
        on_step(times[0], first)
    for k in range(1, times.size):
        states[k] = system.advance(times[k])
        if on_step is not None:
            on_step(times[k], states[k].copy())
    return states
