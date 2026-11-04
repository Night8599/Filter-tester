"""Data source that simulates a true system observed by sensors."""

from __future__ import annotations

from typing import Iterator, Sequence

import numpy as np
from numpy.typing import ArrayLike

from ..core.step import Step
from ..sensors.sensor import Sensor
from ..systems.base import InitialState, RngLike, System

_MERGE_TOL = 1e-9  # relative tolerance for treating two times as the same instant


class SimulatedSource:
    """Runs a true system and its sensors and yields one :class:`Step` per instant.

    The system is evaluated at the union of:

    - ``times``: the truth grid, which also sets the start and end of the
      run. Make it finer than the sensors to see the truth in high
      resolution;
    - every measurement time of every sensor inside ``[times[0], times[-1]]``.

    Instants that coincide (within a small relative tolerance, so that e.g.
    ``0.1 + 0.2`` and ``0.3`` match) form a single step, holding the
    measurements of every sensor that measured then. Steps that come only
    from the truth grid have no measurements.

    Randomness is split into independent streams for the system and for
    each sensor. Switching one sensor's noise off therefore leaves the true
    trajectory and the other sensors' noise unchanged for the same seed.
    """

    def __init__(self, system: System, sensors: Sequence[Sensor], times: ArrayLike):
        sensors = list(sensors)
        names = [s.name for s in sensors]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(f"sensor names must be unique, repeated: {duplicates}")
        times = np.asarray(times, dtype=float).reshape(-1)
        if times.size == 0:
            raise ValueError("times must not be empty")
        if np.any(np.diff(times) < 0):
            raise ValueError("times must be non-decreasing")
        self.system = system
        self.sensors = sensors
        self.times = times

    @property
    def t_start(self) -> float:
        return float(self.times[0])

    @property
    def t_end(self) -> float:
        return float(self.times[-1])

    def schedule(self) -> list[tuple[float, list[Sensor]]]:
        """The instants of a run, each with the sensors that measure then."""
        events = [(t, None) for t in self.times]
        for sensor in self.sensors:
            events += [(t, sensor) for t in sensor.measurement_times(self.t_start, self.t_end)]
        events.sort(key=lambda e: e[0])

        merged: list[tuple[float, list[Sensor], bool]] = []  # (t, sensors, from truth grid)
        for t, sensor in events:
            if merged and _same_instant(merged[-1][0], t):
                t_kept, group, on_grid = merged[-1]
                if sensor is None and not on_grid:
                    t_kept, on_grid = t, True  # prefer the exact truth-grid time
                if sensor is not None and sensor not in group:
                    group.append(sensor)
                merged[-1] = (t_kept, group, on_grid)
            else:
                merged.append((t, [] if sensor is None else [sensor], sensor is None))
        return [(t, group) for t, group, _ in merged]

    def steps(self, rng: RngLike = None, x0: InitialState | None = None) -> Iterator[Step]:
        """Run the simulation lazily, one step at a time.

        ``x0`` overrides the system's initial state for this run.
        """
        system_seed, *sensor_seeds = _spawn(rng, 1 + len(self.sensors))
        sensor_rngs = {s.name: np.random.default_rng(seed) for s, seed in zip(self.sensors, sensor_seeds)}
        schedule = self.schedule()
        x = self.system.reset(schedule[0][0], np.random.default_rng(system_seed), x0=x0)
        for k, (t, group) in enumerate(schedule):
            if k > 0:
                x = self.system.advance(t)
            measurements = {s.name: s.measure(x, t, sensor_rngs[s.name]) for s in group}
            yield Step(t=t, measurements=measurements, truth=x.copy())

    def run(self, rng: RngLike = None, x0: InitialState | None = None) -> list[Step]:
        """Run the whole simulation and return all steps."""
        return list(self.steps(rng, x0))


def _same_instant(a: float, b: float) -> bool:
    return abs(a - b) <= _MERGE_TOL * max(1.0, abs(a), abs(b))


def _spawn(rng: RngLike, n: int) -> list[np.random.SeedSequence]:
    """Independent seed sequences derived from ``rng`` (a Generator, a seed or None)."""
    if isinstance(rng, np.random.Generator):
        rng = int(rng.integers(2**63))
    return list(np.random.SeedSequence(rng).spawn(n))
