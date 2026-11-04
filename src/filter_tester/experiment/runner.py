"""Run a data source against several filters and record what happened."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

from ..core.estimate import Estimate
from ..core.step import Step
from ..filters.base import Filter

#: Called after every step with the step and each filter's estimate.
Observer = Callable[[Step, Mapping[str, Estimate]], None]


@dataclass
class Trajectory:
    """Everything one filter produced during a run, as arrays.

    With ``k`` steps and an ``n``-dimensional state:

    - ``t``: ``(k,)`` step times;
    - ``mean``: ``(k, n)`` estimated states;
    - ``covariance``: ``(k, n, n)``;
    - ``truth``: ``(k, n)`` true states, with ``NaN`` rows where the truth is
      unknown, or ``None`` if the source has no truth at all;
    - ``innovations``: ``(t, sensor, y, S)`` for every measurement the
      filter used;
    - ``step_time``: ``(k,)`` wall-clock seconds the filter spent on each step.
    """

    name: str
    t: np.ndarray
    mean: np.ndarray
    covariance: np.ndarray
    truth: np.ndarray | None
    innovations: list[tuple[float, str, np.ndarray, np.ndarray]] = field(default_factory=list)
    step_time: np.ndarray = field(default_factory=lambda: np.empty(0))

    @property
    def std(self) -> np.ndarray:
        """``(k, n)`` standard deviation of each state component."""
        return np.sqrt(np.clip(np.diagonal(self.covariance, axis1=1, axis2=2), 0.0, None))

    @property
    def has_truth(self) -> np.ndarray:
        """``(k,)`` boolean mask of the steps whose truth is known."""
        if self.truth is None:
            return np.zeros(self.t.size, dtype=bool)
        return np.all(np.isfinite(self.truth), axis=1)

    @property
    def error(self) -> np.ndarray:
        """``(k, n)`` estimation error ``mean - truth`` (``NaN`` where truth is unknown)."""
        if self.truth is None:
            return np.full_like(self.mean, np.nan)
        return self.mean - self.truth


@dataclass
class RunResult:
    """The steps of a run and one :class:`Trajectory` per filter."""

    steps: list[Step]
    trajectories: dict[str, Trajectory]

    def __getitem__(self, name: str) -> Trajectory:
        return self.trajectories[name]

    @property
    def filter_names(self) -> list[str]:
        return list(self.trajectories)


class Runner:
    """Feeds every step of ``source`` to every filter, in time order.

    ::

        runner = Runner(source, {"KF": kf, "KF (wrong R)": kf_bad})
        result = runner.run(rng=0)          # rng / x0 are passed to simulated sources
        result["KF"].mean                    # (k, n)

    Filters are reset at the start of each run and see exactly the same
    steps. They receive only ``(t, measurements)``, never the truth.
    ``observers`` are called after every step with the step and all the
    estimates, e.g. a live plot.
    """

    def __init__(self, source, filters: Mapping[str, Filter]):
        if not filters:
            raise ValueError("give at least one filter")
        if len({id(f) for f in filters.values()}) != len(filters):
            raise ValueError("each filter must be a separate object (filters keep internal state)")
        self.source = source
        self.filters = dict(filters)

    def run(self, rng=None, x0=None, observers: Iterable[Observer] = ()) -> RunResult:
        """Run once and return the recorded :class:`RunResult`.

        ``rng`` and ``x0`` are forwarded to the source's ``steps()`` when
        given (simulated sources accept them; recorded data does not).
        """
        observers = list(observers)
        recorder = self.start(rng, x0)
        for step, estimates in recorder:
            for observer in observers:
                observer(step, estimates)
        return recorder.result()

    def start(self, rng=None, x0=None) -> "RunRecorder":
        """Start a run that is advanced step by step (e.g. by an interactive UI).

        Iterating the returned :class:`RunRecorder` processes one step at a
        time; :meth:`RunRecorder.result` gives the result of the steps so far.
        """
        kwargs = {k: v for k, v in (("rng", rng), ("x0", x0)) if v is not None}
        for f in self.filters.values():
            f.reset()
        return RunRecorder(self.source.steps(**kwargs), self.filters)


class RunRecorder:
    """A run in progress: feeds steps to the filters and records the estimates."""

    def __init__(self, steps: Iterable[Step], filters: Mapping[str, Filter]):
        self._steps = iter(steps)
        self.filters = filters
        self.steps: list[Step] = []
        self._records: dict[str, list] = {name: [] for name in filters}
        self.done = False

    def __iter__(self):
        return self

    def __next__(self) -> tuple[Step, dict[str, Estimate]]:
        """Process the next step; returns ``(step, {filter name: estimate})``."""
        try:
            step = next(self._steps)
        except StopIteration:
            self.done = True
            raise
        self.steps.append(step)
        estimates = {}
        for name, f in self.filters.items():
            start = time.perf_counter()
            estimate = f.step(step.t, step.measurements)
            self._records[name].append((estimate, time.perf_counter() - start))
            estimates[name] = estimate
        return step, estimates

    def result(self) -> RunResult:
        """The :class:`RunResult` of the steps processed so far."""
        if not self.steps:
            raise ValueError("the source produced no steps")
        truth = _truth_array(self.steps)
        return RunResult(
            steps=list(self.steps),
            trajectories={name: _trajectory(name, recs, truth) for name, recs in self._records.items()},
        )


def _truth_array(steps: Sequence[Step]) -> np.ndarray | None:
    known = [s.truth for s in steps if s.truth is not None]
    if not known:
        return None
    n = np.asarray(known[0]).size
    truth = np.full((len(steps), n), np.nan)
    for k, s in enumerate(steps):
        if s.truth is not None:
            truth[k] = s.truth
    return truth


def _trajectory(name: str, records, truth: np.ndarray | None) -> Trajectory:
    estimates = [e for e, _ in records]
    mean = np.array([e.mean for e in estimates])
    if truth is not None and truth.shape[1] != mean.shape[1]:
        raise ValueError(
            f"filter {name!r} estimates {mean.shape[1]} states but the truth has {truth.shape[1]}"
        )
    innovations = [
        (e.t, sensor, np.asarray(y), np.asarray(S))
        for e in estimates
        for sensor, (y, S) in e.extras.get("innovations", {}).items()
    ]
    return Trajectory(
        name=name,
        t=np.array([e.t for e in estimates]),
        mean=mean,
        covariance=np.array([e.covariance for e in estimates]),
        truth=truth,
        innovations=innovations,
        step_time=np.array([dt for _, dt in records]),
    )
