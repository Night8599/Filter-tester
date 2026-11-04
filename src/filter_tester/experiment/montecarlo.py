"""Monte Carlo: repeat a simulated experiment with different seeds and aggregate."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping

import numpy as np

from ..filters.base import Filter
from .runner import Runner, RunResult, Trajectory


@dataclass
class FilterStats:
    """Per-run statistics of one filter, kept for every Monte Carlo run.

    With ``N`` runs, ``K`` steps per run, ``M`` innovations per run and an
    ``n``-dimensional state:

    - ``nees``: ``(N, K)`` NEES at every step (``NaN`` where the truth is unknown);
    - ``nis``: ``(N, M)`` NIS of every innovation, at times ``nis_t`` ``(M,)``,
      from sensors ``nis_sensor`` ``(M,)`` with dimensions ``nis_dim`` ``(M,)``;
    - ``rmse``: ``(N, n)`` RMSE of each state component in each run;
    - ``rmse_total``: ``(N,)`` RMSE of the error vector in each run;
    - ``step_time``: ``(N,)`` mean wall-clock time per step in each run.
    """

    name: str
    nees: np.ndarray
    nis: np.ndarray
    nis_t: np.ndarray
    nis_sensor: np.ndarray
    nis_dim: np.ndarray
    rmse: np.ndarray
    rmse_total: np.ndarray
    step_time: np.ndarray


@dataclass
class MonteCarloResult:
    """Aggregated results of a Monte Carlo experiment.

    ``t`` are the step times (the same in every run), ``seeds`` the seed of
    each run, ``stats[name]`` the :class:`FilterStats` of each filter, and
    ``runs`` the full :class:`RunResult` of every run when ``keep_runs=True``
    (otherwise ``None``).
    """

    t: np.ndarray
    seeds: list[int]
    state_dim: int
    stats: dict[str, FilterStats]
    runs: list[RunResult] | None = None

    @property
    def n_runs(self) -> int:
        return len(self.seeds)

    @property
    def filter_names(self) -> list[str]:
        return list(self.stats)

    @property
    def has_truth(self) -> bool:
        return bool(np.isfinite(next(iter(self.stats.values())).nees).any())

    def __getitem__(self, name: str) -> FilterStats:
        return self.stats[name]


#: Called after every run with the run index, that run's result and the aggregate so far.
RunObserver = Callable[[int, RunResult, MonteCarloResult], None]


class MonteCarlo:
    """Repeats a simulated experiment ``runs`` times; run ``k`` uses seed ``first_seed + k``.

    ::

        mc = MonteCarlo(source, {"KF": kf}, runs=100)
        result = mc.run()                    # seeds 0 .. 99
        other = mc.run(first_seed=100)       # an independent experiment: seeds 100 .. 199
        result["KF"].nees                    # (100, K)

    Any single run can be reproduced with ``Runner(source, filters).run(rng=k)``.
    The source must be simulated (it needs a seed) and must produce the same
    step times in every run. By default only the statistics needed for the
    metrics and plots are kept; ``keep_runs=True`` also keeps every run's
    full :class:`RunResult`. ``observers`` are called after every run, e.g.
    by a live plot of the partial metrics.
    """

    def __init__(self, source, filters: Mapping[str, Filter], runs: int):
        if int(runs) < 1:
            raise ValueError("runs must be at least 1")
        self.runner = Runner(source, filters)
        self.runs = int(runs)

    def run(
        self,
        x0=None,
        keep_runs: bool = False,
        observers: Iterable[RunObserver] = (),
        first_seed: int = 0,
    ) -> MonteCarloResult:
        observers = list(observers)
        # The partial aggregate is only built when someone looks at it.
        for k, run, partial in self.iter_runs(x0, keep_runs, first_seed, build_partial=bool(observers)):
            for observer in observers:
                observer(k, run, partial)
        return self.last_result()

    def iter_runs(self, x0=None, keep_runs: bool = False, first_seed: int = 0, build_partial: bool = True):
        """Yield ``(k, run, partial aggregate)`` after each run, e.g. for an interactive UI.

        With ``build_partial=False`` the partial aggregate is ``None`` (cheaper);
        the final aggregate is then available as ``self.last_result()``.
        """
        acc = _Accumulator()
        kept: list[RunResult] | None = [] if keep_runs else None
        self._acc, self._kept = acc, kept
        for k in range(self.runs):
            seed = int(first_seed) + k
            run = self.runner.run(rng=seed, x0=x0)
            acc.add(seed, run)
            if kept is not None:
                kept.append(run)
            yield k, run, (acc.result(kept) if build_partial else None)

    def last_result(self) -> MonteCarloResult:
        """Aggregate of the runs done by the latest :meth:`run` / :meth:`iter_runs`."""
        return self._acc.result(self._kept)


@dataclass
class _Accumulator:
    seeds: list[int] = field(default_factory=list)
    t: np.ndarray | None = None
    rows: dict[str, dict[str, list]] = field(default_factory=dict)
    nis_layout: dict[str, tuple] = field(default_factory=dict)

    def add(self, seed: int, run: RunResult) -> None:
        t = np.array([s.t for s in run.steps])
        if self.t is None:
            self.t = t
        elif t.shape != self.t.shape or not np.allclose(t, self.t, rtol=0, atol=1e-9):
            raise ValueError("every Monte Carlo run must produce the same step times")
        for name, traj in run.trajectories.items():
            row = self.rows.setdefault(name, {k: [] for k in ("nees", "nis", "rmse", "rmse_total", "step_time")})
            nees, rmse, rmse_total = _truth_stats(traj)
            row["nees"].append(nees)
            row["rmse"].append(rmse)
            row["rmse_total"].append(rmse_total)
            row["step_time"].append(float(np.mean(traj.step_time)))
            layout = (
                np.array([i[0] for i in traj.innovations]),
                np.array([i[1] for i in traj.innovations], dtype=object),
                np.array([i[2].size for i in traj.innovations], dtype=int),
            )
            if name not in self.nis_layout:
                self.nis_layout[name] = layout
            elif not (
                layout[0].shape == self.nis_layout[name][0].shape
                and np.allclose(layout[0], self.nis_layout[name][0], rtol=0, atol=1e-9)
                and np.array_equal(layout[1], self.nis_layout[name][1])
            ):
                raise ValueError(f"filter {name!r} used different measurements in different runs")
            row["nis"].append(np.array([float(y @ np.linalg.solve(S, y)) for _, _, y, S in traj.innovations]))
        self.seeds.append(seed)

    def result(self, runs: list[RunResult] | None) -> MonteCarloResult:
        stats = {}
        for name, row in self.rows.items():
            nis_t, nis_sensor, nis_dim = self.nis_layout[name]
            stats[name] = FilterStats(
                name=name,
                nees=np.array(row["nees"]),
                nis=np.array(row["nis"]).reshape(len(self.seeds), nis_t.size),
                nis_t=nis_t,
                nis_sensor=nis_sensor,
                nis_dim=nis_dim,
                rmse=np.array(row["rmse"]),
                rmse_total=np.array(row["rmse_total"]),
                step_time=np.array(row["step_time"]),
            )
        n = next(iter(stats.values())).rmse.shape[1]
        return MonteCarloResult(
            t=self.t, seeds=list(self.seeds), state_dim=n, stats=stats,
            runs=None if runs is None else list(runs),
        )


def _truth_stats(traj: Trajectory):
    k, n = traj.mean.shape
    nees = np.full(k, np.nan)
    mask = traj.has_truth
    if not mask.any():
        return nees, np.full(n, np.nan), np.nan
    e = traj.error[mask]
    P = traj.covariance[mask]
    nees[mask] = np.einsum("ki,ki->k", e, np.linalg.solve(P, e[..., None])[..., 0])
    rmse = np.sqrt(np.mean(e**2, axis=0))
    rmse_total = float(np.sqrt(np.mean(np.sum(e**2, axis=1))))
    return nees, rmse, rmse_total
