"""Plots of Monte Carlo results.

- :func:`plot_mc_consistency`: NEES and NIS averaged over the runs at each
  instant, with the chi-square bounds (:func:`plot_mc_nees`, :func:`plot_mc_nis`).
- :func:`plot_rmse_boxplot`: distribution of the per-run RMSE of each filter.
- :func:`plot_mc_summary_table`: the Monte Carlo metrics table as a figure.
- :class:`LiveMonteCarloPlot`: the same metrics updating as runs finish.

Every filter keeps the same color in every figure.
"""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from ..experiment.montecarlo import MonteCarloResult
from ..experiment.runner import RunResult
from ..metrics.montecarlo import mc_nees, mc_nis, mc_summary_table
from ._live import LiveFigure
from .filters import draw_table
from .systems import resolve_labels


def plot_mc_nees(result: MonteCarloResult, ax=None, filters=None, confidence: float = 0.95, log: bool = True):
    """Run-averaged NEES over time for each filter, with the chi-square bounds."""
    ax = ax if ax is not None else plt.subplots(figsize=(8, 3.5))[1]
    named = _named_colors(result, filters)
    for color, name in named:
        t, mean, lower, upper = mc_nees(result, name, confidence)
        ax.plot(t, mean, color=color, lw=1.1, label=name)
    _bounds(ax, t, lower, upper, result.state_dim, confidence)  # same for every filter
    _finish(ax, f"NEES averaged over {result.n_runs} runs", "NEES", log)
    return ax


def plot_mc_nis(result: MonteCarloResult, ax=None, filters=None, confidence: float = 0.95, log: bool = True):
    """Run-averaged NIS at each innovation for each filter, with the chi-square bounds.

    When filters use measurements of different dimensions, each filter's
    bounds are drawn in its own color.
    """
    ax = ax if ax is not None else plt.subplots(figsize=(8, 3.5))[1]
    named = _named_colors(result, filters)
    dims = {name: tuple(result[name].nis_dim) for _, name in named}
    shared = len(set(dims.values())) == 1 and len(set(next(iter(dims.values())))) == 1
    for color, name in named:
        t, mean, lower, upper = mc_nis(result, name, confidence)
        ax.plot(t, mean, ".", color=color, ms=3, label=name)
        if not shared:
            ax.plot(t, lower, "--", color=color, lw=0.8)
            ax.plot(t, upper, "--", color=color, lw=0.8)
    if shared:
        first = named[0][1]
        t, _, lower, upper = mc_nis(result, first, confidence)
        _bounds(ax, t, lower, upper, result[first].nis_dim[0], confidence)
    _finish(ax, f"NIS averaged over {result.n_runs} runs", "NIS", log)
    return ax


def plot_mc_consistency(result: MonteCarloResult, filters=None, confidence: float = 0.95, log: bool = True):
    """NEES (if the truth is known) and NIS over time in one figure. Returns ``(fig, axes)``."""
    panels = (["NEES"] if result.has_truth else []) + ["NIS"]
    fig, axes = plt.subplots(len(panels), 1, sharex=True, squeeze=False, figsize=(8, 1.2 + 2.8 * len(panels)))
    axes = list(axes[:, 0])
    for ax, panel in zip(axes, panels):
        draw = plot_mc_nees if panel == "NEES" else plot_mc_nis
        draw(result, ax=ax, filters=filters, confidence=confidence, log=log)
    axes[-1].set_xlabel("t")
    fig.suptitle(f"Consistency over {result.n_runs} Monte Carlo runs ({confidence:.0%} bounds)")
    fig.tight_layout()
    return fig, axes


def plot_rmse_boxplot(result: MonteCarloResult, labels: Sequence[str] | None = None, filters=None):
    """One panel per state component plus the total: boxplot of the per-run RMSE of each filter.

    Returns ``(fig, axes)``.
    """
    if not result.has_truth:
        raise ValueError("plot_rmse_boxplot needs the ground truth")
    n = result.state_dim
    titles = [f"RMSE {l}" for l in resolve_labels(n, labels)] + ["RMSE total"]
    named = _named_colors(result, filters)
    fig, axes = plt.subplots(1, n + 1, squeeze=False, figsize=(2.2 + 2.4 * (n + 1), 3.8))
    axes = list(axes[0])
    for j, (ax, title) in enumerate(zip(axes, titles)):
        data = [result[name].rmse[:, j] if j < n else result[name].rmse_total for _, name in named]
        _boxplot(ax, data, named)
        ax.set_title(title)
    fig.suptitle(f"Per-run RMSE over {result.n_runs} Monte Carlo runs")
    fig.tight_layout()
    return fig, axes


def plot_mc_summary_table(
    result: MonteCarloResult, labels: Sequence[str] | None = None, confidence: float = 0.95, title=None
):
    """Draw the Monte Carlo metrics table. Returns ``(fig, ax)``."""
    header, body = mc_summary_table(result, list(labels) if labels else None, confidence)
    return draw_table(header, body, title or f"Filter comparison — {result.n_runs} Monte Carlo runs")


class LiveMonteCarloPlot(LiveFigure):
    """Partial Monte Carlo metrics, updated as runs finish.

    Pass it to :meth:`MonteCarlo.run` as an observer::

        live = LiveMonteCarloPlot()
        result = MonteCarlo(source, filters, runs=200).run(observers=[live])
        live.finish()

    It shows the run-averaged NEES and NIS so far with their bounds (which
    tighten as runs accumulate) and a boxplot of the per-run total RMSE.
    """

    def __init__(self, confidence: float = 0.95, log: bool = True, refresh_interval: float = 0.25):
        super().__init__(refresh_interval, speed=None)
        self.confidence = confidence
        self.log = log
        self._latest: MonteCarloResult | None = None

    def __call__(self, k: int, run: RunResult, partial: MonteCarloResult) -> None:
        self._check_open()
        if self.fig is None:
            panels = 3 if partial.has_truth else 1
            self.fig = plt.figure(figsize=(11, 2.0 + 2.4 * min(panels, 2)))
            if partial.has_truth:
                grid = self.fig.add_gridspec(2, 3)
                self.axes = {
                    "nees": self.fig.add_subplot(grid[0, :2]),
                    "nis": self.fig.add_subplot(grid[1, :2]),
                    "rmse": self.fig.add_subplot(grid[:, 2]),
                }
            else:
                self.axes = {"nis": self.fig.add_subplot(1, 1, 1)}
        self._latest = partial
        self._tick(float(k))

    def _redraw(self) -> None:
        result = self._latest
        if result is None:
            return
        for ax in self.axes.values():
            ax.clear()
        if "nees" in self.axes:
            plot_mc_nees(result, ax=self.axes["nees"], confidence=self.confidence, log=self.log)
            named = _named_colors(result, None)
            _boxplot(self.axes["rmse"], [result[name].rmse_total for _, name in named], named)
            self.axes["rmse"].set_title("RMSE total per run")
        plot_mc_nis(result, ax=self.axes["nis"], confidence=self.confidence, log=self.log)
        self.fig.suptitle(f"Monte Carlo — {result.n_runs} runs so far")
        self.fig.tight_layout()
        self._show()


def _named_colors(result: MonteCarloResult, filters) -> list[tuple[str, str]]:
    names = result.filter_names
    selected = names if filters is None else list(filters)
    unknown = [n for n in selected if n not in names]
    if unknown:
        raise ValueError(f"unknown filters {unknown}; available: {names}")
    return [(f"C{i % 10}", n) for i, n in enumerate(names) if n in selected]


def _bounds(ax, t, lower, upper, dim, confidence) -> None:
    ax.fill_between(t, lower, upper, color="0.5", alpha=0.15, lw=0, label=f"{confidence:.0%} bounds")
    ax.axhline(dim, color="k", lw=0.8, ls=":", label=f"ideal ({dim:g})")


def _finish(ax, title, ylabel, log) -> None:
    if log:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="upper right", fontsize=8)


def _boxplot(ax, data, named) -> None:
    box = ax.boxplot(data, patch_artist=True, widths=0.6)
    for patch, (color, _) in zip(box["boxes"], named):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    for median in box["medians"]:
        median.set_color("k")
    ax.set_xticks(range(1, len(named) + 1), [name for _, name in named], rotation=20)
    ax.grid(True, axis="y", alpha=0.3)
