"""Plots of filter results.

- :func:`plot_estimates`: estimate of each filter with a ±kσ band, against the truth.
- :func:`plot_errors`: estimation error of each filter with its ±kσ band.
- :func:`plot_summary_table`: the metrics table as a figure.
- :class:`LiveFilterPlot`: estimates vs. truth updating while a :class:`Runner` runs.

Every filter keeps the same color in all plots (matplotlib's color cycle in
the order of the filters).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

from ..core.estimate import Estimate
from ..core.step import Step
from ..experiment.runner import RunResult
from ..metrics.metrics import summary_table
from ._live import LiveFigure
from .systems import _get_axes, resolve_labels


def plot_estimates(
    result: RunResult,
    labels: Sequence[str] | None = None,
    sigma: float = 3.0,
    filters: Sequence[str] | None = None,
    title: str | None = "Estimates",
):
    """One subplot per state: the truth (if known) and each filter's estimate ± ``sigma``·σ.

    Returns ``(fig, axes)``.
    """
    names = _select(result, filters)
    n = result[names[0]].mean.shape[1]
    fig, axes = _get_axes(n, resolve_labels(n, labels), title, None)
    first = result[names[0]]
    if first.truth is not None:
        known = first.has_truth
        for i, ax in enumerate(axes):
            ax.plot(first.t[known], first.truth[known, i], color="k", lw=1.2, label="truth")
    for color, name in zip(_colors(result), result.filter_names, strict=True):
        if name not in names:
            continue
        traj = result[name]
        for i, ax in enumerate(axes):
            m, s = traj.mean[:, i], sigma * traj.std[:, i]
            ax.plot(traj.t, m, color=color, lw=1.2, label=name)
            ax.fill_between(traj.t, m - s, m + s, color=color, alpha=0.18, lw=0)
    _legend(axes[0], sigma)
    return fig, axes


def plot_errors(
    result: RunResult,
    labels: Sequence[str] | None = None,
    sigma: float = 3.0,
    filters: Sequence[str] | None = None,
    title: str | None = "Estimation error",
):
    """One subplot per state: each filter's error ``estimate - truth`` and its ± ``sigma``·σ band.

    A consistent filter keeps its error inside its own band most of the time.
    Requires the truth. Returns ``(fig, axes)``.
    """
    names = _select(result, filters)
    first = result[names[0]]
    if not first.has_truth.any():
        raise ValueError("plot_errors needs the ground truth")
    n = first.mean.shape[1]
    fig, axes = _get_axes(n, [f"error {label}" for label in resolve_labels(n, labels)], title, None)
    for color, name in zip(_colors(result), result.filter_names, strict=True):
        if name not in names:
            continue
        traj = result[name]
        known = traj.has_truth
        t, e, s = traj.t[known], traj.error[known], sigma * traj.std[known]
        for i, ax in enumerate(axes):
            ax.plot(t, e[:, i], color=color, lw=1.0, label=name)
            ax.fill_between(t, -s[:, i], s[:, i], color=color, alpha=0.15, lw=0)
    for ax in axes:
        ax.axhline(0.0, color="k", lw=0.8)
    _legend(axes[0], sigma)
    return fig, axes


def plot_summary_table(
    result: RunResult,
    labels: Sequence[str] | None = None,
    confidence: float = 0.95,
    title: str | None = "Filter comparison",
):
    """Draw the metrics table (RMSE, NEES, NIS, time per step) as a figure.

    Returns ``(fig, ax)``.
    """
    header, body = summary_table(result, list(labels) if labels else None, confidence)
    return draw_table(header, body, title)


def draw_table(header: list[str], body: list[list[str]], title: str | None = None):
    """Draw a metrics table; NEES / NIS verdict cells are colored. Returns ``(fig, ax)``."""
    fig, ax = plt.subplots(figsize=(min(1.0 + 1.35 * len(header), 22), 0.9 + 0.4 * (len(body) + 1)))
    ax.axis("off")
    table = ax.table(cellText=body, colLabels=header, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.auto_set_column_width(range(len(header)))
    table.scale(1.0, 1.4)
    for (row, _), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
    verdict_cols = [i for i, h in enumerate(header) if h.startswith(("NEES", "NIS", "ANEES", "ANIS"))]
    for k, row in enumerate(body, start=1):  # color the consistency verdicts
        for col in verdict_cols:
            text = row[col]
            if "optimistic" in text or "pessimistic" in text:
                table[k, col].set_facecolor("#f8d7d3")
            elif "consistent" in text:
                table[k, col].set_facecolor("#d8eedb")
    if title:
        ax.set_title(title, weight="bold")
    fig.tight_layout()
    return fig, ax


class LiveFilterPlot(LiveFigure):
    """Estimates vs. truth, updating while a run is in progress.

    Pass it to :meth:`Runner.run` as an observer::

        live = LiveFilterPlot(labels=["position", "velocity"], speed=1.0)
        result = runner.run(rng=0, observers=[live])
        live.finish()

    The figure is created at the first step, when the state dimension and the
    filters are known. ``t_range`` fixes the time axis in advance.
    """

    def __init__(
        self,
        labels: Sequence[str] | None = None,
        sigma: float = 3.0,
        title: str | None = "Estimates (live)",
        t_range: tuple[float, float] | None = None,
        refresh_interval: float = 1 / 30,
        speed: float | None = None,
    ):
        super().__init__(refresh_interval, speed)
        self.labels = labels
        self.sigma = float(sigma)
        self.title = title
        self.t_range = t_range
        self._names: list[str] = []
        self._t: list[float] = []
        self._truth_t: list[float] = []
        self._truth: list[np.ndarray] = []
        self._mean: dict[str, list[np.ndarray]] = {}
        self._std: dict[str, list[np.ndarray]] = {}

    def __call__(self, step: Step, estimates: Mapping[str, Estimate]) -> None:
        self._check_open()
        if self.fig is None:
            self._create(estimates)
        self._t.append(step.t)
        if step.truth is not None:
            self._truth_t.append(step.t)
            self._truth.append(np.asarray(step.truth, dtype=float))
        for name in self._names:
            self._mean[name].append(estimates[name].mean)
            self._std[name].append(estimates[name].std)
        self._tick(step.t)

    def _create(self, estimates: Mapping[str, Estimate]) -> None:
        self._names = list(estimates)
        n = next(iter(estimates.values())).mean.size
        self.fig, self.axes = _get_axes(n, resolve_labels(n, self.labels), self.title, None)
        self._truth_lines = [ax.plot([], [], color="k", lw=1.2, label="truth")[0] for ax in self.axes]
        colors = [f"C{i % 10}" for i in range(len(self._names))]
        self._colors = dict(zip(self._names, colors, strict=True))
        self._lines = {
            name: [ax.plot([], [], color=self._colors[name], lw=1.2, label=name)[0] for ax in self.axes]
            for name in self._names
        }
        self._bands: dict[str, list] = {name: [None] * n for name in self._names}
        for name in self._names:
            self._mean[name], self._std[name] = [], []
        self._fixed_t = self.t_range is not None and self.t_range[1] > self.t_range[0]
        if self._fixed_t:
            for ax in self.axes:
                ax.set_xlim(*self.t_range)
        _legend(self.axes[0], self.sigma)

    def _redraw(self) -> None:
        if self.fig is None or not self._t:
            return
        t = np.array(self._t)
        if self._truth:
            truth = np.array(self._truth)
            for i, line in enumerate(self._truth_lines):
                line.set_data(self._truth_t, truth[:, i])
        for name in self._names:
            mean, std = np.array(self._mean[name]), self.sigma * np.array(self._std[name])
            for i, ax in enumerate(self.axes):
                self._lines[name][i].set_data(t, mean[:, i])
                if self._bands[name][i] is not None:
                    self._bands[name][i].remove()
                self._bands[name][i] = ax.fill_between(
                    t,
                    mean[:, i] - std[:, i],
                    mean[:, i] + std[:, i],
                    color=self._colors[name],
                    alpha=0.18,
                    lw=0,
                )
        for ax in self.axes:
            ax.relim()
            ax.autoscale_view(scalex=not self._fixed_t)
        self._show()


def _select(result: RunResult, filters: Sequence[str] | None) -> list[str]:
    names = result.filter_names if filters is None else list(filters)
    unknown = [n for n in names if n not in result.trajectories]
    if unknown:
        raise ValueError(f"unknown filters {unknown}; available: {result.filter_names}")
    if not names:
        raise ValueError("no filters to plot")
    return names


def _colors(result: RunResult) -> list[str]:
    return [f"C{i % 10}" for i in range(len(result.filter_names))]


def _legend(ax, sigma: float) -> None:
    handles, labels = ax.get_legend_handles_labels()
    handles.append(plt.Rectangle((0, 0), 1, 1, color="0.5", alpha=0.3))
    labels.append(f"±{sigma:g}σ")
    ax.legend(handles, labels, loc="upper left", fontsize=8)
