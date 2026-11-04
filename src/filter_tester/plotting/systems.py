"""Plots of true-system states over time.

- :func:`plot_states` plots states that were already simulated.
- :func:`plot_system` simulates a system and plots it, optionally live.
- :class:`LiveStatePlot` is the live figure itself, fed one ``(t, x)`` at a time.

All plots have one subplot per state component, sharing the time axis.
Axis labels come from ``labels`` if given, otherwise from the system's
``state_names``, otherwise ``x[0]``, ``x[1]``, ...
"""

from __future__ import annotations

import time
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike

from ..systems.base import RngLike, System
from ..systems.simulate import simulate


def plot_states(
    times: ArrayLike,
    states: ArrayLike,
    labels: Sequence[str] | None = None,
    title: str | None = None,
    axes: Sequence[plt.Axes] | None = None,
    **line_kwargs,
):
    """Plot ``states`` (shape ``(len(times), n)``) against ``times``.

    Pass ``axes`` from a previous call to draw on the same figure, e.g. to
    overlay several runs. Extra keyword arguments go to ``Axes.plot``.
    Returns ``(fig, axes)``.
    """
    times = np.asarray(times, dtype=float).reshape(-1)
    states = np.asarray(states, dtype=float)
    if states.ndim == 1:
        states = states[:, None]
    if states.shape[0] != times.size:
        raise ValueError(f"states has {states.shape[0]} rows but there are {times.size} times")
    n = states.shape[1]
    fig, axes = _get_axes(n, resolve_labels(n, labels), title, axes)
    for i, ax in enumerate(axes):
        ax.plot(times, states[:, i], **line_kwargs)
    return fig, axes


def plot_system(
    system: System,
    times: ArrayLike,
    rng: RngLike = None,
    labels: Sequence[str] | None = None,
    title: str | None = None,
    live: bool = False,
    refresh_interval: float = 1 / 30,
    **line_kwargs,
):
    """Simulate ``system`` over ``times`` and plot its states.

    With ``live=True`` the figure is shown and updated while the simulation
    runs (at most once every ``refresh_interval`` seconds), and stays open as
    a static figure at the end. Call ``plt.show()`` afterwards to keep the
    window open in a script.

    Returns ``(fig, axes, states)``.
    """
    times = np.asarray(times, dtype=float).reshape(-1)
    labels = labels if labels is not None else system.state_names
    title = title if title is not None else system.name
    if not live:
        states = simulate(system, times, rng)
        fig, axes = plot_states(times, states, labels, title, **line_kwargs)
        return fig, axes, states

    plot: LiveStatePlot | None = None

    def on_step(t, x):
        nonlocal plot
        if plot is None:  # state dimension is only known after reset()
            plot = LiveStatePlot(
                x.size, labels, title, t_range=(times[0], times[-1]),
                refresh_interval=refresh_interval, **line_kwargs,
            )
        plot.update(t, x)

    states = simulate(system, times, rng, on_step=on_step)
    plot.finish()
    return plot.fig, plot.axes, states


class LiveStatePlot:
    """A figure that updates as states arrive, one ``(t, x)`` at a time.

    ::

        plot = LiveStatePlot(n_states=2, labels=["position", "velocity"])
        for t in times:
            plot.update(t, system.advance(t))
        plot.finish()

    Redraws are throttled to one every ``refresh_interval`` seconds so that
    plotting does not dominate the simulation time. ``t_range`` fixes the
    time axis in advance; without it the axis grows with the data.
    """

    def __init__(
        self,
        n_states: int,
        labels: Sequence[str] | None = None,
        title: str | None = None,
        t_range: tuple[float, float] | None = None,
        refresh_interval: float = 1 / 30,
        **line_kwargs,
    ):
        self._was_interactive = plt.isinteractive()
        plt.ion()
        self.fig, self.axes = _get_axes(n_states, resolve_labels(n_states, labels), title, None)
        self.lines = [ax.plot([], [], **line_kwargs)[0] for ax in self.axes]
        if t_range is not None and t_range[1] > t_range[0]:
            for ax in self.axes:
                ax.set_xlim(*t_range)
        self._fixed_t = t_range is not None and t_range[1] > t_range[0]
        self.refresh_interval = float(refresh_interval)
        self._t: list[float] = []
        self._x: list[np.ndarray] = []
        self._last_draw = -np.inf
        self._finished = False

    def update(self, t: float, x: ArrayLike) -> None:
        """Add the state ``x`` at time ``t`` and redraw if enough time passed."""
        if self._finished:
            raise RuntimeError("the plot is already finished")
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.size != len(self.lines):
            raise ValueError(f"expected {len(self.lines)} state components, got {x.size}")
        self._t.append(float(t))
        self._x.append(x)
        now = time.perf_counter()
        if now - self._last_draw >= self.refresh_interval:
            self._redraw()
            self._last_draw = now

    def finish(self) -> None:
        """Draw all received data and leave the figure as a static plot."""
        if self._finished:
            return
        self._redraw()
        self._finished = True
        if not self._was_interactive:
            plt.ioff()

    def _redraw(self) -> None:
        if not self._t:
            return
        data = np.array(self._x)
        for i, (ax, line) in enumerate(zip(self.axes, self.lines)):
            line.set_data(self._t, data[:, i])
            ax.relim()
            ax.autoscale_view(scalex=not self._fixed_t)
        self.fig.canvas.draw_idle()
        plt.pause(1e-3)  # let the GUI process events and show the update


def resolve_labels(n: int, labels: Sequence[str] | None) -> list[str]:
    """Return ``labels`` checked against ``n``, or default ``x[i]`` names."""
    if labels is None:
        return [f"x[{i}]" for i in range(n)]
    labels = [str(label) for label in labels]
    if len(labels) != n:
        raise ValueError(f"{len(labels)} labels given for {n} state components")
    return labels


def _get_axes(n, labels, title, axes):
    if axes is not None:
        axes = list(np.ravel(axes))
        if len(axes) != n:
            raise ValueError(f"{len(axes)} axes given for {n} state components")
        return axes[0].figure, axes
    fig, grid = plt.subplots(n, 1, sharex=True, squeeze=False, figsize=(8, 1.2 + 1.8 * n))
    axes = list(grid[:, 0])
    for ax, label in zip(axes, labels):
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("t")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig, axes
