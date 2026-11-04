"""Shared machinery for figures that update while a run is in progress."""

from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np


class LiveFigure:
    """Base for live figures: throttled redraws, optional wall-clock pacing.

    Subclasses create ``self.fig`` after calling ``super().__init__`` (which
    switches matplotlib to interactive mode), store incoming data, call
    :meth:`_tick` with the time of each new sample, and implement
    :meth:`_redraw`.

    - After each redraw, the next one waits at least ``refresh_interval``
      seconds, so plotting does not dominate the run time.
    - ``speed`` paces the run against the wall clock: ``None`` does not wait,
      ``1.0`` is real time, ``2.0`` twice as fast. If the run is slower than
      the requested speed, it simply goes as fast as it can.
    """

    def __init__(self, refresh_interval: float = 1 / 30, speed: float | None = None):
        if speed is not None and not speed > 0:
            raise ValueError("speed must be positive or None")
        self.refresh_interval = float(refresh_interval)
        self.speed = None if speed is None else float(speed)
        self._was_interactive = plt.isinteractive()
        plt.ion()
        self.fig = None
        self._last_draw = -np.inf
        self._clock_start: tuple[float, float] | None = None  # (t, wall time) of the first sample
        self._finished = False

    def _check_open(self) -> None:
        if self._finished:
            raise RuntimeError("the plot is already finished")

    def _tick(self, t: float) -> None:
        """Redraw if enough time passed, then wait for the wall clock if paced."""
        if time.perf_counter() - self._last_draw >= self.refresh_interval:
            self._redraw()
            # Measured after drawing, so slow redraws cannot take all the time.
            self._last_draw = time.perf_counter()
        self._wait_for(t)

    def _wait_for(self, t: float) -> None:
        if self.speed is None:
            return
        now = time.perf_counter()
        if self._clock_start is None:
            self._clock_start = (t, now)
            return
        t0, wall0 = self._clock_start
        remaining = wall0 + (t - t0) / self.speed - now
        if remaining > 0 and self.fig is not None:
            # Waits while letting the GUI process events, so the window stays responsive.
            self.fig.canvas.start_event_loop(remaining)

    def finish(self) -> None:
        """Draw all received data and leave the figure as a static plot."""
        if self._finished:
            return
        self._redraw()
        self._finished = True
        if not self._was_interactive:
            plt.ioff()

    def _show(self) -> None:
        self.fig.canvas.draw_idle()
        plt.pause(1e-3)  # let the GUI process events and show the update

    def _redraw(self) -> None:
        raise NotImplementedError
