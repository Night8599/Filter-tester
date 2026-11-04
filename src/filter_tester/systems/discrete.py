"""Discrete-time systems."""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from numpy.typing import ArrayLike

from .base import InitialState, System, sample_initial_state
from .noise import NoiseLike, as_sampler

TransitionFn = Callable[[np.ndarray, "np.ndarray | None", float, float], ArrayLike]


class DiscreteSystem(System):
    """Discrete-time system ``x_next = f(x, w, t, dt)``.

    One call to ``f`` is made per :meth:`advance`, with ``dt`` the time since
    the previous call, so both fixed and variable time steps work. ``w`` is a
    fresh noise sample per step (``None`` if no noise is given), and may
    enter ``f`` in any way. Additive Gaussian noise, for example::

        A = np.array([[1, dt], [0, 1]])
        DiscreteSystem(
            f=lambda x, w, t, dt: A @ x + w,
            x0=[0.0, 1.0],
            noise=GaussianNoise(np.diag([1e-4, 1e-3])),
        )

    ``x0`` is either a fixed vector or a noise / callable ``rng -> x0`` that
    samples a random initial state at every :meth:`reset`.
    """

    def __init__(
        self,
        f: TransitionFn,
        x0: InitialState,
        noise: NoiseLike = None,
        name: str | None = None,
        state_names: Sequence[str] | None = None,
    ):
        super().__init__(name, state_names)
        if not callable(f):
            raise TypeError("f must be callable as f(x, w, t, dt)")
        self.f = f
        self.x0 = x0
        self._noise = as_sampler(noise)

    def _initial_state(self, t0, rng):
        return sample_initial_state(self.x0, rng)

    def _advance(self, x, t, t_next, rng):
        w = None if self._noise is None else self._noise(rng)
        x_next = np.asarray(self.f(x, w, t, t_next - t), dtype=float).reshape(-1)
        if x_next.shape != x.shape:
            raise ValueError(f"f changed the state shape from {x.shape} to {x_next.shape}")
        return x_next
