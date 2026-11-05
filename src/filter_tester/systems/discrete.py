"""Discrete-time systems."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import ArrayLike

from ..noise import NoiseLike, NoiseTerm
from .base import InitialState, System, sample_initial_state

TransitionFn = Callable[[np.ndarray, "np.ndarray | None", float, float], ArrayLike]


class DiscreteSystem(System):
    """Discrete-time system ``x_next = f(x, w, t, dt)``.

    One call to ``f`` is made per :meth:`advance`, with ``dt`` the time since
    the previous call, so both fixed and variable time steps work. ``w`` is a
    fresh noise sample per step (``None`` if no noise is given, zeros if
    noise is disabled), and may
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
        noise_enabled: bool = True,
    ):
        super().__init__(name, state_names, noise_enabled)
        if not callable(f):
            raise TypeError("f must be callable as f(x, w, t, dt)")
        self.f = f
        self.x0 = x0
        self._noise = NoiseTerm(noise)

    def _initial_state(self, t0, rng):
        return sample_initial_state(self.x0, rng)

    def _advance(self, x, t, t_next, rng):
        w = self._noise.draw(rng, self.noise_enabled)
        x_next = np.asarray(self.f(x, w, t, t_next - t), dtype=float).reshape(-1)
        if x_next.shape != x.shape:
            raise ValueError(f"f changed the state shape from {x.shape} to {x_next.shape}")
        return x_next
