"""Continuous-time stochastic systems (SDEs)."""

from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np
from numpy.typing import ArrayLike

from .base import InitialState, System, sample_initial_state

DriftFn = Callable[[np.ndarray, float], ArrayLike]
DiffusionFn = Callable[[np.ndarray, float], ArrayLike]


class SDESystem(System):
    """Itô SDE ``dx = f(x, t) dt + G(x, t) dW``, integrated with Euler–Maruyama.

    - ``f(x, t)`` (drift) returns a vector of shape ``(n,)``.
    - ``G(x, t)`` (diffusion) returns either an ``(n, m)`` matrix, driven by an
      ``m``-dimensional Wiener process, or an ``(n,)`` vector, read as a diagonal
      matrix. ``G`` may depend on the state, so multiplicative noise is supported.
      If ``G`` is ``None``, or noise is disabled, the system is deterministic.

    Between two calls to :meth:`advance`, the interval is split into equal
    sub-steps no longer than ``max_step``. Smaller values are more accurate
    and slower. Euler–Maruyama has strong order 0.5 (weak order 1).

    Example, an Ornstein–Uhlenbeck process::

        SDESystem(f=lambda x, t: -theta * x, G=lambda x, t: [sigma], x0=[1.0], max_step=1e-3)
    """

    def __init__(
        self,
        f: DriftFn,
        x0: InitialState,
        G: DiffusionFn | None = None,
        max_step: float = 1e-3,
        name: str | None = None,
        state_names: Sequence[str] | None = None,
        noise_enabled: bool = True,
    ):
        super().__init__(name, state_names, noise_enabled)
        if not callable(f):
            raise TypeError("f must be callable as f(x, t)")
        if G is not None and not callable(G):
            raise TypeError("G must be callable as G(x, t) or None")
        if not max_step > 0:
            raise ValueError("max_step must be positive")
        self.f = f
        self.G = G
        self.x0 = x0
        self.max_step = float(max_step)

    def _initial_state(self, t0, rng):
        return sample_initial_state(self.x0, rng)

    def _advance(self, x, t, t_next, rng):
        n_sub = max(1, math.ceil((t_next - t) / self.max_step - 1e-12))
        h = (t_next - t) / n_sub
        sqrt_h = math.sqrt(h)
        for i in range(n_sub):
            s = t + i * h
            drift = np.asarray(self.f(x, s), dtype=float).reshape(-1)
            if drift.shape != x.shape:
                raise ValueError(f"f returned shape {drift.shape}, expected {x.shape}")
            x_new = x + drift * h
            if self.G is not None and self.noise_enabled:
                x_new = x_new + self._diffusion_increment(x, s, sqrt_h, rng)
            x = x_new
        return x

    def _diffusion_increment(self, x, s, sqrt_h, rng):
        G = np.asarray(self.G(x, s), dtype=float)
        n = x.shape[0]
        if G.ndim <= 1:  # diagonal diffusion
            G = np.broadcast_to(G.reshape(-1), (n,))
            return G * rng.standard_normal(n) * sqrt_h
        if G.ndim != 2 or G.shape[0] != n:
            raise ValueError(f"G returned shape {G.shape}, expected ({n}, m) or ({n},)")
        return G @ rng.standard_normal(G.shape[1]) * sqrt_h
