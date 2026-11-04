"""Systems whose state is an explicit function of time."""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from numpy.typing import ArrayLike

from .base import System
from ..noise import NoiseLike, NoiseTerm

AnalyticFn = Callable[[float, "np.ndarray | None"], ArrayLike]


class AnalyticSystem(System):
    """True state given directly by ``x = g(t, w)``.

    ``w`` is a fresh noise sample drawn at every evaluated time (``None`` if
    no noise is given, zeros if noise is disabled), so the noise may enter ``g`` in any way, e.g.::

        AnalyticSystem(lambda t, w: np.array([np.sin(t)]) + w, GaussianNoise([[0.01]]))
        AnalyticSystem(lambda t, w: np.array([np.sin(t + w[0])]), GaussianNoise([[0.1]]))

    Samples at different times are independent, so this models a signal plus
    white perturbations rather than a process with memory.
    """

    def __init__(
        self,
        g: AnalyticFn,
        noise: NoiseLike = None,
        name: str | None = None,
        state_names: Sequence[str] | None = None,
        noise_enabled: bool = True,
    ):
        super().__init__(name, state_names, noise_enabled)
        if not callable(g):
            raise TypeError("g must be callable as g(t, w)")
        self.g = g
        self._noise = NoiseTerm(noise)

    def _evaluate(self, t: float, rng: np.random.Generator) -> ArrayLike:
        w = self._noise.draw(rng, self.noise_enabled)
        return self.g(t, w)

    def _initial_state(self, t0, rng):
        return self._evaluate(t0, rng)

    def _advance(self, x, t, t_next, rng):
        return self._evaluate(t_next, rng)
