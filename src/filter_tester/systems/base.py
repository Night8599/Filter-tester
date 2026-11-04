"""Base class shared by all true systems."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Sequence, Union

import numpy as np
from numpy.typing import ArrayLike

from .noise import Noise

# An initial state is either a fixed vector or something that samples one.
InitialState = Union[ArrayLike, Noise, Callable[[np.random.Generator], np.ndarray]]
RngLike = Union[np.random.Generator, int, None]


class System(ABC):
    """A true system: produces the true state at increasing times ``t``.

    Usage::

        system.reset(t0=0.0, rng=42)   # sets the initial state
        x = system.advance(0.1)       # true state at t = 0.1
        x = system.advance(0.25)      # times must not decrease

    ``state_names`` optionally names each state component (e.g.
    ``["position", "velocity"]``); plots use them as axis labels.

    Subclasses implement :meth:`_initial_state` and :meth:`_advance`.
    """

    def __init__(self, name: str | None = None, state_names: Sequence[str] | None = None):
        self.name = name or type(self).__name__
        self.state_names = None if state_names is None else [str(n) for n in state_names]
        self._t: float | None = None
        self._x: np.ndarray | None = None
        self._rng: np.random.Generator | None = None

    # ------------------------------------------------------------------ API
    def reset(self, t0: float = 0.0, rng: RngLike = None) -> np.ndarray:
        """Start a new run at time ``t0`` and return the initial state.

        ``rng`` may be a ``numpy.random.Generator``, an integer seed, or
        ``None`` (fresh, non-reproducible randomness).
        """
        self._rng = np.random.default_rng(rng)
        self._t = float(t0)
        self._x = _as_state(self._initial_state(self._t, self._rng))
        if self.state_names is not None and len(self.state_names) != self._x.size:
            raise ValueError(
                f"{len(self.state_names)} state_names given for a state of dimension {self._x.size}"
            )
        return self._x.copy()

    def advance(self, t: float) -> np.ndarray:
        """Move the system to time ``t`` and return the true state there."""
        if self._t is None:
            raise RuntimeError("call reset() before advance()")
        t = float(t)
        if t < self._t:
            raise ValueError(f"time must not decrease (current t={self._t}, requested t={t})")
        if t > self._t:
            self._x = _as_state(self._advance(self._x, self._t, t, self._rng))
            self._t = t
        return self._x.copy()

    @property
    def t(self) -> float | None:
        """Current time (``None`` before :meth:`reset`)."""
        return self._t

    @property
    def state(self) -> np.ndarray | None:
        """Current true state (``None`` before :meth:`reset`)."""
        return None if self._x is None else self._x.copy()

    # ------------------------------------------------------- for subclasses
    @abstractmethod
    def _initial_state(self, t0: float, rng: np.random.Generator) -> ArrayLike:
        """Return the state at ``t0``."""

    @abstractmethod
    def _advance(self, x: np.ndarray, t: float, t_next: float, rng: np.random.Generator) -> ArrayLike:
        """Return the state at ``t_next`` given state ``x`` at ``t`` (``t_next > t``)."""


def sample_initial_state(x0: InitialState, rng: np.random.Generator) -> np.ndarray:
    """Resolve an :data:`InitialState` into a concrete vector."""
    if callable(x0):
        return _as_state(x0(rng))
    return _as_state(x0)


def _as_state(x: ArrayLike) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim > 1:
        x = x.reshape(-1)
    return np.atleast_1d(x).copy()
