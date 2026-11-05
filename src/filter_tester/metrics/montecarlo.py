"""Metrics over Monte Carlo runs.

Runs use independent seeds, so at each instant the NEES (NIS) averaged over
the ``N`` runs follows ``chi2(N n) / N`` for a consistent filter. This gives
an exact interval per instant, unlike the single-run check. A consistent
filter keeps the run-averaged NEES inside it at about ``confidence`` of the
instants.

The overall verdict uses ANEES / ANIS (the average over runs and time) with
an interval whose sample size is ``N`` times the effective number of
independent instants of the run-averaged series.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2

from ..experiment.montecarlo import MonteCarloResult
from .metrics import _fmt_time, effective_sample_size


def mc_nees(result: MonteCarloResult, name: str, confidence: float = 0.95):
    """``(t, mean NEES over runs, lower, upper)`` at every step with known truth."""
    stats = result[name]
    mask = np.all(np.isfinite(stats.nees), axis=0)
    if not mask.any():
        raise ValueError(f"{name}: no ground truth, NEES needs it")
    N, n = result.n_runs, result.state_dim
    lower, upper = _bounds(np.full(mask.sum(), n), N, confidence)
    return result.t[mask], stats.nees[:, mask].mean(axis=0), lower, upper


def mc_nis(result: MonteCarloResult, name: str, confidence: float = 0.95):
    """``(t, mean NIS over runs, lower, upper)`` at every innovation (bounds depend on its dimension)."""
    stats = result[name]
    if stats.nis.shape[1] == 0:
        raise ValueError(f"{name}: no innovations")
    lower, upper = _bounds(stats.nis_dim, result.n_runs, confidence)
    return stats.nis_t, stats.nis.mean(axis=0), lower, upper


@dataclass(frozen=True)
class MonteCarloConsistency:
    """ANEES or ANIS of a Monte Carlo experiment and its verdict."""

    mean: float
    expected: float
    lower: float
    upper: float
    fraction_inside: float  # share of instants whose run-averaged value is inside its bounds
    confidence: float

    @property
    def consistent(self) -> bool:
        """Whether ANEES / ANIS lies inside its chi-square interval."""
        return self.lower <= self.mean <= self.upper

    @property
    def verdict(self) -> str:
        """``consistent``, ``optimistic`` (above the interval) or ``pessimistic`` (below)."""
        if self.consistent:
            return "consistent"
        return "optimistic" if self.mean > self.upper else "pessimistic"


def mc_nees_consistency(result: MonteCarloResult, name: str, confidence: float = 0.95) -> MonteCarloConsistency:
    """ANEES of one filter with its interval, verdict and share of instants inside the bounds."""
    t, mean, lower, upper = mc_nees(result, name, confidence)
    dof = np.full(mean.size, result.state_dim)
    return _overall(mean, dof, lower, upper, result.n_runs, confidence)


def mc_nis_consistency(result: MonteCarloResult, name: str, confidence: float = 0.95) -> MonteCarloConsistency:
    """ANIS of one filter with its interval, verdict and share of instants inside the bounds."""
    t, mean, lower, upper = mc_nis(result, name, confidence)
    return _overall(mean, result[name].nis_dim, lower, upper, result.n_runs, confidence)


def _bounds(dims: np.ndarray, N: int, confidence: float):
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    alpha = 1 - confidence
    dims = np.asarray(dims, dtype=float)
    return chi2.ppf(alpha / 2, N * dims) / N, chi2.ppf(1 - alpha / 2, N * dims) / N


def _overall(series, dof, lower, upper, N, confidence) -> MonteCarloConsistency:
    K_eff = effective_sample_size(series)
    total = N * K_eff
    expected = float(np.mean(dof))
    alpha = 1 - confidence
    lo, hi = chi2.ppf([alpha / 2, 1 - alpha / 2], expected * total) / total
    inside = (series >= lower) & (series <= upper)
    return MonteCarloConsistency(
        mean=float(series.mean()),
        expected=expected,
        lower=float(lo),
        upper=float(hi),
        fraction_inside=float(inside.mean()),
        confidence=confidence,
    )


# ------------------------------------------------------------------ summary
def mc_summary_table(
    result: MonteCarloResult, state_names: list[str] | None = None, confidence: float = 0.95
) -> tuple[list[str], list[list[str]]]:
    """Metrics over all runs as ``(header, rows)`` of strings.

    RMSE columns show the mean ± standard deviation of the per-run RMSE.
    NEES / NIS columns show ANEES / ANIS, the interval, the verdict and the
    share of instants inside their per-instant bounds (ideal ≈ confidence).
    """
    n = result.state_dim
    names = state_names or [f"x[{i}]" for i in range(n)]
    pct = f"{confidence:.0%}"
    header = (
        ["Filter"]
        + [f"RMSE {s}" for s in names]
        + ["RMSE total", f"ANEES (ideal {n}, {pct})", f"ANIS ({pct})", "Time/step"]
    )
    body = []
    for name, stats in result.stats.items():
        if result.has_truth:
            rm = [_pm(stats.rmse[:, i]) for i in range(n)] + [_pm(stats.rmse_total)]
            nees = _fmt(mc_nees_consistency(result, name, confidence))
        else:
            rm, nees = ["—"] * (n + 1), "—"
        nis = _fmt(mc_nis_consistency(result, name, confidence), show_expected=True) if stats.nis.shape[1] else "—"
        body.append([name, *rm, nees, nis, _fmt_time(float(np.mean(stats.step_time)))])
    return header, body


def format_mc_summary(result: MonteCarloResult, state_names: list[str] | None = None, confidence: float = 0.95) -> str:
    """The Monte Carlo summary as aligned plain text."""
    header, body = mc_summary_table(result, state_names, confidence)
    widths = [max(len(row[i]) for row in [header, *body]) for i in range(len(header))]
    line = lambda cells: "  ".join(c.ljust(w) for c, w in zip(cells, widths, strict=True))  # noqa: E731
    title = f"Monte Carlo: {result.n_runs} runs (seeds {result.seeds[0]}–{result.seeds[-1]})"
    return "\n".join([title, line(header), "  ".join("-" * w for w in widths), *map(line, body)])


def _pm(values: np.ndarray) -> str:
    return f"{np.mean(values):.4g} ± {np.std(values):.2g}"


def _fmt(c: MonteCarloConsistency, show_expected: bool = False) -> str:
    expected = f"ideal {c.expected:.3g}, " if show_expected else ""
    return f"{c.mean:.3g} [{expected}{c.lower:.3g}–{c.upper:.3g}] {c.verdict}, {c.fraction_inside:.0%} inside"
