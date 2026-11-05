"""Metrics computed from a filter's :class:`~filter_tester.experiment.Trajectory`.

- **RMSE**: how far the estimate is from the truth (needs the truth).
- **NEES**: whether the covariance matches the actual error (needs the truth);
  a consistent filter has mean NEES close to the state dimension ``n``.
- **NIS**: whether the innovation covariance matches the innovations (no truth
  needed); a consistent filter has mean NIS close to the measurement dimension.
- **Step time**: average wall-clock time per step.

The consistency checks compare the time-averaged NEES / NIS of a single run
with a two-sided chi-square interval. Consecutive samples of one run are
correlated in time (especially NEES on a fine truth grid), so the interval
uses an *effective sample size* estimated from the autocorrelation of the
sequence instead of the raw number of samples. Without that correction a
filter with the exact model is flagged far too often. The verdict is still
an approximation; Monte Carlo runs give a stricter test.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2

from ..experiment.runner import RunResult, Trajectory


def rmse(traj: Trajectory) -> np.ndarray:
    """RMSE of each state component over the steps with known truth, shape ``(n,)``."""
    e = _known_errors(traj)
    return np.sqrt(np.mean(e**2, axis=0))


def rmse_total(traj: Trajectory) -> float:
    """RMSE of the error vector: ``sqrt(mean_k ||x_hat_k - x_k||^2)``."""
    e = _known_errors(traj)
    return float(np.sqrt(np.mean(np.sum(e**2, axis=1))))


def nees(traj: Trajectory) -> tuple[np.ndarray, np.ndarray]:
    """``(t, NEES)`` at every step with known truth."""
    mask = traj.has_truth
    if not mask.any():
        raise ValueError(f"{traj.name}: no ground truth, NEES needs it")
    e = traj.error[mask]
    P = traj.covariance[mask]
    values = np.einsum("ki,ki->k", e, np.linalg.solve(P, e[..., None])[..., 0])
    return traj.t[mask], values


def nis(traj: Trajectory, sensor: str | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(t, NIS, measurement dimension)`` for every innovation (of one sensor, if given)."""
    rows = [(t, y, S) for t, s, y, S in traj.innovations if sensor is None or s == sensor]
    if not rows:
        raise ValueError(f"{traj.name}: no innovations" + (f" for sensor {sensor!r}" if sensor else ""))
    t = np.array([r[0] for r in rows])
    values = np.array([float(y @ np.linalg.solve(S, y)) for _, y, S in rows])
    dims = np.array([y.size for _, y, _ in rows])
    return t, values, dims


def mean_step_time(traj: Trajectory) -> float:
    """Average wall-clock seconds per step."""
    return float(np.mean(traj.step_time))


@dataclass(frozen=True)
class Consistency:
    """Time-averaged NEES or NIS of one run compared with its chi-square interval."""

    mean: float
    expected: float  # dimension: n for NEES, average measurement dimension for NIS
    lower: float
    upper: float
    samples: int
    effective_samples: float

    @property
    def consistent(self) -> bool:
        """Whether the mean lies inside the chi-square interval."""
        return self.lower <= self.mean <= self.upper

    @property
    def verdict(self) -> str:
        """``consistent``, ``optimistic`` (mean above the interval) or ``pessimistic`` (below)."""
        if self.consistent:
            return "consistent"
        return "optimistic" if self.mean > self.upper else "pessimistic"


def nees_consistency(traj: Trajectory, confidence: float = 0.95) -> Consistency:
    """Mean NEES vs. the chi-square interval for ``K`` samples of dimension ``n``."""
    _, values = nees(traj)
    n = traj.mean.shape[1]
    return _consistency(values, dof=np.full(values.size, n), confidence=confidence)


def nis_consistency(traj: Trajectory, sensor: str | None = None, confidence: float = 0.95) -> Consistency:
    """Mean NIS vs. the chi-square interval (handles sensors of different dimensions)."""
    _, values, dims = nis(traj, sensor)
    return _consistency(values, dof=dims, confidence=confidence)


def _consistency(values: np.ndarray, dof: np.ndarray, confidence: float) -> Consistency:
    """Chi-square interval for the mean of ``values`` (sample ``k`` has ``dof[k]`` degrees of freedom)."""
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    K = values.size
    expected = float(dof.mean())
    K_eff = effective_sample_size(values)
    alpha = 1 - confidence
    # The sum of K_eff independent samples is chi-square with expected * K_eff
    # degrees of freedom; divide by K_eff for the interval of the mean.
    lower, upper = chi2.ppf([alpha / 2, 1 - alpha / 2], expected * K_eff) / K_eff
    return Consistency(
        mean=float(values.mean()),
        expected=expected,
        lower=float(lower),
        upper=float(upper),
        samples=K,
        effective_samples=K_eff,
    )


def effective_sample_size(values: np.ndarray) -> float:
    """``K / (1 + 2 sum_k rho_k)``, summing autocorrelations up to the first non-positive one."""
    v = np.asarray(values, dtype=float) - np.mean(values)
    K = v.size
    var = v @ v / K
    if K < 3 or var == 0:
        return float(K)
    rho_sum = 0.0
    for lag in range(1, K // 2):
        rho = (v[:-lag] @ v[lag:]) / (K * var)
        if rho <= 0:
            break
        rho_sum += rho
    return float(K / (1 + 2 * rho_sum))


def _known_errors(traj: Trajectory) -> np.ndarray:
    mask = traj.has_truth
    if not mask.any():
        raise ValueError(f"{traj.name}: no ground truth, RMSE needs it")
    return traj.error[mask]


# ------------------------------------------------------------------ summary
def summary(result: RunResult, confidence: float = 0.95) -> list[dict]:
    """One row of metrics per filter. Metrics that cannot be computed are ``None``."""
    rows = []
    for name, traj in result.trajectories.items():
        row = {"filter": name, "rmse": None, "rmse_total": None, "nees": None, "nis": None}
        if traj.has_truth.any():
            row["rmse"] = rmse(traj)
            row["rmse_total"] = rmse_total(traj)
            row["nees"] = nees_consistency(traj, confidence)
        if traj.innovations:
            row["nis"] = nis_consistency(traj, confidence=confidence)
        row["step_time"] = mean_step_time(traj)
        rows.append(row)
    return rows


def summary_table(
    result: RunResult, state_names: list[str] | None = None, confidence: float = 0.95
) -> tuple[list[str], list[list[str]]]:
    """Metrics as ``(header, rows)`` of strings, ready to print or draw."""
    rows = summary(result, confidence)
    n = next(iter(result.trajectories.values())).mean.shape[1]
    names = state_names or [f"x[{i}]" for i in range(n)]
    pct = f"{confidence:.0%}"
    header = (
        ["Filter"]
        + [f"RMSE {s}" for s in names]
        + ["RMSE total", f"NEES (ideal {n}, {pct})", f"NIS ({pct})", "Time/step"]
    )
    body = []
    for r in rows:
        rm = ["—"] * n if r["rmse"] is None else [f"{v:.4g}" for v in r["rmse"]]
        body.append(
            [r["filter"]]
            + rm
            + ["—" if r["rmse_total"] is None else f"{r['rmse_total']:.4g}"]
            + [_fmt_consistency(r["nees"]), _fmt_consistency(r["nis"], show_expected=True)]
            + [_fmt_time(r["step_time"])]
        )
    return header, body


def format_summary(result: RunResult, state_names: list[str] | None = None, confidence: float = 0.95) -> str:
    """The summary table as aligned plain text."""
    header, body = summary_table(result, state_names, confidence)
    widths = [max(len(row[i]) for row in [header, *body]) for i in range(len(header))]
    line = lambda cells: "  ".join(c.ljust(w) for c, w in zip(cells, widths, strict=True))  # noqa: E731
    return "\n".join([line(header), "  ".join("-" * w for w in widths), *map(line, body)])


def _fmt_consistency(c: Consistency | None, show_expected: bool = False) -> str:
    if c is None:
        return "—"
    expected = f"ideal {c.expected:.3g}, " if show_expected else ""
    return f"{c.mean:.3g} [{expected}{c.lower:.3g}–{c.upper:.3g}] {c.verdict}"


def _fmt_time(seconds: float) -> str:
    for unit, scale in (("s", 1.0), ("ms", 1e-3), ("µs", 1e-6)):
        if seconds >= scale:
            return f"{seconds / scale:.3g} {unit}"
    return f"{seconds / 1e-9:.3g} ns"
