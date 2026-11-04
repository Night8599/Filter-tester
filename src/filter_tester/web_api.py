"""JSON API used by the web interface (runs in the browser through Pyodide).

Every public function takes and returns JSON strings, so the JavaScript side
never handles Python objects. A session is advanced in small time slices so
the page stays responsive and plots update in real time::

    config = {
        "scenario": "constant_velocity",
        "system": {...},                       # true constants (missing keys -> defaults)
        "filters": [{"name": "KF", "type": "kf", "params": {...}}],
        "mode": "single" | "montecarlo",
        "seed": 0,                             # single run: seed; Monte Carlo: first seed
        "runs": 100,                           # Monte Carlo only
    }
    sid = create_session(json.dumps(config))
    update = json.loads(advance(sid, json.dumps({"budget": 0.03})))
"""

from __future__ import annotations

import itertools
import json
import math
import time
from typing import Any

import numpy as np

from .experiment.montecarlo import MonteCarlo
from .experiment.runner import Runner
from .metrics.metrics import summary_table
from .metrics.montecarlo import mc_nees, mc_nis, mc_summary_table
from .scenarios import SCENARIOS, get_scenario

_sessions: dict[int, "_Session"] = {}
_ids = itertools.count(1)


# ------------------------------------------------------------------ public
def scenarios() -> str:
    """All scenarios with their adjustable constants."""
    return _dump([s.to_dict() for s in SCENARIOS.values()])


def filter_defaults(scenario: str, filter_type: str, system_json: str = "{}") -> str:
    """Filter constants matching the given true system (a "matched" filter)."""
    return _dump(get_scenario(scenario).filter_defaults(filter_type, json.loads(system_json)))


def create_session(config_json: str) -> int:
    config = json.loads(config_json)
    mode = config.get("mode", "single")
    if mode == "single":
        session: _Session = SingleRunSession(config)
    elif mode == "montecarlo":
        session = MonteCarloSession(config)
    else:
        raise ValueError(f"unknown mode {mode!r}")
    sid = next(_ids)
    _sessions[sid] = session
    return sid


def advance(session_id: int, args_json: str = "{}") -> str:
    """Advance a session for at most ``budget`` seconds of computation and return the new data."""
    args = json.loads(args_json)
    return _dump(_sessions[int(session_id)].advance(**args))


def close_session(session_id: int) -> None:
    _sessions.pop(int(session_id), None)


# ---------------------------------------------------------------- sessions
class _Session:
    def __init__(self, config: dict[str, Any]):
        self.scenario = get_scenario(config["scenario"])
        self.system = self.scenario.system_values(config.get("system"))
        specs = config.get("filters") or []
        if not specs:
            raise ValueError("choose at least one filter")
        names = [f["name"] for f in specs]
        if len(set(names)) != len(names) or any(not str(n).strip() for n in names):
            raise ValueError("filter names must be unique and non-empty")
        self.filter_specs = specs
        self.seed = int(config.get("seed", 0))

    def make_filters(self):
        return {f["name"]: self.scenario.make_filter(f.get("type", "kf"), f.get("params")) for f in self.filter_specs}

    def info(self) -> dict[str, Any]:
        source = self.scenario.source(self.system)
        return {
            "state_names": list(self.scenario.state_names),
            "filters": [f["name"] for f in self.filter_specs],
            "sensors": [s.name for s in source.sensors],
            "sensor_state": dict(self.scenario.sensor_state),
            "t_start": source.t_start,
            "t_end": source.t_end,
        }


class SingleRunSession(_Session):
    """One run, streamed step by step."""

    def __init__(self, config):
        super().__init__(config)
        self.source = self.scenario.source(self.system)
        self.runner = Runner(self.source, self.make_filters())
        self.recorder = self.runner.start(rng=self.seed)
        self._sent_info = False

    def advance(self, budget: float = 0.03, until_t: float | None = None, metrics: bool = False):
        """Process steps for up to ``budget`` seconds (and not beyond ``until_t``)."""
        deadline = time.perf_counter() + float(budget)
        out: dict[str, Any] = {"t": [], "truth": [], "measurements": [], "estimates": {}}
        est_rows: dict[str, dict[str, list]] = {n: {"mean": [], "var": []} for n in self.runner.filters}
        if not self._sent_info:
            out["info"] = self.info()
            self._sent_info = True
        last = None
        while not self.recorder.done and time.perf_counter() < deadline:
            if until_t is not None and self.recorder.steps and self.recorder.steps[-1].t >= until_t:
                break
            try:
                step, estimates = next(self.recorder)
            except StopIteration:
                break
            out["t"].append(step.t)
            out["truth"].append(None if step.truth is None else step.truth)
            for sensor, z in step.measurements.items():
                out["measurements"].append({"t": step.t, "sensor": sensor, "z": z})
            for name, e in estimates.items():
                est_rows[name]["mean"].append(e.mean)
                est_rows[name]["var"].append(np.diag(e.covariance))
            last = estimates
        out["estimates"] = est_rows
        if last is not None:
            out["current"] = {n: {"mean": e.mean, "var": np.diag(e.covariance)} for n, e in last.items()}
        out["done"] = self.recorder.done
        t_now = self.recorder.steps[-1].t if self.recorder.steps else self.source.t_start
        span = self.source.t_end - self.source.t_start
        out["t_now"] = t_now
        out["progress"] = 1.0 if span <= 0 else (t_now - self.source.t_start) / span
        if (metrics or out["done"]) and self.recorder.steps:
            header, rows = summary_table(self.recorder.result(), list(self.scenario.state_names))
            out["metrics"] = {"header": header, "rows": rows}
        return out


class MonteCarloSession(_Session):
    """Many runs (seeds ``seed .. seed + runs - 1``), aggregated as they finish."""

    def __init__(self, config):
        super().__init__(config)
        runs = int(config.get("runs", 100))
        if not 1 <= runs <= 10_000:
            raise ValueError("runs must be between 1 and 10000")
        self.mc = MonteCarlo(self.scenario.source(self.system), self.make_filters(), runs)
        self._iter = self.mc.iter_runs(first_seed=self.seed, build_partial=False)
        self.runs_done = 0
        self.done = False
        self._sent_info = False

    def advance(self, budget: float = 0.1, **_ignored):
        """Do whole runs for up to ``budget`` seconds (at least one) and return the aggregate."""
        deadline = time.perf_counter() + float(budget)
        out: dict[str, Any] = {}
        if not self._sent_info:
            out["info"] = self.info()
            self._sent_info = True
        did = False
        while not self.done and (not did or time.perf_counter() < deadline):
            try:
                next(self._iter)
                self.runs_done += 1
                did = True
            except StopIteration:
                self.done = True
        out.update({"runs_done": self.runs_done, "runs_total": self.mc.runs, "done": self.done})
        if self.runs_done == 0:
            return out
        result = self.mc.last_result()
        names = result.filter_names
        out["nees"], out["nis"] = {}, {}
        for name in names:
            if result.has_truth:
                t, mean, lo, hi = mc_nees(result, name)
                out["nees"][name] = {"t": t, "mean": mean, "lower": lo, "upper": hi}
            if result[name].nis.shape[1]:
                t, mean, lo, hi = mc_nis(result, name)
                out["nis"][name] = {"t": t, "mean": mean, "lower": lo, "upper": hi}
        out["rmse"] = {n: result[n].rmse for n in names}
        out["rmse_total"] = {n: result[n].rmse_total for n in names}
        out["rmse_stats"] = {
            n: {
                "mean": np.append(result[n].rmse.mean(axis=0), result[n].rmse_total.mean()),
                "var": np.append(result[n].rmse.var(axis=0), result[n].rmse_total.var()),
            }
            for n in names
        }
        header, rows = mc_summary_table(result, list(self.scenario.state_names))
        out["metrics"] = {"header": header, "rows": rows}
        return out


# ------------------------------------------------------------------ JSON
def _clean(obj):
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _dump(obj) -> str:
    return json.dumps(_clean(obj), allow_nan=False, separators=(",", ":"))
