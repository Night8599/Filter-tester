import json
import math

import numpy as np
import pytest
from scipy.linalg import expm

from filter_tester import web_api
from filter_tester.experiment import MonteCarlo
from filter_tester.metrics import mc_nees_consistency, mc_nis_consistency
from filter_tester.scenarios import SCENARIOS, Param, discretize, get_scenario


# ------------------------------------------------------------------ params
def test_param_coercion_and_bounds():
    p = Param("a", "A", 1.0, 0.0, 2.0)
    assert p.coerce("1.5") == 1.5
    with pytest.raises(ValueError):
        p.coerce(3.0)
    assert Param("n", "N", 3, 1, 10, kind="int").coerce(4.6) == 5
    assert Param("b", "B", True, kind="bool").coerce(0) is False


def test_unknown_parameters_rejected():
    with pytest.raises(ValueError):
        get_scenario("constant_velocity").system_values({"nope": 1})
    with pytest.raises(KeyError):
        get_scenario("missing")


# --------------------------------------------------------------- scenarios
@pytest.mark.parametrize("key", list(SCENARIOS))
def test_every_scenario_builds_and_runs_every_filter(key):
    sc = get_scenario(key)
    source = sc.source({"duration": 3.0})
    steps = source.run(rng=0)
    assert steps[0].truth.size == len(sc.state_names)
    types = [ft.key for ft in sc.filter_types]
    assert {"ekf", "ukf", "pf"} <= set(types)
    for ft in types:
        f = sc.make_filter(ft, sc.filter_defaults(ft))
        est = [f.step(s.t, s.measurements) for s in steps]
        assert np.all(np.isfinite(est[-1].mean)), (key, ft)
    d = sc.to_dict()
    json.dumps(d)  # JSON-friendly
    for ft in d["filter_types"]:
        assert ft["defaults"].keys() == {p["key"] for p in ft["params"]}


def test_kf_only_where_a_linear_model_exists():
    assert {k for k, sc in SCENARIOS.items() if any(ft.key == "kf" for ft in sc.filter_types)} == {
        "constant_velocity",
        "ornstein_uhlenbeck",
        "mass_spring",
        "pendulum",
    }
    pf = {p.key for p in get_scenario("lorenz").filter_type("pf").params}
    assert {"n_particles", "resample_threshold", "regularization", "seed"} <= pf


def test_lorenz_nonlinear_filters_ranking():
    """On the chaotic Lorenz system the particle filter is the most consistent, the EKF the least."""
    sc = get_scenario("lorenz")
    system = {"duration": 6.0}
    filters = {ft: sc.make_filter(ft, sc.filter_defaults(ft, system)) for ft in ("ekf", "ukf", "pf")}
    result = MonteCarlo(sc.source(system), filters, runs=12).run()
    inside = {n: mc_nees_consistency(result, n).fraction_inside for n in filters}
    assert inside["pf"] > 0.8 and inside["pf"] >= inside["ekf"]
    for n in filters:
        assert mc_nis_consistency(result, n).fraction_inside > 0.8, n


def test_filter_defaults_follow_the_true_system():
    sc = get_scenario("mass_spring")
    d = sc.filter_defaults("kf", {"omega": 5.0, "sensor_R": 0.5})
    assert d["omega"] == 5.0 and d["sensor_R"] == 0.5


def test_discretize_matches_known_solutions():
    # Constant velocity: F = [[0,1],[0,0]], L = [0,1] -> the white-noise acceleration Q.
    A, Q = discretize(np.array([[0.0, 1.0], [0.0, 0.0]]), np.array([[0.0], [1.0]]), 0.3, 0.2)
    assert np.allclose(A, [[1, 0.2], [0, 1]])
    assert np.allclose(Q, 0.3 * np.array([[0.2**3 / 3, 0.2**2 / 2], [0.2**2 / 2, 0.2]]))
    # Scalar OU: F = -theta.
    A, Q = discretize(np.array([[-2.0]]), np.array([[1.0]]), 0.5**2, 0.1)
    assert np.isclose(A[0, 0], math.exp(-0.2))
    assert np.isclose(Q[0, 0], 0.25 / 4 * (1 - math.exp(-0.4)))
    F = np.array([[0.0, 1.0], [-4.0, -0.4]])
    assert np.allclose(discretize(F, np.array([[0.0], [1.0]]), 1.0, 0.05)[0], expm(F * 0.05))


@pytest.mark.parametrize("key", ["constant_velocity", "ornstein_uhlenbeck", "mass_spring"])
def test_matched_filters_are_consistent(key):
    """For the linear scenarios the default (matched) filter must pass the Monte Carlo test."""
    sc = get_scenario(key)
    system = {"duration": 6.0}
    if key == "constant_velocity":
        system["speed_on"] = True
    mc = MonteCarlo(sc.source(system), {"KF": sc.make_filter("kf", sc.filter_defaults("kf", system))}, runs=60)
    # The initial estimate is deliberately not matched to the true x0, so skip nothing but
    # check the innovations, which are the most robust indicator.
    result = mc.run()
    assert mc_nis_consistency(result, "KF").fraction_inside >= 0.85


def test_pendulum_small_angle_model_degrades_with_angle():
    sc = get_scenario("pendulum")

    def mean_nis(angle):
        system = {"duration": 8.0, "angle0": angle, "sigma": 0.01}
        params = sc.filter_defaults("kf", system) | {"x0_0": math.radians(angle), "P0_0": 1e-3, "P0_1": 1e-3}
        mc = MonteCarlo(sc.source(system), {"KF": sc.make_filter("kf", params)}, runs=10)
        return mc_nis_consistency(mc.run(), "KF").mean

    small, large = mean_nis(5.0), mean_nis(120.0)
    assert small < 3 and large > 10 * small


# ----------------------------------------------------------------- web API
def config(**kw):
    base = {
        "scenario": "constant_velocity",
        "system": {"duration": 4.0},
        "filters": [
            {"name": "A", "type": "kf", "params": {}},
            {"name": "B", "type": "kf", "params": {"gps_R": 4.0}},
        ],
        "mode": "single",
        "seed": 3,
    }
    base.update(kw)
    return json.dumps(base)


def test_scenarios_endpoint():
    data = json.loads(web_api.scenarios())
    assert [d["key"] for d in data] == list(SCENARIOS)
    assert json.loads(web_api.filter_defaults("ornstein_uhlenbeck", "kf", '{"theta": 3}'))["theta"] == 3


def test_single_run_session_streams_everything():
    sid = web_api.create_session(config())
    chunks = []
    while True:
        chunk = json.loads(web_api.advance(sid, json.dumps({"budget": 0.002})))
        chunks.append(chunk)
        if chunk["done"]:
            break
    web_api.close_session(sid)
    assert "info" in chunks[0] and "info" not in chunks[1]
    assert chunks[0]["info"]["filters"] == ["A", "B"] and chunks[0]["info"]["sensors"] == ["gps"]
    t = sum((c["t"] for c in chunks), [])
    assert np.allclose(t, np.arange(0, 4.0 + 0.025, 0.05))
    means = sum((c["estimates"]["A"]["mean"] for c in chunks), [])
    assert len(means) == len(t) and len(means[0]) == 2
    z = sum((c["measurements"] for c in chunks), [])
    assert [m["t"] for m in z] == pytest.approx(list(np.arange(0, 4.01, 0.5)))
    final = chunks[-1]
    assert final["progress"] == pytest.approx(1.0)
    assert final["metrics"]["header"][0] == "Filter" and len(final["metrics"]["rows"]) == 2
    assert set(final["current"]) == {"A", "B"} and len(final["current"]["A"]["var"]) == 2


def test_single_run_matches_runner_with_same_seed():
    from filter_tester.experiment import Runner

    sid = web_api.create_session(config())
    means = []
    while True:
        chunk = json.loads(web_api.advance(sid, '{"budget": 1}'))
        means += chunk["estimates"]["B"]["mean"]
        if chunk["done"]:
            break
    sc = get_scenario("constant_velocity")
    system = {"duration": 4.0}
    kf = sc.make_filter("kf", {"gps_R": 4.0})
    expected = Runner(sc.source(system), {"B": kf}).run(rng=3)["B"].mean
    assert np.allclose(means, expected)


def test_until_t_paces_the_run():
    sid = web_api.create_session(config())
    chunk = json.loads(web_api.advance(sid, json.dumps({"budget": 1, "until_t": 1.0})))
    assert chunk["t"][-1] == pytest.approx(1.0) and not chunk["done"]


def test_monte_carlo_session():
    sid = web_api.create_session(config(mode="montecarlo", runs=6, seed=10))
    first = json.loads(web_api.advance(sid, '{"budget": 0}'))
    assert first["runs_done"] == 1 and not first["done"] and "info" in first
    while True:
        chunk = json.loads(web_api.advance(sid, '{"budget": 1}'))
        if chunk["done"]:
            break
    assert chunk["runs_done"] == 6
    assert set(chunk["nees"]) == {"A", "B"} and len(chunk["rmse_total"]["A"]) == 6
    stats = chunk["rmse_stats"]["A"]
    assert len(stats["mean"]) == 3 and np.isclose(stats["var"][2], np.var(chunk["rmse_total"]["A"]))
    assert chunk["metrics"]["header"][4].startswith("ANEES")


@pytest.mark.parametrize(
    "bad",
    [
        dict(filters=[]),
        dict(filters=[{"name": "A", "type": "kf"}, {"name": "A", "type": "kf"}]),
        dict(mode="other"),
        dict(mode="montecarlo", runs=0),
        dict(filters=[{"name": "A", "type": "nope"}]),
    ],
)
def test_invalid_configs(bad):
    with pytest.raises((ValueError, KeyError)):
        web_api.create_session(config(**bad))


def test_json_has_no_nan():
    from filter_tester.web_api import _dump

    assert json.loads(_dump({"a": np.array([1.0, np.nan]), "b": np.float64(np.inf)})) == {"a": [1.0, None], "b": None}


def test_current_estimate_is_kept_in_a_final_empty_chunk():
    sid = web_api.create_session(config())
    json.loads(web_api.advance(sid, '{"budget": 5}'))  # processes every step
    final = json.loads(web_api.advance(sid, '{"budget": 5}'))  # nothing left to do
    assert final["done"] and final["t"] == [] and set(final["current"]) == {"A", "B"}
