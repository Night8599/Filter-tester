import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from filter_tester.experiment import Runner  # noqa: E402
from filter_tester.filters import KalmanFilter  # noqa: E402
from filter_tester.metrics import (  # noqa: E402
    format_summary,
    mean_step_time,
    nees,
    nees_consistency,
    nis,
    nis_consistency,
    rmse,
    rmse_total,
    summary,
    summary_table,
)
from filter_tester.models import LinearMeasurementModel, LinearProcessModel  # noqa: E402
from filter_tester.noise import GaussianNoise  # noqa: E402
from filter_tester.plotting import (  # noqa: E402
    LiveFilterPlot,
    plot_errors,
    plot_estimates,
    plot_summary_table,
)
from filter_tester.sensors import Sensor  # noqa: E402
from filter_tester.sources import CSVSource, SimulatedSource  # noqa: E402
from filter_tester.systems import DiscreteSystem  # noqa: E402


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def cv_A(dt):
    return np.array([[1.0, dt], [0.0, 1.0]])


def cv_Q(dt, q=0.1):
    return q * np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]])


def make_kf(R=0.25, Q_scale=1.0):
    return KalmanFilter(
        process=LinearProcessModel(A=cv_A, Q=lambda dt: Q_scale * cv_Q(dt)),
        measurements={"gps": LinearMeasurementModel(H=[[1.0, 0.0]], R=[[R]])},
        x0=[0.0, 0.0],
        P0=np.eye(2),
    )


def make_source(duration=20.0):
    dt = 0.1
    system = DiscreteSystem(
        f=lambda x, w, t, dt: cv_A(dt) @ x + w, x0=GaussianNoise(np.eye(2)), noise=GaussianNoise(cv_Q(dt))
    )
    gps = Sensor("gps", lambda x, v, t: x[:1] + v, noise=GaussianNoise([[0.25]]), period=0.2)
    return SimulatedSource(system, [gps], times=np.arange(0.0, duration + dt / 2, dt))


# ------------------------------------------------------------------- runner
def test_runner_records_one_trajectory_per_filter():
    source = make_source(5.0)
    result = Runner(source, {"good": make_kf(), "bad": make_kf(R=0.01)}).run(rng=0)
    assert result.filter_names == ["good", "bad"]
    traj = result["good"]
    k = len(result.steps)
    assert traj.mean.shape == (k, 2) and traj.covariance.shape == (k, 2, 2)
    assert traj.truth.shape == (k, 2) and traj.has_truth.all()
    assert traj.step_time.shape == (k,) and np.all(traj.step_time >= 0)
    assert len(traj.innovations) == sum(s.has_measurements for s in result.steps)
    assert np.allclose(traj.std, np.sqrt(np.diagonal(traj.covariance, axis1=1, axis2=2)))


def test_runner_matches_manual_loop():
    source = make_source(3.0)
    result = Runner(source, {"KF": make_kf()}).run(rng=4)
    kf = make_kf()
    manual = np.array([kf.step(s.t, s.measurements).mean for s in source.steps(rng=4)])
    assert np.allclose(result["KF"].mean, manual)


def test_runner_resets_filters_between_runs():
    runner = Runner(make_source(3.0), {"KF": make_kf()})
    a, b = runner.run(rng=1), runner.run(rng=1)
    assert np.array_equal(a["KF"].mean, b["KF"].mean)


def test_runner_calls_observers_every_step():
    calls = []
    result = Runner(make_source(1.0), {"KF": make_kf()}).run(
        rng=0, observers=[lambda step, est: calls.append((step.t, set(est)))]
    )
    assert [t for t, _ in calls] == [s.t for s in result.steps]
    assert all(names == {"KF"} for _, names in calls)


def test_runner_rejects_shared_filter_objects():
    kf = make_kf()
    with pytest.raises(ValueError):
        Runner(make_source(1.0), {"a": kf, "b": kf})
    with pytest.raises(ValueError):
        Runner(make_source(1.0), {})


def test_runner_with_csv_without_truth(tmp_path):
    (tmp_path / "gps.csv").write_text("t,x\n0,0.1\n0.5,0.6\n1.0,0.9\n")
    result = Runner(CSVSource(tmp_path / "gps.csv"), {"KF": make_kf()}).run()
    traj = result["KF"]
    assert traj.truth is None and not traj.has_truth.any()
    row = summary(result)[0]
    assert row["rmse"] is None and row["nees"] is None and row["nis"] is not None
    with pytest.raises(ValueError):
        rmse(traj)
    with pytest.raises(TypeError):  # recorded data takes no rng
        Runner(CSVSource(tmp_path / "gps.csv"), {"KF": make_kf()}).run(rng=0)


# ------------------------------------------------------------------ metrics
def test_rmse_values():
    result = Runner(make_source(5.0), {"KF": make_kf()}).run(rng=0)
    traj = result["KF"]
    e = traj.mean - traj.truth
    assert np.allclose(rmse(traj), np.sqrt((e**2).mean(axis=0)))
    assert rmse_total(traj) == pytest.approx(np.sqrt((e**2).sum(axis=1).mean()))


def test_nees_and_nis_definitions():
    traj = Runner(make_source(2.0), {"KF": make_kf()}).run(rng=0)["KF"]
    t, values = nees(traj)
    k = 5
    e = traj.mean[k] - traj.truth[k]
    assert values[k] == pytest.approx(e @ np.linalg.inv(traj.covariance[k]) @ e)
    t_nis, v_nis, dims = nis(traj)
    _, _, y, S = traj.innovations[0]
    assert v_nis[0] == pytest.approx(float(y @ np.linalg.inv(S) @ y)) and np.all(dims == 1)
    with pytest.raises(ValueError):
        nis(traj, sensor="missing")


def test_consistency_verdicts():
    result = Runner(
        make_source(200.0),
        {"matched": make_kf(), "optimistic": make_kf(R=0.02), "pessimistic": make_kf(R=4.0, Q_scale=20.0)},
    ).run(rng=0)
    assert nees_consistency(result["matched"]).expected == 2
    verdicts = {name: nis_consistency(result[name]).verdict for name in result.filter_names}
    assert verdicts == {"matched": "consistent", "optimistic": "optimistic", "pessimistic": "pessimistic"}
    assert nees_consistency(result["optimistic"]).verdict == "optimistic"


def test_consistency_interval_shrinks_with_more_samples():
    short = nees_consistency(Runner(make_source(2.0), {"KF": make_kf()}).run(rng=0)["KF"])
    long = nees_consistency(Runner(make_source(50.0), {"KF": make_kf()}).run(rng=0)["KF"])
    assert (long.upper - long.lower) < (short.upper - short.lower)
    assert short.lower < 2 < short.upper


def test_summary_table_and_text():
    result = Runner(make_source(5.0), {"A": make_kf(), "B": make_kf(R=1.0)}).run(rng=0)
    header, body = summary_table(result, state_names=["pos", "vel"])
    assert header[:3] == ["Filter", "RMSE pos", "RMSE vel"]
    assert [row[0] for row in body] == ["A", "B"]
    assert all(len(row) == len(header) for row in body)
    text = format_summary(result, ["pos", "vel"])
    assert "RMSE pos" in text and "A" in text and len(text.splitlines()) == 4
    assert mean_step_time(result["A"]) > 0


# -------------------------------------------------------------------- plots
def test_plot_estimates_and_errors():
    result = Runner(make_source(2.0), {"A": make_kf(), "B": make_kf(R=1.0)}).run(rng=0)
    fig, axes = plot_estimates(result, labels=["pos", "vel"])
    assert len(axes) == 2 and [l.get_label() for l in axes[0].lines] == ["truth", "A", "B"]
    fig, axes = plot_errors(result, filters=["B"])
    assert [l.get_label() for l in axes[0].lines if not l.get_label().startswith("_")] == ["B"]
    with pytest.raises(ValueError):
        plot_estimates(result, filters=["C"])


def test_plot_errors_requires_truth(tmp_path):
    (tmp_path / "gps.csv").write_text("t,x\n0,0.1\n1,0.9\n")
    result = Runner(CSVSource(tmp_path / "gps.csv"), {"KF": make_kf()}).run()
    plot_estimates(result)  # fine without truth
    with pytest.raises(ValueError):
        plot_errors(result)


def test_plot_summary_table():
    result = Runner(make_source(2.0), {"A": make_kf()}).run(rng=0)
    fig, ax = plot_summary_table(result, labels=["pos", "vel"])
    assert ax.tables


def test_live_filter_plot_shows_final_data():
    live = LiveFilterPlot(labels=["pos", "vel"])
    result = Runner(make_source(2.0), {"A": make_kf(), "B": make_kf(R=1.0)}).run(rng=0, observers=[live])
    live.finish()
    for i, ax in enumerate(live.axes):
        assert np.allclose(live._lines["B"][i].get_ydata(), result["B"].mean[:, i])
        assert np.allclose(live._truth_lines[i].get_ydata(), result["B"].truth[:, i])
    with pytest.raises(RuntimeError):
        live(result.steps[0], {})


def test_effective_sample_size():
    from filter_tester.metrics import effective_sample_size

    rng = np.random.default_rng(0)
    white = rng.standard_normal(5000)
    assert effective_sample_size(white) == pytest.approx(5000, rel=0.1)
    # AR(1) with phi = 0.9: K_eff ~ K (1 - phi) / (1 + phi)
    ar = np.zeros(20000)
    for k in range(1, ar.size):
        ar[k] = 0.9 * ar[k - 1] + rng.standard_normal()
    assert effective_sample_size(ar) == pytest.approx(20000 * 0.1 / 1.9, rel=0.25)


def test_matched_filter_not_flagged_on_fine_truth_grid():
    """Regression: NEES on a fine grid is strongly correlated; most matched runs must pass."""
    dt = 0.05
    system = DiscreteSystem(
        f=lambda x, w, t, dt: cv_A(dt) @ x + w, x0=GaussianNoise(np.eye(2)), noise=GaussianNoise(cv_Q(dt))
    )
    gps = Sensor("gps", lambda x, v, t: x[:1] + v, noise=GaussianNoise([[0.25]]), period=0.5)
    source = SimulatedSource(system, [gps], times=np.arange(0.0, 30.0 + dt / 2, dt))
    runner = Runner(source, {"KF": make_kf()})
    passed = sum(nees_consistency(runner.run(rng=s)["KF"]).consistent for s in range(30))
    assert passed >= 24  # nominal 95%; the uncorrected interval passes only ~25%


def test_summary_table_colors_only_verdict_cells():
    result = Runner(make_source(20.0), {"optimistic": make_kf(R=0.01)}).run(rng=0)
    fig, ax = plot_summary_table(result)
    table = ax.tables[0]
    white = (1.0, 1.0, 1.0, 1.0)
    assert table[1, 0].get_facecolor() == white  # the filter name is not colored
    assert any(table[1, c].get_facecolor() != white for c in range(1, len(table.get_celld()) // 2))
