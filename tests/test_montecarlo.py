import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from filter_tester.experiment import MonteCarlo, Runner  # noqa: E402
from filter_tester.filters import KalmanFilter  # noqa: E402
from filter_tester.metrics import (  # noqa: E402
    format_mc_summary,
    mc_nees,
    mc_nees_consistency,
    mc_nis,
    mc_nis_consistency,
    mc_summary_table,
    nees,
    rmse,
)
from filter_tester.models import LinearMeasurementModel, LinearProcessModel  # noqa: E402
from filter_tester.noise import GaussianNoise  # noqa: E402
from filter_tester.plotting import (  # noqa: E402
    LiveMonteCarloPlot,
    plot_mc_consistency,
    plot_mc_summary_table,
    plot_rmse_boxplot,
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


def make_source(duration=5.0, dt=0.1):
    system = DiscreteSystem(
        f=lambda x, w, t, dt: cv_A(dt) @ x + w, x0=GaussianNoise(np.eye(2)), noise=GaussianNoise(cv_Q(dt))
    )
    gps = Sensor("gps", lambda x, v, t: x[:1] + v, noise=GaussianNoise([[0.25]]), period=0.5)
    return SimulatedSource(system, [gps], times=np.arange(0.0, duration + dt / 2, dt))


FILTERS = lambda: {"matched": make_kf(), "optimistic": make_kf(R=0.01), "pessimistic": make_kf(R=4.0, Q_scale=20)}  # noqa: E731


def test_shapes_and_seeds():
    source = make_source()
    result = MonteCarlo(source, {"KF": make_kf()}, runs=7).run()
    s = result["KF"]
    K = len(result.t)
    assert result.seeds == list(range(7)) and result.n_runs == 7
    assert s.nees.shape == (7, K)
    assert s.nis.shape == (7, s.nis_t.size) and np.all(s.nis_dim == 1)
    assert s.rmse.shape == (7, 2) and s.rmse_total.shape == (7,) and s.step_time.shape == (7,)
    assert result.runs is None


def test_run_k_matches_runner_with_seed_k():
    source = make_source()
    result = MonteCarlo(source, {"KF": make_kf()}, runs=3).run(keep_runs=True)
    single = Runner(source, {"KF": make_kf()}).run(rng=2)["KF"]
    assert np.allclose(result["KF"].nees[2], nees(single)[1])
    assert np.allclose(result["KF"].rmse[2], rmse(single))
    assert len(result.runs) == 3
    assert np.array_equal(result.runs[2]["KF"].mean, single.mean)


def test_requires_a_simulated_source(tmp_path):
    (tmp_path / "gps.csv").write_text("t,x\n0,0.1\n1,0.9\n")
    with pytest.raises(TypeError):
        MonteCarlo(CSVSource(tmp_path / "gps.csv"), {"KF": make_kf()}, runs=2).run()
    with pytest.raises(ValueError):
        MonteCarlo(make_source(), {"KF": make_kf()}, runs=0)


def test_per_instant_bounds():
    result = MonteCarlo(make_source(), {"KF": make_kf()}, runs=50).run()
    t, mean, lower, upper = mc_nees(result, "KF")
    from scipy.stats import chi2

    assert np.allclose(lower, chi2.ppf(0.025, 100) / 50) and np.allclose(upper, chi2.ppf(0.975, 100) / 50)
    t, mean, lower, upper = mc_nis(result, "KF")
    assert np.allclose(upper, chi2.ppf(0.975, 50) / 50)


def test_verdicts_distinguish_matched_optimistic_pessimistic():
    result = MonteCarlo(make_source(10.0), FILTERS(), runs=100).run()
    nees_v = {name: mc_nees_consistency(result, name).verdict for name in result.filter_names}
    nis_v = {name: mc_nis_consistency(result, name).verdict for name in result.filter_names}
    expected = {"matched": "consistent", "optimistic": "optimistic", "pessimistic": "pessimistic"}
    assert nees_v == expected and nis_v == expected
    # The matched filter stays inside its per-instant bounds about 95% of the time.
    assert mc_nees_consistency(result, "matched").fraction_inside >= 0.85
    assert mc_nees_consistency(result, "optimistic").fraction_inside < 0.2


def test_matched_filter_false_alarm_rate():
    """The overall verdict for a matched filter should be 'consistent' in most experiments."""
    mc = MonteCarlo(make_source(5.0), {"KF": make_kf()}, runs=20)
    ok = sum(mc_nees_consistency(mc.run(first_seed=20 * e), "KF").consistent for e in range(20))
    assert ok >= 16  # nominal 19/20


def test_first_seed_offsets_the_seeds():
    result = MonteCarlo(make_source(), {"KF": make_kf()}, runs=3).run(first_seed=10)
    assert result.seeds == [10, 11, 12]


def test_summary_table():
    result = MonteCarlo(make_source(), FILTERS(), runs=10).run()
    header, body = mc_summary_table(result, ["pos", "vel"])
    assert header[:3] == ["Filter", "RMSE pos", "RMSE vel"] and "ANEES" in header[4]
    assert [r[0] for r in body] == ["matched", "optimistic", "pessimistic"]
    assert "±" in body[0][1] and "inside" in body[0][4]
    text = format_mc_summary(result, ["pos", "vel"])
    assert text.startswith("Monte Carlo: 10 runs (seeds 0–9)")


def test_observers_see_growing_partial_results():
    seen = []
    MonteCarlo(make_source(), {"KF": make_kf()}, runs=4).run(
        observers=[lambda k, run, partial: seen.append((k, partial.n_runs, partial["KF"].nees.shape[0]))]
    )
    assert seen == [(0, 1, 1), (1, 2, 2), (2, 3, 3), (3, 4, 4)]


def test_plots():
    result = MonteCarlo(make_source(), FILTERS(), runs=10).run()
    fig, axes = plot_mc_consistency(result)
    assert len(axes) == 2 and axes[0].get_yscale() == "log"
    fig, axes = plot_rmse_boxplot(result, labels=["pos", "vel"])
    assert [ax.get_title() for ax in axes] == ["RMSE pos", "RMSE vel", "RMSE total"]
    fig, ax = plot_mc_summary_table(result, ["pos", "vel"])
    assert ax.tables
    with pytest.raises(ValueError):
        plot_mc_consistency(result, filters=["nope"])


def test_live_plot_tracks_runs():
    live = LiveMonteCarloPlot(refresh_interval=0)
    MonteCarlo(make_source(), FILTERS(), runs=5).run(observers=[live])
    live.finish()
    assert live._latest.n_runs == 5
    assert "5 runs so far" in live.fig._suptitle.get_text()
    with pytest.raises(RuntimeError):
        live(5, None, live._latest)
