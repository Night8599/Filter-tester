import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from filter_tester.plotting import LiveStatePlot, plot_states, plot_system  # noqa: E402
from filter_tester.systems import DiscreteSystem, GaussianNoise, simulate  # noqa: E402


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def make_cv(state_names=None):
    A = np.array([[1.0, 0.1], [0.0, 1.0]])
    return DiscreteSystem(
        lambda x, w, t, dt: A @ x + w,
        x0=[0.0, 1.0],
        noise=GaussianNoise(np.diag([1e-4, 1e-3])),
        state_names=state_names,
    )


TIMES = np.arange(0.0, 5.0, 0.1)


def ylabels(axes):
    return [ax.get_ylabel() for ax in axes]


def test_plot_states_one_subplot_per_state():
    states = simulate(make_cv(), TIMES, rng=0)
    fig, axes = plot_states(TIMES, states)
    assert len(axes) == 2
    assert ylabels(axes) == ["x[0]", "x[1]"]
    assert axes[-1].get_xlabel() == "t"
    for i, ax in enumerate(axes):
        x, y = ax.lines[0].get_data()
        assert np.allclose(x, TIMES) and np.allclose(y, states[:, i])


def test_plot_states_accepts_1d_states():
    fig, axes = plot_states(TIMES, np.sin(TIMES))
    assert len(axes) == 1


def test_plot_states_overlay_on_existing_axes():
    fig, axes = plot_states(TIMES, simulate(make_cv(), TIMES, rng=0))
    fig2, axes2 = plot_states(TIMES, simulate(make_cv(), TIMES, rng=1), axes=axes)
    assert fig2 is fig
    assert all(len(ax.lines) == 2 for ax in axes)


@pytest.mark.parametrize(
    "kwargs",
    [dict(labels=["only one"]), dict(axes=[None])],
)
def test_plot_states_rejects_wrong_counts(kwargs):
    states = simulate(make_cv(), TIMES, rng=0)
    if "axes" in kwargs:
        _, one_axis = plot_states(TIMES, states[:, :1])
        kwargs = dict(axes=one_axis)
    with pytest.raises(ValueError):
        plot_states(TIMES, states, **kwargs)


def test_plot_states_rejects_length_mismatch():
    with pytest.raises(ValueError):
        plot_states(TIMES, np.zeros((len(TIMES) - 1, 2)))


def test_labels_come_from_system_and_can_be_overridden():
    system = make_cv(state_names=["position", "velocity"])
    _, axes, _ = plot_system(system, TIMES, rng=0)
    assert ylabels(axes) == ["position", "velocity"]
    _, axes, _ = plot_system(system, TIMES, rng=0, labels=["p", "v"])
    assert ylabels(axes) == ["p", "v"]


def test_system_rejects_wrong_number_of_state_names():
    with pytest.raises(ValueError):
        make_cv(state_names=["only one"]).reset()


@pytest.mark.parametrize("live", [False, True])
def test_plot_system_returns_simulated_states(live):
    fig, axes, states = plot_system(make_cv(), TIMES, rng=3, live=live)
    assert np.array_equal(states, simulate(make_cv(), TIMES, rng=3))
    for i, ax in enumerate(axes):
        x, y = ax.lines[0].get_data()
        assert np.allclose(x, TIMES) and np.allclose(y, states[:, i])


def test_live_plot_restores_interactive_mode():
    plt.ioff()
    plot_system(make_cv(), TIMES, rng=0, live=True)
    assert not plt.isinteractive()


def test_live_plot_throttles_redraws(monkeypatch):
    plot = LiveStatePlot(2, refresh_interval=3600)
    draws = []
    monkeypatch.setattr(plot, "_redraw", lambda: draws.append(len(plot._t)))
    for t in TIMES:
        plot.update(t, [t, 2 * t])
    assert draws == [1]  # only the first update is drawn immediately
    plot.finish()
    assert draws == [1, len(TIMES)]  # finish() draws everything


def test_live_plot_validates_input():
    plot = LiveStatePlot(2)
    with pytest.raises(ValueError):
        plot.update(0.0, [1.0, 2.0, 3.0])
    plot.finish()
    with pytest.raises(RuntimeError):
        plot.update(0.1, [1.0, 2.0])


def test_simulate_calls_on_step_for_every_time():
    seen = []
    states = simulate(make_cv(), TIMES, rng=0, on_step=lambda t, x: seen.append((t, x)))
    assert [t for t, _ in seen] == list(TIMES)
    assert np.allclose([x for _, x in seen], states)
