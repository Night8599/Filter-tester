import numpy as np
import pytest

from filter_tester.core import Step, measurement_series, truth_series
from filter_tester.noise import GaussianNoise
from filter_tester.sensors import Sensor
from filter_tester.sources import SimulatedSource
from filter_tester.systems import AnalyticSystem, DiscreteSystem


def make_cv(**kwargs):
    A = lambda dt: np.array([[1.0, dt], [0.0, 1.0]])  # noqa: E731
    return DiscreteSystem(
        lambda x, w, t, dt: A(dt) @ x + w,
        x0=[0.0, 1.0],
        noise=GaussianNoise(np.diag([1e-4, 1e-3])),
        **kwargs,
    )


def position_sensor(name="gps", period=0.5, **kwargs):
    return Sensor(name, lambda x, v, t: x[:1] + v, noise=GaussianNoise([[0.25]]), period=period, **kwargs)


def test_truth_grid_and_sensor_times_are_merged():
    times = np.arange(0.0, 1.01, 0.1)  # floating-point grid: 0.30000000000000004 etc.
    source = SimulatedSource(make_cv(), [position_sensor(period=0.3)], times)
    steps = source.run(rng=0)
    assert len(steps) == len(times)  # every sensor time lands on the grid
    assert np.allclose([s.t for s in steps], times)
    measured = [round(s.t, 9) for s in steps if s.has_measurements]
    assert measured == [0.0, 0.3, 0.6, 0.9]


def test_sensor_times_off_the_grid_add_steps():
    source = SimulatedSource(make_cv(), [position_sensor(period=0.25)], [0.0, 0.5, 1.0])
    steps = source.run(rng=0)
    assert [s.t for s in steps] == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert all(s.has_measurements for s in steps)


def test_simultaneous_measurements_share_one_step():
    gps = position_sensor("gps", period=0.5)
    speed = Sensor("speedometer", lambda x, v, t: x[1:], period=0.25)
    steps = SimulatedSource(make_cv(), [gps, speed], [0.0, 1.0]).run(rng=0)
    by_time = {s.t: set(s.measurements) for s in steps}
    assert by_time[0.5] == {"gps", "speedometer"}
    assert by_time[0.25] == {"speedometer"}
    assert len(steps) == 5


def test_measurements_come_from_the_true_state():
    exact = Sensor("exact", lambda x, v, t: x.copy(), period=0.1)
    steps = SimulatedSource(make_cv(), [exact], np.linspace(0, 1, 11)).run(rng=3)
    for s in steps:
        assert np.array_equal(s.measurements["exact"], s.truth)


def test_series_helpers():
    steps = SimulatedSource(make_cv(), [position_sensor(period=0.5)], np.linspace(0, 2, 41)).run(rng=0)
    t, x = truth_series(steps)
    assert x.shape == (41, 2)
    tz, z = measurement_series(steps, "gps")
    assert np.allclose(tz, [0.0, 0.5, 1.0, 1.5, 2.0]) and z.shape == (5, 1)
    assert measurement_series(steps, "missing")[0].size == 0
    assert truth_series([Step(t=0.0)])[0].size == 0


def test_runs_are_reproducible():
    source = SimulatedSource(make_cv(), [position_sensor()], np.linspace(0, 2, 21))
    a, b = source.run(rng=7), source.run(rng=7)
    assert all(np.array_equal(sa.truth, sb.truth) for sa, sb in zip(a, b, strict=True))
    gen_runs = [source.run(rng=np.random.default_rng(7)) for _ in range(2)]
    assert np.array_equal(truth_series(gen_runs[0])[1], truth_series(gen_runs[1])[1])


def test_sensor_noise_does_not_change_the_truth():
    gps = position_sensor()
    source = SimulatedSource(make_cv(), [gps], np.linspace(0, 2, 21))
    noisy = source.run(rng=1)
    gps.noise_enabled = False
    clean = source.run(rng=1)
    assert np.array_equal(truth_series(noisy)[1], truth_series(clean)[1])
    for s in clean:
        if s.has_measurements:
            assert np.allclose(s.measurements["gps"], s.truth[:1])


def test_initial_state_override():
    steps = SimulatedSource(make_cv(), [], [0.0, 1.0]).run(rng=0, x0=[10.0, 0.0])
    assert np.array_equal(steps[0].truth, [10.0, 0.0])


def test_steps_are_lazy():
    source = SimulatedSource(make_cv(), [position_sensor()], np.linspace(0, 1000, 100001))
    first = next(source.steps(rng=0))
    assert first.t == 0.0


def test_analytic_system_works_as_source():
    wave = AnalyticSystem(lambda t, w: [np.sin(t)])
    steps = SimulatedSource(wave, [Sensor("s", lambda x, v, t: x, period=1.0)], np.linspace(0, 3, 31)).run()
    assert np.allclose(truth_series(steps)[1][:, 0], np.sin(np.linspace(0, 3, 31)))


@pytest.mark.parametrize("times", [[], [1.0, 0.0]])
def test_invalid_times_rejected(times):
    with pytest.raises(ValueError):
        SimulatedSource(make_cv(), [], times)


def test_duplicate_sensor_names_rejected():
    with pytest.raises(ValueError):
        SimulatedSource(make_cv(), [position_sensor("a"), position_sensor("a")], [0.0, 1.0])
