import numpy as np
import pytest

from filter_tester.noise import GaussianNoise, NoiseTerm
from filter_tester.sensors import Sensor
from filter_tester.systems import AnalyticSystem, DiscreteSystem, SDESystem, simulate


# ------------------------------------------------------------- measurement times
def test_period_grid_is_absolute_and_inclusive():
    sensor = Sensor("s", lambda x, v, t: x, period=0.5)
    assert np.allclose(sensor.measurement_times(0.0, 2.0), [0.0, 0.5, 1.0, 1.5, 2.0])
    assert np.allclose(sensor.measurement_times(0.2, 1.2), [0.5, 1.0])


def test_period_with_offset():
    sensor = Sensor("s", lambda x, v, t: x, period=1.0, offset=0.25)
    assert np.allclose(sensor.measurement_times(0.0, 3.0), [0.25, 1.25, 2.25])


def test_period_grid_does_not_drift_on_long_runs():
    sensor = Sensor("s", lambda x, v, t: x, period=0.1)
    times = sensor.measurement_times(0.0, 1000.0)
    assert times.size == 10001
    assert times[-1] == pytest.approx(1000.0, abs=1e-9)
    # Endpoints that are multiples of the period are included despite rounding.
    assert sensor.measurement_times(0.3, 0.3).size == 1


def test_explicit_times_are_filtered_to_window():
    sensor = Sensor("s", lambda x, v, t: x, times=[0.1, 0.7, 2.0, 5.0])
    assert np.allclose(sensor.measurement_times(0.5, 2.0), [0.7, 2.0])
    assert sensor.measurement_times(3.0, 4.0).size == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(),
        dict(period=1.0, times=[0.0]),
        dict(period=0.0),
        dict(times=[0.0, 0.0]),
        dict(times=[0.0, 1.0], offset=0.5),
    ],
)
def test_invalid_timing_is_rejected(kwargs):
    with pytest.raises(ValueError):
        Sensor("s", lambda x, v, t: x, **kwargs)


# ----------------------------------------------------------------- measurements
def test_measure_deterministic():
    sensor = Sensor("s", lambda x, v, t: [x[0] + t], period=1.0)
    assert np.allclose(sensor.measure([2.0, 5.0], 3.0), [5.0])


def test_measure_additive_noise_statistics():
    sensor = Sensor("s", lambda x, v, t: x[:1] + v, noise=GaussianNoise([[0.25]]), period=1.0)
    rng = np.random.default_rng(0)
    z = np.array([sensor.measure([1.0, 0.0], 0.0, rng)[0] for _ in range(20000)])
    assert abs(z.mean() - 1.0) < 0.01
    assert abs(z.var() - 0.25) < 0.01


def test_measure_non_additive_noise():
    # Range measurement with multiplicative noise.
    sensor = Sensor("s", lambda x, v, t: [np.linalg.norm(x) * (1 + v[0])], noise=GaussianNoise([[0.01]]), period=1.0)
    z = sensor.measure([3.0, 4.0], 0.0, np.random.default_rng(1))
    assert z.shape == (1,) and z[0] != 5.0


def test_measure_requires_rng_only_when_noise_is_drawn():
    noisy = Sensor("s", lambda x, v, t: x + v, noise=GaussianNoise([[1.0]]), period=1.0)
    with pytest.raises(ValueError):
        noisy.measure([0.0], 0.0)
    noisy.noise_enabled = False
    assert np.allclose(noisy.measure([0.0], 0.0), [0.0])


def test_measurement_names_checked():
    sensor = Sensor("s", lambda x, v, t: x, period=1.0, measurement_names=["a", "b"])
    with pytest.raises(ValueError):
        sensor.measure([1.0], 0.0)


# --------------------------------------------------------------- noise switch
def test_disabled_sensor_noise_gives_nominal_measurement():
    sensor = Sensor("s", lambda x, v, t: x[:1] + v, noise=GaussianNoise([[0.25]]), period=1.0, noise_enabled=False)
    assert np.allclose(sensor.measure([1.0, 0.0], 0.0, np.random.default_rng(0)), [1.0])


def make_cv(**kwargs):
    A = np.array([[1.0, 0.1], [0.0, 1.0]])
    return DiscreteSystem(
        lambda x, w, t, dt: A @ x + w, x0=[0.0, 1.0], noise=GaussianNoise(np.diag([1e-2, 1e-2])), **kwargs
    )


def test_disabled_system_noise_gives_nominal_trajectory():
    times = np.arange(0.0, 1.05, 0.1)
    nominal = simulate(make_cv(noise_enabled=False), times, rng=0)
    assert np.allclose(nominal[-1], [1.0, 1.0])
    noisy = simulate(make_cv(), times, rng=0)
    assert not np.allclose(noisy, nominal)


def test_system_noise_can_be_toggled_between_runs():
    system = make_cv()
    times = np.arange(0.0, 1.05, 0.1)
    system.noise_enabled = False
    a = simulate(system, times, rng=0)
    system.noise_enabled = True
    b = simulate(system, times, rng=0)
    assert np.allclose(a[-1], [1.0, 1.0]) and not np.allclose(a, b)


def test_disabled_noise_in_analytic_and_sde():
    wave = AnalyticSystem(lambda t, w: [np.sin(t + w[0])], GaussianNoise([[0.1]]), noise_enabled=False)
    times = np.linspace(0, 3, 10)
    assert np.allclose(simulate(wave, times, rng=0)[:, 0], np.sin(times))

    ou = SDESystem(f=lambda x, t: -x, G=lambda x, t: [1.0], x0=[1.0], max_step=1e-4, noise_enabled=False)
    ou.reset(0.0, rng=0)
    assert np.isclose(ou.advance(1.0)[0], np.exp(-1.0), rtol=1e-3)


def test_noise_term_zero_shape_from_callable_does_not_touch_run_rng():
    term = NoiseTerm(lambda rng: rng.normal(size=3))
    rng = np.random.default_rng(5)
    state_before = rng.bit_generator.state
    assert np.array_equal(term.draw(rng, enabled=False), np.zeros(3))
    assert rng.bit_generator.state == state_before


@pytest.mark.parametrize("name", [None, "", "   ", 3])
def test_name_is_required(name):
    with pytest.raises(ValueError):
        Sensor(name, lambda x, v, t: x, period=1.0)
