import math

import numpy as np
import pytest

from filter_tester.experiment import MonteCarlo, Runner
from filter_tester.filters import ExtendedKalmanFilter, KalmanFilter, ParticleFilter, UnscentedKalmanFilter
from filter_tester.metrics import mc_nees_consistency, mc_nis_consistency
from filter_tester.models import (
    LinearMeasurementModel,
    LinearProcessModel,
    MeasurementModel,
    ProcessModel,
    as_process_model,
    numerical_jacobian,
    rk4,
)
from filter_tester.noise import GaussianNoise
from filter_tester.sensors import Sensor
from filter_tester.sources import SimulatedSource
from filter_tester.systems import DiscreteSystem, SDESystem


def cv_A(dt):
    return np.array([[1.0, dt], [0.0, 1.0]])


def cv_Q(dt, q=0.1):
    return q * np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]])


LINEAR = dict(
    process=LinearProcessModel(A=cv_A, Q=cv_Q),
    measurements={"gps": LinearMeasurementModel(H=[[1.0, 0.0]], R=[[0.25]])},
    x0=[0.0, 0.0],
    P0=np.eye(2),
)


def cv_source(duration=5.0):
    system = DiscreteSystem(
        f=lambda x, w, t, dt: cv_A(dt) @ x + w, x0=GaussianNoise(np.eye(2)), noise=GaussianNoise(cv_Q(0.1))
    )
    gps = Sensor("gps", lambda x, v, t: x[:1] + v, noise=GaussianNoise([[0.25]]), period=0.5)
    return SimulatedSource(system, [gps], times=np.arange(0.0, duration + 0.05, 0.1))


# ------------------------------------------------------------------- models
def test_numerical_jacobian():
    f = lambda x: np.array([x[0] ** 2 * x[1], np.sin(x[1])])  # noqa: E731
    x = np.array([1.5, 0.3])
    J = numerical_jacobian(f, x)
    assert np.allclose(J, [[2 * 1.5 * 0.3, 1.5**2], [0.0, math.cos(0.3)]], atol=1e-6)


def test_vectorized_and_plain_models_agree():
    plain = ProcessModel(lambda x, dt: np.array([x[0] + dt * x[1], math.sin(x[1])]), Q=np.eye(2))
    vec = ProcessModel(lambda X, dt: np.stack([X[..., 0] + dt * X[..., 1], np.sin(X[..., 1])], -1), Q=np.eye(2), vectorized=True)
    X = np.random.default_rng(0).normal(size=(5, 2))
    assert np.allclose(plain.propagate(X, 0.1), vec.propagate(X, 0.1))
    assert np.allclose(plain.f(X[0], 0.1), vec.f(X[0], 0.1))
    assert np.allclose(plain.jacobian(X[0], 0.1), vec.jacobian(X[0], 0.1), atol=1e-6)
    hm = MeasurementModel(lambda x: [np.hypot(x[0], x[1])], R=[[1.0]])
    hv = MeasurementModel(lambda X: np.hypot(X[..., 0], X[..., 1])[..., None], R=[[1.0]], vectorized=True)
    assert np.allclose(hm.measure_batch(X), hv.measure_batch(X)) and hv.h(X[0]).shape == (1,)


def test_linear_models_convert_exactly():
    pm = as_process_model(LINEAR["process"])
    assert np.allclose(pm.jacobian(np.zeros(2), 0.3), cv_A(0.3))
    assert np.allclose(pm.propagate(np.ones((3, 2)), 0.3), np.ones((3, 2)) @ cv_A(0.3).T)


def test_rk4_matches_exponential():
    f = rk4(lambda x: -2.0 * x, max_step=0.01)
    assert np.allclose(f(np.array([1.0]), 1.0), math.exp(-2.0), rtol=1e-8)
    assert f(np.ones((4, 1)), 0.5).shape == (4, 1)


# ---------------------------------------- linear systems: all equal the KF
def run(filter_, source, seed=3):
    return Runner(source, {"F": filter_}).run(rng=seed)["F"]


def test_ekf_equals_kf_on_linear_models():
    src = cv_source()
    kf, ekf = run(KalmanFilter(**LINEAR), src), run(ExtendedKalmanFilter(**LINEAR), src)
    assert np.allclose(kf.mean, ekf.mean) and np.allclose(kf.covariance, ekf.covariance)


def test_ukf_equals_kf_on_linear_models():
    src = cv_source()
    kf = run(KalmanFilter(**LINEAR), src)
    for alpha, kappa in [(1.0, 0.0), (0.5, 1.0)]:
        ukf = run(UnscentedKalmanFilter(**LINEAR, alpha=alpha, kappa=kappa), src)
        assert np.allclose(kf.mean, ukf.mean, atol=1e-9) and np.allclose(kf.covariance, ukf.covariance, atol=1e-9)


def test_ekf_with_numerical_jacobian_equals_kf():
    src = cv_source()
    process = ProcessModel(lambda x, dt: cv_A(dt) @ x, Q=cv_Q)  # no Jacobian given
    meas = {"gps": MeasurementModel(lambda x: x[:1], R=[[0.25]])}
    ekf = run(ExtendedKalmanFilter(process, meas, [0.0, 0.0], np.eye(2)), src)
    assert np.allclose(ekf.mean, run(KalmanFilter(**LINEAR), src).mean, atol=1e-6)


def test_particle_filter_approaches_kf():
    src = cv_source()
    kf = run(KalmanFilter(**LINEAR), src)
    pf = run(ParticleFilter(**LINEAR, n_particles=20000, seed=1), src)
    assert np.max(np.abs(pf.mean - kf.mean)) < 0.08
    assert np.allclose(pf.std, kf.std, rtol=0.15)


# ----------------------------------------------------------- unscented details
def test_sigma_points_reproduce_mean_and_covariance():
    ukf = UnscentedKalmanFilter(**LINEAR, alpha=0.7, beta=2.0, kappa=1.0)
    x, P = np.array([1.0, -2.0]), np.array([[2.0, 0.3], [0.3, 0.5]])
    X = ukf.sigma_points(x, P)
    assert X.shape == (5, 2)
    assert np.allclose(ukf.Wm @ X, x)
    D = X - x
    # The covariance weights add (1 - alpha^2 + beta) on the central point, which has zero deviation.
    assert np.allclose((ukf.Wc[:, None] * D).T @ D, P)
    assert np.isclose(ukf.Wm.sum(), 1.0)


@pytest.mark.parametrize("kwargs", [dict(alpha=0.0), dict(alpha=1.5), dict(alpha=0.1, kappa=-2.0)])
def test_ukf_parameter_validation(kwargs):
    with pytest.raises(ValueError):
        UnscentedKalmanFilter(**LINEAR, **kwargs)


# ------------------------------------------------------------ particle filter
def test_particle_filter_is_reproducible_and_resets():
    src = cv_source(2.0)
    a = run(ParticleFilter(**LINEAR, n_particles=300, seed=5), src)
    pf = ParticleFilter(**LINEAR, n_particles=300, seed=5)
    b = run(pf, src)
    c = run(pf, src)  # the Runner resets the filter, restarting the same random sequence
    assert np.array_equal(a.mean, b.mean) and np.array_equal(b.mean, c.mean)
    d = run(ParticleFilter(**LINEAR, n_particles=300, seed=6), src)
    assert not np.array_equal(a.mean, d.mean)


def test_particle_filter_weights_and_resampling():
    pf = ParticleFilter(**LINEAR, n_particles=1000, resample_threshold=1.0, seed=0)
    pf.step(0.0, {})
    assert np.isclose(pf.weights.sum(), 1.0) and pf.effective_sample_size == pytest.approx(1000)
    pf.step(0.1, {"gps": [3.0]})  # threshold 1.0: always resamples after an update
    assert pf.resample_count == 1 and pf.effective_sample_size == pytest.approx(1000)
    never = ParticleFilter(**LINEAR, n_particles=1000, resample_threshold=0.0, seed=0)
    never.step(0.0, {"gps": [3.0]})
    assert never.resample_count == 0 and never.effective_sample_size < 1000
    assert np.isclose(never.weights.sum(), 1.0)


@pytest.mark.parametrize("resampling", ["systematic", "multinomial"])
def test_resampling_schemes_keep_the_distribution(resampling):
    pf = ParticleFilter(**LINEAR, n_particles=5000, resampling=resampling, resample_threshold=0.0, seed=2)
    pf.step(0.0, {"gps": [1.0]})
    mean_before = pf.x.copy()
    pf._resample()
    pf._summarize()
    assert np.allclose(pf.x, mean_before, atol=0.05)


def test_particle_filter_handles_zero_process_noise():
    process = LinearProcessModel(A=cv_A, Q=np.zeros((2, 2)))
    pf = ParticleFilter(process, LINEAR["measurements"], [0.0, 0.0], np.eye(2), n_particles=200, resample_threshold=1.0)
    for k in range(30):
        est = pf.step(0.1 * k, {"gps": [0.0]})
    assert np.all(np.isfinite(est.mean)) and np.all(np.linalg.eigvalsh(est.covariance) > 0)


def test_particle_filter_survives_an_impossible_measurement():
    pf = ParticleFilter(**LINEAR, n_particles=100, seed=0)
    est = pf.step(0.0, {"gps": [1e9]})  # every likelihood underflows
    assert np.all(np.isfinite(est.mean)) and np.isclose(pf.weights.sum(), 1.0)


@pytest.mark.parametrize("kwargs", [dict(n_particles=1), dict(resample_threshold=1.5), dict(resampling="other")])
def test_particle_filter_validation(kwargs):
    with pytest.raises(ValueError):
        ParticleFilter(**LINEAR, **kwargs)


# ---------------------------------------------------- a strongly nonlinear case
def pendulum_setup(angle0_deg=120.0):
    g_L, c, sigma = 9.81, 0.1, 0.05
    deriv = lambda X: np.stack([X[..., 1], -g_L * np.sin(X[..., 0]) - c * X[..., 1]], -1)  # noqa: E731
    system = SDESystem(
        f=lambda x, t: deriv(x), G=lambda x, t: [0.0, sigma],
        x0=GaussianNoise(np.diag([0.01, 0.01]), mean=[math.radians(angle0_deg), 0.0]), max_step=0.002,
    )
    enc = Sensor("enc", lambda x, v, t: x[:1] + v, noise=GaussianNoise([[0.003]]), period=0.1)
    source = SimulatedSource(system, [enc], times=np.arange(0.0, 6.0 + 0.01, 0.02))
    Q = lambda dt: sigma**2 * np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]])  # noqa: E731
    process = ProcessModel(rk4(deriv, 0.005), Q=Q, vectorized=True)
    meas = {"enc": MeasurementModel(lambda X: X[..., :1], R=[[0.003]], vectorized=True)}
    lin = LinearProcessModel(A=lambda dt: _small_angle_A(g_L, c, dt), Q=Q)  # sin(angle) ~ angle
    init = dict(x0=[math.radians(angle0_deg), 0.0], P0=np.diag([0.01, 0.01]))
    return source, process, meas, lin, init


def _small_angle_A(g_L, c, dt):
    from scipy.linalg import expm

    return expm(np.array([[0.0, 1.0], [-g_L, -c]]) * dt)


def test_nonlinear_filters_beat_the_small_angle_kf_on_a_large_swing():
    source, process, meas, lin, init = pendulum_setup()
    filters = {
        "KF": KalmanFilter(lin, {"enc": LinearMeasurementModel([[1.0, 0.0]], [[0.003]])}, **init),
        "EKF": ExtendedKalmanFilter(process, meas, **init),
        "UKF": UnscentedKalmanFilter(process, meas, **init),
        "PF": ParticleFilter(process, meas, **init, n_particles=400, seed=0),
    }
    result = MonteCarlo(source, filters, runs=15).run()
    nis = {name: mc_nis_consistency(result, name) for name in filters}
    assert nis["KF"].verdict == "optimistic" and nis["KF"].mean > 20
    for name in ("EKF", "UKF", "PF"):
        assert nis[name].mean < 3, (name, nis[name])
    rmse = {name: result[name].rmse_total.mean() for name in filters}
    assert max(rmse["EKF"], rmse["UKF"], rmse["PF"]) < rmse["KF"] / 3
    assert mc_nees_consistency(result, "UKF").fraction_inside > 0.6


def test_batch_jacobian_matches_plain():
    vec = ProcessModel(lambda X, dt: np.stack([X[..., 0] * X[..., 1], np.sin(X[..., 1]) * dt], -1), Q=np.eye(2), vectorized=True)
    plain = ProcessModel(lambda x, dt: np.array([x[0] * x[1], math.sin(x[1]) * dt]), Q=np.eye(2))
    x = np.array([0.7, -1.2])
    assert np.allclose(vec.jacobian(x, 0.3), plain.jacobian(x, 0.3), atol=1e-8)
    hv = MeasurementModel(lambda X: (X[..., 0] ** 2 + X[..., 1])[..., None], R=[[1.0]], vectorized=True)
    assert np.allclose(hv.jacobian(x), [[2 * 0.7, 1.0]], atol=1e-6)


def test_regularization_keeps_particles_diverse():
    process = LinearProcessModel(A=cv_A, Q=np.zeros((2, 2)))  # no process noise: the worst case
    def distinct(reg):
        pf = ParticleFilter(process, LINEAR["measurements"], [0.0, 0.0], np.eye(2), n_particles=300,
                            resample_threshold=1.0, regularization=reg, seed=0)
        for k in range(20):
            pf.step(0.1 * k, {"gps": [0.0]})
        return len(np.unique(pf.particles.round(12), axis=0))
    assert distinct(0.0) < 100 and distinct(1.0) == 300
    with pytest.raises(ValueError):
        ParticleFilter(**LINEAR, regularization=-1.0)
