import numpy as np
import pytest

from filter_tester.core import Estimate
from filter_tester.filters import KalmanFilter
from filter_tester.models import LinearMeasurementModel, LinearProcessModel
from filter_tester.noise import GaussianNoise
from filter_tester.sensors import Sensor
from filter_tester.sources import SimulatedSource
from filter_tester.systems import DiscreteSystem


def cv_A(dt):
    return np.array([[1.0, dt], [0.0, 1.0]])


def cv_Q(dt, q=0.1):
    return q * np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]])


def make_kf(**kwargs):
    defaults = dict(
        process=LinearProcessModel(A=cv_A, Q=cv_Q),
        measurements={"gps": LinearMeasurementModel(H=[[1.0, 0.0]], R=[[0.25]])},
        x0=[0.0, 0.0],
        P0=np.eye(2),
    )
    defaults.update(kwargs)
    return KalmanFilter(**defaults)


# ------------------------------------------------------------------ Estimate
def test_estimate_validates_and_exposes_std():
    e = Estimate(t=1, mean=[1.0, 2.0], covariance=np.diag([4.0, 9.0]))
    assert np.allclose(e.std, [2.0, 3.0])
    with pytest.raises(ValueError):
        Estimate(t=0, mean=[1.0, 2.0], covariance=np.eye(3))


# ------------------------------------------------------------------- basics
def test_static_state_converges_to_sample_mean():
    # x constant, measured directly: the KF reduces to a running average.
    kf = KalmanFilter(
        process=LinearProcessModel(A=[[1.0]], Q=[[0.0]]),
        measurements={"s": LinearMeasurementModel(H=[[1.0]], R=[[1.0]])},
        x0=[0.0],
        P0=[[1e9]],
    )
    rng = np.random.default_rng(0)
    z = 5.0 + rng.standard_normal(400)
    for k, zk in enumerate(z):
        est = kf.step(float(k), {"s": [zk]})
    assert est.mean[0] == pytest.approx(z.mean(), abs=1e-6)
    assert est.covariance[0, 0] == pytest.approx(1.0 / len(z), rel=1e-3)


def test_first_step_starts_from_x0_p0_without_predicting():
    kf = make_kf()
    est = kf.step(5.0, {})
    assert est.t == 5.0
    assert np.array_equal(est.mean, [0.0, 0.0]) and np.array_equal(est.covariance, np.eye(2))


def test_first_step_with_measurement_updates_immediately():
    est = make_kf().step(0.0, {"gps": [1.0]})
    # Scalar update: gain P / (P + R) = 1 / 1.25 on the position.
    assert est.mean[0] == pytest.approx(0.8)
    assert est.covariance[0, 0] == pytest.approx(0.2)


def test_step_without_measurements_only_predicts():
    kf = make_kf(x0=[1.0, 2.0])
    kf.step(0.0, {})
    est = kf.step(0.5, {})
    A = cv_A(0.5)
    assert np.allclose(est.mean, A @ [1.0, 2.0])
    assert np.allclose(est.covariance, A @ np.eye(2) @ A.T + cv_Q(0.5))
    assert est.extras["innovations"] == {}


def test_predict_uses_the_elapsed_dt():
    seen = []
    kf = make_kf(process=LinearProcessModel(A=lambda dt: seen.append(dt) or np.eye(2), Q=np.zeros((2, 2))))
    for t in [0.0, 0.1, 0.4, 0.4, 1.0]:
        kf.step(t, {})
    assert np.allclose(seen, [0.1, 0.3, 0.6])  # same time twice: no prediction


def test_fixed_matrices_are_accepted():
    kf = make_kf(process=LinearProcessModel(A=cv_A(0.1), Q=cv_Q(0.1)))
    kf.step(0.0, {})
    assert np.allclose(kf.step(0.1, {}).covariance, cv_A(0.1) @ cv_A(0.1).T + cv_Q(0.1))


def test_unknown_sensors_are_ignored():
    kf = make_kf()
    est = kf.step(0.0, {"gps": [1.0], "speedometer": [3.0]})
    assert set(est.extras["innovations"]) == {"gps"}
    assert np.allclose(est.mean, make_kf().step(0.0, {"gps": [1.0]}).mean)


def test_two_sensors_in_one_step():
    models = {
        "gps": LinearMeasurementModel(H=[[1.0, 0.0]], R=[[0.25]]),
        "speedometer": LinearMeasurementModel(H=[[0.0, 1.0]], R=[[0.01]]),
    }
    est = make_kf(measurements=models).step(0.0, {"gps": [1.0], "speedometer": [2.0]})
    assert set(est.extras["innovations"]) == {"gps", "speedometer"}
    assert est.mean[1] == pytest.approx(2.0 / 1.01)


def test_reset_restores_initial_estimate():
    kf = make_kf()
    kf.step(0.0, {"gps": [3.0]})
    kf.step(1.0, {"gps": [4.0]})
    kf.reset()
    est = kf.step(7.0, {})
    assert np.array_equal(est.mean, [0.0, 0.0]) and est.t == 7.0


def test_innovation_matches_definition():
    kf = make_kf(x0=[1.0, 0.0])
    y, S = kf.step(0.0, {"gps": [3.0]}).extras["innovations"]["gps"]
    assert np.allclose(y, [2.0]) and np.allclose(S, [[1.25]])


def test_covariance_stays_symmetric_positive_definite():
    kf = make_kf()
    rng = np.random.default_rng(1)
    for k in range(2000):
        est = kf.step(0.01 * k, {"gps": [rng.normal()]})
    P = est.covariance
    assert np.array_equal(P, P.T)
    assert np.all(np.linalg.eigvalsh(P) > 0)


# --------------------------------------------------------------- validation
def test_time_must_not_decrease():
    kf = make_kf()
    kf.step(1.0, {})
    with pytest.raises(ValueError):
        kf.step(0.5, {})


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(P0=np.eye(3)),
        dict(P0=[[1.0, 0.5], [0.0, 1.0]]),
        dict(measurements={"gps": LinearMeasurementModel(H=[[1.0, 0.0, 0.0]], R=[[1.0]])}),
    ],
)
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        make_kf(**kwargs)


def test_measurement_model_validates_r():
    with pytest.raises(ValueError):
        LinearMeasurementModel(H=[[1.0, 0.0]], R=np.eye(2))


def test_wrong_measurement_size():
    with pytest.raises(ValueError):
        make_kf().step(0.0, {"gps": [1.0, 2.0]})


def test_wrong_process_matrix_size():
    kf = make_kf(process=LinearProcessModel(A=np.eye(3), Q=np.eye(3)))
    kf.step(0.0, {})
    with pytest.raises(ValueError):
        kf.step(1.0, {})


# -------------------------------------------------------------- consistency
def test_consistent_with_matched_model():
    """With the true model, average NEES ~ state dim and NIS ~ measurement dim."""
    dt, R = 0.1, 0.25
    system = DiscreteSystem(
        f=lambda x, w, t, dt: cv_A(dt) @ x + w,
        x0=GaussianNoise(np.eye(2)),  # matches the filter's x0, P0
        noise=GaussianNoise(cv_Q(dt)),
    )
    gps = Sensor("gps", lambda x, v, t: x[:1] + v, noise=GaussianNoise([[R]]), period=2 * dt)
    source = SimulatedSource(system, [gps], times=np.arange(0.0, 5.0 + dt / 2, dt))

    nees, nis = [], []
    kf = make_kf()
    for run in range(300):
        kf.reset()
        for step in source.steps(rng=run):
            est = kf.step(step.t, step.measurements)
            e = step.truth - est.mean
            nees.append(e @ np.linalg.solve(est.covariance, e))
            for y, S in est.extras["innovations"].values():
                nis.append(float(y @ np.linalg.solve(S, y)))
    assert np.mean(nees) == pytest.approx(2.0, rel=0.06)
    assert np.mean(nis) == pytest.approx(1.0, rel=0.06)
