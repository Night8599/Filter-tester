"""Ready-made scenarios: a true system, its sensors and matching filter models.

Each :class:`Scenario` exposes the constants a user can change (with ranges,
for sliders), builds the simulated data source from the *true* constants and
builds filters from the constants each filter *believes*. Filter constants
default to the true ones, so a new filter starts with a matched model.

Scenarios:

- ``constant_velocity``: position / velocity with random acceleration.
- ``ornstein_uhlenbeck``: a process pulled back to zero (SDE).
- ``mass_spring``: damped oscillator driven by random force (SDE).
- ``pendulum``: nonlinear pendulum (SDE); the linear Kalman filter uses the
  small-angle model, which breaks down at large angles, while the EKF, UKF
  and particle filter use the nonlinear model.
- ``lorenz``: the chaotic Lorenz system (SDE), for the nonlinear filters only.

Every scenario offers each filter that applies to it (KF, EKF, UKF, PF).
They share the same model constants, and the UKF and PF add their own.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.linalg import expm

from .filters.base import Filter
from .filters.ekf import ExtendedKalmanFilter
from .filters.kalman import KalmanFilter
from .filters.particle import ParticleFilter
from .filters.ukf import UnscentedKalmanFilter
from .models.linear import LinearMeasurementModel, LinearProcessModel
from .models.nonlinear import ProcessModel, rk4
from .noise import GaussianNoise
from .sensors.sensor import Sensor
from .sources.simulated import SimulatedSource
from .systems.continuous import SDESystem
from .systems.discrete import DiscreteSystem


@dataclass(frozen=True)
class Param:
    """A user-adjustable constant.

    ``kind`` is ``"float"`` (slider between ``min`` and ``max``, logarithmic
    if ``log``), ``"int"`` or ``"bool"``. ``group`` groups parameters in the UI.
    """

    key: str
    label: str
    default: float | int | bool
    min: float | None = None
    max: float | None = None
    step: float | None = None
    log: bool = False
    kind: str = "float"
    group: str = ""
    help: str = ""

    def coerce(self, value) -> float | int | bool:
        """Convert ``value`` to this parameter's kind and check its range."""
        if self.kind == "bool":
            return bool(value)
        v = int(round(float(value))) if self.kind == "int" else float(value)
        if self.min is not None and v < self.min:
            raise ValueError(f"{self.label} must be >= {self.min}, got {v}")
        if self.max is not None and v > self.max:
            raise ValueError(f"{self.label} must be <= {self.max}, got {v}")
        return v

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly description, used by the web interface."""
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass(frozen=True)
class FilterType:
    """A kind of filter a scenario can use, with its adjustable constants."""

    key: str
    label: str
    params: tuple[Param, ...]
    build: Callable[[Mapping[str, Any]], Filter]
    description: str = ""


@dataclass(frozen=True)
class Scenario:
    """A true system with its sensors, plus the filters that can estimate it.

    ``system_params`` are the true constants (they build the data source);
    each :class:`FilterType` has the constants a filter believes. Filter
    constants listed in ``matched_defaults`` default to the true values.
    """

    key: str
    title: str
    description: str
    state_names: tuple[str, ...]
    system_params: tuple[Param, ...]
    build_source: Callable[[Mapping[str, Any]], SimulatedSource]
    filter_types: tuple[FilterType, ...]
    #: For plotting: which state component each sensor measures directly.
    sensor_state: Mapping[str, int] = field(default_factory=dict)
    #: Maps a filter constant to the true constant it defaults to.
    matched_defaults: Mapping[str, str] = field(default_factory=dict)

    def system_values(self, values: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """System constants with defaults filled in and validated."""
        return _values(self.system_params, values)

    def filter_type(self, key: str) -> FilterType:
        """The :class:`FilterType` with the given key (``kf``, ``ekf``, ``ukf`` or ``pf``)."""
        for ft in self.filter_types:
            if ft.key == key:
                return ft
        raise KeyError(f"scenario {self.key!r} has no filter type {key!r}")

    def filter_defaults(self, filter_type: str, system: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Filter constants that match the (given or default) true system."""
        system = self.system_values(system)
        defaults = {p.key: p.default for p in self.filter_type(filter_type).params}
        for fkey, skey in self.matched_defaults.items():
            if fkey in defaults:
                defaults[fkey] = system[skey]
        return defaults

    def source(self, values: Mapping[str, Any] | None = None) -> SimulatedSource:
        """The simulated data source for the given true constants (missing keys use defaults)."""
        return self.build_source(self.system_values(values))

    def make_filter(self, filter_type: str, values: Mapping[str, Any] | None = None) -> Filter:
        """Build a filter of the given type with the given constants (missing keys use defaults)."""
        ft = self.filter_type(filter_type)
        return ft.build(_values(ft.params, values))

    def to_dict(self) -> dict[str, Any]:
        """Description for a user interface (JSON-friendly)."""
        return {
            "key": self.key,
            "title": self.title,
            "description": self.description,
            "state_names": list(self.state_names),
            "system_params": [p.to_dict() for p in self.system_params],
            "filter_types": [
                {
                    "key": ft.key,
                    "label": ft.label,
                    "description": ft.description,
                    "params": [p.to_dict() for p in ft.params],
                    "defaults": self.filter_defaults(ft.key),
                }
                for ft in self.filter_types
            ],
            "sensor_state": dict(self.sensor_state),
            "matched_defaults": dict(self.matched_defaults),
        }


def _values(params, values) -> dict[str, Any]:
    values = dict(values or {})
    unknown = set(values) - {p.key for p in params}
    if unknown:
        raise ValueError(f"unknown parameters {sorted(unknown)}")
    return {p.key: p.coerce(values.get(p.key, p.default)) for p in params}


# ----------------------------------------------------------- shared pieces
def _run_params(duration=30.0, dt=0.05):
    return (
        Param("duration", "Duration [s]", duration, 2.0, 200.0, 1.0, group="Run"),
        Param(
            "dt",
            "Truth time step [s]",
            dt,
            0.005,
            0.5,
            0.005,
            group="Run",
            help="Resolution of the true trajectory; the filters also predict at these times.",
        ),
    )


def _sensor_params(key, label, R, period, group, enabled=None):
    params = []
    if enabled is not None:
        params.append(Param(f"{key}_on", f"Use {label}", enabled, kind="bool", group=group))
    params += [
        Param(f"{key}_R", f"{label} noise variance R", R, 1e-4, 10.0, log=True, group=group),
        Param(f"{key}_period", f"{label} period [s]", period, 0.01, 10.0, 0.01, group=group),
    ]
    return tuple(params)


def _filter_init_params(names, x0, P0, group="Initial estimate", limit=20.0):
    params = []
    for i, n in enumerate(names):
        params.append(Param(f"x0_{i}", f"Initial {n}", x0[i], -limit, limit, 0.1, group=group))
    for i, n in enumerate(names):
        params.append(Param(f"P0_{i}", f"Initial variance of {n}", P0[i], 1e-4, 100.0, log=True, group=group))
    return tuple(params)


def _x0(v, n):
    return np.array([v[f"x0_{i}"] for i in range(n)])


def _P0(v, n):
    return np.diag([v[f"P0_{i}"] for i in range(n)])


def _times(v):
    return np.arange(0.0, v["duration"] + v["dt"] / 2, v["dt"])


def _sensor(name, index, v):
    return Sensor(
        name,
        lambda x, w, t, i=index: x[i : i + 1] + w,
        noise=GaussianNoise([[v[f"{name}_R"]]]),
        period=v[f"{name}_period"],
    )


def _H(index, n):
    H = np.zeros((1, n))
    H[0, index] = 1.0
    return H


UKF_PARAMS = (
    Param(
        "alpha",
        "Spread α",
        1.0,
        0.01,
        1.0,
        log=True,
        group="Unscented transform",
        help="How far the sigma points are from the mean (small α keeps them close).",
    ),
    Param(
        "beta",
        "Prior knowledge β",
        2.0,
        0.0,
        4.0,
        0.1,
        group="Unscented transform",
        help="2 is optimal for Gaussian distributions.",
    ),
    Param("kappa", "Secondary scaling κ", 0.0, 0.0, 3.0, 0.1, group="Unscented transform"),
)

PF_PARAMS = (
    Param(
        "n_particles",
        "Number of particles",
        500,
        20,
        5000,
        10,
        kind="int",
        group="Particles",
        help="More particles: more accurate and slower.",
    ),
    Param(
        "resample_threshold",
        "Resample when N_eff / N below",
        0.5,
        0.0,
        1.0,
        0.05,
        group="Particles",
        help="Resampling fights weight degeneracy; 1 = every update, 0 = never.",
    ),
    Param(
        "regularization",
        "Regularization (jitter after resampling)",
        0.5,
        0.0,
        2.0,
        0.05,
        group="Particles",
        help="0 = plain bootstrap filter; 1 = optimal-bandwidth jitter, which keeps the cloud diverse.",
    ),
    Param("seed", "Random seed", 0, 0, 100000, 1, kind="int", group="Particles"),
)

FILTER_DESCRIPTIONS = {
    "ekf": "Linearizes the model around the current estimate (Jacobians).",
    "ukf": "Propagates sigma points through the nonlinear model; no Jacobians.",
    "pf": "A weighted cloud of samples; handles any nonlinearity, at a higher cost.",
}


def _family(n, model_params, build_models, build_linear=None, kf_label="Kalman filter (KF)", kf_description=""):
    """The filter types of a scenario: KF (if a linear model exists), EKF, UKF and PF.

    ``build_models(values)`` returns ``(process, {sensor: measurement model})``,
    used by the EKF, UKF and PF; ``build_linear(values)`` returns the linear
    models used by the KF.
    """

    def init(v):
        return _x0(v, n), _P0(v, n)

    types = []
    if build_linear is not None:
        types.append(
            FilterType("kf", kf_label, model_params, lambda v: KalmanFilter(*build_linear(v), *init(v)), kf_description)
        )
    types += [
        FilterType(
            "ekf",
            "Extended Kalman filter (EKF)",
            model_params,
            lambda v: ExtendedKalmanFilter(*build_models(v), *init(v)),
            FILTER_DESCRIPTIONS["ekf"],
        ),
        FilterType(
            "ukf",
            "Unscented Kalman filter (UKF)",
            model_params + UKF_PARAMS,
            lambda v: UnscentedKalmanFilter(
                *build_models(v), *init(v), alpha=v["alpha"], beta=v["beta"], kappa=v["kappa"]
            ),
            FILTER_DESCRIPTIONS["ukf"],
        ),
        FilterType(
            "pf",
            "Particle filter (PF)",
            model_params + PF_PARAMS,
            lambda v: ParticleFilter(
                *build_models(v),
                *init(v),
                n_particles=v["n_particles"],
                resample_threshold=v["resample_threshold"],
                regularization=v["regularization"],
                seed=v["seed"],
            ),
            FILTER_DESCRIPTIONS["pf"],
        ),
    ]
    return tuple(types)


def _white_acceleration_Q(sigma):
    return lambda dt: sigma**2 * np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]])


def discretize(F: np.ndarray, L: np.ndarray, q: float, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Exact discretization of ``dx = F x dt + L dW`` with ``E[dW dW^T] = q dt``.

    Returns ``(A, Q)`` with ``A = exp(F dt)`` and ``Q`` from Van Loan's method.
    """
    n = F.shape[0]
    M = np.zeros((2 * n, 2 * n))
    M[:n, :n] = -F
    M[:n, n:] = q * (L @ L.T)
    M[n:, n:] = F.T
    E = expm(M * dt)
    A = E[n:, n:].T
    Q = A @ E[:n, n:]
    return A, 0.5 * (Q + Q.T)


# ------------------------------------------------------- constant velocity
def _cv_matrices(q):
    A = lambda dt: np.array([[1.0, dt], [0.0, 1.0]])  # noqa: E731
    Q = lambda dt: q * np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]])  # noqa: E731
    return A, Q


def _cv_source(v):
    A, Q = _cv_matrices(v["q"])
    system = DiscreteSystem(
        f=lambda x, w, t, h: A(h) @ x + np.linalg.cholesky(Q(h) + 1e-15 * np.eye(2)) @ w,
        x0=[v["x0_pos"], v["x0_vel"]],
        noise=GaussianNoise(np.eye(2)),
        state_names=["position", "velocity"],
    )
    sensors = [_sensor("gps", 0, v)]
    if v["speed_on"]:
        sensors.append(_sensor("speed", 1, v))
    return SimulatedSource(system, sensors, _times(v))


def _cv_models(v):
    A, Q = _cv_matrices(v["q"])
    models = {}
    if v["use_gps"]:
        models["gps"] = LinearMeasurementModel(_H(0, 2), [[v["gps_R"]]])
    if v["use_speed"]:
        models["speed"] = LinearMeasurementModel(_H(1, 2), [[v["speed_R"]]])
    return LinearProcessModel(A, Q), models


CONSTANT_VELOCITY = Scenario(
    key="constant_velocity",
    title="Constant velocity",
    description="An object moving in a line with random acceleration (white-noise acceleration model). "
    "A GPS measures the position and, optionally, a speedometer measures the velocity.",
    state_names=("position", "velocity"),
    system_params=_run_params()
    + (
        Param("q", "Acceleration noise intensity q", 0.05, 1e-4, 10.0, log=True, group="System"),
        Param("x0_pos", "Initial position", 0.0, -20.0, 20.0, 0.1, group="System"),
        Param("x0_vel", "Initial velocity", 1.0, -10.0, 10.0, 0.1, group="System"),
    )
    + _sensor_params("gps", "GPS", 0.25, 0.5, "GPS (position)")
    + _sensor_params("speed", "Speedometer", 0.01, 0.2, "Speedometer (velocity)", enabled=False),
    build_source=_cv_source,
    filter_types=_family(
        2,
        (
            Param(
                "q",
                "Process noise intensity q",
                0.05,
                1e-4,
                10.0,
                log=True,
                group="Process model",
                help="How much the filter expects the velocity to change.",
            ),
            Param("use_gps", "Use GPS", True, kind="bool", group="Measurements"),
            Param("gps_R", "GPS noise variance R", 0.25, 1e-4, 10.0, log=True, group="Measurements"),
            Param("use_speed", "Use speedometer", True, kind="bool", group="Measurements"),
            Param("speed_R", "Speedometer noise variance R", 0.01, 1e-4, 10.0, log=True, group="Measurements"),
        )
        + _filter_init_params(("position", "velocity"), (0.0, 0.0), (1.0, 1.0)),
        build_models=_cv_models,
        build_linear=_cv_models,
        kf_label="Kalman filter (KF)",
        kf_description="Constant-velocity model with white-noise acceleration.",
    ),
    sensor_state={"gps": 0, "speed": 1},
    matched_defaults={"q": "q", "gps_R": "gps_R", "speed_R": "speed_R"},
)


# ------------------------------------------------------- Ornstein-Uhlenbeck
def _ou_source(v):
    theta, sigma = v["theta"], v["sigma"]
    system = SDESystem(
        method="heun",
        f=lambda x, t: -theta * x,
        G=lambda x, t: [sigma],
        x0=[v["x0"]],
        max_step=min(v["dt"] / 4, 0.01),
        state_names=["x"],
    )
    return SimulatedSource(system, [_sensor("sensor", 0, v)], _times(v))


def _ou_models(v):
    theta, sigma = v["theta"], v["sigma"]
    process = LinearProcessModel(
        A=lambda dt: np.array([[math.exp(-theta * dt)]]),
        Q=lambda dt: np.array([[sigma**2 / (2 * theta) * (1 - math.exp(-2 * theta * dt))]]),
    )
    return process, {"sensor": LinearMeasurementModel([[1.0]], [[v["sensor_R"]]])}


ORNSTEIN_UHLENBECK = Scenario(
    key="ornstein_uhlenbeck",
    title="Ornstein–Uhlenbeck process",
    description="dx = −θ x dt + σ dW: a noisy quantity that keeps being pulled back to zero "
    "(e.g. a sensor bias or a velocity with friction). One sensor measures x.",
    state_names=("x",),
    system_params=_run_params(duration=20.0, dt=0.02)
    + (
        Param("theta", "Mean reversion θ", 1.0, 0.01, 10.0, log=True, group="System"),
        Param("sigma", "Volatility σ", 0.5, 0.01, 5.0, log=True, group="System"),
        Param("x0", "Initial x", 2.0, -10.0, 10.0, 0.1, group="System"),
    )
    + _sensor_params("sensor", "Sensor", 0.05, 0.25, "Sensor"),
    build_source=_ou_source,
    filter_types=_family(
        1,
        (
            Param("theta", "Model mean reversion θ", 1.0, 0.01, 10.0, log=True, group="Process model"),
            Param("sigma", "Model volatility σ", 0.5, 0.01, 5.0, log=True, group="Process model"),
            Param("sensor_R", "Sensor noise variance R", 0.05, 1e-4, 10.0, log=True, group="Measurements"),
        )
        + _filter_init_params(("x",), (0.0,), (4.0,)),
        build_models=_ou_models,
        build_linear=_ou_models,
        kf_label="Kalman filter (KF)",
        kf_description="Exact discretization of the OU process.",
    ),
    sensor_state={"sensor": 0},
    matched_defaults={"theta": "theta", "sigma": "sigma", "sensor_R": "sensor_R"},
)


# ------------------------------------------------------------- mass-spring
def _spring_F(omega, zeta):
    return np.array([[0.0, 1.0], [-(omega**2), -2.0 * zeta * omega]])


_L2 = np.array([[0.0], [1.0]])


def _linear_models_2d(F, sigma, v, sensor):
    cache: dict[float, tuple[np.ndarray, np.ndarray]] = {}

    def AQ(dt):  # the time steps repeat, so each (A, Q) is computed once
        key = round(dt, 12)
        if key not in cache:
            cache[key] = discretize(F, _L2, sigma**2, dt)
        return cache[key]

    process = LinearProcessModel(A=lambda dt: AQ(dt)[0], Q=lambda dt: AQ(dt)[1])
    return process, {sensor: LinearMeasurementModel(_H(0, 2), [[v[f"{sensor}_R"]]])}


def _spring_source(v):
    F = _spring_F(v["omega"], v["zeta"])
    sigma = v["sigma"]
    system = SDESystem(
        method="heun",
        f=lambda x, t: F @ x,
        G=lambda x, t: [0.0, sigma],
        x0=[v["x0_pos"], v["x0_vel"]],
        max_step=min(v["dt"] / 4, 0.005),
        state_names=["position", "velocity"],
    )
    return SimulatedSource(system, [_sensor("sensor", 0, v)], _times(v))


def _spring_models(v):
    return _linear_models_2d(_spring_F(v["omega"], v["zeta"]), v["sigma"], v, "sensor")


MASS_SPRING = Scenario(
    key="mass_spring",
    title="Mass–spring–damper",
    description="A damped oscillator ẍ = −ω² x − 2ζω ẋ + σ w driven by a random force. A sensor measures the position.",
    state_names=("position", "velocity"),
    system_params=_run_params(duration=20.0, dt=0.02)
    + (
        Param("omega", "Natural frequency ω [rad/s]", 2.0, 0.1, 20.0, log=True, group="System"),
        Param("zeta", "Damping ratio ζ", 0.1, 0.0, 2.0, 0.01, group="System"),
        Param("sigma", "Random force σ", 0.5, 0.01, 10.0, log=True, group="System"),
        Param("x0_pos", "Initial position", 1.0, -10.0, 10.0, 0.1, group="System"),
        Param("x0_vel", "Initial velocity", 0.0, -10.0, 10.0, 0.1, group="System"),
    )
    + _sensor_params("sensor", "Position sensor", 0.01, 0.1, "Sensor"),
    build_source=_spring_source,
    filter_types=_family(
        2,
        (
            Param("omega", "Model frequency ω [rad/s]", 2.0, 0.1, 20.0, log=True, group="Process model"),
            Param("zeta", "Model damping ratio ζ", 0.1, 0.0, 2.0, 0.01, group="Process model"),
            Param("sigma", "Model random force σ", 0.5, 0.01, 10.0, log=True, group="Process model"),
            Param("sensor_R", "Sensor noise variance R", 0.01, 1e-4, 10.0, log=True, group="Measurements"),
        )
        + _filter_init_params(("position", "velocity"), (0.0, 0.0), (1.0, 1.0)),
        build_models=_spring_models,
        build_linear=_spring_models,
        kf_label="Kalman filter (KF)",
        kf_description="Exact discretization of the oscillator.",
    ),
    sensor_state={"sensor": 0},
    matched_defaults={"omega": "omega", "zeta": "zeta", "sigma": "sigma", "sensor_R": "sensor_R"},
)


# ---------------------------------------------------------------- pendulum
G_ACCEL = 9.81


def _pendulum_source(v):
    w2, c, sigma = G_ACCEL / v["length"], v["damping"], v["sigma"]
    system = SDESystem(
        method="heun",
        f=lambda x, t: np.array([x[1], -w2 * math.sin(x[0]) - c * x[1]]),
        G=lambda x, t: [0.0, sigma],
        x0=[math.radians(v["angle0"]), v["omega0"]],
        max_step=min(v["dt"] / 4, 0.005),
        state_names=["angle [rad]", "angular velocity [rad/s]"],
    )
    return SimulatedSource(system, [_sensor("encoder", 0, v)], _times(v))


def _pendulum_linear(v):
    # Small-angle model: sin(angle) ~ angle, i.e. a mass-spring with w^2 = g / L, 2 zeta w = c.
    w = math.sqrt(G_ACCEL / v["length"])
    F = np.array([[0.0, 1.0], [-(w**2), -v["damping"]]])
    return _linear_models_2d(F, v["sigma"], v, "encoder")


def _pendulum_nonlinear(v):
    w2, c = G_ACCEL / v["length"], v["damping"]

    def deriv(X):
        return np.stack([X[..., 1], -w2 * np.sin(X[..., 0]) - c * X[..., 1]], axis=-1)

    process = ProcessModel(rk4(deriv, max_step=0.01), Q=_white_acceleration_Q(v["sigma"]), vectorized=True)
    return process, {"encoder": LinearMeasurementModel(_H(0, 2), [[v["encoder_R"]]])}


PENDULUM = Scenario(
    key="pendulum",
    title="Pendulum (nonlinear)",
    description="θ̈ = −(g/L) sin θ − c θ̇ + σ w. The linear Kalman filter uses the small-angle model "
    "sin θ ≈ θ: it works for small swings and degrades as the initial angle grows. An encoder measures the angle.",
    state_names=("angle [rad]", "angular velocity [rad/s]"),
    system_params=_run_params(duration=15.0, dt=0.02)
    + (
        Param("length", "Length L [m]", 1.0, 0.1, 10.0, log=True, group="System"),
        Param("damping", "Damping c [1/s]", 0.1, 0.0, 2.0, 0.01, group="System"),
        Param("sigma", "Random torque σ", 0.2, 0.001, 5.0, log=True, group="System"),
        Param("angle0", "Initial angle [deg]", 60.0, -179.0, 179.0, 1.0, group="System"),
        Param("omega0", "Initial angular velocity [rad/s]", 0.0, -10.0, 10.0, 0.1, group="System"),
    )
    + _sensor_params("encoder", "Encoder", 0.003, 0.1, "Encoder (angle)"),
    build_source=_pendulum_source,
    filter_types=_family(
        2,
        (
            Param("length", "Model length L [m]", 1.0, 0.1, 10.0, log=True, group="Process model"),
            Param("damping", "Model damping c [1/s]", 0.1, 0.0, 2.0, 0.01, group="Process model"),
            Param("sigma", "Model random torque σ", 0.2, 0.001, 5.0, log=True, group="Process model"),
            Param("encoder_R", "Encoder noise variance R", 0.003, 1e-4, 10.0, log=True, group="Measurements"),
        )
        + _filter_init_params(("angle", "angular velocity"), (0.0, 0.0), (1.0, 1.0)),
        build_models=_pendulum_nonlinear,
        build_linear=_pendulum_linear,
        kf_label="Kalman filter (KF, small-angle model)",
        kf_description="Linearized pendulum: sin θ ≈ θ.",
    ),
    sensor_state={"encoder": 0},
    matched_defaults={"length": "length", "damping": "damping", "sigma": "sigma", "encoder_R": "encoder_R"},
)


# ------------------------------------------------------------------- Lorenz
def _lorenz_deriv(sigma, rho, beta):
    def deriv(X):
        x, y, z = X[..., 0], X[..., 1], X[..., 2]
        return np.stack([sigma * (y - x), x * (rho - z) - y, x * y - beta * z], axis=-1)

    return deriv


def _lorenz_source(v):
    deriv = _lorenz_deriv(v["sigma"], v["rho"], v["beta"])
    noise = v["noise"]
    system = SDESystem(
        method="heun",
        f=lambda x, t: deriv(x),
        G=lambda x, t: [noise, noise, noise],
        x0=[v["x0_x"], v["x0_y"], v["x0_z"]],
        max_step=min(v["dt"] / 4, 0.0025),
        state_names=["x", "y", "z"],
    )
    sensors = [_sensor("x_sensor", 0, v)]
    if v["z_sensor_on"]:
        sensors.append(_sensor("z_sensor", 2, v))
    return SimulatedSource(system, sensors, _times(v))


def _lorenz_models(v):
    noise = v["noise"]
    process = ProcessModel(
        rk4(_lorenz_deriv(v["sigma"], v["rho"], v["beta"]), max_step=0.005),
        Q=lambda dt: noise**2 * dt * np.eye(3),
        vectorized=True,
    )
    models = {}
    if v["use_x_sensor"]:
        models["x_sensor"] = LinearMeasurementModel(_H(0, 3), [[v["x_sensor_R"]]])
    if v["use_z_sensor"]:
        models["z_sensor"] = LinearMeasurementModel(_H(2, 3), [[v["z_sensor_R"]]])
    return process, models


LORENZ = Scenario(
    key="lorenz",
    title="Lorenz system (chaotic)",
    description="ẋ = σ(y − x), ẏ = x(ρ − z) − y, ż = xy − βz, plus random noise. Small errors grow "
    "exponentially, so the filter must keep correcting. A sensor measures x and, optionally, another measures z. "
    "Only the nonlinear filters apply.",
    state_names=("x", "y", "z"),
    system_params=_run_params(duration=10.0, dt=0.01)
    + (
        Param("sigma", "σ", 10.0, 0.1, 30.0, 0.1, group="System"),
        Param("rho", "ρ", 28.0, 0.5, 50.0, 0.5, group="System"),
        Param("beta", "β", 8.0 / 3.0, 0.1, 10.0, 0.01, group="System"),
        Param("noise", "Random noise σ_w", 0.5, 0.001, 10.0, log=True, group="System"),
        Param("x0_x", "Initial x", -8.0, -30.0, 30.0, 0.1, group="System"),
        Param("x0_y", "Initial y", 7.0, -30.0, 30.0, 0.1, group="System"),
        Param("x0_z", "Initial z", 27.0, 0.0, 60.0, 0.1, group="System"),
    )
    + _sensor_params("x_sensor", "x sensor", 1.0, 0.05, "x sensor")
    + _sensor_params("z_sensor", "z sensor", 1.0, 0.1, "z sensor", enabled=False),
    build_source=_lorenz_source,
    filter_types=_family(
        3,
        (
            Param("sigma", "Model σ", 10.0, 0.1, 30.0, 0.1, group="Process model"),
            Param("rho", "Model ρ", 28.0, 0.5, 50.0, 0.5, group="Process model"),
            Param("beta", "Model β", 8.0 / 3.0, 0.1, 10.0, 0.01, group="Process model"),
            Param("noise", "Model noise σ_w", 0.5, 0.001, 10.0, log=True, group="Process model"),
            Param("use_x_sensor", "Use x sensor", True, kind="bool", group="Measurements"),
            Param("x_sensor_R", "x sensor noise variance R", 1.0, 1e-4, 10.0, log=True, group="Measurements"),
            Param("use_z_sensor", "Use z sensor", True, kind="bool", group="Measurements"),
            Param("z_sensor_R", "z sensor noise variance R", 1.0, 1e-4, 10.0, log=True, group="Measurements"),
        )
        + _filter_init_params(("x", "y", "z"), (-7.0, 6.0, 26.0), (4.0, 4.0, 4.0), limit=60.0),
        build_models=_lorenz_models,
    ),
    sensor_state={"x_sensor": 0, "z_sensor": 2},
    matched_defaults={
        "sigma": "sigma",
        "rho": "rho",
        "beta": "beta",
        "noise": "noise",
        "x_sensor_R": "x_sensor_R",
        "z_sensor_R": "z_sensor_R",
    },
)


SCENARIOS: dict[str, Scenario] = {
    s.key: s for s in (CONSTANT_VELOCITY, ORNSTEIN_UHLENBECK, MASS_SPRING, PENDULUM, LORENZ)
}


def get_scenario(key: str) -> Scenario:
    """The scenario with the given key, e.g. ``"pendulum"``."""
    try:
        return SCENARIOS[key]
    except KeyError:
        raise KeyError(f"unknown scenario {key!r}; available: {list(SCENARIOS)}") from None
