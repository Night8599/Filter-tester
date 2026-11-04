"""Ready-made scenarios: a true system, its sensors and matching filter models.

Each :class:`Scenario` exposes the constants a user can change (with ranges,
for sliders), builds the simulated data source from the *true* constants and
builds filters from the constants each filter *believes*. Filter constants
default to the true ones, so a new filter starts with a matched model.

Scenarios:

- ``constant_velocity``: position / velocity with random acceleration.
- ``ornstein_uhlenbeck``: a process pulled back to zero (SDE).
- ``mass_spring``: damped oscillator driven by random force (SDE).
- ``pendulum``: nonlinear pendulum (SDE); the Kalman filter uses the
  small-angle linear model, which breaks down at large angles.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import numpy as np
from scipy.linalg import expm

from .filters.base import Filter
from .filters.kalman import KalmanFilter
from .models.linear import LinearMeasurementModel, LinearProcessModel
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
        if self.kind == "bool":
            return bool(value)
        v = int(round(float(value))) if self.kind == "int" else float(value)
        if self.min is not None and v < self.min:
            raise ValueError(f"{self.label} must be >= {self.min}, got {v}")
        if self.max is not None and v > self.max:
            raise ValueError(f"{self.label} must be <= {self.max}, got {v}")
        return v

    def to_dict(self) -> dict[str, Any]:
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
        return self.build_source(self.system_values(values))

    def make_filter(self, filter_type: str, values: Mapping[str, Any] | None = None) -> Filter:
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
        Param("dt", "Truth time step [s]", dt, 0.005, 0.5, 0.005, group="Run",
              help="Resolution of the true trajectory; the filters also predict at these times."),
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


def _filter_init_params(names, x0, P0, group="Initial estimate"):
    params = []
    for i, n in enumerate(names):
        params.append(Param(f"x0_{i}", f"Initial {n}", x0[i], -20.0, 20.0, 0.1, group=group))
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
        lambda x, w, t, i=index: x[i:i + 1] + w,
        noise=GaussianNoise([[v[f"{name}_R"]]]),
        period=v[f"{name}_period"],
    )


def _H(index, n):
    H = np.zeros((1, n))
    H[0, index] = 1.0
    return H


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


def _cv_filter(v):
    A, Q = _cv_matrices(v["q"])
    models = {}
    if v["use_gps"]:
        models["gps"] = LinearMeasurementModel(_H(0, 2), [[v["gps_R"]]])
    if v["use_speed"]:
        models["speed"] = LinearMeasurementModel(_H(1, 2), [[v["speed_R"]]])
    return KalmanFilter(LinearProcessModel(A, Q), models, _x0(v, 2), _P0(v, 2))


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
    filter_types=(
        FilterType(
            key="kf",
            label="Kalman filter (linear)",
            description="Constant-velocity model with white-noise acceleration.",
            params=(
                Param("q", "Process noise intensity q", 0.05, 1e-4, 10.0, log=True, group="Process model",
                      help="How much the filter expects the velocity to change."),
                Param("use_gps", "Use GPS", True, kind="bool", group="Measurements"),
                Param("gps_R", "GPS noise variance R", 0.25, 1e-4, 10.0, log=True, group="Measurements"),
                Param("use_speed", "Use speedometer", True, kind="bool", group="Measurements"),
                Param("speed_R", "Speedometer noise variance R", 0.01, 1e-4, 10.0, log=True, group="Measurements"),
            )
            + _filter_init_params(("position", "velocity"), (0.0, 0.0), (1.0, 1.0)),
            build=_cv_filter,
        ),
    ),
    sensor_state={"gps": 0, "speed": 1},
    matched_defaults={"q": "q", "gps_R": "gps_R", "speed_R": "speed_R"},
)


# ------------------------------------------------------- Ornstein-Uhlenbeck
def _ou_source(v):
    theta, sigma = v["theta"], v["sigma"]
    system = SDESystem(
        f=lambda x, t: -theta * x,
        G=lambda x, t: [sigma],
        x0=[v["x0"]],
        max_step=min(v["dt"] / 4, 0.01),
        state_names=["x"],
    )
    return SimulatedSource(system, [_sensor("sensor", 0, v)], _times(v))


def _ou_filter(v):
    theta, sigma = v["theta"], v["sigma"]
    process = LinearProcessModel(
        A=lambda dt: np.array([[math.exp(-theta * dt)]]),
        Q=lambda dt: np.array([[sigma**2 / (2 * theta) * (1 - math.exp(-2 * theta * dt))]]),
    )
    return KalmanFilter(
        process, {"sensor": LinearMeasurementModel([[1.0]], [[v["sensor_R"]]])}, _x0(v, 1), _P0(v, 1)
    )


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
    filter_types=(
        FilterType(
            key="kf",
            label="Kalman filter (linear)",
            description="Exact discretization of the OU process.",
            params=(
                Param("theta", "Model mean reversion θ", 1.0, 0.01, 10.0, log=True, group="Process model"),
                Param("sigma", "Model volatility σ", 0.5, 0.01, 5.0, log=True, group="Process model"),
                Param("sensor_R", "Sensor noise variance R", 0.05, 1e-4, 10.0, log=True, group="Measurements"),
            )
            + _filter_init_params(("x",), (0.0,), (4.0,)),
            build=_ou_filter,
        ),
    ),
    sensor_state={"sensor": 0},
    matched_defaults={"theta": "theta", "sigma": "sigma", "sensor_R": "sensor_R"},
)


# ------------------------------------------------------------- mass-spring
def _spring_F(omega, zeta):
    return np.array([[0.0, 1.0], [-(omega**2), -2.0 * zeta * omega]])


_L2 = np.array([[0.0], [1.0]])


def _linear_kf_2d(F, sigma, v, sensor):
    cache: dict[float, tuple[np.ndarray, np.ndarray]] = {}

    def AQ(dt):  # the time steps repeat, so each (A, Q) is computed once
        key = round(dt, 12)
        if key not in cache:
            cache[key] = discretize(F, _L2, sigma**2, dt)
        return cache[key]

    process = LinearProcessModel(A=lambda dt: AQ(dt)[0], Q=lambda dt: AQ(dt)[1])
    return KalmanFilter(
        process, {sensor: LinearMeasurementModel(_H(0, 2), [[v[f"{sensor}_R"]]])}, _x0(v, 2), _P0(v, 2)
    )


def _spring_source(v):
    F = _spring_F(v["omega"], v["zeta"])
    sigma = v["sigma"]
    system = SDESystem(
        f=lambda x, t: F @ x,
        G=lambda x, t: [0.0, sigma],
        x0=[v["x0_pos"], v["x0_vel"]],
        max_step=min(v["dt"] / 4, 0.005),
        state_names=["position", "velocity"],
    )
    return SimulatedSource(system, [_sensor("sensor", 0, v)], _times(v))


def _spring_filter(v):
    return _linear_kf_2d(_spring_F(v["omega"], v["zeta"]), v["sigma"], v, "sensor")


MASS_SPRING = Scenario(
    key="mass_spring",
    title="Mass–spring–damper",
    description="A damped oscillator ẍ = −ω² x − 2ζω ẋ + σ w driven by a random force. "
    "A sensor measures the position.",
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
    filter_types=(
        FilterType(
            key="kf",
            label="Kalman filter (linear)",
            description="Exact discretization of the oscillator.",
            params=(
                Param("omega", "Model frequency ω [rad/s]", 2.0, 0.1, 20.0, log=True, group="Process model"),
                Param("zeta", "Model damping ratio ζ", 0.1, 0.0, 2.0, 0.01, group="Process model"),
                Param("sigma", "Model random force σ", 0.5, 0.01, 10.0, log=True, group="Process model"),
                Param("sensor_R", "Sensor noise variance R", 0.01, 1e-4, 10.0, log=True, group="Measurements"),
            )
            + _filter_init_params(("position", "velocity"), (0.0, 0.0), (1.0, 1.0)),
            build=_spring_filter,
        ),
    ),
    sensor_state={"sensor": 0},
    matched_defaults={"omega": "omega", "zeta": "zeta", "sigma": "sigma", "sensor_R": "sensor_R"},
)


# ---------------------------------------------------------------- pendulum
G_ACCEL = 9.81


def _pendulum_source(v):
    w2, c, sigma = G_ACCEL / v["length"], v["damping"], v["sigma"]
    system = SDESystem(
        f=lambda x, t: np.array([x[1], -w2 * math.sin(x[0]) - c * x[1]]),
        G=lambda x, t: [0.0, sigma],
        x0=[math.radians(v["angle0"]), v["omega0"]],
        max_step=min(v["dt"] / 4, 0.005),
        state_names=["angle [rad]", "angular velocity [rad/s]"],
    )
    return SimulatedSource(system, [_sensor("encoder", 0, v)], _times(v))


def _pendulum_filter(v):
    # Small-angle model: sin(angle) ~ angle, i.e. a mass-spring with w^2 = g / L, 2 zeta w = c.
    w = math.sqrt(G_ACCEL / v["length"])
    F = np.array([[0.0, 1.0], [-(w**2), -v["damping"]]])
    return _linear_kf_2d(F, v["sigma"], v, "encoder")


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
    filter_types=(
        FilterType(
            key="kf",
            label="Kalman filter (linear, small-angle model)",
            description="Linearized pendulum: sin θ ≈ θ.",
            params=(
                Param("length", "Model length L [m]", 1.0, 0.1, 10.0, log=True, group="Process model"),
                Param("damping", "Model damping c [1/s]", 0.1, 0.0, 2.0, 0.01, group="Process model"),
                Param("sigma", "Model random torque σ", 0.2, 0.001, 5.0, log=True, group="Process model"),
                Param("encoder_R", "Encoder noise variance R", 0.003, 1e-4, 10.0, log=True, group="Measurements"),
            )
            + _filter_init_params(("angle", "angular velocity"), (0.0, 0.0), (1.0, 1.0)),
            build=_pendulum_filter,
        ),
    ),
    sensor_state={"encoder": 0},
    matched_defaults={"length": "length", "damping": "damping", "sigma": "sigma", "encoder_R": "encoder_R"},
)


SCENARIOS: dict[str, Scenario] = {
    s.key: s for s in (CONSTANT_VELOCITY, ORNSTEIN_UHLENBECK, MASS_SPRING, PENDULUM)
}


def get_scenario(key: str) -> Scenario:
    try:
        return SCENARIOS[key]
    except KeyError:
        raise KeyError(f"unknown scenario {key!r}; available: {list(SCENARIOS)}") from None
