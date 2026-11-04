# Filter Tester

A framework for testing and comparing **stochastic filters** (Kalman filters, particle filters, or any algorithm that turns noisy inputs into a state estimate) against **different kinds of systems**, using either simulated data or real data loaded from files.

> **Status: under construction.** The project is being rewritten from scratch, module by module. This README describes the target architecture; see the [Roadmap](#roadmap) for what is already implemented.

---

## Goals

- **Any filter.** Anything that receives noisy measurements over time and returns a state estimate can be plugged in, whether it is Bayesian (KF, EKF, UKF, particle filter, …) or not (moving average, low-pass, …).
- **Any system.** A system only has to produce a state for a time `t`. It can be an analytic function of time, a discrete-time model, or a continuous-time model that is integrated numerically.
- **Simulated or real data.** Filters consume the same input whether it comes from a simulation or from a recorded file.
- **Model mismatch.** The "true" system that generates the data and the model the filter *believes in* are separate objects. This makes it possible to test filters with wrong dynamics, wrong `Q`, or wrong `R`.
- **Two ways to run experiments.** Use a Python API, or declarative configuration files built on top of the same API.

---

## How it works

Every experiment follows the same data flow:

```
 Data source ──► sequence of Steps ──► N filters ──► Trajectories ──► Metrics / Plots
 (simulated       (t, measurement,                   (estimates
  or file)         control, truth?)                   per step)
```

The key rule: **a filter never knows where its data comes from.** At each step it only receives the time, the measurement (which may be missing) and, optionally, a control input. It returns an estimate. The ground truth, when available, is used only afterwards to compute metrics.

---

## Architecture

### 1. Core data types (`core/`)

| Type | Contents |
|---|---|
| `Step` *(implemented)* | `t`, `measurements` (dict keyed by sensor name; empty when no sensor measured, and several sensors measuring at the same instant share one step), `truth` (optional, since real data may not have it) |
| `Estimate` | `t`, `mean` and `covariance` (mandatory: the covariance is the filter's measure of uncertainty), plus optional `particles` / `weights`, `extras` |
| `Trajectory` | The full history of steps and estimates of one run, stored as arrays |

### 2. True systems (`systems/`) — implemented

These generate the true state. All of them share one interface:

```python
x0 = system.reset(t0=0.0, rng=42)   # start a run; rng: Generator, int seed or None
x  = system.advance(t)              # true state at time t (times must not decrease)
```

Implementations:

| Class | Model | Noise |
|---|---|---|
| `AnalyticSystem(g, noise)` | `x = g(t, w)` | `w` drawn independently at each evaluated time |
| `DiscreteSystem(f, x0, noise)` | `x_next = f(x, w, t, dt)` | one `w` per step; `dt` is the time since the previous call, so fixed and variable steps both work |
| `SDESystem(f, x0, G, max_step)` | `dx = f(x, t) dt + G(x, t) dW` (Itô) | Wiener process, integrated with Euler–Maruyama in sub-steps of at most `max_step` |

Noise is **general, not only additive**: `w` is passed to the model function, which decides how it enters the dynamics. A noise source is a `Noise` object from `filter_tester.noise` (`GaussianNoise(cov, mean)`, `UniformNoise(low, high)`) or any callable `rng -> np.ndarray`. In `SDESystem`, `G` may depend on the state (multiplicative noise) and may be an `(n, m)` matrix or an `(n,)` diagonal. The initial state `x0` is either a fixed vector or a noise / callable that samples a random one at each `reset`.

Noise is **optional and easy to switch off**: set `system.noise_enabled = False` (or pass `noise_enabled=False`). The model then receives a zero noise vector (in `SDESystem`, the diffusion term is skipped), giving the nominal, noiseless trajectory. A random initial state is still sampled.

The **initial state** is easy to choose per run: `system.reset(x0=...)`, `simulate(..., x0=...)`, `plot_system(..., x0=...)` and `SimulatedSource.run(x0=...)` all accept a fixed vector or a noise / callable for a random one, overriding the system's own `x0` for that run only. (`AnalyticSystem` doesn't accept it, since its state at `t0` is given by `g`.)

`simulate(system, times, rng)` runs a system over a time grid and returns an array of shape `(len(times), n)`.

Every system accepts an optional `state_names` list (e.g. `["position", "velocity"]`), used as axis labels in plots. `simulate` also accepts an `on_step(t, x)` callback, called after every time point.

Control inputs are not supported yet.

### 3. Sensors (`sensors/`) — implemented

These turn the true state into measurements: `Sensor(h, noise, period | times)` computes `z = h(x, v, t)`. As with systems, the noise `v` is general (passed to `h`), optional, and can be switched off with `sensor.noise_enabled = False`.

When a sensor measures is defined in one of two ways:

- `period=...` (with optional `offset`): measurements at `offset + k * period`, an absolute grid independent of when a run starts;
- `times=[...]`: an explicit list of measurement times.

`sensor.measurement_times(t_start, t_end)` lists the times in a window, and `sensor.measure(x, t, rng)` returns one measurement. Sensors are separate from systems so that the same system can be observed by different sensors. An experiment can combine **several sensors with different sampling rates** (multi-sensor fusion): each `Step` carries the measurements available at that time and which sensor produced them.

### 4. Data sources (`sources/`)

These produce the sequence of `Step`s that the filters consume.

- **`SimulatedSource(system, sensors, times)`** *(implemented)* runs a true system and its sensors. The system is evaluated on the `times` grid, which also sets the start and end of the run and can be finer than the sensors to show the truth in high resolution, plus at every sensor measurement time. Instants that coincide become **one step** holding all the measurements taken then. Randomness is split into independent streams for the system and for each sensor, so turning one sensor's noise off leaves the truth unchanged for the same seed. `source.run(rng, x0)` returns all steps, and `source.steps(rng, x0)` yields them lazily. `truth_series(steps)` and `measurement_series(steps, sensor_name)` from `filter_tester.core` turn steps into arrays.
- **`FileSource(path, column_mapping)`** reads recorded data from **CSV** files. A time column is required, measurement columns are mapped to sensors, and ground-truth columns are optional.

### 5. Filter models (`models/`)

These are what a filter believes about the world. They are independent from the true system.

- **`ProcessModel`**: transition function `f`, process noise `Q`, and optionally a Jacobian and a sampling function.
- **`MeasurementModel`**: measurement function `h`, measurement noise `R`, and optionally a Jacobian.

Each filter uses only what it needs. For example, a linear KF needs matrices, an EKF needs Jacobians (or numerical differentiation), and a particle filter needs to sample the noise.

**Filter state vs. true state.** The filter estimates the same state as the true system (same dimension and variables), so estimates and ground truth are compared directly.

**Control inputs.** A filter may receive either the true control input or a noisy version of it; this is configured per experiment.

### 6. Filters (`filters/`)

There are two layers:

```python
class Filter(ABC):
    """Any filter: noisy inputs in, state estimate out."""
    def reset(self, t0, rng): ...
    def step(self, t, measurement, control=None) -> Estimate: ...


class BayesFilter(Filter):
    """Filters with an explicit predict / update cycle."""
    def predict(self, t, control=None): ...
    def update(self, measurement): ...
    # step() calls predict(), then update() only if a measurement is present
```

Because `predict` and `update` are separate, the framework naturally handles steps without measurements and irregularly sampled data.

### 7. Experiments (`experiment/`)

- **`Runner`** runs one data source against N filters and returns one `Trajectory` per filter.
- **`MonteCarlo`** repeats a simulated experiment with different random seeds.
- The `Runner` notifies **observers** after every step, so results can be consumed while the experiment is running (e.g. by a live plot) and not only at the end.

### 8. Metrics and plots (`metrics/`, `plotting/`)

These are pure functions over `Trajectory` objects, for example RMSE, NEES, NIS and execution time. Metrics that need the ground truth are skipped automatically when it is not available, as with real data.

Two kinds of plotting are planned:

- **Real-time plots** that update while the experiment runs (as a `Runner` observer).
- **Report figures** generated afterwards that illustrate the strong and weak points of each filter, e.g. estimate vs. truth, error with ±3σ bounds from the covariance, and consistency (NEES/NIS).

### 9. Configuration (`config/`)

The Python API is the foundation. On top of it there is a name-based registry (e.g. `@register("kalman")`) and a **YAML** configuration loader that builds the same objects.

---

## Requirements

- Python ≥ 3.10
- numpy, scipy, matplotlib, pyyaml
- pytest (for tests)

Plots, both real-time and report figures, use matplotlib.

---

## Planned project layout

```
src/filter_tester/
  core/          # Step, Estimate, Trajectory
  systems/       # true systems (analytic, discrete, continuous)
  sensors/       # measurement generation
  sources/       # simulated and file-based data sources
  models/        # process / measurement models used by filters
  filters/       # Filter, BayesFilter and implementations
  experiment/    # Runner, MonteCarlo
  metrics/       # RMSE, NEES, NIS, ...
  plotting/
  config/        # registry and config-file loader
  cli.py
examples/        # example scripts and configuration files
tests/
```

---

## Installation

```bash
pip install -e ".[test]"
pytest
```

## Usage

### True systems (available now)

```python
import numpy as np
from filter_tester.systems import AnalyticSystem, DiscreteSystem, SDESystem, GaussianNoise, simulate

dt = 0.1
times = np.arange(0.0, 10.0, dt)

# Constant-velocity model with additive Gaussian process noise
cv = DiscreteSystem(
    f=lambda x, w, t, dt: np.array([[1.0, dt], [0.0, 1.0]]) @ x + w,  # uses the step's dt
    x0=[0.0, 1.0],
    noise=GaussianNoise(np.diag([1e-4, 1e-3])),
)

# Non-additive noise: a sine wave with a random phase perturbation
wave = AnalyticSystem(lambda t, w: [np.sin(t + w[0])], noise=GaussianNoise([[0.01]]))

# Ornstein–Uhlenbeck process: dx = -theta x dt + sigma dW
ou = SDESystem(f=lambda x, t: -1.5 * x, G=lambda x, t: [0.8], x0=[2.0], max_step=1e-3)

states = simulate(cv, times, rng=42)   # shape (len(times), 2)
```

### Sensors (available now)

```python
from filter_tester.noise import GaussianNoise
from filter_tester.sensors import Sensor

# Noisy position measurement every 0.5 s
gps = Sensor(h=lambda x, v, t: x[:1] + v, noise=GaussianNoise([[0.25]]), period=0.5,
             measurement_names=["position"])

rng = np.random.default_rng(0)
for t in gps.measurement_times(0.0, 2.0):          # [0.0, 0.5, 1.0, 1.5, 2.0]
    z = gps.measure(states[int(round(t / dt))], t, rng)

gps.noise_enabled = False                           # noiseless measurements
```

Connecting systems and sensors automatically over a run is the job of the upcoming simulated data source.

### Simulated data source (available now)

```python
from filter_tester.core import measurement_series, truth_series
from filter_tester.sources import SimulatedSource

gps.name = "gps"
source = SimulatedSource(cv, [gps], times=np.linspace(0.0, 10.0, 1001))  # fine truth grid
steps = source.run(rng=0, x0=[0.0, 2.0])     # x0 is optional

t, x = truth_series(steps)                   # truth in high resolution
tz, z = measurement_series(steps, "gps")     # measurements every 0.5 s
```

See `examples/simulated_source.py` for two sensors at different rates plotted over the truth.

### Plotting systems (available now)

Both functions draw one subplot per state component against time:

```python
import matplotlib.pyplot as plt
from filter_tester.plotting import plot_states, plot_system

# Simulate and plot in one call; live=True updates the figure while simulating.
# speed=1.0 paces it in real time (None = as fast as possible, 2.0 = twice as fast)
fig, axes, states = plot_system(cv, times, rng=42, live=True, speed=1.0)

# Plot states you already have, e.g. overlaying another run on the same axes
plot_states(times, simulate(cv, times, rng=7), axes=axes, label="seed 7")

plt.show()
```

Axis labels come from the `labels` argument if given, otherwise from the system's `state_names`, otherwise `x[0]`, `x[1]`, …. For custom live views, `LiveStatePlot(n_states, labels).update(t, x)` can be fed one state at a time. See `examples/plot_systems.py` for a complete script (`--static` disables live plotting, `--speed 1` paces it in real time).

### Full pipeline (planned API)

> The examples below illustrate the intended design for the modules that are not implemented yet. Names may change during implementation.



```python
import numpy as np
from filter_tester.systems import DiscreteSystem, GaussianNoise
from filter_tester.sensors import Sensor
from filter_tester.sources import SimulatedSource
from filter_tester.models import LinearProcessModel, LinearMeasurementModel
from filter_tester.filters import KalmanFilter
from filter_tester.experiment import Runner
from filter_tester.metrics import rmse, nees

dt = 0.1
A = np.array([[1, dt], [0, 1]])
H = np.array([[1, 0]])

# True system that generates the data
system = DiscreteSystem(f=lambda x, w, t, dt: A @ x + w, x0=[0.0, 1.0], noise=GaussianNoise(np.diag([1e-4, 1e-3])))
sensor = Sensor(h=lambda x, v, t: H @ x + v, noise=GaussianNoise([[0.25]]), period=dt, name="gps")
source = SimulatedSource(system, [sensor], times=np.arange(0, 10, dt))

# The filter's own (possibly mismatched) model
kf = KalmanFilter(
    process=LinearProcessModel(A=A, Q=np.diag([1e-3, 1e-2])),
    measurement=LinearMeasurementModel(H=H, R=np.array([[0.25]])),
    x0=np.zeros(2), P0=np.eye(2),
)

results = Runner(source, {"KF": kf}).run(seed=42)
print(rmse(results["KF"]), nees(results["KF"]))
```

### Configuration file

```yaml
source:
  type: simulated
  system: {type: discrete, ...}
  sensors: [{type: linear, ...}]
filters:
  KF: {type: kalman, ...}
metrics: [rmse, nees]
```

### Real data

```python
source = FileSource("data/run01.csv", column_mapping={...})
results = Runner(source, {"KF": kf}).run()
# Truth-based metrics are only computed if the file contains ground truth.
```

---

## Extending the framework

- **New filter:** subclass `Filter` (or `BayesFilter` if it has a predict/update cycle) and register it by name.
- **New system:** subclass `System` and implement `_initial_state(t0, rng)` and `_advance(x, t, t_next, rng)`. Often you don't need a subclass: pass your own functions to `AnalyticSystem`, `DiscreteSystem` or `SDESystem`.
- **New metric:** write a function that takes a `Trajectory` and returns a value.

---

## Roadmap

The framework is built module by module, starting with the data generators.

- [x] True systems: analytic, discrete-time, SDE (Euler–Maruyama), general noise
- [x] Optional, switchable noise in systems and sensors
- [x] Sensors (period or explicit times, general noise)
- [x] Simulated data source (truth grid + sensor times, one step per instant), per-run initial state
- [ ] CSV data source
- [x] Plots of true systems (states vs. time, static and live)
- [ ] Core types, filters, runner, metrics, filter plots, configuration

---

## License

See [LICENSE](LICENSE).
