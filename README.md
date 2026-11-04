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
| `Estimate` *(implemented)* | `t`, `mean` and `covariance` (mandatory: the covariance is the filter's measure of uncertainty), `std` (square root of the diagonal), and `extras` for filter-specific data. Kalman-type filters put each sensor's innovation `(y, S)` in `extras["innovations"]` |
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

These turn the true state into measurements: `Sensor(name, h, noise, period | times)` computes `z = h(x, v, t)`. The `name` is required and must be unique within a data source, since it identifies the sensor's measurements in each step. As with systems, the noise `v` is general (passed to `h`), optional, and can be switched off with `sensor.noise_enabled = False`.

When a sensor measures is defined in one of two ways:

- `period=...` (with optional `offset`): measurements at `offset + k * period`, an absolute grid independent of when a run starts;
- `times=[...]`: an explicit list of measurement times.

`sensor.measurement_times(t_start, t_end)` lists the times in a window, and `sensor.measure(x, t, rng)` returns one measurement. Sensors are separate from systems so that the same system can be observed by different sensors. An experiment can combine **several sensors with different sampling rates** (multi-sensor fusion): each `Step` carries the measurements available at that time and which sensor produced them.

### 4. Data sources (`sources/`)

These produce the sequence of `Step`s that the filters consume.

- **`SimulatedSource(system, sensors, times)`** *(implemented)* runs a true system and its sensors. The system is evaluated on the `times` grid, which also sets the start and end of the run and can be finer than the sensors to show the truth in high resolution, plus at every sensor measurement time. Instants that coincide become **one step** holding all the measurements taken then. Randomness is split into independent streams for the system and for each sensor, so turning one sensor's noise off leaves the truth unchanged for the same seed. `source.run(rng, x0)` returns all steps, and `source.steps(rng, x0)` yields them lazily. `truth_series(steps)` and `measurement_series(steps, sensor_name)` from `filter_tester.core` turn steps into arrays.
- **`CSVSource(files | mapping, time_column, time_unit)`** *(implemented)* reads recorded data from **CSV** files (header row, `,` separator), typically **one file per sensor**, plus optional ground truth. Times are converted to seconds from `time_unit` (`s`, `ms`, `us`, `ns`, `min`, `h`). The time column is detected automatically (`t`, `time` or `timestamp`, case-insensitive) unless `time_column` is given.
  - *Automatic mode* (`CSVSource(files)`): a column `prefix.col` belongs to group `prefix`, and a column without prefix belongs to the group named after its file (`gps.csv` → sensor `gps`). The group `truth` is the ground truth, so it can come from its own `truth.csv` or from `truth.*` columns inside another file.
  - *Explicit mode* (`CSVSource(mapping={"gps": {"file": ..., "columns": [...]}, "truth": {...}})`): only the listed columns are read. Each entry may set its own `time_column`.
  - A row whose group columns are all empty (or `NaN`) means that group has no sample at that time, so several sensors with different rates can share a file. Samples at the same instant across files form one step, and steps carry the truth only where it was recorded.

### 5. Filter models (`models/`)

These are what a filter believes about the world. They are independent from the true system.

- **`LinearProcessModel(A, Q)`** *(implemented)*: `x_next = A x + w`, `w ~ N(0, Q)`. `A` and `Q` are fixed matrices or functions of `dt`.
- **`LinearMeasurementModel(H, R)`** *(implemented)*: `z = H x + v`, `v ~ N(0, R)`.
- *Planned:* nonlinear process / measurement models (`f`, `h`, optional Jacobians and sampling functions) for EKF, UKF and particle filters.

Each filter uses only what it needs. For example, a linear KF needs matrices, an EKF needs Jacobians (or numerical differentiation), and a particle filter needs to sample the noise.

**Filter state vs. true state.** The filter estimates the same state as the true system (same dimension and variables), so estimates and ground truth are compared directly.

**Control inputs.** A filter may receive either the true control input or a noisy version of it; this is configured per experiment.

### 6. Filters (`filters/`) — base classes and linear KF implemented

There are two layers:

```python
class Filter(ABC):
    """Any filter: noisy measurements in, state estimate out."""
    def reset(self): ...
    def step(self, t, measurements) -> Estimate: ...   # measurements: {sensor name: z}, may be empty


class BayesFilter(Filter):
    """Filters with an initial estimate (x0, P0) and a predict / update cycle."""
    def predict(self, dt): ...
    def update(self, sensor, z): ...
    def knows_sensor(self, sensor) -> bool: ...
```

A filter receives only the time and the measurements of each step, never the truth. `BayesFilter.step` works like this:

- On the **first step**, the estimate starts from `x0`, `P0` at that step's time, with no prediction. If the step has measurements, they are used right away.
- On **later steps**, it predicts by the elapsed `dt` (skipped when `dt = 0`), then updates once per measurement.
- **Measurements from sensors without a model are ignored**, so each filter can use a different subset of the sensors.

Filters receive **every** step, including those without measurements (for example the points of a fine truth grid). On those steps a Bayesian filter only predicts, which also yields estimates at the full resolution of the truth.

Implemented filters:

- **`KalmanFilter(process, measurements, x0, P0)`**: linear Kalman filter. `measurements` maps sensor names to `LinearMeasurementModel`s. The covariance update uses the Joseph form for numerical robustness.

### 7. Experiments (`experiment/`) — Runner implemented

- **`Runner(source, {name: filter})`** resets the filters, feeds every step of the source to every filter, and returns a `RunResult`:
  - `result.steps` holds the steps of the run;
  - `result[name]` holds one `Trajectory` per filter, with arrays `t`, `mean`, `covariance`, `std`, `truth` (NaN rows where unknown, or `None`), `error`, the `innovations` used, and the `step_time` of each step.
  - `runner.run(rng, x0)` forwards `rng` and `x0` to simulated sources.
- **Observers** (`runner.run(observers=[...])`) are called after every step with the step and all the estimates, e.g. by a live plot.
- **`MonteCarlo(source, {name: filter}, runs=N).run(...)`** *(implemented)* repeats a simulated experiment N times.
  - Run `k` uses seed `k` (`first_seed=` shifts them for an independent experiment), so `Runner(...).run(rng=k)` reproduces any single run.
  - It keeps, per filter: the NEES at every step `(N, K)`, the NIS at every innovation `(N, M)`, and each run's RMSE and mean step time.
  - With `keep_runs=True` it also keeps every run's full `RunResult`.
  - Observers are called after each run with the partial aggregate.

### 8. Metrics and plots (`metrics/`, `plotting/`) — implemented

| Metric | Measures | Needs truth | Ideal |
|---|---|---|---|
| `rmse`, `rmse_total` | accuracy, per state component and for the whole error vector | yes | lower is better |
| `nees`, `nees_consistency` | whether the covariance matches the actual error | yes | mean ≈ state dimension `n` |
| `nis`, `nis_consistency` | whether the innovation covariance matches the innovations | no, so it also works on real data | mean ≈ measurement dimension |
| `mean_step_time` | computational cost | no | lower is better |

The consistency checks compare a run's mean NEES / NIS with a chi-square interval (95% by default) and give a verdict:
- **consistent**: the mean is inside the interval;
- **optimistic**: the mean is above it, so the filter underestimates its error;
- **pessimistic**: the mean is below it, so the filter overestimates its error.

Samples within one run are correlated in time, so the interval uses an effective sample size estimated from the autocorrelation. Without this correction, a filter with the exact model was flagged in about 75% of runs on a fine truth grid. With it, about 5% are flagged, as expected at 95%. A single run still detects a pessimistic filter by NEES only about half the time (NIS catches it), so use Monte Carlo for a strict test.

**Monte Carlo metrics.** Runs are independent, so at every instant the NEES (NIS) averaged over the N runs is checked against the exact interval `chi2(N·n)/N`. A consistent filter stays inside it at about 95% of the instants. The table reports:
- RMSE as mean ± std over runs;
- ANEES / ANIS (the average over runs and time) with an interval and verdict;
- the share of instants inside their bounds.

In tests with 20-run experiments, the per-instant check gave 96% inside for a matched filter. Optimistic and pessimistic filters were flagged in 50/50 experiments, and the overall ANEES verdict wrongly flagged a matched filter in about 14% of experiments (nominal 5%). Functions: `mc_nees`, `mc_nis`, `mc_nees_consistency`, `mc_nis_consistency`, `mc_summary_table`, `format_mc_summary`.

Monte Carlo plots:
- `plot_mc_consistency(result)`: run-averaged NEES and NIS over time with their bounds, on a log scale;
- `plot_rmse_boxplot(result)`: the per-run RMSE distribution of each filter, per state and total;
- `plot_mc_summary_table(result)`;
- `LiveMonteCarloPlot()`: an observer that shows the partial metrics as runs finish.

`summary(result)` returns the metrics per filter, and `format_summary(result)` prints them as a table. Metrics that need the truth are left out when it is not available.

Plots (each filter keeps the same color in every figure):
- **`plot_estimates(result)`**: the truth and each filter's estimate with a ±3σ band.
- **`plot_errors(result)`**: each filter's error with its own ±3σ band. A consistent filter keeps its error inside the band.
- **`plot_summary_table(result)`**: the metrics table as a figure, with consistency verdicts in green or red.
- **`LiveFilterPlot(labels, speed=...)`**: estimates vs. truth while the `Runner` runs, passed as an observer, with the same `speed` pacing as the live system plot.

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
gps = Sensor("gps", h=lambda x, v, t: x[:1] + v, noise=GaussianNoise([[0.25]]), period=0.5,
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

source = SimulatedSource(cv, [gps], times=np.linspace(0.0, 10.0, 1001))  # fine truth grid
steps = source.run(rng=0, x0=[0.0, 2.0])     # x0 is optional

t, x = truth_series(steps)                   # truth in high resolution
tz, z = measurement_series(steps, "gps")     # measurements every 0.5 s
```

See `examples/simulated_source.py` for two sensors at different rates plotted over the truth.

### Recorded data from CSV (available now)

```python
from filter_tester.sources import CSVSource

# One file per sensor; truth.csv (optional) holds the ground truth
source = CSVSource(["examples/data/gps.csv", "examples/data/speedometer.csv", "examples/data/truth.csv"])
steps = source.run()                         # same Step objects as SimulatedSource
source.sensor_names                          # ['gps', 'speedometer']

# Explicit columns, other names and units
source = CSVSource(mapping={
    "gps":   {"file": "examples/data/gps.csv", "columns": ["position"]},
    "truth": {"file": "examples/data/truth.csv", "columns": ["position", "velocity"]},
}, time_unit="s")
```

See `examples/csv_source.py` for reading and plotting the sample files in `examples/data/`.

### Kalman filter (available now)

```python
from filter_tester.filters import KalmanFilter
from filter_tester.models import LinearMeasurementModel, LinearProcessModel

kf = KalmanFilter(
    process=LinearProcessModel(
        A=lambda dt: np.array([[1.0, dt], [0.0, 1.0]]),
        Q=lambda dt: np.diag([1e-3, 1e-2]) * dt,
    ),
    measurements={"gps": LinearMeasurementModel(H=[[1.0, 0.0]], R=[[0.04]])},
    x0=[0.0, 0.0],
    P0=np.eye(2),
)

kf.reset()
estimates = [kf.step(s.t, s.measurements) for s in source.run()]   # any data source
estimates[-1].mean, estimates[-1].covariance, estimates[-1].std
```

See `examples/kalman_filter.py` for the filter on the sample CSV data, plotted with ±3σ bounds against the truth.

### Comparing filters (available now)

```python
from filter_tester.experiment import Runner
from filter_tester.metrics import format_summary
from filter_tester.plotting import LiveFilterPlot, plot_errors, plot_estimates, plot_summary_table

runner = Runner(source, {"KF": kf})                        # any number of filters
live = LiveFilterPlot(labels=["position", "velocity"])     # optional, speed=1.0 for real time
result = runner.run(observers=[live])                      # rng=..., x0=... for simulated sources
live.finish()

print(format_summary(result, ["position", "velocity"]))    # RMSE, NEES, NIS, time per step
plot_estimates(result, ["position", "velocity"])
plot_errors(result, ["position", "velocity"])
plot_summary_table(result, ["position", "velocity"])
```

See `examples/compare_filters.py`: three Kalman filters (matched, optimistic and pessimistic measurement noise) on the same simulated data, shown live and then compared.

### Monte Carlo (available now)

```python
from filter_tester.experiment import MonteCarlo
from filter_tester.metrics import format_mc_summary
from filter_tester.plotting import LiveMonteCarloPlot, plot_mc_consistency, plot_mc_summary_table, plot_rmse_boxplot

mc = MonteCarlo(source, {"KF": kf}, runs=100)    # simulated source; run k uses seed k
live = LiveMonteCarloPlot()                      # optional: partial metrics while running
result = mc.run(observers=[live])                # keep_runs=True keeps every run
live.finish()

print(format_mc_summary(result, ["position", "velocity"]))
plot_mc_consistency(result)                      # NEES / NIS over time with chi-square bounds
plot_rmse_boxplot(result, ["position", "velocity"])
plot_mc_summary_table(result, ["position", "velocity"])
```

See `examples/monte_carlo.py` (matched, optimistic and pessimistic filters over 100 runs).

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

### Full pipeline

> Everything below except the configuration files is implemented; see the sections above for the current API.



```python
import numpy as np
from filter_tester.systems import DiscreteSystem, GaussianNoise
from filter_tester.sensors import Sensor
from filter_tester.sources import SimulatedSource
from filter_tester.models import LinearProcessModel, LinearMeasurementModel
from filter_tester.filters import KalmanFilter
from filter_tester.experiment import Runner
from filter_tester.metrics import rmse, nees_consistency

dt = 0.1
A = np.array([[1, dt], [0, 1]])
H = np.array([[1, 0]])

# True system that generates the data
system = DiscreteSystem(f=lambda x, w, t, dt: A @ x + w, x0=[0.0, 1.0], noise=GaussianNoise(np.diag([1e-4, 1e-3])))
sensor = Sensor("gps", h=lambda x, v, t: H @ x + v, noise=GaussianNoise([[0.25]]), period=dt)
source = SimulatedSource(system, [sensor], times=np.arange(0, 10, dt))

# The filter's own (possibly mismatched) model
kf = KalmanFilter(
    process=LinearProcessModel(A=A, Q=np.diag([1e-3, 1e-2])),
    measurements={"gps": LinearMeasurementModel(H=H, R=np.array([[0.25]]))},
    x0=np.zeros(2), P0=np.eye(2),
)

results = Runner(source, {"KF": kf}).run(rng=42)
print(rmse(results["KF"]), nees_consistency(results["KF"]).verdict)
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
source = CSVSource(["gps.csv", "speedometer.csv", "truth.csv"])   # available now
results = Runner(source, {"KF": kf}).run()
# Truth-based metrics are only computed if the file contains ground truth.
```

---

## Extending the framework

- **New filter:** subclass `Filter` and implement `reset()` and `step(t, measurements) -> Estimate`. If it has a predict/update cycle, subclass `BayesFilter` and implement `predict(dt)`, `update(sensor, z)` and `knows_sensor(sensor)` (registration by name comes with the configuration files).
- **New system:** subclass `System` and implement `_initial_state(t0, rng)` and `_advance(x, t, t_next, rng)`. Often you don't need a subclass: pass your own functions to `AnalyticSystem`, `DiscreteSystem` or `SDESystem`.
- **New metric:** write a function that takes a `Trajectory` and returns a value.

---

## Roadmap

The framework is built module by module, starting with the data generators.

- [x] True systems: analytic, discrete-time, SDE (Euler–Maruyama), general noise
- [x] Optional, switchable noise in systems and sensors
- [x] Sensors (period or explicit times, general noise)
- [x] Simulated data source (truth grid + sensor times, one step per instant), per-run initial state
- [x] CSV data source (one file per sensor, automatic or explicit columns, optional truth)
- [x] Plots of true systems (states vs. time, static and live)
- [x] `Estimate`, filter base classes (`Filter`, `BayesFilter`), linear Kalman filter
- [x] Runner with observers, metrics (RMSE, NEES, NIS, step time) with consistency verdicts, filter plots (estimates, errors, table, live)
- [x] Monte Carlo (seeds 0..N-1, per-instant chi-square bounds, ANEES/ANIS, RMSE boxplots, live partial metrics)
- [ ] Nonlinear filters (EKF, UKF, particle filter)
- [ ] Configuration files

---

## License

See [LICENSE](LICENSE).
