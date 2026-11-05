# Changelog

## 0.2.0 — 2026-11-05

- **Nonlinear filters**: `ExtendedKalmanFilter` (analytic or finite-difference Jacobians), `UnscentedKalmanFilter` (scaled sigma points, α/β/κ) and `ParticleFilter` (systematic or multinomial resampling at an `N_eff` threshold, optional regularization, seeded).
- **Nonlinear models**: `ProcessModel` / `MeasurementModel` with additive Gaussian noise and optional vectorization; `rk4` to integrate continuous-time models; linear models work with every filter.
- **`SDESystem(method="heun")`**: second-order drift with Itô noise. The scenarios now use it: Euler–Maruyama's integration error was larger than the process noise on chaotic systems and made every filter look inconsistent.
- **Web interface**: every scenario offers KF/EKF/UKF/PF with their own constants, plus a new **Lorenz** scenario.
- Example `examples/nonlinear_filters.py`.
- Linting and formatting with ruff, checked in CI; docstrings for the public API.
- README: animated demo and screenshots of the web interface.

## 0.1.0 — 2026-11-05

First release of the rewritten framework.

- **Systems**: `AnalyticSystem`, `DiscreteSystem`, `SDESystem` (Euler–Maruyama), with general noise that can be switched off and a per-run initial state.
- **Sensors** with a period or explicit times. **Sources**: `SimulatedSource` (truth grid + sensor times) and `CSVSource` (one file per sensor, optional truth).
- **Filters**: `Filter` / `BayesFilter` interfaces and a linear `KalmanFilter` (Joseph form).
- **Experiments**: `Runner` with observers, and `MonteCarlo`.
- **Metrics**: RMSE, NEES, NIS and step time, with chi-square consistency verdicts. Single runs use an effective-sample-size correction; Monte Carlo uses per-instant bounds.
- **Plots** (optional matplotlib): systems, estimates ±3σ, errors, tables, NEES/NIS over time, RMSE boxplots, and live views.
- **Web interface** on GitHub Pages (Pyodide), with four scenarios: constant velocity, Ornstein–Uhlenbeck, mass–spring–damper and a nonlinear pendulum.
