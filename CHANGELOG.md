# Changelog

## 0.1.0 — 2026-11-05

First release of the rewritten framework.

- **Systems**: `AnalyticSystem`, `DiscreteSystem`, `SDESystem` (Euler–Maruyama), with general noise that can be switched off and a per-run initial state.
- **Sensors** with a period or explicit times. **Sources**: `SimulatedSource` (truth grid + sensor times) and `CSVSource` (one file per sensor, optional truth).
- **Filters**: `Filter` / `BayesFilter` interfaces and a linear `KalmanFilter` (Joseph form).
- **Experiments**: `Runner` with observers, and `MonteCarlo`.
- **Metrics**: RMSE, NEES, NIS and step time, with chi-square consistency verdicts. Single runs use an effective-sample-size correction; Monte Carlo uses per-instant bounds.
- **Plots** (optional matplotlib): systems, estimates ±3σ, errors, tables, NEES/NIS over time, RMSE boxplots, and live views.
- **Web interface** on GitHub Pages (Pyodide), with four scenarios: constant velocity, Ornstein–Uhlenbeck, mass–spring–damper and a nonlinear pendulum.
