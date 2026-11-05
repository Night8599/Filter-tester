"""A linear Kalman filter on the recorded data in examples/data/.

Two sensors (GPS for position every 0.5 s, speedometer for velocity every
0.2 s) plus the truth (every 0.05 s). The filter predicts on the truth-only
steps too, so its estimate has the truth's resolution. Requires
`pip install -e ".[plot]"` first.

    python examples/kalman_filter.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from filter_tester.core import truth_series
from filter_tester.filters import KalmanFilter
from filter_tester.models import LinearMeasurementModel, LinearProcessModel
from filter_tester.sources import CSVSource

data = Path(__file__).parent / "data"
source = CSVSource([data / "gps.csv", data / "speedometer.csv", data / "truth.csv"])

# Constant-velocity model for any dt, x = [position, velocity]
kf = KalmanFilter(
    process=LinearProcessModel(
        A=lambda dt: np.array([[1.0, dt], [0.0, 1.0]]),
        Q=lambda dt: np.diag([1e-3, 1e-2]) * dt,
    ),
    measurements={
        "gps": LinearMeasurementModel(H=[[1.0, 0.0]], R=[[0.04]]),
        "speedometer": LinearMeasurementModel(H=[[0.0, 1.0]], R=[[0.01]]),
    },
    x0=[0.0, 0.0],
    P0=np.diag([1.0, 1.0]),
)

steps = source.run()
estimates = [kf.step(s.t, s.measurements) for s in steps]

t = np.array([e.t for e in estimates])
mean = np.array([e.mean for e in estimates])
std = np.array([e.std for e in estimates])
tt, truth = truth_series(steps)
rmse = np.sqrt(np.mean((mean - truth) ** 2, axis=0))
print(f"RMSE position = {rmse[0]:.3f}, velocity = {rmse[1]:.3f}")

fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 5))
for i, (ax, label) in enumerate(zip(axes, source.columns("truth"), strict=True)):
    ax.plot(tt, truth[:, i], "k", lw=1, label="truth")
    ax.plot(t, mean[:, i], label="KF estimate")
    ax.fill_between(t, mean[:, i] - 3 * std[:, i], mean[:, i] + 3 * std[:, i], alpha=0.25, label="±3σ")
    ax.set_ylabel(label)
    ax.grid(True, alpha=0.3)
axes[0].legend(loc="upper left")
axes[-1].set_xlabel("t")
fig.suptitle("Kalman filter on recorded data")
fig.tight_layout()
plt.show()
