"""Monte Carlo comparison of three Kalman filters.

The same simulated experiment is repeated N times (run k uses seed k):

- "matched":     the filter's model equals the true system;
- "optimistic":  it believes the GPS is 10x more precise than it is;
- "pessimistic": it believes the GPS is 10x noisier than it is.

The partial metrics are shown live as runs finish, then the final figures and
table. Requires `pip install -e .` first.

    python examples/monte_carlo.py              # 100 runs, live
    python examples/monte_carlo.py --runs 300
    python examples/monte_carlo.py --static     # no live plot
"""

import sys

import matplotlib.pyplot as plt
import numpy as np

from filter_tester.experiment import MonteCarlo
from filter_tester.filters import KalmanFilter
from filter_tester.metrics import format_mc_summary
from filter_tester.models import LinearMeasurementModel, LinearProcessModel
from filter_tester.noise import GaussianNoise
from filter_tester.plotting import (
    LiveMonteCarloPlot,
    plot_mc_consistency,
    plot_mc_summary_table,
    plot_rmse_boxplot,
)
from filter_tester.sensors import Sensor
from filter_tester.sources import SimulatedSource
from filter_tester.systems import DiscreteSystem

live = "--static" not in sys.argv
runs = int(sys.argv[sys.argv.index("--runs") + 1]) if "--runs" in sys.argv else 100
labels = ["position", "velocity"]


def A(dt):
    return np.array([[1.0, dt], [0.0, 1.0]])


def Q(dt, q=0.05):  # white-noise acceleration
    return q * np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]])


dt, R_true = 0.1, 0.25
system = DiscreteSystem(
    f=lambda x, w, t, dt: A(dt) @ x + w,
    x0=GaussianNoise(np.eye(2), mean=[0.0, 1.0]),  # random initial state, matching the filters' x0 / P0
    noise=GaussianNoise(Q(dt)),
)
gps = Sensor("gps", lambda x, v, t: x[:1] + v, noise=GaussianNoise([[R_true]]), period=0.5)
source = SimulatedSource(system, [gps], times=np.arange(0.0, 20.0 + dt / 2, dt))


def kf(R):
    return KalmanFilter(
        process=LinearProcessModel(A=A, Q=Q),
        measurements={"gps": LinearMeasurementModel(H=[[1.0, 0.0]], R=[[R]])},
        x0=[0.0, 1.0],
        P0=np.eye(2),
    )


mc = MonteCarlo(source, {"matched": kf(R_true), "optimistic": kf(R_true / 10), "pessimistic": kf(R_true * 10)}, runs)

observers = []
if live:
    live_plot = LiveMonteCarloPlot()
    observers.append(live_plot)
result = mc.run(observers=observers)
if live:
    live_plot.finish()

print(format_mc_summary(result, labels))
plot_mc_consistency(result)
plot_rmse_boxplot(result, labels)
plot_mc_summary_table(result, labels)
plt.show()
