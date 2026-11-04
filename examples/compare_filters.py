"""Compare three Kalman filters on the same simulated data.

- "matched":     the filter's model equals the true system;
- "optimistic":  it believes the GPS is 10x more precise than it is (R too small);
- "pessimistic": it believes the GPS is 10x noisier than it is (R too large).

The run is shown live, then the static figures and the metrics table.
Requires `pip install -e .` first.

    python examples/compare_filters.py             # live, as fast as possible
    python examples/compare_filters.py --speed 2   # live, twice real time
    python examples/compare_filters.py --static    # no live plot
"""

import sys

import matplotlib.pyplot as plt
import numpy as np

from filter_tester.experiment import Runner
from filter_tester.filters import KalmanFilter
from filter_tester.metrics import format_summary
from filter_tester.models import LinearMeasurementModel, LinearProcessModel
from filter_tester.noise import GaussianNoise
from filter_tester.plotting import LiveFilterPlot, plot_errors, plot_estimates, plot_summary_table
from filter_tester.sensors import Sensor
from filter_tester.sources import SimulatedSource
from filter_tester.systems import DiscreteSystem

live = "--static" not in sys.argv
speed = float(sys.argv[sys.argv.index("--speed") + 1]) if "--speed" in sys.argv else None
labels = ["position", "velocity"]


def A(dt):
    return np.array([[1.0, dt], [0.0, 1.0]])


def Q(dt, q=0.05):  # white-noise acceleration
    return q * np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]])


# True system and sensor
dt, R_true = 0.05, 0.25
system = DiscreteSystem(
    f=lambda x, w, t, dt: A(dt) @ x + w,
    x0=[0.0, 1.0],
    noise=GaussianNoise(Q(dt)),
    state_names=labels,
)
gps = Sensor("gps", lambda x, v, t: x[:1] + v, noise=GaussianNoise([[R_true]]), period=0.5)
source = SimulatedSource(system, [gps], times=np.arange(0.0, 30.0 + dt / 2, dt))


def kf(R):
    return KalmanFilter(
        process=LinearProcessModel(A=A, Q=Q),
        measurements={"gps": LinearMeasurementModel(H=[[1.0, 0.0]], R=[[R]])},
        x0=[0.0, 0.0],
        P0=np.diag([1.0, 1.0]),
    )


runner = Runner(source, {"matched": kf(R_true), "optimistic": kf(R_true / 10), "pessimistic": kf(R_true * 10)})

observers = []
if live:
    live_plot = LiveFilterPlot(labels=labels, t_range=(source.t_start, source.t_end), speed=speed)
    observers.append(live_plot)
result = runner.run(rng=1, observers=observers)
if live:
    live_plot.finish()

print(format_summary(result, labels))
plot_estimates(result, labels)
plot_errors(result, labels)
plot_summary_table(result, labels)
plt.show()
