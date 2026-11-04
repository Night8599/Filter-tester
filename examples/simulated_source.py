"""A constant-velocity system observed by two sensors at different rates.

The truth is simulated on a fine grid (high resolution), and the sensors add
steps at their own measurement times. Requires `pip install -e ".[plot]"` first.

    python examples/simulated_source.py
"""

import matplotlib.pyplot as plt
import numpy as np

from filter_tester.core import measurement_series, truth_series
from filter_tester.noise import GaussianNoise
from filter_tester.plotting import plot_states
from filter_tester.sensors import Sensor
from filter_tester.sources import SimulatedSource
from filter_tester.systems import DiscreteSystem

# x = [position, velocity]; transition for any dt, so the truth grid and the
# sensor times can be mixed freely.
system = DiscreteSystem(
    f=lambda x, w, t, dt: np.array([[1.0, dt], [0.0, 1.0]]) @ x + np.sqrt(dt) * w,
    x0=[0.0, 1.0],
    noise=GaussianNoise(np.diag([1e-3, 1e-2])),
    state_names=["position", "velocity"],
)
gps = Sensor("gps", lambda x, v, t: x[:1] + v, noise=GaussianNoise([[0.04]]), period=0.5)
speedometer = Sensor("speedometer", lambda x, v, t: x[1:] + v, noise=GaussianNoise([[0.01]]), period=0.2)

source = SimulatedSource(system, [gps, speedometer], times=np.linspace(0.0, 10.0, 1001))
steps = source.run(rng=0)
print(f"{len(steps)} steps, {sum(s.has_measurements for s in steps)} with measurements, "
      f"{sum(len(s.measurements) == 2 for s in steps)} with both sensors")

t, x = truth_series(steps)
fig, axes = plot_states(t, x, labels=system.state_names, title="Truth and measurements", label="truth")
for ax, sensor in zip(axes, ["gps", "speedometer"]):
    tz, z = measurement_series(steps, sensor)
    ax.plot(tz, z[:, 0], ".", label=sensor)
    ax.legend(loc="upper left")
plt.show()
