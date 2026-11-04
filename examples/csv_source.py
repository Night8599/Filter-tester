"""Read recorded data from CSV files (one file per sensor, plus the truth).

The files in examples/data/ were produced by a simulation:

    gps.csv          t, position   (every 0.5 s)
    speedometer.csv  t, velocity   (every 0.2 s)
    truth.csv        t, position, velocity (every 0.05 s, optional)

Requires `pip install -e .` first.

    python examples/csv_source.py
"""

from pathlib import Path

import matplotlib.pyplot as plt

from filter_tester.core import measurement_series, truth_series
from filter_tester.plotting import plot_states
from filter_tester.sources import CSVSource

data = Path(__file__).parent / "data"

# Automatic mode: sensor names come from the file names, truth.csv is the truth.
source = CSVSource([data / "gps.csv", data / "speedometer.csv", data / "truth.csv"])
steps = source.run()
print(f"sensors: {source.sensor_names}, truth: {source.has_truth}, {len(steps)} steps")

t, x = truth_series(steps)
fig, axes = plot_states(t, x, labels=source.columns("truth"), title="Recorded data", label="truth")
for ax, sensor in zip(axes, ["gps", "speedometer"]):
    tz, z = measurement_series(steps, sensor)
    ax.plot(tz, z[:, 0], ".", label=sensor)
    ax.legend(loc="upper left")
plt.show()
