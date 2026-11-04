"""Simulate and plot the three kinds of true systems.

Run from the repository root:

    python examples/plot_systems.py          # live plots
    python examples/plot_systems.py --static # static plots only
"""

import sys

import matplotlib.pyplot as plt
import numpy as np

from filter_tester.plotting import plot_states, plot_system
from filter_tester.systems import AnalyticSystem, DiscreteSystem, GaussianNoise, SDESystem, simulate

live = "--static" not in sys.argv

# Constant-velocity model, x_next = A x + w
dt = 0.05
A = np.array([[1.0, dt], [0.0, 1.0]])
cv = DiscreteSystem(
    f=lambda x, w, t, dt: A @ x + w,
    x0=[0.0, 1.0],
    noise=GaussianNoise(np.diag([1e-4, 1e-3])),
    name="Constant velocity",
    state_names=["position", "velocity"],
)
plot_system(cv, np.arange(0.0, 10.0, dt), rng=1, live=live)

# Ornstein-Uhlenbeck process, dx = -theta x dt + sigma dW, five runs overlaid
ou = SDESystem(f=lambda x, t: -1.5 * x, G=lambda x, t: [0.8], x0=[2.0], name="Ornstein-Uhlenbeck")
times = np.linspace(0.0, 5.0, 500)
fig, axes, _ = plot_system(ou, times, rng=0, live=live, label="seed 0")
for seed in range(1, 5):  # the remaining runs are drawn on the same axes
    plot_states(times, simulate(ou, times, rng=seed), axes=axes, label=f"seed {seed}")
axes[0].legend()

# Sine wave with random phase noise; axis label overridden in the plot call
wave = AnalyticSystem(lambda t, w: [np.sin(t + w[0])], noise=GaussianNoise([[0.01]]), name="Noisy sine")
plot_system(wave, np.linspace(0.0, 10.0, 300), rng=2, live=live, labels=["amplitude"])

plt.show()
