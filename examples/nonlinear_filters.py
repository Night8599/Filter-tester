"""KF vs. EKF vs. UKF vs. particle filter on the pendulum and the Lorenz system.

Uses the ready-made scenarios (the same ones as the web interface) and a
Monte Carlo experiment for each case, then prints the comparison tables.
Requires `pip install -e .`; `--plot` also draws the NEES / NIS figures
(needs the "plot" extra).

    python examples/nonlinear_filters.py              # 30 runs per case
    python examples/nonlinear_filters.py --runs 100 --plot
"""

import sys

from filter_tester.experiment import MonteCarlo
from filter_tester.metrics import format_mc_summary
from filter_tester.scenarios import get_scenario

runs = int(sys.argv[sys.argv.index("--runs") + 1]) if "--runs" in sys.argv else 30
plot = "--plot" in sys.argv

cases = [
    ("pendulum", {"angle0": 10.0, "duration": 8.0}, "Pendulum, 10° swing: almost linear, every filter works"),
    ("pendulum", {"angle0": 120.0, "duration": 8.0}, "Pendulum, 120° swing: the small-angle KF breaks down"),
    ("lorenz", {"duration": 6.0}, "Lorenz system: chaotic; no linear model at all"),
]

results = []
for key, system, title in cases:
    scenario = get_scenario(key)
    filters = {
        ft.key.upper(): scenario.make_filter(ft.key, scenario.filter_defaults(ft.key, system))
        for ft in scenario.filter_types
    }
    result = MonteCarlo(scenario.source(system), filters, runs=runs).run()
    print(f"\n{title}\n" + format_mc_summary(result, list(scenario.state_names)))
    results.append((title, result))

if plot:
    import matplotlib.pyplot as plt

    from filter_tester.plotting import plot_mc_consistency

    for title, result in results:
        fig, _ = plot_mc_consistency(result)
        fig.suptitle(title)
    plt.show()
