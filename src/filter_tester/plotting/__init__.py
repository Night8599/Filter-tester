"""Plotting utilities (matplotlib).

matplotlib is an optional dependency: install it with ``pip install "filter-tester[plot]"``
(or ``pip install matplotlib``).
"""

try:
    import matplotlib  # noqa: F401
except ImportError as err:  # pragma: no cover - depends on the environment
    raise ImportError(
        'filter_tester.plotting needs matplotlib: pip install matplotlib (or install the package with the "plot" extra)'
    ) from err

from .filters import LiveFilterPlot, plot_errors, plot_estimates, plot_summary_table
from .montecarlo import (
    LiveMonteCarloPlot,
    plot_mc_consistency,
    plot_mc_nees,
    plot_mc_nis,
    plot_mc_summary_table,
    plot_rmse_boxplot,
)
from .systems import LiveStatePlot, plot_states, plot_system

__all__ = [
    "plot_states",
    "plot_system",
    "LiveStatePlot",
    "plot_estimates",
    "plot_errors",
    "plot_summary_table",
    "LiveFilterPlot",
    "plot_mc_consistency",
    "plot_mc_nees",
    "plot_mc_nis",
    "plot_rmse_boxplot",
    "plot_mc_summary_table",
    "LiveMonteCarloPlot",
]
