"""Graph Forecasting Package.

A package for predicting and analyzing the evolution of complex networks.

Modules
-------
metrics
    Network metric calculations
predictors
    Network forecasting models
visualization
    Plotting and visualization utilities
"""

from .metrics import get_network_metrics, calculate_error_metrics
from .predictors import BaseNetworkPredictor, WeightedAveragePredictor
from .visualization import plot_metric_evolution

__version__ = "0.1.0"
__author__ = "Your Name"

__all__ = [
    "get_network_metrics",
    "calculate_error_metrics",
    "BaseNetworkPredictor",
    "WeightedAveragePredictor",
    "plot_metric_evolution",
]
