"""Graph Forecasting Package.

A package for predicting and analyzing the evolution of complex networks.

Modules
-------
generators
    Network generation functions
metrics
    Network metric calculations
predictors
    Network forecasting models
visualization
    Plotting and visualization utilities
"""

from .generators import generate_ba_network, generate_evolving_ba_network
from .metrics import get_network_metrics, calculate_error_metrics
from .predictors import BaseNetworkPredictor, WeightedAveragePredictor
from .visualization import plot_metric_evolution

__version__ = "0.1.0"
__author__ = "Your Name"

__all__ = [
    "generate_ba_network",
    "generate_evolving_ba_network",
    "get_network_metrics",
    "calculate_error_metrics",
    "BaseNetworkPredictor",
    "WeightedAveragePredictor",
    "plot_metric_evolution",
]
