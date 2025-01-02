"""Visualization utilities for network forecasting.

This module provides functions for visualizing network metrics,
prediction results, and evolution of network properties.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any, Tuple
from .metrics import get_network_metrics


def plot_metric_evolution(
    actual_series: List[Dict[str, Any]],
    predictions: List[Dict[str, Any]],
    min_history: int,
    figsize: Tuple[int, int] = (15, 25),
) -> None:
    """Plot the evolution of multiple network metrics over time.

    Parameters
    ----------
    actual_series : List[Dict[str, Any]]
        Complete actual network time series
    predictions : List[Dict[str, Any]]
        List of predictions
    min_history : int
        Minimum history points needed
    figsize : Tuple[int, int], optional
        Figure size (width, height), by default (15, 25)
    """
    # Calculate metrics
    actual_metrics = [get_network_metrics(net["graph"]) for net in actual_series]
    pred_metrics = [get_network_metrics(p["graph"]) for p in predictions]
    pred_times = [p["time"] for p in predictions]
    history_sizes = [p["history_size"] for p in predictions]

    # Setup subplots
    fig, axs = plt.subplots(4, 2, figsize=figsize)
    fig.suptitle(
        "Evolution of Network Metrics Over Time\nwith Expanding History",
        fontsize=16,
        y=1.02,
    )

    # Define metrics to plot
    metrics = [
        ("avg_degree", "Average Degree", "Degree"),
        ("clustering", "Clustering Coefficient", "Coefficient"),
        ("avg_betweenness", "Average Betweenness", "Centrality"),
        ("spectral_gap", "Spectral Gap", "Value"),
        ("algebraic_connectivity", "Algebraic Connectivity", "Value"),
        ("density", "Network Density", "Density"),
    ]

    times = list(range(len(actual_series)))

    # Plot each metric
    for idx, (metric_name, title, ylabel) in enumerate(metrics):
        _plot_metric(
            axs[idx // 2, idx % 2],
            times,
            pred_times,
            min_history,
            actual_metrics,
            pred_metrics,
            metric_name,
            title,
            ylabel,
        )

    # Plot history size
    _plot_history_size(axs[3, 0], pred_times, history_sizes)

    # Plot prediction error
    _plot_prediction_error(
        axs[3, 1], pred_times, predictions, actual_series[min_history:]
    )

    plt.tight_layout()
    plt.show()


def _plot_metric(
    ax: plt.Axes,
    times: List[int],
    pred_times: List[int],
    min_history: int,
    actual_metrics: List[Dict[str, float]],
    pred_metrics: List[Dict[str, float]],
    metric_name: str,
    title: str,
    ylabel: str,
) -> None:
    """Plot a single metric's evolution.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes to plot on
    times : List[int]
        Time points for actual series
    pred_times : List[int]
        Time points for predictions
    min_history : int
        Minimum history points
    actual_metrics : List[Dict[str, float]]
        Actual metric values
    pred_metrics : List[Dict[str, float]]
        Predicted metric values
    metric_name : str
        Name of metric to plot
    title : str
        Plot title
    ylabel : str
        Y-axis label
    """
    # Get metric values
    actual_values = [m[metric_name] for m in actual_metrics]
    pred_values = [m[metric_name] for m in pred_metrics]

    # Plot actual values
    ax.plot(times, actual_values, "b-", label="Actual", linewidth=2, alpha=0.7)

    # Plot predicted values
    ax.plot(pred_times, pred_values, "r--", label="Predicted", linewidth=2, alpha=0.7)

    # Add vertical line at minimum history
    ax.axvline(x=min_history, color="g", linestyle=":", label="Prediction Start")

    # Add labels and title
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlabel("Time Step", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)

    # Add legend and grid
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Set y-limits with padding
    ymin = min(min(actual_values), min(pred_values))
    ymax = max(max(actual_values), max(pred_values))
    padding = (ymax - ymin) * 0.1
    ax.set_ylim(ymin - padding, ymax + padding)


def _plot_history_size(
    ax: plt.Axes, pred_times: List[int], history_sizes: List[int]
) -> None:
    """Plot the evolution of training history size.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes to plot on
    pred_times : List[int]
        Time points for predictions
    history_sizes : List[int]
        Number of historical points used for each prediction
    """
    ax.plot(pred_times, history_sizes, "g-", label="History Size", linewidth=2)
    ax.set_title("Training History Size", fontsize=12, pad=10)
    ax.set_xlabel("Time Step", fontsize=10)
    ax.set_ylabel("Number of Points", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)


def _plot_prediction_error(
    ax: plt.Axes,
    pred_times: List[int],
    predictions: List[Dict[str, Any]],
    actual_series: List[Dict[str, Any]],
) -> None:
    """Plot the evolution of prediction error.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes to plot on
    pred_times : List[int]
        Time points for predictions
    predictions : List[Dict[str, Any]]
        Prediction data
    actual_series : List[Dict[str, Any]]
        Actual network data
    """
    errors = []
    for p, actual in zip(predictions, actual_series):
        pred_metrics = get_network_metrics(p["graph"])
        actual_metrics = get_network_metrics(actual["graph"])
        error = abs(pred_metrics["avg_degree"] - actual_metrics["avg_degree"])
        errors.append(error)

    ax.plot(pred_times, errors, "r-", label="Prediction Error", linewidth=2)
    ax.set_title("Average Degree Prediction Error", fontsize=12, pad=10)
    ax.set_xlabel("Time Step", fontsize=10)
    ax.set_ylabel("Absolute Error", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
