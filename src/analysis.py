# src/analysis.py

"""Analysis module for computing statistics on martingale plots."""

import numpy as np
from scipy.stats import gaussian_kde
import logging
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

logger = logging.getLogger(__name__)


def align_martingales(
    actual_sum: np.ndarray,
    pred_sum: np.ndarray,
    time_points_actual: range,
    time_points_pred: range,
    prediction_window: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Align actual and predicted martingales to the same timescale.

    Args:
        actual_sum: Array of actual sum martingale values
        pred_sum: Array of predicted sum martingale values
        time_points_actual: Range of timesteps for actual values
        time_points_pred: Range of timesteps for predicted values
        prediction_window: Number of steps predicted ahead

    Returns:
        Tuple of aligned (actual_sum, pred_sum) arrays
    """
    # Convert ranges to arrays for easier manipulation
    actual_times = np.array(list(time_points_actual))
    pred_times = np.array(list(time_points_pred))

    # Shift predicted times forward by prediction_window
    aligned_pred_times = pred_times + prediction_window

    # Find overlapping time range
    start_time = max(actual_times[0], aligned_pred_times[0])
    end_time = min(actual_times[-1], aligned_pred_times[-1])

    # Get indices for overlapping range
    actual_mask = (actual_times >= start_time) & (actual_times <= end_time)
    pred_mask = (aligned_pred_times >= start_time) & (aligned_pred_times <= end_time)

    return actual_sum[actual_mask], pred_sum[pred_mask]


def compute_kl_divergence(
    actual_sum: np.ndarray,
    pred_sum: np.ndarray,
    n_bins: int = 50,
    bandwidth: Optional[float] = None,
) -> Dict[str, float]:
    """Compute KL divergence between actual and predicted martingale distributions.

    Args:
        actual_sum: Array of actual sum martingale values
        pred_sum: Array of predicted sum martingale values
        n_bins: Number of bins for histogram approximation
        bandwidth: Optional bandwidth for kernel density estimation

    Returns:
        Dict containing:
            - kl_div_hist: KL divergence using histogram approximation
            - kl_div_kde: KL divergence using kernel density estimation
            - js_div: Jensen-Shannon divergence
    """
    # Ensure arrays are same length
    if len(actual_sum) != len(pred_sum):
        raise ValueError("Arrays must be of same length after alignment")

    # 1. Histogram-based KL divergence
    # Get common bin edges for both distributions
    min_val = min(actual_sum.min(), pred_sum.min())
    max_val = max(actual_sum.max(), pred_sum.max())
    bins = np.linspace(min_val, max_val, n_bins)

    # Compute histograms
    hist_actual, _ = np.histogram(actual_sum, bins=bins, density=True)
    hist_pred, _ = np.histogram(pred_sum, bins=bins, density=True)

    # Add small constant to avoid division by zero
    eps = 1e-10
    hist_actual = hist_actual + eps
    hist_pred = hist_pred + eps

    # Normalize
    hist_actual = hist_actual / hist_actual.sum()
    hist_pred = hist_pred / hist_pred.sum()

    # Compute KL divergence using histograms
    kl_div_hist = np.sum(hist_actual * np.log(hist_actual / hist_pred))

    # 2. KDE-based KL divergence
    if bandwidth is None:
        # Scott's rule for bandwidth selection
        bandwidth = (
            0.9
            * min(np.std(actual_sum), np.std(pred_sum))
            * len(actual_sum) ** (-1 / 5)
        )

    kde_actual = gaussian_kde(actual_sum, bw_method=bandwidth)
    kde_pred = gaussian_kde(pred_sum, bw_method=bandwidth)

    # Evaluate KDEs on a grid
    grid = np.linspace(min_val, max_val, n_bins)
    pdf_actual = kde_actual(grid)
    pdf_pred = kde_pred(grid)

    # Normalize
    pdf_actual = pdf_actual / pdf_actual.sum()
    pdf_pred = pdf_pred / pdf_pred.sum()

    # Compute KL divergence using KDEs
    kl_div_kde = np.sum(pdf_actual * np.log(pdf_actual / pdf_pred))

    # 3. Jensen-Shannon divergence
    m = 0.5 * (pdf_actual + pdf_pred)
    js_div = 0.5 * (
        np.sum(pdf_actual * np.log(pdf_actual / m))
        + np.sum(pdf_pred * np.log(pdf_pred / m))
    )

    return {
        "kl_div_hist": float(kl_div_hist),
        "kl_div_kde": float(kl_div_kde),
        "js_div": float(js_div),
    }


def analyze_martingale_distributions(
    actual_sum: np.ndarray,
    pred_sum: np.ndarray,
    time_points_actual: range,
    time_points_pred: range,
    prediction_window: int,
    n_bins: int = 50,
    bandwidth: Optional[float] = None,
) -> Dict[str, float]:
    """Analyze the distributions of actual and predicted martingales.

    Args:
        actual_sum: Array of actual sum martingale values
        pred_sum: Array of predicted sum martingale values
        time_points_actual: Range of timesteps for actual values
        time_points_pred: Range of timesteps for predicted values
        prediction_window: Number of steps predicted ahead
        n_bins: Number of bins for histogram approximation
        bandwidth: Optional bandwidth for kernel density estimation

    Returns:
        Dict containing various statistical measures
    """
    # First align the martingales
    aligned_actual, aligned_pred = align_martingales(
        actual_sum, pred_sum, time_points_actual, time_points_pred, prediction_window
    )

    # Compute divergence measures
    divergences = compute_kl_divergence(
        aligned_actual, aligned_pred, n_bins=n_bins, bandwidth=bandwidth
    )

    # Additional statistical measures
    stats = {"correlation": float(np.corrcoef(aligned_actual, aligned_pred)[0, 1])}

    return {**divergences, **stats}


def visualize_distribution_analysis(
    actual_sum: np.ndarray,
    pred_sum: np.ndarray,
    actual_avg: np.ndarray,
    pred_avg: np.ndarray,
    time_points_actual: range,
    time_points_pred: range,
    prediction_window: int,
    output_path: str,
    n_bins: int = 50,
) -> None:
    """Create a comprehensive visualization of martingale distributions and their metrics.

    Args:
        actual_sum: Array of actual sum martingale values
        pred_sum: Array of predicted sum martingale values
        actual_avg: Array of actual average martingale values
        pred_avg: Array of predicted average martingale values
        time_points_actual: Range of timesteps for actual values
        time_points_pred: Range of timesteps for predicted values
        prediction_window: Number of steps predicted ahead
        output_path: Path to save the visualization
        n_bins: Number of bins for histograms
    """
    # Compute distribution analysis
    aligned_actual, aligned_pred = align_martingales(
        actual_sum, pred_sum, time_points_actual, time_points_pred, prediction_window
    )
    metrics = compute_kl_divergence(aligned_actual, aligned_pred, n_bins=n_bins)
    correlation = float(np.corrcoef(aligned_actual, aligned_pred)[0, 1])

    # Create figure with GridSpec
    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(3, 2, figure=fig)

    # 1. Time series plot (top row, spans both columns)
    ax_time = fig.add_subplot(gs[0, :])
    ax_time.plot(
        time_points_actual,
        actual_sum,
        label="Sum Martingale",
        color="blue",
        linewidth=2,
    )
    ax_time.plot(
        time_points_pred, pred_sum, label="Predicted Sum", color="orange", linewidth=2
    )
    ax_time.plot(
        time_points_actual,
        actual_avg,
        label="Avg Martingale",
        color="green",
        linestyle="--",
        linewidth=2,
    )
    ax_time.plot(
        time_points_pred,
        pred_avg,
        label="Predicted Avg",
        color="red",
        linestyle="--",
        linewidth=2,
    )

    # Add metrics text box to top-left of time series plot
    metrics_text = (
        f"Distribution Metrics:\n"
        f'KL Divergence (Hist): {metrics["kl_div_hist"]:.3f}\n'
        f'KL Divergence (KDE): {metrics["kl_div_kde"]:.3f}\n'
        f'Jensen-Shannon Div: {metrics["js_div"]:.3f}\n'
        f"Correlation: {correlation:.3f}"
    )
    ax_time.text(
        0.02,
        0.98,
        metrics_text,
        transform=ax_time.transAxes,  # Use axes coordinates
        fontsize=10,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray", pad=1.5),
    )

    ax_time.set_title("Martingale Evolution Over Time", fontsize=12)
    ax_time.set_xlabel("Time", fontsize=10)
    ax_time.set_ylabel("Martingale Value", fontsize=10)
    ax_time.legend(fontsize=10, loc="upper right")

    # 2. Distribution plots (middle row)
    # Sum martingales
    ax_sum_dist = fig.add_subplot(gs[1, 0])
    ax_sum_dist.hist(
        actual_sum,
        bins=n_bins,
        alpha=0.5,
        density=True,
        label="Actual Sum",
        color="blue",
    )
    ax_sum_dist.hist(
        pred_sum,
        bins=n_bins,
        alpha=0.5,
        density=True,
        label="Predicted Sum",
        color="orange",
    )
    ax_sum_dist.set_title("Sum Martingale Distribution", fontsize=12)
    ax_sum_dist.set_xlabel("Martingale Value", fontsize=10)
    ax_sum_dist.set_ylabel("Density", fontsize=10)
    ax_sum_dist.legend(fontsize=10)

    # Average martingales
    ax_avg_dist = fig.add_subplot(gs[1, 1])
    ax_avg_dist.hist(
        actual_avg,
        bins=n_bins,
        alpha=0.5,
        density=True,
        label="Actual Avg",
        color="green",
    )
    ax_avg_dist.hist(
        pred_avg,
        bins=n_bins,
        alpha=0.5,
        density=True,
        label="Predicted Avg",
        color="red",
    )
    ax_avg_dist.set_title("Average Martingale Distribution", fontsize=12)
    ax_avg_dist.set_xlabel("Martingale Value", fontsize=10)
    ax_avg_dist.set_ylabel("Density", fontsize=10)
    ax_avg_dist.legend(fontsize=10)

    # 3. KDE plots (bottom row)
    # Sum martingales KDE
    ax_sum_kde = fig.add_subplot(gs[2, 0])
    kde_actual_sum = gaussian_kde(actual_sum)
    kde_pred_sum = gaussian_kde(pred_sum)
    x_grid = np.linspace(
        min(actual_sum.min(), pred_sum.min()),
        max(actual_sum.max(), pred_sum.max()),
        200,
    )
    ax_sum_kde.plot(x_grid, kde_actual_sum(x_grid), label="Actual Sum", color="blue")
    ax_sum_kde.plot(x_grid, kde_pred_sum(x_grid), label="Predicted Sum", color="orange")
    ax_sum_kde.set_title("Sum Martingale KDE", fontsize=12)
    ax_sum_kde.set_xlabel("Martingale Value", fontsize=10)
    ax_sum_kde.set_ylabel("Density", fontsize=10)
    ax_sum_kde.legend(fontsize=10)

    # Average martingales KDE
    ax_avg_kde = fig.add_subplot(gs[2, 1])
    kde_actual_avg = gaussian_kde(actual_avg)
    kde_pred_avg = gaussian_kde(pred_avg)
    x_grid = np.linspace(
        min(actual_avg.min(), pred_avg.min()),
        max(actual_avg.max(), pred_avg.max()),
        200,
    )
    ax_avg_kde.plot(x_grid, kde_actual_avg(x_grid), label="Actual Avg", color="green")
    ax_avg_kde.plot(x_grid, kde_pred_avg(x_grid), label="Predicted Avg", color="red")
    ax_avg_kde.set_title("Average Martingale KDE", fontsize=12)
    ax_avg_kde.set_xlabel("Martingale Value", fontsize=10)
    ax_avg_kde.set_ylabel("Density", fontsize=10)
    ax_avg_kde.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
