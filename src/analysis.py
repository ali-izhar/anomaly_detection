# src/analysis.py

"""Analysis module for computing statistics on martingale plots."""

import numpy as np
from scipy.stats import gaussian_kde
import logging
from typing import Dict, Tuple, Optional, List
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
    kl_div_hist: float,
    kl_div_kde: float,
    js_div: float,
    correlation: float,
    change_points: List[int],
    n_bins: int = 50,
) -> None:
    """Create a compact visualization of martingale evolution with metrics."""
    # Set high-quality plotting defaults
    plt.rcParams["figure.dpi"] = 600
    plt.rcParams["savefig.dpi"] = 600
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["axes.linewidth"] = 0.5
    plt.rcParams["lines.linewidth"] = 0.8
    plt.rcParams["grid.linewidth"] = 0.5
    plt.rcParams["xtick.major.width"] = 0.5
    plt.rcParams["ytick.major.width"] = 0.5
    plt.rcParams["axes.unicode_minus"] = False

    # Create figure
    fig = plt.figure(figsize=(7, 2.5))  # Double column width, reduced height
    ax = fig.add_subplot(111)

    # Plot actual values
    ax.plot(
        time_points_actual,
        actual_sum,
        label="Sum Mart.",
        color="blue",
        linewidth=0.8,
        alpha=0.8,
        solid_capstyle="round",
        solid_joinstyle="round",
    )
    ax.plot(
        time_points_actual,
        actual_avg,
        label="Avg Mart.",
        color="#2ecc71",  # Green
        linewidth=0.8,
        linestyle="--",
        alpha=0.8,
        dash_capstyle="round",
        dash_joinstyle="round",
    )

    # Plot predicted values
    ax.plot(
        time_points_pred,
        pred_sum,
        label="Pred. Sum",
        color="orange",
        linewidth=0.8,
        alpha=0.8,
        solid_capstyle="round",
        solid_joinstyle="round",
    )
    ax.plot(
        time_points_pred,
        pred_avg,
        label="Pred. Avg",
        color="#9b59b6",  # Purple
        linewidth=0.8,
        linestyle="--",
        alpha=0.8,
        dash_capstyle="round",
        dash_joinstyle="round",
    )

    # Add change points
    for cp in change_points:
        ax.axvline(
            x=cp,
            color="r",
            alpha=0.5,
            linestyle="--",
            linewidth=0.5,
            label="Change Point" if cp == change_points[0] else "",
        )

    # Add title and labels
    ax.set_title("Martingale Evolution", fontsize=8, pad=3)
    ax.set_xlabel("Time", fontsize=8)
    ax.set_ylabel("Martingale Value", fontsize=8)

    # Add right-side legend (vertical)
    ax.legend(
        fontsize=6,
        ncol=1,  # Vertical layout
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),  # Place legend to the right of the plot
        borderaxespad=0.1,
        handlelength=1.5,
        columnspacing=1.0,
    )

    # Add left-side metrics legend
    metrics_text = (
        "Metrics:\n"
        f"KL (hist) = {kl_div_hist:.3f}\n"
        f"KL (kde) = {kl_div_kde:.3f}\n"
        f"JS = {js_div:.3f}\n"
        f"Corr = {correlation:.3f}"
    )
    ax.text(
        -0.15,  # Position to the left of the plot
        0.5,  # Vertical center
        metrics_text,
        transform=ax.transAxes,
        fontsize=6,
        verticalalignment="center",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=3),
    )

    # Customize grid and ticks
    ax.tick_params(axis="both", which="major", labelsize=6, pad=2)
    ax.grid(True, linestyle=":", alpha=0.3, linewidth=0.5)

    # Save with high quality settings
    plt.savefig(
        output_path,
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.02,
        format="png",
        metadata={"Creator": "Matplotlib"},
    )
    plt.close()

    # Reset rcParams to default
    plt.rcdefaults()
