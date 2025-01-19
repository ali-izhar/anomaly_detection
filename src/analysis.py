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
    """Create a compact visualization of martingale distributions."""
    # Set high-quality plotting defaults
    plt.rcParams['figure.dpi'] = 600  # High DPI for the figure
    plt.rcParams['savefig.dpi'] = 600  # High DPI for saving
    plt.rcParams['pdf.fonttype'] = 42  # Ensure text is exported as text, not paths
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['axes.linewidth'] = 0.5  # Thinner spines
    plt.rcParams['lines.linewidth'] = 0.8  # Default line width
    plt.rcParams['grid.linewidth'] = 0.5  # Thinner grid lines
    plt.rcParams['xtick.major.width'] = 0.5  # Thinner ticks
    plt.rcParams['ytick.major.width'] = 0.5
    plt.rcParams['axes.unicode_minus'] = False  # Ensure proper minus signs

    # Create figure with 2x1 layout (time series on top, distributions on bottom)
    fig = plt.figure(figsize=(7, 4))  # Double column width, compact height
    gs = plt.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.4)

    # 1. Time series plot (top)
    ax_time = fig.add_subplot(gs[0])
    
    # Plot actual values with higher quality lines
    ax_time.plot(
        time_points_actual,
        actual_sum,
        label="Act. Sum",
        color="blue",
        linewidth=0.8,
        alpha=0.8,
        solid_capstyle='round',
        solid_joinstyle='round',
    )
    ax_time.plot(
        time_points_actual,
        actual_avg,
        label="Act. Avg",
        color="green",
        linewidth=0.8,
        alpha=0.8,
        solid_capstyle='round',
        solid_joinstyle='round',
    )
    
    # Plot predicted values with higher quality lines
    ax_time.plot(
        time_points_pred,
        pred_sum,
        label="Pred. Sum",
        color="orange",
        linewidth=0.8,
        linestyle="--",
        alpha=0.8,
        dash_capstyle='round',
        dash_joinstyle='round',
    )
    ax_time.plot(
        time_points_pred,
        pred_avg,
        label="Pred. Avg",
        color="red",
        linewidth=0.8,
        linestyle="--",
        alpha=0.8,
        dash_capstyle='round',
        dash_joinstyle='round',
    )
    
    ax_time.set_title("Martingale Evolution", fontsize=8, pad=3)
    ax_time.set_xlabel("Time", fontsize=8)
    ax_time.set_ylabel("Martingale Value", fontsize=8)
    ax_time.legend(fontsize=6, ncol=2, loc="upper right", 
                  borderaxespad=0.1, handlelength=1.5, 
                  columnspacing=1.0)
    ax_time.tick_params(axis='both', which='major', labelsize=6, pad=2)
    ax_time.grid(True, linestyle=':', alpha=0.3, linewidth=0.5)

    # 2. Combined distribution plot (bottom)
    ax_dist = fig.add_subplot(gs[1])
    
    # Plot sum distributions with refined settings
    ax_dist.hist(
        actual_sum,
        bins=n_bins,
        alpha=0.3,
        density=True,
        label="Act. Sum",
        color="blue",
        edgecolor='none',
    )
    ax_dist.hist(
        pred_sum,
        bins=n_bins,
        alpha=0.3,
        density=True,
        label="Pred. Sum",
        color="orange",
        edgecolor='none',
    )
    
    # Plot average distributions with different hatching
    ax_dist.hist(
        actual_avg,
        bins=n_bins,
        alpha=0.3,
        density=True,
        label="Act. Avg",
        color="green",
        hatch="//",
        edgecolor='darkgreen',
        linewidth=0.5,
    )
    ax_dist.hist(
        pred_avg,
        bins=n_bins,
        alpha=0.3,
        density=True,
        label="Pred. Avg",
        color="red",
        hatch="\\\\",
        edgecolor='darkred',
        linewidth=0.5,
    )
    
    # Add KDE curves with refined settings
    for data, color in [(actual_sum, "blue"), (pred_sum, "orange"), 
                       (actual_avg, "green"), (pred_avg, "red")]:
        kde = gaussian_kde(data)
        x_range = np.linspace(min(data), max(data), 300)  # More points for smoother curve
        ax_dist.plot(x_range, kde(x_range), color=color, linewidth=0.8,
                    solid_capstyle='round', solid_joinstyle='round')
    
    ax_dist.set_title("Martingale Distributions", fontsize=8, pad=3)
    ax_dist.set_xlabel("Martingale Value", fontsize=8)
    ax_dist.set_ylabel("Density", fontsize=8)
    ax_dist.legend(fontsize=6, ncol=2, loc="upper right", 
                  borderaxespad=0.1, handlelength=1.5, 
                  columnspacing=1.0)
    ax_dist.tick_params(axis='both', which='major', labelsize=6, pad=2)
    ax_dist.grid(True, linestyle=':', alpha=0.3, linewidth=0.5)

    # Ensure the figure is tight and save with high quality
    plt.tight_layout()
    
    # Save with high quality settings
    plt.savefig(
        output_path,
        dpi=600,  # High DPI
        bbox_inches="tight",
        pad_inches=0.02,  # Minimal padding
        format='png',  # Use PNG format for sharp lines
        metadata={'Creator': 'Matplotlib'}  # Add metadata
    )
    plt.close()

    # Reset rcParams to default
    plt.rcdefaults()
