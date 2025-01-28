# src/metrics/feature_metrics.py

"""Metrics for evaluating feature prediction accuracy."""

from dataclasses import dataclass
from typing import Dict, List, Optional, TypeVar, Union
import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Type aliases for better readability
FeatureDict = Dict[str, Union[float, List[float]]]
T = TypeVar("T", bound=Union[float, List[float]])

# Constants
EPSILON = 1e-8
DEFAULT_HIST_BINS = 50
MIN_SAMPLES_FOR_REGRESSION = 3


@dataclass
class FeatureMetrics:
    """Container for feature prediction metrics."""

    mse: float
    rmse: float
    mae: float
    mape: float
    r2: float
    mean_bias: float
    std_error: float


@dataclass
class DistributionMetrics:
    """Container for distribution-based metrics."""

    kl_divergence: float
    js_divergence: float
    wasserstein: float
    ks_statistic: float
    mean_diff: float
    std_diff: float


def compute_feature_metrics(
    actual_features: List[FeatureDict],
    predicted_features: List[FeatureDict],
    feature_names: Optional[List[str]] = None,
) -> Dict[str, FeatureMetrics]:
    """Compute metrics comparing actual vs predicted features.

    Args:
        actual_features: List of dictionaries containing actual feature values
        predicted_features: List of dictionaries containing predicted feature values
        feature_names: Optional list of feature names to evaluate. If None, uses all features.

    Returns:
        Dictionary mapping feature names to their metrics.

    Raises:
        ValueError: If input lengths mismatch or empty features provided.
    """
    if not actual_features or not predicted_features:
        raise ValueError("Empty feature lists provided")

    if len(actual_features) != len(predicted_features):
        raise ValueError(
            f"Length mismatch: actual ({len(actual_features)}) vs "
            f"predicted ({len(predicted_features)})"
        )

    feature_names = feature_names or list(actual_features[0].keys())
    metrics: Dict[str, FeatureMetrics] = {}

    for feature in feature_names:
        actual_values, predicted_values = _extract_feature_values(
            actual_features, predicted_features, feature
        )

        metrics[feature] = _compute_single_feature_metrics(
            np.array(actual_values), np.array(predicted_values)
        )

    return metrics


def compute_feature_distribution_metrics(
    actual_features: List[FeatureDict],
    predicted_features: List[FeatureDict],
    feature_names: Optional[List[str]] = None,
) -> Dict[str, DistributionMetrics]:
    """Compute distribution-based metrics for features that are lists.

    Args:
        actual_features: List of dictionaries containing actual feature values
        predicted_features: List of dictionaries containing predicted feature values
        feature_names: Optional list of feature names to evaluate. If None, uses all features.

    Returns:
        Dictionary mapping feature names to their distribution metrics.

    Raises:
        ValueError: If input lengths mismatch or empty features provided.
    """
    if not actual_features or not predicted_features:
        raise ValueError("Empty feature lists provided")

    if len(actual_features) != len(predicted_features):
        raise ValueError(
            f"Length mismatch: actual ({len(actual_features)}) vs "
            f"predicted ({len(predicted_features)})"
        )

    feature_names = feature_names or list(actual_features[0].keys())
    metrics: Dict[str, DistributionMetrics] = {}

    for feature in feature_names:
        # Skip non-list features
        if not isinstance(actual_features[0][feature], list):
            continue

        timestep_metrics = _compute_timestep_distribution_metrics(
            actual_features, predicted_features, feature
        )

        if timestep_metrics:
            metrics[feature] = _average_distribution_metrics(timestep_metrics)

    return metrics


def _extract_feature_values(
    actual_features: List[FeatureDict],
    predicted_features: List[FeatureDict],
    feature: str,
) -> tuple[List[float], List[float]]:
    """Extract and process feature values from actual and predicted features."""
    actual_values = []
    predicted_values = []

    for actual, predicted in zip(actual_features, predicted_features):
        # Handle both scalar and list features
        actual_val = (
            np.mean(actual[feature])
            if isinstance(actual[feature], list)
            else actual[feature]
        )
        pred_val = (
            np.mean(predicted[feature])
            if isinstance(predicted[feature], list)
            else predicted[feature]
        )

        actual_values.append(actual_val)
        predicted_values.append(pred_val)

    return actual_values, predicted_values


def _compute_single_feature_metrics(
    actual_values: NDArray[np.float64],
    predicted_values: NDArray[np.float64],
) -> FeatureMetrics:
    """Compute metrics for a single feature."""
    mse = mean_squared_error(actual_values, predicted_values)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual_values, predicted_values)

    # Compute MAPE, handling zero values
    with np.errstate(divide="ignore", invalid="ignore"):
        mape = np.mean(np.abs((actual_values - predicted_values) / actual_values)) * 100
        mape = np.nan_to_num(mape, nan=np.inf)

    # Compute R-squared
    ss_res = np.sum((actual_values - predicted_values) ** 2)
    ss_tot = np.sum((actual_values - np.mean(actual_values)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    # Compute bias and error statistics
    errors = predicted_values - actual_values
    mean_bias = float(np.mean(errors))
    std_error = float(np.std(errors))

    return FeatureMetrics(
        mse=float(mse),
        rmse=float(rmse),
        mae=float(mae),
        mape=float(mape),
        r2=float(r2),
        mean_bias=mean_bias,
        std_error=std_error,
    )


def _compute_timestep_distribution_metrics(
    actual_features: List[FeatureDict],
    predicted_features: List[FeatureDict],
    feature: str,
) -> List[DistributionMetrics]:
    """Compute distribution metrics for each timestep."""
    metrics = []

    for actual, predicted in zip(actual_features, predicted_features):
        actual_dist = np.array(actual[feature])
        pred_dist = np.array(predicted[feature])

        if len(actual_dist) == 0 or len(pred_dist) == 0:
            continue

        metrics.append(
            DistributionMetrics(
                kl_divergence=_compute_kl_divergence(actual_dist, pred_dist),
                js_divergence=_compute_js_divergence(actual_dist, pred_dist),
                wasserstein=_compute_wasserstein(actual_dist, pred_dist),
                ks_statistic=_compute_ks_statistic(actual_dist, pred_dist),
                mean_diff=float(np.mean(pred_dist) - np.mean(actual_dist)),
                std_diff=float(np.std(pred_dist) - np.std(actual_dist)),
            )
        )

    return metrics


def _average_distribution_metrics(
    metrics: List[DistributionMetrics],
) -> DistributionMetrics:
    """Average distribution metrics across timesteps."""
    return DistributionMetrics(
        kl_divergence=float(np.mean([m.kl_divergence for m in metrics])),
        js_divergence=float(np.mean([m.js_divergence for m in metrics])),
        wasserstein=float(np.mean([m.wasserstein for m in metrics])),
        ks_statistic=float(np.mean([m.ks_statistic for m in metrics])),
        mean_diff=float(np.mean([m.mean_diff for m in metrics])),
        std_diff=float(np.mean([m.std_diff for m in metrics])),
    )


def _compute_wasserstein(
    actual: NDArray[np.float64], predicted: NDArray[np.float64]
) -> float:
    """Compute Wasserstein distance between two distributions."""
    actual_sorted = np.sort(actual)
    predicted_sorted = np.sort(predicted)

    # Interpolate to same length if necessary
    if len(actual_sorted) != len(predicted_sorted):
        n = max(len(actual_sorted), len(predicted_sorted))
        actual_interp = np.interp(
            np.linspace(0, 1, n), np.linspace(0, 1, len(actual_sorted)), actual_sorted
        )
        predicted_interp = np.interp(
            np.linspace(0, 1, n),
            np.linspace(0, 1, len(predicted_sorted)),
            predicted_sorted,
        )
    else:
        actual_interp = actual_sorted
        predicted_interp = predicted_sorted

    return float(np.mean(np.abs(actual_interp - predicted_interp)))


def _compute_ks_statistic(
    actual: NDArray[np.float64], predicted: NDArray[np.float64]
) -> float:
    """Compute Kolmogorov-Smirnov statistic between two distributions."""
    actual_sorted = np.sort(actual)
    predicted_sorted = np.sort(predicted)

    n_actual = len(actual_sorted)
    n_predicted = len(predicted_sorted)

    actual_cdf = np.arange(1, n_actual + 1) / n_actual
    predicted_cdf = np.arange(1, n_predicted + 1) / n_predicted

    # Interpolate to same length if necessary
    if n_actual != n_predicted:
        n = max(n_actual, n_predicted)
        actual_cdf_interp = np.interp(
            np.linspace(0, 1, n), np.linspace(0, 1, n_actual), actual_cdf
        )
        predicted_cdf_interp = np.interp(
            np.linspace(0, 1, n), np.linspace(0, 1, n_predicted), predicted_cdf
        )
    else:
        actual_cdf_interp = actual_cdf
        predicted_cdf_interp = predicted_cdf

    return float(np.max(np.abs(actual_cdf_interp - predicted_cdf_interp)))


def _compute_kl_divergence(
    actual: NDArray[np.float64],
    predicted: NDArray[np.float64],
    bins: int = DEFAULT_HIST_BINS,
) -> float:
    """Compute KL divergence between two distributions using histogram approximation."""
    min_val = min(actual.min(), predicted.min())
    max_val = max(actual.max(), predicted.max())

    actual_hist, _ = np.histogram(
        actual, bins=bins, range=(min_val, max_val), density=True
    )
    predicted_hist, _ = np.histogram(
        predicted, bins=bins, range=(min_val, max_val), density=True
    )

    # Add small constant to avoid division by zero
    actual_hist = actual_hist + EPSILON
    predicted_hist = predicted_hist + EPSILON

    # Normalize
    actual_hist = actual_hist / actual_hist.sum()
    predicted_hist = predicted_hist / predicted_hist.sum()

    return float(np.sum(actual_hist * np.log(actual_hist / predicted_hist)))


def _compute_js_divergence(
    actual: NDArray[np.float64],
    predicted: NDArray[np.float64],
    bins: int = DEFAULT_HIST_BINS,
) -> float:
    """Compute Jensen-Shannon divergence between two distributions."""
    min_val = min(actual.min(), predicted.min())
    max_val = max(actual.max(), predicted.max())

    actual_hist, _ = np.histogram(
        actual, bins=bins, range=(min_val, max_val), density=True
    )
    predicted_hist, _ = np.histogram(
        predicted, bins=bins, range=(min_val, max_val), density=True
    )

    # Add small constant to avoid division by zero
    actual_hist = actual_hist + EPSILON
    predicted_hist = predicted_hist + EPSILON

    # Normalize
    actual_hist = actual_hist / actual_hist.sum()
    predicted_hist = predicted_hist / predicted_hist.sum()

    # Compute midpoint distribution
    m = 0.5 * (actual_hist + predicted_hist)

    return float(
        0.5 * np.sum(actual_hist * np.log(actual_hist / m))
        + 0.5 * np.sum(predicted_hist * np.log(predicted_hist / m))
    )
