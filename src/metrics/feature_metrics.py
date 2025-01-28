"""Metrics for evaluating feature prediction accuracy."""

from typing import Dict, List, Tuple
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


def compute_feature_metrics(
    actual_features: List[Dict],
    predicted_features: List[Dict],
    feature_names: List[str] = None,
) -> Dict[str, Dict[str, float]]:
    """Compute comprehensive metrics comparing actual vs predicted features.

    Args:
        actual_features: List of dictionaries containing actual feature values
        predicted_features: List of dictionaries containing predicted feature values
        feature_names: Optional list of feature names to evaluate. If None, uses all features.

    Returns:
        Dictionary mapping feature names to their metrics:
        {
            'feature_name': {
                'mse': Mean squared error,
                'rmse': Root mean squared error,
                'mae': Mean absolute error,
                'mape': Mean absolute percentage error,
                'r2': R-squared score,
                'mean_bias': Mean bias (predicted - actual),
                'std_error': Standard deviation of error
            }
        }
    """
    if len(actual_features) != len(predicted_features):
        raise ValueError(
            f"Length mismatch: actual ({len(actual_features)}) vs "
            f"predicted ({len(predicted_features)})"
        )

    if not actual_features:
        raise ValueError("Empty feature lists provided")

    # Get feature names if not provided
    if feature_names is None:
        feature_names = list(actual_features[0].keys())

    metrics = {}
    for feature in feature_names:
        # Extract values, handling both scalar and list features
        actual_values = []
        predicted_values = []

        for actual, predicted in zip(actual_features, predicted_features):
            # For list features (like degrees), use mean
            if isinstance(actual[feature], list):
                actual_values.append(np.mean(actual[feature]))
                predicted_values.append(np.mean(predicted[feature]))
            else:
                actual_values.append(actual[feature])
                predicted_values.append(predicted[feature])

        actual_values = np.array(actual_values)
        predicted_values = np.array(predicted_values)

        # Compute basic error metrics
        mse = mean_squared_error(actual_values, predicted_values)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual_values, predicted_values)

        # Compute MAPE, handling zero values
        with np.errstate(divide="ignore", invalid="ignore"):
            mape = (
                np.mean(np.abs((actual_values - predicted_values) / actual_values))
                * 100
            )
            mape = np.nan_to_num(mape, nan=np.inf)  # Replace NaN with inf

        # Compute R-squared
        ss_res = np.sum((actual_values - predicted_values) ** 2)
        ss_tot = np.sum((actual_values - np.mean(actual_values)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        # Compute bias and error statistics
        errors = predicted_values - actual_values
        mean_bias = np.mean(errors)
        std_error = np.std(errors)

        metrics[feature] = {
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "mape": float(mape),
            "r2": float(r2),
            "mean_bias": float(mean_bias),
            "std_error": float(std_error),
        }

    return metrics


def compute_feature_distribution_metrics(
    actual_features: List[Dict],
    predicted_features: List[Dict],
    feature_names: List[str] = None,
) -> Dict[str, Dict[str, float]]:
    """Compute distribution-based metrics for features that are lists (e.g., degrees).

    Args:
        actual_features: List of dictionaries containing actual feature values
        predicted_features: List of dictionaries containing predicted feature values
        feature_names: Optional list of feature names to evaluate. If None, uses all features.

    Returns:
        Dictionary mapping feature names to their distribution metrics:
        {
            'feature_name': {
                'kl_divergence': KL divergence between distributions,
                'js_divergence': Jensen-Shannon divergence,
                'wasserstein': Wasserstein distance,
                'ks_statistic': Kolmogorov-Smirnov statistic,
                'mean_diff': Difference in means,
                'std_diff': Difference in standard deviations
            }
        }
    """
    if len(actual_features) != len(predicted_features):
        raise ValueError(
            f"Length mismatch: actual ({len(actual_features)}) vs "
            f"predicted ({len(predicted_features)})"
        )

    if not actual_features:
        raise ValueError("Empty feature lists provided")

    # Get feature names if not provided
    if feature_names is None:
        feature_names = list(actual_features[0].keys())

    metrics = {}
    for feature in feature_names:
        # Only process list features
        if not isinstance(actual_features[0][feature], list):
            continue

        # Compute metrics for each timestep and average
        timestep_metrics = []
        for actual, predicted in zip(actual_features, predicted_features):
            actual_dist = np.array(actual[feature])
            pred_dist = np.array(predicted[feature])

            if len(actual_dist) == 0 or len(pred_dist) == 0:
                continue

            # Compute basic statistics differences
            mean_diff = np.mean(pred_dist) - np.mean(actual_dist)
            std_diff = np.std(pred_dist) - np.std(actual_dist)

            # Compute Wasserstein distance
            wasserstein = _compute_wasserstein(actual_dist, pred_dist)

            # Compute KS statistic
            ks_stat = _compute_ks_statistic(actual_dist, pred_dist)

            # Compute KL and JS divergences
            kl_div = _compute_kl_divergence(actual_dist, pred_dist)
            js_div = _compute_js_divergence(actual_dist, pred_dist)

            timestep_metrics.append(
                {
                    "kl_divergence": kl_div,
                    "js_divergence": js_div,
                    "wasserstein": wasserstein,
                    "ks_statistic": ks_stat,
                    "mean_diff": mean_diff,
                    "std_diff": std_diff,
                }
            )

        # Average metrics across timesteps
        if timestep_metrics:
            metrics[feature] = {
                metric: float(np.mean([m[metric] for m in timestep_metrics]))
                for metric in timestep_metrics[0].keys()
            }

    return metrics


def _compute_wasserstein(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Compute Wasserstein distance between two distributions."""
    # Sort the distributions
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


def _compute_ks_statistic(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Compute Kolmogorov-Smirnov statistic between two distributions."""
    # Compute empirical CDFs
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
    actual: np.ndarray, predicted: np.ndarray, bins: int = 50
) -> float:
    """Compute KL divergence between two distributions using histogram approximation."""
    # Compute histogram range
    min_val = min(actual.min(), predicted.min())
    max_val = max(actual.max(), predicted.max())

    # Compute histograms
    actual_hist, _ = np.histogram(
        actual, bins=bins, range=(min_val, max_val), density=True
    )
    predicted_hist, _ = np.histogram(
        predicted, bins=bins, range=(min_val, max_val), density=True
    )

    # Add small constant to avoid division by zero
    epsilon = 1e-10
    actual_hist = actual_hist + epsilon
    predicted_hist = predicted_hist + epsilon

    # Normalize
    actual_hist = actual_hist / actual_hist.sum()
    predicted_hist = predicted_hist / predicted_hist.sum()

    # Compute KL divergence
    return float(np.sum(actual_hist * np.log(actual_hist / predicted_hist)))


def _compute_js_divergence(
    actual: np.ndarray, predicted: np.ndarray, bins: int = 50
) -> float:
    """Compute Jensen-Shannon divergence between two distributions."""
    # Compute histogram range
    min_val = min(actual.min(), predicted.min())
    max_val = max(actual.max(), predicted.max())

    # Compute histograms
    actual_hist, _ = np.histogram(
        actual, bins=bins, range=(min_val, max_val), density=True
    )
    predicted_hist, _ = np.histogram(
        predicted, bins=bins, range=(min_val, max_val), density=True
    )

    # Add small constant to avoid division by zero
    epsilon = 1e-10
    actual_hist = actual_hist + epsilon
    predicted_hist = predicted_hist + epsilon

    # Normalize
    actual_hist = actual_hist / actual_hist.sum()
    predicted_hist = predicted_hist / predicted_hist.sum()

    # Compute midpoint distribution
    m = 0.5 * (actual_hist + predicted_hist)

    # Compute JS divergence
    return float(
        0.5 * np.sum(actual_hist * np.log(actual_hist / m))
        + 0.5 * np.sum(predicted_hist * np.log(predicted_hist / m))
    )
