"""Conformal prediction: strangeness scores and p-values."""

import random
from typing import Optional, Literal
import numpy as np
from scipy.spatial.distance import mahalanobis


DistanceMetric = Literal["euclidean", "mahalanobis", "cosine", "chebyshev"]


def compute_distance(
    x: np.ndarray,
    center: np.ndarray,
    metric: DistanceMetric = "mahalanobis",
    cov_inv: Optional[np.ndarray] = None,
) -> float:
    """Compute distance between observation and center.

    St = ||Xt - Ct|| where ||.|| is the distance metric.

    Args:
        x: Observation vector (d,)
        center: Cluster center vector (d,)
        metric: Distance metric to use
        cov_inv: Inverse covariance matrix for Mahalanobis (d, d)

    Returns:
        Distance value
    """
    x = np.asarray(x).ravel()
    center = np.asarray(center).ravel()

    if metric == "euclidean":
        return float(np.linalg.norm(x - center))

    elif metric == "mahalanobis":
        if cov_inv is None:
            # Fall back to Euclidean if no covariance provided
            return float(np.linalg.norm(x - center))
        try:
            return float(mahalanobis(x, center, cov_inv))
        except (ValueError, np.linalg.LinAlgError):
            return float(np.linalg.norm(x - center))

    elif metric == "cosine":
        norm_x = np.linalg.norm(x)
        norm_c = np.linalg.norm(center)
        if norm_x < 1e-10 or norm_c < 1e-10:
            return 0.0
        similarity = np.dot(x, center) / (norm_x * norm_c)
        return float(1.0 - similarity)

    elif metric == "chebyshev":
        return float(np.max(np.abs(x - center)))

    else:
        raise ValueError(f"Unknown distance metric: {metric}")


def compute_nonconformity_score(
    history: np.ndarray,
    new_point: np.ndarray,
    metric: DistanceMetric = "mahalanobis",
) -> float:
    """Compute non-conformity score for new observation.

    St = ||Xt - Ct|| where:
    - Ct = cluster center (mean) from historical observations {X1, ..., X_{t-1}}
    - ||.|| is the configured distance metric

    Args:
        history: Historical observations (n_samples, n_features)
        new_point: New observation (n_features,)
        metric: Distance metric to use

    Returns:
        Non-conformity score St
    """
    history = np.atleast_2d(history)
    new_point = np.asarray(new_point).ravel()

    if history.shape[0] == 0:
        return 0.0

    # Ct = mean of historical observations (K-means with K=1)
    center = np.mean(history, axis=0)

    # Compute inverse covariance for Mahalanobis
    cov_inv = None
    if metric == "mahalanobis" and history.shape[0] > history.shape[1]:
        try:
            cov = np.cov(history, rowvar=False)
            if cov.ndim == 0:
                cov = np.array([[cov]])
            elif cov.ndim == 1:
                cov = np.diag(cov)
            # Add small regularization for numerical stability
            cov += np.eye(cov.shape[0]) * 1e-6
            cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            cov_inv = None

    return compute_distance(new_point, center, metric, cov_inv)


def compute_pvalue(
    historical_scores: np.ndarray,
    current_score: float,
    random_state: Optional[int] = None,
) -> float:
    """Compute conformal p-value.

    pt = (#{s : Ss > St} + θt * #{s : Ss = St}) / t

    where θt ~ Unif(0,1) is drawn independently at time t to break ties.

    Args:
        historical_scores: Array of historical non-conformity scores {S1, ..., S_{t-1}}
        current_score: Current non-conformity score St
        random_state: Random seed for reproducibility

    Returns:
        P-value in [0, 1]
    """
    if len(historical_scores) == 0:
        return 0.5

    if random_state is not None:
        random.seed(random_state + len(historical_scores))

    n_larger = np.sum(historical_scores > current_score)
    n_equal = np.sum(historical_scores == current_score)
    theta = random.random()

    # Add 1 to denominator to account for the new observation
    t = len(historical_scores) + 1
    return (n_larger + theta * (n_equal + 1)) / t
