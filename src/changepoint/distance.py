# src/changepoint/distance.py

"""Distance computation utilities for change point detection.

This module provides a collection of distance metrics for computing dissimilarity
between data points and cluster centers in the context of change point detection.
The distances are used in strangeness measure computation and conformal prediction.

Mathematical Framework:
---------------------
For two points x, y ∈ ℝᵈ, the following distances are supported:

1. Euclidean Distance (L₂):
   d(x,y) = √(Σᵢ(xᵢ-yᵢ)²)

2. Mahalanobis Distance:
   d(x,y) = √((x-y)ᵀΣ⁻¹(x-y))
   where Σ is the covariance matrix

3. Manhattan Distance (L₁):
   d(x,y) = Σᵢ|xᵢ-yᵢ|

4. Cosine Distance:
   d(x,y) = 1 - cos(x,y) = 1 - (x·y)/(||x||·||y||)

5. Minkowski Distance (Lₚ):
   d(x,y) = (Σᵢ|xᵢ-yᵢ|ᵖ)^(1/p)

6. Chebyshev Distance (L∞):
   d(x,y) = maxᵢ|xᵢ-yᵢ|

Numerical Considerations:
----------------------
1. Small constants are added to denominators to prevent division by zero
2. Covariance matrices are regularized for numerical stability
3. Input arrays are validated for correct dimensions and types
4. Edge cases (zero vectors, identical points) are handled gracefully

References:
----------
[1] Cha, S.-H. (2007). "Comprehensive Survey on Distance/Similarity Measures
    between Probability Density Functions"
[2] Deza, M. M., & Deza, E. (2009). "Encyclopedia of Distances"
"""

from dataclasses import dataclass
from typing import Union, Optional, Literal, get_args
import logging
import numpy as np
from numpy.linalg import LinAlgError
from sklearn.cluster import KMeans, MiniBatchKMeans

logger = logging.getLogger(__name__)

# Define allowed distance metric names using Literal types for better type safety.
DistanceMetric = Literal[
    "euclidean", "mahalanobis", "manhattan", "cosine", "minkowski", "chebyshev"
]
# Retrieve the valid metric names as a tuple
VALID_METRICS: tuple[str, ...] = get_args(DistanceMetric)


@dataclass(frozen=True)
class DistanceConfig:
    """Configuration for distance computation.

    Attributes:
        metric: Distance metric to use
        p: Order parameter for Minkowski distance
        eps: Small constant for numerical stability
        cov_reg: Regularization parameter for covariance matrix
        use_correlation: Whether to use correlation instead of covariance
    """

    metric: DistanceMetric = "euclidean"
    p: float = 2.0
    eps: float = 1e-8
    cov_reg: float = 1e-6
    use_correlation: bool = False

    def __post_init__(self):
        """Validate configuration parameters."""
        # Check if the provided metric is one of the allowed metrics
        if self.metric not in VALID_METRICS:
            raise ValueError(
                f"Invalid metric: {self.metric}. Must be one of {VALID_METRICS}"
            )
        # Ensure the Minkowski parameter p is strictly positive
        if self.p <= 0:
            raise ValueError(f"p must be positive, got {self.p}")
        # Ensure the small constant eps is strictly positive
        if self.eps <= 0:
            raise ValueError(f"eps must be positive, got {self.eps}")
        # Regularization parameter for the covariance must be non-negative
        if self.cov_reg < 0:
            raise ValueError(f"cov_reg must be non-negative, got {self.cov_reg}")


def compute_pairwise_distances(
    x: np.ndarray,
    y: np.ndarray,
    config: Optional[DistanceConfig] = None,
) -> np.ndarray:
    """Compute pairwise distances between points in x and y.

    Args:
        x: Array of shape (n_samples_x, n_features)
        y: Array of shape (n_samples_y, n_features)
        config: Distance computation configuration

    Returns:
        Array of shape (n_samples_x, n_samples_y) containing pairwise distances

    Raises:
        ValueError: If input dimensions are invalid
        RuntimeError: If distance computation fails
    """
    # Use the provided configuration or default to a Euclidean configuration
    config = config or DistanceConfig()

    # Convert inputs to numpy arrays if they aren't already
    x = np.asarray(x)
    y = np.asarray(y)

    # Validate input dimensions
    if x.ndim > 2 or y.ndim > 2:
        raise ValueError(f"Expected 2D arrays, got x.ndim={x.ndim}, y.ndim={y.ndim}")
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("Expected 2D arrays")

    try:
        # Validate feature dimensions match
        if x.shape[1] != y.shape[1]:
            raise ValueError(
                f"Feature dimension mismatch: x={x.shape[1]}, y={y.shape[1]}"
            )

        # Dispatch to the proper function based on the selected metric
        if config.metric == "euclidean":
            return _compute_euclidean(x, y, config.eps)
        elif config.metric == "mahalanobis":
            return _compute_mahalanobis(x, y, config)
        elif config.metric == "manhattan":
            return _compute_manhattan(x, y)
        elif config.metric == "cosine":
            return _compute_cosine(x, y, config.eps)
        elif config.metric == "minkowski":
            return _compute_minkowski(x, y, config.p)
        else:  # Must be chebyshev if not any of the above
            return _compute_chebyshev(x, y)

    except Exception as e:
        # Only wrap non-ValueError exceptions in RuntimeError
        if isinstance(e, ValueError):
            raise
        logger.error(f"Distance computation failed: {str(e)}")
        raise RuntimeError(f"Distance computation failed: {str(e)}")


def compute_cluster_distances(
    data_array: np.ndarray,
    model: Union[KMeans, MiniBatchKMeans],
    config: Optional[DistanceConfig] = None,
) -> np.ndarray:
    """Compute distances from each point to cluster centers.

    Args:
        data_array: Array of shape (n_samples, n_features)
        model: Fitted clustering model with cluster centers
        config: Distance computation configuration

    Returns:
        Array of shape (n_samples, n_clusters) containing distances to centers

    Raises:
        TypeError: If input type is invalid
        ValueError: If input dimensions are invalid
        RuntimeError: If distance computation fails
    """
    # Use default configuration if none is provided
    config = config or DistanceConfig()

    # Check input type before any conversion
    if not isinstance(data_array, np.ndarray):
        raise TypeError("data_array must be a numpy array")

    # Store original dimension for validation
    orig_ndim = data_array.ndim

    # Reshape 1D array to 2D if needed
    if data_array.ndim == 1:
        data_array = data_array.reshape(-1, 1)

    # Validate input dimensions
    if orig_ndim != 2 and orig_ndim != 1:
        raise ValueError(f"Expected 1D or 2D array, got {orig_ndim}D")

    try:
        # Retrieve cluster centers from the model
        centers = model.cluster_centers_

        # Check feature dimensions match
        if data_array.shape[1] != centers.shape[1]:
            raise ValueError(
                f"Feature dimension mismatch: data={data_array.shape[1]}, centers={centers.shape[1]}"
            )

        # For Euclidean distances, if not using correlation adjustments,
        # use the clustering model's built-in transform for efficiency
        if config.metric == "euclidean" and not config.use_correlation:
            try:
                return model.transform(data_array)
            except Exception:
                # Fall back to manual computation if transform fails
                return compute_pairwise_distances(data_array, centers, config)

        # Otherwise, compute the distances manually using pairwise distances
        return compute_pairwise_distances(data_array, centers, config)

    except (TypeError, ValueError) as e:
        # Re-raise TypeError and ValueError as is
        raise
    except Exception as e:
        # Wrap other exceptions in RuntimeError
        logger.error(f"Cluster distance computation failed: {str(e)}")
        raise RuntimeError(f"Cluster distance computation failed: {str(e)}")


def _compute_euclidean(x: np.ndarray, y: np.ndarray, eps: float) -> np.ndarray:
    """Compute Euclidean (L₂) distances."""
    # Compute squared norms for x and y.
    x2 = np.sum(x * x, axis=1, keepdims=True)  # Shape (n_x, 1)
    y2 = np.sum(y * y, axis=1, keepdims=True).T  # Shape (1, n_y)
    # Compute the dot product between x and y.
    xy = np.dot(x, y.T)  # Shape (n_x, n_y)

    # Calculate squared distances and take the square root.
    # Use np.maximum to ensure non-negative values inside sqrt
    squared_dist = np.maximum(x2 + y2 - 2 * xy, 0)
    # Only add eps to exact zeros
    squared_dist = np.where(squared_dist == 0, 0, squared_dist)
    return np.sqrt(squared_dist)


def _compute_mahalanobis(
    x: np.ndarray,
    y: np.ndarray,
    config: DistanceConfig,
) -> np.ndarray:
    """Compute Mahalanobis distances using global covariance."""
    # Special handling for 1D case
    if x.shape[1] == 1:
        x_flat = x.ravel()
        y_flat = y.ravel()
        var = np.var(np.concatenate([x_flat, y_flat])) + config.eps
        diff = x_flat[:, np.newaxis] - y_flat[np.newaxis, :]
        return np.abs(diff) / np.sqrt(var)

    # Regular computation for higher dimensions
    combined = np.vstack([x, y])
    if config.use_correlation:
        cov = np.corrcoef(combined, rowvar=False)
    else:
        cov = np.cov(combined, rowvar=False)
        cov += np.eye(cov.shape[0]) * config.cov_reg

    try:
        inv_cov = np.linalg.pinv(cov)
    except LinAlgError:
        logger.warning(
            "Covariance matrix inversion failed, falling back to Euclidean distance"
        )
        return _compute_euclidean(x, y, config.eps)

    diff = x[:, np.newaxis, :] - y[np.newaxis, :, :]
    return np.sqrt(
        np.sum(np.sum(diff[:, :, :, np.newaxis] * inv_cov, axis=2) * diff, axis=2)
    )


def _compute_manhattan(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute Manhattan (L₁) distances."""
    # Use broadcasting to compute pairwise differences
    diff = x[:, np.newaxis, :] - y[np.newaxis, :, :]  # Shape (n_x, n_y, n_features)
    # Sum absolute differences along feature axis
    distances = np.sum(np.abs(diff), axis=2)
    # Ensure exact zeros for identical points
    return np.where(np.all(diff == 0, axis=2), 0, distances)


def _compute_cosine(x: np.ndarray, y: np.ndarray, eps: float) -> np.ndarray:
    """Compute cosine distances."""
    # Compute the L2 norms of x and y.
    x_norm = np.sqrt(np.sum(x * x, axis=1, keepdims=True))
    y_norm = np.sqrt(np.sum(y * y, axis=1, keepdims=True))

    # Normalize the vectors; add eps to avoid division by zero.
    x_normalized = x / (x_norm + eps)
    y_normalized = y / (y_norm + eps)

    # Compute cosine similarity as the dot product of the normalized vectors.
    similarity = np.dot(x_normalized, y_normalized.T)
    # Clip the similarity values to the valid range [-1, 1] to handle numerical imprecision.
    similarity = np.clip(similarity, -1.0, 1.0)
    # Convert similarity to distance.
    return 1.0 - similarity


def _compute_minkowski(x: np.ndarray, y: np.ndarray, p: float) -> np.ndarray:
    """Compute Minkowski (Lₚ) distances."""
    # Calculate the p-th power of the absolute differences, sum them, then take the 1/p root.
    return np.power(
        np.sum(np.power(np.abs(x[:, np.newaxis, :] - y[np.newaxis, :, :]), p), axis=2),
        1.0 / p,
    )


def _compute_chebyshev(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute Chebyshev (L∞) distances."""
    # For each pair, take the maximum absolute difference across features.
    return np.max(np.abs(x[:, np.newaxis, :] - y[np.newaxis, :, :]), axis=2)
