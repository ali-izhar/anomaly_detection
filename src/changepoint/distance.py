# src/changepoint/distance.py

"""Distance computation utilities for change point detection."""

from typing import Union

import logging
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans

logger = logging.getLogger(__name__)


def compute_cluster_distances(
    data_array: np.ndarray,
    model: Union[KMeans, MiniBatchKMeans],
    distance_measure: str = "euclidean",
    p: float = 2.0,
) -> np.ndarray:
    """
    Compute distances from each point in `data_array` to each cluster center
    using the specified distance measure.

    Parameters
    ----------
    data_array : np.ndarray
        2D numpy array of shape (N, d) where N is the number of points.
    model : Union[KMeans, MiniBatchKMeans]
        A fitted clustering model (e.g., KMeans) whose cluster centers will be used.
    distance_measure : str, optional
        The distance metric to use. Options are:
          - "euclidean" (default): Uses the model's transform.
          - "mahalanobis": Uses a global covariance matrix computed over data_array.
          - "manhattan": L1 norm.
          - "cosine": 1 minus the cosine similarity.
          - "minkowski": Minkowski distance with order p (default p=2.0 is Euclidean).
    p : float, optional
        The order for Minkowski distance (only used when distance_measure=="minkowski").

    Returns
    -------
    np.ndarray
        A (N x n_clusters) array of distances.
    """
    # Ensure that the input data is a numpy array.
    if not isinstance(data_array, np.ndarray):
        logger.error("data_array must be a numpy array")
        raise TypeError("data_array must be a numpy array")

    # Check that the data array is two-dimensional.
    if data_array.ndim != 2:
        logger.error(f"Invalid data dimensions: {data_array.ndim}D. Expected 2D array.")
        raise ValueError(
            f"Invalid data dimensions: {data_array.ndim}D. Expected 2D array."
        )

    # Unpack the shape of the data array: N points with d features each.
    N, d = data_array.shape

    # Retrieve cluster centers from the clustering model.
    centers = model.cluster_centers_
    n_clusters, d_centers = centers.shape

    # Validate that the feature dimensions of the data and the cluster centers match.
    if d != d_centers:
        logger.error(f"Feature dimension mismatch: data ({d}) != centers ({d_centers})")
        raise ValueError(
            f"Feature dimension mismatch: data ({d}) != centers ({d_centers})"
        )

    # === Euclidean Distance ===
    # For Euclidean distance, we take advantage of the model's efficient transform method.
    if distance_measure == "euclidean":
        distances = model.transform(data_array)
        if distances.shape != (N, n_clusters):
            logger.error(
                f"Invalid euclidean distances shape: {distances.shape}. Expected: ({N}, {n_clusters})"
            )
            raise ValueError(
                f"Invalid euclidean distances shape: {distances.shape}. Expected: ({N}, {n_clusters})"
            )
        return distances

    # === Mahalanobis Distance ===
    elif distance_measure == "mahalanobis":
        # For single-feature data, use variance instead of covariance matrix
        if d == 1:
            # Compute variance, adding small constant to avoid division by zero
            var = np.var(data_array, axis=0) + 1e-8
            distances = np.zeros((N, n_clusters))
            for i in range(N):
                for j in range(n_clusters):
                    diff = data_array[i] - centers[j]
                    distances[i, j] = np.sqrt(np.sum((diff**2) / var))
            return distances

        # For multi-feature data, use full covariance matrix
        # Compute the global covariance matrix over all data points
        cov = np.cov(data_array, rowvar=False)
        if cov.shape != (d, d):
            logger.error(
                f"Invalid covariance matrix shape: {cov.shape}. Expected: ({d}, {d})"
            )
            raise ValueError(
                f"Invalid covariance matrix shape: {cov.shape}. Expected: ({d}, {d})"
            )

        # Add small constant to diagonal to ensure matrix is invertible
        cov = cov + np.eye(d) * 1e-8

        # Compute the pseudo-inverse of the covariance matrix to handle singular cases
        inv_cov = np.linalg.pinv(cov)
        if inv_cov.shape != (d, d):
            logger.error(
                f"Invalid inverse covariance matrix shape: {inv_cov.shape}. Expected: ({d}, {d})"
            )
            raise ValueError(
                f"Invalid inverse covariance matrix shape: {inv_cov.shape}. Expected: ({d}, {d})"
            )

        # Initialize an array to store distances from each point to each cluster center.
        distances = np.zeros((N, n_clusters))
        # Compute the Mahalanobis distance for every point-center pair.
        for i in range(N):
            for j in range(n_clusters):
                # Calculate the difference vector between the point and the cluster center.
                diff = data_array[i] - centers[j]
                if diff.shape != (d,):
                    logger.error(
                        f"Invalid difference vector shape: {diff.shape}. Expected: ({d},)"
                    )
                    raise ValueError(
                        f"Invalid difference vector shape: {diff.shape}. Expected: ({d},)"
                    )
                # Compute the Mahalanobis distance using the quadratic form.
                distances[i, j] = np.sqrt(diff @ inv_cov @ diff)

        return distances

    # === Manhattan Distance (L1 Norm) ===
    elif distance_measure == "manhattan":
        # Initialize the distances array.
        distances = np.zeros((N, n_clusters))
        # Compute L1 distance (sum of absolute differences) for each point-center pair.
        for i in range(N):
            for j in range(n_clusters):
                diff = data_array[i] - centers[j]
                if diff.shape != (d,):
                    logger.error(
                        f"Invalid difference vector shape: {diff.shape}. Expected: ({d},)"
                    )
                    raise ValueError(
                        f"Invalid difference vector shape: {diff.shape}. Expected: ({d},)"
                    )
                distances[i, j] = np.sum(np.abs(diff))
        return distances

    # === Cosine Distance ===
    # Cosine distance is defined as 1 minus the cosine similarity.
    elif distance_measure == "cosine":
        # Initialize an array for the cosine distances.
        distances = np.zeros((N, n_clusters))
        # Iterate over every point and cluster center.
        for i in range(N):
            for j in range(n_clusters):
                if data_array[i].shape != (d,) or centers[j].shape != (d,):
                    logger.error(
                        f"Invalid vector shapes: data {data_array[i].shape}, center {centers[j].shape}. Expected: ({d},)"
                    )
                    raise ValueError(
                        f"Invalid vector shapes: data {data_array[i].shape}, center {centers[j].shape}. Expected: ({d},)"
                    )

                # Compute the dot product between the data point and the cluster center.
                numerator = np.dot(data_array[i], centers[j])
                # Compute the norms of the data point and the cluster center.
                norm_i = np.linalg.norm(data_array[i])
                norm_j = np.linalg.norm(centers[j])

                # Handle potential division by zero by setting cosine similarity to zero.
                if norm_i == 0 or norm_j == 0:
                    cosine_similarity = 0
                else:
                    cosine_similarity = numerator / (norm_i * norm_j)
                # Cosine distance is 1 minus the cosine similarity.
                distances[i, j] = 1 - cosine_similarity

        return distances

    # === Minkowski Distance ===
    elif distance_measure == "minkowski":
        # Initialize an array to hold the Minkowski distances.
        distances = np.zeros((N, n_clusters))
        # Calculate the Minkowski distance (of order p) for each point-center pair.
        for i in range(N):
            for j in range(n_clusters):
                diff = data_array[i] - centers[j]
                if diff.shape != (d,):
                    logger.error(
                        f"Invalid difference vector shape: {diff.shape}. Expected: ({d},)"
                    )
                    raise ValueError(
                        f"Invalid difference vector shape: {diff.shape}. Expected: ({d},)"
                    )
                # Minkowski distance: sum(|diff|^p) raised to the power of 1/p.
                distances[i, j] = np.sum(np.abs(diff) ** p) ** (1.0 / p)

        return distances

    # === Chebyshev Distance (L∞ norm) ===
    elif distance_measure == "chebyshev":
        # Initialize an array to hold the Chebyshev distances.
        distances = np.zeros((N, n_clusters))
        # Calculate the Chebyshev distance (maximum absolute difference) for each point-center pair.
        for i in range(N):
            for j in range(n_clusters):
                diff = data_array[i] - centers[j]
                if diff.shape != (d,):
                    logger.error(
                        f"Invalid difference vector shape: {diff.shape}. Expected: ({d},)"
                    )
                    raise ValueError(
                        f"Invalid difference vector shape: {diff.shape}. Expected: ({d},)"
                    )
                # Chebyshev distance: max(|diff|)
                distances[i, j] = np.max(np.abs(diff))

        return distances

    else:
        logger.error(f"Invalid distance_measure: {distance_measure}")
        raise ValueError(f"Invalid distance_measure: {distance_measure}")
