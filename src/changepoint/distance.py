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
    # Validate input dimensions
    if not isinstance(data_array, np.ndarray):
        logger.error("data_array must be a numpy array")
        raise TypeError("data_array must be a numpy array")

    if data_array.ndim != 2:
        logger.error(f"Invalid data dimensions: {data_array.ndim}D. Expected 2D array.")
        raise ValueError(
            f"Invalid data dimensions: {data_array.ndim}D. Expected 2D array."
        )

    N, d = data_array.shape
    centers = model.cluster_centers_
    n_clusters, d_centers = centers.shape

    # Validate cluster centers dimensions
    if d != d_centers:
        logger.error(f"Feature dimension mismatch: data ({d}) != centers ({d_centers})")
        raise ValueError(
            f"Feature dimension mismatch: data ({d}) != centers ({d_centers})"
        )

    logger.debug("Input dimensions:")
    logger.debug(f"- Data shape: {data_array.shape}")
    logger.debug(f"- Cluster centers shape: {centers.shape}")

    # For Euclidean distance, use the clustering model's efficient transform.
    if distance_measure == "euclidean":
        distances = model.transform(data_array)
        # Validate output shape
        if distances.shape != (N, n_clusters):
            logger.error(
                f"Invalid euclidean distances shape: {distances.shape}. Expected: ({N}, {n_clusters})"
            )
            raise ValueError(
                f"Invalid euclidean distances shape: {distances.shape}. Expected: ({N}, {n_clusters})"
            )
        return distances

    # Global Mahalanobis distance
    elif distance_measure == "mahalanobis":
        # Compute global covariance matrix over the data.
        cov = np.cov(data_array, rowvar=False)
        if cov.shape != (d, d):
            logger.error(
                f"Invalid covariance matrix shape: {cov.shape}. Expected: ({d}, {d})"
            )
            raise ValueError(
                f"Invalid covariance matrix shape: {cov.shape}. Expected: ({d}, {d})"
            )

        # Use the pseudo-inverse to handle cases where the covariance matrix is singular.
        inv_cov = np.linalg.pinv(cov)
        if inv_cov.shape != (d, d):
            logger.error(
                f"Invalid inverse covariance matrix shape: {inv_cov.shape}. Expected: ({d}, {d})"
            )
            raise ValueError(
                f"Invalid inverse covariance matrix shape: {inv_cov.shape}. Expected: ({d}, {d})"
            )

        distances = np.zeros((N, n_clusters))
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
                distances[i, j] = np.sqrt(diff @ inv_cov @ diff)

        return distances

    # Manhattan distance (L1 norm)
    elif distance_measure == "manhattan":
        distances = np.zeros((N, n_clusters))
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

    # Cosine distance: 1 - cosine similarity
    elif distance_measure == "cosine":
        distances = np.zeros((N, n_clusters))
        for i in range(N):
            for j in range(n_clusters):
                if data_array[i].shape != (d,) or centers[j].shape != (d,):
                    logger.error(
                        f"Invalid vector shapes: data {data_array[i].shape}, center {centers[j].shape}. Expected: ({d},)"
                    )
                    raise ValueError(
                        f"Invalid vector shapes: data {data_array[i].shape}, center {centers[j].shape}. Expected: ({d},)"
                    )

                numerator = np.dot(data_array[i], centers[j])
                norm_i = np.linalg.norm(data_array[i])
                norm_j = np.linalg.norm(centers[j])

                # Prevent division by zero.
                if norm_i == 0 or norm_j == 0:
                    cosine_similarity = 0
                else:
                    cosine_similarity = numerator / (norm_i * norm_j)
                distances[i, j] = 1 - cosine_similarity

        return distances

    # Minkowski distance
    elif distance_measure == "minkowski":
        distances = np.zeros((N, n_clusters))
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
                distances[i, j] = np.sum(np.abs(diff) ** p) ** (1.0 / p)

        return distances

    else:
        logger.error(f"Invalid distance_measure: {distance_measure}")
        raise ValueError(f"Invalid distance_measure: {distance_measure}")
