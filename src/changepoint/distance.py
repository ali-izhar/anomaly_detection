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

    # For Euclidean distance, use the clustering model's efficient transform.
    if distance_measure == "euclidean":
        distances = model.transform(data_array)
        return distances

    # Global Mahalanobis distance:
    # This implementation computes a single covariance matrix over all points in data_array.
    # Note: If a local (cluster-specific) covariance is desired, this implementation would need to be modified.
    elif distance_measure == "mahalanobis":
        centers = model.cluster_centers_
        n_clusters = centers.shape[0]
        N = data_array.shape[0]
        # Compute global covariance matrix over the data.
        cov = np.cov(data_array, rowvar=False)
        # Use the pseudo-inverse to handle cases where the covariance matrix is singular.
        inv_cov = np.linalg.pinv(cov)

        distances = np.zeros((N, n_clusters))
        for i in range(N):
            for j in range(n_clusters):
                diff = data_array[i] - centers[j]

                distances[i, j] = np.sqrt(diff @ inv_cov @ diff)
        return distances

    # Manhattan distance (L1 norm)
    elif distance_measure == "manhattan":
        centers = model.cluster_centers_
        n_clusters = centers.shape[0]
        N = data_array.shape[0]
        distances = np.zeros((N, n_clusters))
        for i in range(N):

            for j in range(n_clusters):
                distances[i, j] = np.sum(np.abs(data_array[i] - centers[j]))

        return distances

    # Cosine distance: 1 - cosine similarity.
    # This is useful when only the angular similarity is of interest.
    elif distance_measure == "cosine":
        centers = model.cluster_centers_
        n_clusters = centers.shape[0]
        N = data_array.shape[0]
        distances = np.zeros((N, n_clusters))
        for i in range(N):

            for j in range(n_clusters):
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

    # Minkowski distance generalizes both Euclidean (p=2) and Manhattan (p=1).
    elif distance_measure == "minkowski":
        centers = model.cluster_centers_
        n_clusters = centers.shape[0]
        N = data_array.shape[0]
        distances = np.zeros((N, n_clusters))
        for i in range(N):
            for j in range(n_clusters):
                distances[i, j] = np.sum(np.abs(data_array[i] - centers[j]) ** p) ** (
                    1.0 / p
                )
        return distances

    else:
        logger.error(f"Invalid distance_measure: {distance_measure}")
        raise ValueError(f"Invalid distance_measure: {distance_measure}")
