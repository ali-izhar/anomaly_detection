# src/changepoint/distance.py

"""Distance computation utilities for change point detection."""

import logging
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from typing import Union

logger = logging.getLogger(__name__)


def compute_cluster_distances(
    data_array: np.ndarray,
    model: Union[KMeans, MiniBatchKMeans],
    distance_measure: str = "euclidean",
) -> np.ndarray:
    """
    Compute distances from each point in `data_array` to each cluster center
    using the specified distance measure.

    :param data_array: 2D numpy array (N x d).
    :param model: Fitted KMeans or MiniBatchKMeans model.
    :param distance_measure: "euclidean" (default) or "mahalanobis".
    :return: A (N x n_clusters) array of distances.
    """
    if distance_measure == "euclidean":
        # Use the model's transform for Euclidean distance
        return model.transform(data_array)

    elif distance_measure == "mahalanobis":
        # Mahalanobis distance uses a global covariance matrix for all data
        centers = model.cluster_centers_
        n_clusters = centers.shape[0]
        N = data_array.shape[0]

        # Compute (pseudo-)inverse of covariance
        cov = np.cov(data_array, rowvar=False)
        inv_cov = np.linalg.pinv(cov)

        # Calculate Mahalanobis distances
        distances = np.zeros((N, n_clusters))
        for i in range(N):
            for j in range(n_clusters):
                diff = data_array[i] - centers[j]
                distances[i, j] = np.sqrt(diff @ inv_cov @ diff)
        return distances

    else:
        logger.error(f"Invalid distance_measure: {distance_measure}")
        raise ValueError(f"Invalid distance_measure: {distance_measure}")
