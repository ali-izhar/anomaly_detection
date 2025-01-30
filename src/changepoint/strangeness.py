# src/changepoint/strangeness.py

"""Strangeness computation for change point detection."""

import logging
import numpy as np
import random
from typing import List, Optional, Union, Any
from sklearn.cluster import KMeans, MiniBatchKMeans

from .distance import compute_cluster_distances

logger = logging.getLogger(__name__)


def strangeness_point(
    data: Union[List[Any], np.ndarray],
    n_clusters: int = 1,
    random_state: Optional[int] = None,
    batch_size: Optional[int] = None,
    distance_measure: str = "euclidean",
) -> np.ndarray:
    """
    Computes the 'strangeness' for each point in `data` as the minimum distance
    to any cluster center. One row per point.

    :param data: 2D or 3D data (N x d) or (N x something x d).
    :param n_clusters: Number of clusters to fit.
    :param random_state: Optional seed.
    :param batch_size: If provided and data is large, use MiniBatchKMeans.
    :param distance_measure: Distance metric to use: "euclidean" or "mahalanobis".
    :return: 1D array of shape (N,) containing the minimum distance (strangeness) for each point.
    """
    if data is None or len(data) == 0:
        logger.error("Empty data sequence")
        raise ValueError("Empty data sequence")

    # Set random seed for reproducibility if provided
    if random_state is not None:
        np.random.seed(random_state)
        random.seed(random_state)

    data_array = np.array(data)
    if data_array.size == 0:
        logger.error("Data array has zero size after np.array conversion")
        raise ValueError("Empty data sequence")

    # Flatten last dimension if data is 3D (e.g., shape (N, something, d))
    # so final shape is (N*, d).
    if data_array.ndim == 3:
        data_array = data_array.reshape(-1, data_array.shape[-1])

    N = data_array.shape[0]
    if batch_size is not None and N > batch_size:
        model = MiniBatchKMeans(
            n_clusters=n_clusters, batch_size=batch_size, random_state=random_state
        )
    else:
        model = KMeans(n_clusters=n_clusters, n_init="auto", random_state=random_state)

    logger.debug(f"Fitting KMeans with n_clusters={n_clusters}")
    model.fit(data_array)

    # Compute distances based on the chosen measure
    distances = compute_cluster_distances(data_array, model, distance_measure)

    # The strangeness is the min distance to any cluster center.
    strangeness_scores = distances.min(axis=1)

    return strangeness_scores


def get_pvalue(strangeness: List[float], random_state: Optional[int] = None) -> float:
    """
    Computes the conformal p-value for the *last* strangeness in the list,
    comparing it to all strangeness values in the same list (including itself).
    p-value formula (Vovk's tie-breaking):
        p = ( (# of points with strangeness > current ) + theta * (# = current ) ) / N

    :param strangeness: The full history of strangeness values, where
                        the last element is the 'new' point's strangeness.
    :param random_state: Optional seed for reproducibility.
    :return: A single float p-value.
    """
    if isinstance(strangeness, np.ndarray):
        if strangeness.size == 0:
            logger.error("Empty numpy array provided for strangeness computation")
            raise ValueError("Strangeness sequence cannot be empty")
    else:
        if not strangeness:
            logger.error("Empty list provided for strangeness computation")
            raise ValueError("Strangeness sequence cannot be empty")

    if random_state is not None:
        np.random.seed(random_state)
        random.seed(random_state)

    # Tie-breaking variable theta ~ Uniform(0,1)
    theta = random.uniform(0, 1)
    current = strangeness[-1]

    # Count how many are strictly larger and how many tie exactly
    num_larger = sum(s > current for s in strangeness)
    num_equal = sum(s == current for s in strangeness)

    # Conformal p-value
    pvalue = (num_larger + theta * num_equal) / len(strangeness)
    logger.debug(f"Computed p-value: {pvalue} (theta={theta}, current={current})")
    return pvalue
