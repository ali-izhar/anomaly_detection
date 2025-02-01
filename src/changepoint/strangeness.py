# src/changepoint/strangeness.py

"""Strangeness computation for change point detection."""

from typing import List, Optional, Union, Any
import logging
import numpy as np
import random
from sklearn.cluster import KMeans, MiniBatchKMeans

from .distance import compute_cluster_distances

logger = logging.getLogger(__name__)


def strangeness_point(
    data: Union[List[Any], np.ndarray],
    n_clusters: int = 1,
    random_state: Optional[int] = None,
    batch_size: Optional[int] = None,
    distance_measure: str = "euclidean",
    p: float = 2.0,
) -> np.ndarray:
    """
    Computes the 'strangeness' for each point in `data` as the minimum distance
    to any cluster center. The distances are computed using a specified metric.

    Parameters
    ----------
    data : Union[List[Any], np.ndarray]
        2D or 3D data with shape (N x d) or (N x something x d).
    n_clusters : int, optional
        Number of clusters to fit (default is 1).
    random_state : int, optional
        Seed for reproducibility.
    batch_size : int, optional
        If provided and the data is large, MiniBatchKMeans is used.
    distance_measure : str, optional
        The distance metric to use. Supported options are:
          "euclidean", "mahalanobis", "manhattan", "cosine", and "minkowski".
          (Default is "euclidean".)
    p : float, optional
        The order for Minkowski distance (only used if distance_measure=="minkowski").
        Default is 2.0.

    Returns
    -------
    np.ndarray
        A 1D array of shape (N,) containing the minimum distance (strangeness) for each point.
    """
    # Validate input data.
    if data is None or len(data) == 0:
        logger.error("Empty data sequence")
        raise ValueError("Empty data sequence")

    # Set random seed for reproducibility, if provided.
    if random_state is not None:
        np.random.seed(random_state)
        random.seed(random_state)

    # Convert data to a numpy array.
    data_array = np.array(data)
    if data_array.size == 0:
        logger.error("Data array has zero size after np.array conversion")
        raise ValueError("Empty data sequence")

    # If data is 3D (e.g., (N, something, d)), flatten the extra dimension.
    if data_array.ndim == 3:
        data_array = data_array.reshape(-1, data_array.shape[-1])

    N = data_array.shape[0]
    # Choose MiniBatchKMeans if batch_size is provided and data is large.
    if batch_size is not None and N > batch_size:
        model = MiniBatchKMeans(
            n_clusters=n_clusters, batch_size=batch_size, random_state=random_state
        )
    else:
        model = KMeans(n_clusters=n_clusters, n_init="auto", random_state=random_state)
    model.fit(data_array)

    # Compute distances from each point to each cluster center using the specified metric.
    distances = compute_cluster_distances(data_array, model, distance_measure, p)

    # The strangeness for each point is the minimum distance to any of the cluster centers.
    strangeness_scores = distances.min(axis=1)
    return strangeness_scores


def get_pvalue(strangeness: List[float], random_state: Optional[int] = None) -> float:
    """
    Computes the conformal p-value for the *last* strangeness value in the list,
    comparing it to all strangeness values in the list (including itself).
    The p-value is computed using Vovk's tie-breaking:
        p = ( (# of points with strangeness > current ) + theta * (# = current ) ) / N
    where theta ~ Uniform(0, 1).

    Parameters
    ----------
    strangeness : List[float]
        The full history of strangeness values, with the last element corresponding
        to the 'new' point.
    random_state : int, optional
        Seed for reproducibility.

    Returns
    -------
    float
        The conformal p-value.
    """
    # Validate the strangeness list or array.
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

    # Generate a random tie-breaker from Uniform(0,1).
    theta = random.uniform(0, 1)
    current = strangeness[-1]

    # Count how many values are strictly larger and how many are equal to the current.
    num_larger = sum(s > current for s in strangeness)
    num_equal = sum(s == current for s in strangeness)
    pvalue = (num_larger + theta * num_equal) / len(strangeness)
    return pvalue
