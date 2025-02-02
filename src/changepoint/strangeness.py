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
    # --- Input Validation ---

    # Check if the provided data is None or an empty sequence.
    if data is None or len(data) == 0:
        logger.error("Empty data sequence")
        raise ValueError("Empty data sequence")

    # Set the random seed for reproducibility if random_state is provided.
    if random_state is not None:
        np.random.seed(random_state)
        random.seed(random_state)

    # Convert the input data to a numpy array to ensure consistent processing.
    data_array = np.array(data)
    if data_array.size == 0:
        logger.error("Data array has zero size after np.array conversion")
        raise ValueError("Empty data sequence")

    # --- Data Dimension Handling ---

    # Check if the data has either 2 or 3 dimensions.
    if data_array.ndim not in [2, 3]:
        logger.error(
            f"Invalid data dimensions: {data_array.ndim}D. Expected 2D or 3D array."
        )
        raise ValueError(
            f"Invalid data dimensions: {data_array.ndim}D. Expected 2D or 3D array."
        )

    # Log the original shape of the input data.
    logger.debug(f"Input data dimensions. Original shape: {data_array.shape}")

    # If the data is 3D (for example, (N, something, d)), flatten the extra dimension
    # so that the data becomes a 2D array of shape (N * something, d).
    if data_array.ndim == 3:
        data_array = data_array.reshape(-1, data_array.shape[-1])
        logger.debug(f"Reshaped to: {data_array.shape}")

    # --- Post-Reshape Validations ---

    # Unpack the shape of the (now 2D) data: N points with d features.
    N, d = data_array.shape

    # Ensure there are at least as many points as the number of clusters.
    if N < n_clusters:
        logger.error(
            f"Number of points ({N}) must be >= number of clusters ({n_clusters})"
        )
        raise ValueError(
            f"Number of points ({N}) must be >= number of clusters ({n_clusters})"
        )

    # The feature dimension must be non-zero.
    if d == 0:
        logger.error("Feature dimension cannot be zero")
        raise ValueError("Feature dimension cannot be zero")

    # --- Clustering Model Selection ---

    # If a batch_size is provided and the number of points exceeds the batch size,
    # use MiniBatchKMeans for more efficient clustering on large datasets.
    if batch_size is not None and N > batch_size:
        model = MiniBatchKMeans(
            n_clusters=n_clusters, batch_size=batch_size, random_state=random_state
        )
    else:
        # Otherwise, use the standard KMeans clustering.
        model = KMeans(n_clusters=n_clusters, n_init="auto", random_state=random_state)

    # Fit the clustering model to the data.
    model.fit(data_array)

    # --- Distance Computation ---

    # Compute the distances from each data point to every cluster center using the
    # specified distance metric.
    distances = compute_cluster_distances(data_array, model, distance_measure, p)

    # Validate that the resulting distance matrix has the expected shape (N, n_clusters).
    if distances.shape != (N, n_clusters):
        logger.error(
            f"Invalid distances shape: {distances.shape}. Expected: ({N}, {n_clusters})"
        )
        raise ValueError(
            f"Invalid distances shape: {distances.shape}. Expected: ({N}, {n_clusters})"
        )

    # --- Strangeness Computation ---

    # For each point, compute its strangeness as the minimum distance to any cluster center.
    strangeness_scores = distances.min(axis=1)

    # Validate the output shape to be a 1D array with one entry per data point.
    if strangeness_scores.shape != (N,):
        logger.error(
            f"Invalid output shape: {strangeness_scores.shape}. Expected: ({N},)"
        )
        raise ValueError(
            f"Invalid output shape: {strangeness_scores.shape}. Expected: ({N},)"
        )

    logger.debug(f"Output strangeness shape: {strangeness_scores.shape}")
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
    # --- Input Validation ---

    # Check if the strangeness input is a numpy array.
    if isinstance(strangeness, np.ndarray):
        # Ensure the numpy array is not empty.
        if strangeness.size == 0:
            logger.error("Empty numpy array provided for strangeness computation")
            raise ValueError("Strangeness sequence cannot be empty")
    else:
        # If it is a list, ensure it is not empty.
        if not strangeness:
            logger.error("Empty list provided for strangeness computation")
            raise ValueError("Strangeness sequence cannot be empty")

    # Set random seeds for reproducibility if random_state is provided.
    if random_state is not None:
        np.random.seed(random_state)
        random.seed(random_state)

    # --- p-value Computation Using Vovk's Tie-Breaking ---

    # Generate a random number theta from Uniform(0,1) to break ties.
    theta = random.uniform(0, 1)

    # The current (new) strangeness value is the last element in the sequence.
    current = strangeness[-1]

    # Count the number of points with strangeness strictly larger than the current value.
    num_larger = sum(s > current for s in strangeness)

    # Count the number of points with strangeness equal to the current value.
    num_equal = sum(s == current for s in strangeness)

    # Compute the p-value using the formula:
    # p = ( (# of points with strangeness > current) + theta * (# with strangeness == current) ) / total number of points.
    pvalue = (num_larger + theta * num_equal) / len(strangeness)

    return pvalue
