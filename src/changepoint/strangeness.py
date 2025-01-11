# src/changepoint/strangeness.py

import logging
import numpy as np
import random
from typing import List, Optional, Union, Any
from sklearn.cluster import KMeans, MiniBatchKMeans

logger = logging.getLogger(__name__)


def strangeness_point(
    data: Union[List[Any], np.ndarray],
    n_clusters: int = 1,
    random_state: Optional[int] = 42,
    batch_size: Optional[int] = None,
    return_all_distances: bool = True,
) -> np.ndarray:
    """Compute distance-based scores for each sample relative to KMeans (or MiniBatchKMeans) cluster centers.

    By default, returns the full (N, n_clusters) distance matrix.
    If `return_all_distances=False`, returns the minimum distance to any center as a 1D array (N,).

    Parameters
    ----------
    data : Union[List[Any], np.ndarray]
        Input data to compute strangeness for.
        Can be 2D (N x features) or 3D (a x b x features). If 3D, it is flattened to 2D.
    n_clusters : int, default=1
        Number of clusters to use in KMeans or MiniBatchKMeans.
    random_state : int, optional
        Random seed for reproducibility.
    batch_size : int, optional
        If provided and data size > batch_size, uses MiniBatchKMeans for efficiency.
    return_all_distances : bool, default=True
        If True, return the full distance matrix (N, n_clusters).
        If False, return the minimum distance per sample as a 1D array (N,).

    Returns
    -------
    np.ndarray
        - If `return_all_distances=False`, shape (N,).
        - If `return_all_distances=True`, shape (N, n_clusters).

    Raises
    ------
    ValueError
        If data is empty or has invalid shape.
    RuntimeError
        If cluster-fitting fails for some reason.
    """
    # 1. Validate data
    if data is None or len(data) == 0:
        logger.error("Empty data sequence")
        raise ValueError("Empty data sequence")

    try:
        data_array = np.array(data)
        if data_array.size == 0:
            logger.error("Data array has zero size after np.array conversion")
            raise ValueError("Empty data sequence")

        # 2. Flatten if 3D
        if data_array.ndim == 3:
            data_array = data_array.reshape(-1, data_array.shape[-1])

        # 3. Decide KMeans vs MiniBatchKMeans
        N = data_array.shape[0]
        if batch_size is not None and N > batch_size:
            model = MiniBatchKMeans(
                n_clusters=n_clusters, batch_size=batch_size, random_state=random_state
            )
        else:
            model = KMeans(
                n_clusters=n_clusters, n_init="auto", random_state=random_state
            )

        logger.debug(f"Fitting model with {n_clusters} cluster(s)")
        # 4. Fit + Transform => distances shape (N, n_clusters)
        distances = model.fit_transform(data_array)

        # 5. Return either all distances or the minimum
        if return_all_distances:
            logger.debug("Returning full (N, n_clusters) distance matrix")
            return distances
        else:
            # If the docstring says “strangeness = min distance to any center,” do .min(axis=1)
            strangeness_scores = distances.min(axis=1)
            logger.debug("Returning 1D array of minimum distances (strangeness)")
            return strangeness_scores

    except Exception as e:
        logger.error(f"Strangeness computation failed: {str(e)}")
        raise RuntimeError(f"Strangeness computation failed: {str(e)}")


def get_pvalue(strangeness: List[float]) -> float:
    """Compute a nonparametric conformal p-value for the latest observation based
    on the empirical distribution of historical strangeness measures.

    Let alpha = (alpha_1, alpha_2, ..., alpha_n) be the sequence of
    strangeness values, where alpha_n is the most recent (current) strangeness.
    We draw a random theta ~ Uniform(0,1) for tie-breaking and define:

        num_larger = |{ alpha_i : alpha_i > alpha_n }|
        num_equal  = |{ alpha_i : alpha_i = alpha_n }|

    Then the p-value is computed as:

        p = ( num_larger + theta * num_equal ) / n

    This is a standard approach in conformal prediction to handle
    tie-breaking in a principled way.

    Parameters
    ----------
    strangeness : List[float]
        A list of strangeness values for all data points seen so far,
        including the most recent point's strangeness (alpha_n) as the last element.
        - The 'history' portion is alpha_1 to alpha_(n-1).
        - The 'current' portion is alpha_n at the end.

    Returns
    -------
    float
        The computed p-value in the interval [0,1].

    Raises
    ------
    ValueError
        If the `strangeness` list is empty.
    """
    if isinstance(strangeness, np.ndarray):
        if strangeness.size == 0:
            logger.error("Empty numpy array provided for strangeness computation")
            raise ValueError("Strangeness sequence cannot be empty")
    else:
        if not strangeness:
            logger.error("Empty list provided for strangeness computation")
            raise ValueError("Strangeness sequence cannot be empty")

    try:
        theta = random.uniform(0, 1)
        current = strangeness[-1]

        num_larger = sum(s > current for s in strangeness)
        num_equal = sum(s == current for s in strangeness)

        pvalue = (num_larger + theta * num_equal) / len(strangeness)
        logger.debug(f"Computed p-value: {pvalue} (theta={theta})")
        return pvalue
    except Exception as e:
        logger.error(f"Failed to compute p-value: {str(e)}")
        raise ValueError(f"P-value computation failed: {str(e)}")
