# src/changepoint/strangeness.py

import logging
import numpy as np
import random
from typing import List, Optional
from sklearn.cluster import KMeans, MiniBatchKMeans

logger = logging.getLogger(__name__)


def strangeness_point(
    data: List[List[float]],
    n_clusters: int = 1,
    random_state: Optional[int] = None,
    batch_size: Optional[int] = None,
) -> np.ndarray:
    """Compute a cluster-based strangeness score for each sample in `data`
    using (MiniBatch)KMeans. The 'strangeness' is defined here as the
    minimum distance from each sample to any of the cluster centers.

    1. We treat `data` as (N x features) or (a x b x features). If it's 3D, we flatten
       it to a 2D array of shape (N, features).
    2. We fit KMeans or MiniBatchKMeans on the data (using `n_clusters`).
    3. For each sample, we compute the distances to the cluster centers. The smallest
       distance is taken as that sample's 'strangeness'.
    4. Returning these strangeness values as a 1D array of length N.

    This approach is a simplistic but effective way to measure how "unusual" a point is,
    in a clustering sense. Points closer to their nearest cluster center have lower strangeness;
    points far from all centers have higher strangeness.

    Parameters
    ----------
    data : List[List[float]]
        A list-of-lists representing the dataset. Can be:
          - 2D: shape (N, num_features)
          - 3D: shape (a, b, num_features) which will be flattened to (N, num_features).
    n_clusters : int, default=1
        Number of clusters to use in KMeans or MiniBatchKMeans.
        - For `n_clusters=1`, we effectively measure distance from the single centroid
          to each sample.
    random_state : int, optional
        Random seed for the clustering algorithm, controlling reproducibility.
    batch_size : int, optional
        If specified and the number of samples N > `batch_size`,
        we use MiniBatchKMeans for efficiency.

    Returns
    -------
    np.ndarray
        A 1D array of length N, where each entry is the strangeness value
        for the corresponding sample.

    Raises
    ------
    ValueError
        If `data` is empty.
    """
    # 1. Basic checks and conversion to array
    data_array = np.array(data)
    if data_array.size == 0:
        raise ValueError("Empty data sequence.")

    # 2. If data is 3D, flatten to 2D => (N, features)
    if data_array.ndim == 3:
        data_array = data_array.reshape(-1, data_array.shape[-1])

    # 3. Decide which KMeans variant to use
    try:
        if batch_size is not None and data_array.shape[0] > batch_size:
            model = MiniBatchKMeans(
                n_clusters=n_clusters, batch_size=batch_size, random_state=random_state
            )
        else:
            model = KMeans(
                n_clusters=n_clusters, n_init="auto", random_state=random_state
            )

        # 4. Fit and compute distances
        distances = model.fit_transform(data_array)  # shape => (N, n_clusters)

        # 5. Return min distance across cluster centers => strangeness
        strangeness_scores = distances.min(axis=1)  # shape => (N,)
        return strangeness_scores

    except Exception as e:
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
