# src/changepoint/strangeness.py

"""Strangeness computation for change point detection.

This module implements strangeness (nonconformity) measures for change point detection
using conformal prediction. The strangeness of a point is computed as its minimum
distance to cluster centers, providing a measure of how "strange" or "nonconforming"
the point is with respect to the existing data.

Mathematical Framework:
---------------------
For a sequence of observations {x₁, ..., xₙ}, the strangeness is computed as:

1. Clustering Step:
   - Fit k clusters to the data using KMeans or MiniBatchKMeans
   - Obtain cluster centers {c₁, ..., cₖ}

2. Distance Computation:
   For each point xᵢ:
   - Compute distances d(xᵢ, cⱼ) to all cluster centers
   - Strangeness α(xᵢ) = min_{j=1..k} d(xᵢ, cⱼ)

3. P-value Computation:
   For a new point xₙ:
   p(xₙ) = (#{i: α(xᵢ) ≥ α(xₙ)} + θU(0,1)) / n
   where θ ~ U(0,1) is used for tie-breaking

Properties:
----------
1. Strangeness values are non-negative
2. Lower values indicate points closer to cluster centers
3. Higher values suggest potential anomalies or changes
4. Scale-invariant when using appropriate distance metrics

References:
----------
[1] Vovk et al. (2005) "Algorithmic Learning in a Random World"
[2] Volkhonskiy et al. (2017) "Inductive Conformal Martingales for Change-Point Detection"
"""

from dataclasses import dataclass
from typing import List, Optional, Union, TypeVar, final
import logging
import numpy as np
import random
from sklearn.cluster import KMeans, MiniBatchKMeans

from .distance import DistanceConfig, compute_cluster_distances

logger = logging.getLogger(__name__)

# Type definitions for data sequence elements (can be a float, list of floats, or numpy array)
T = TypeVar("T", bound=Union[float, List[float], np.ndarray])
DataSequence = List[T]


@dataclass(frozen=True)
class StrangenessConfig:
    """Configuration for strangeness computation.

    Attributes:
        n_clusters: Number of clusters for strangeness computation
        batch_size: Batch size for MiniBatchKMeans (if None, use standard KMeans)
        random_state: Random seed for reproducibility
        distance_config: Configuration for distance computation
    """

    n_clusters: int = 1
    batch_size: Optional[int] = None
    random_state: Optional[int] = None
    distance_config: Optional[DistanceConfig] = None

    def __post_init__(self):
        """Validate configuration parameters."""
        # Ensure at least one cluster is requested.
        if self.n_clusters < 1:
            raise ValueError(f"n_clusters must be positive, got {self.n_clusters}")
        # If a batch size is specified, it must be positive.
        if self.batch_size is not None and self.batch_size < 1:
            raise ValueError(
                f"batch_size must be positive if specified, got {self.batch_size}"
            )


@final
def strangeness_point(
    data: Union[DataSequence, np.ndarray],
    config: Optional[StrangenessConfig] = None,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Compute strangeness values for each point in the sequence.

    The strangeness of a point is defined as its minimum distance to any cluster center,
    providing a measure of how "nonconforming" the point is with respect to the data.

    Args:
        data: Input sequence of shape (n_samples, n_features) or (n_samples, time, n_features)
        config: Configuration for strangeness computation
        random_state: Optional random seed that overrides config's random_state

    Returns:
        Array of shape (n_samples,) containing strangeness values

    Raises:
        ValueError: If input validation fails
        RuntimeError: If computation fails
    """
    # Use provided configuration or fall back to defaults.
    config = config or StrangenessConfig()

    # Override config's random_state if one is provided
    if random_state is not None:
        # Create a copy of the config with the new random_state
        config = StrangenessConfig(
            n_clusters=config.n_clusters,
            batch_size=config.batch_size,
            random_state=random_state,  # Use the provided random_state
            distance_config=config.distance_config,
        )

    try:
        # Handle empty input cases first
        if isinstance(data, (list, np.ndarray)) and len(data) == 0:
            raise ValueError("Empty data sequence")

        # Convert input to a numpy array for uniform processing.
        data_array = np.asarray(data)

        # Validate dimensions before any reshaping
        if data_array.ndim not in [2, 3]:
            raise ValueError(f"Expected 2D or 3D array, got {data_array.ndim}D")

        # For 2D arrays, check feature dimension first
        if data_array.ndim == 2:
            if data_array.shape[1] == 0:
                raise ValueError("Feature dimension cannot be zero")

        # For 3D data, check feature dimension
        if data_array.ndim == 3:
            n_samples, time_steps, n_features = data_array.shape
            if n_features == 0:
                raise ValueError("Feature dimension cannot be zero")

        # Check for empty array after dimension validation
        if data_array.size == 0:
            raise ValueError("Empty data sequence")

        # For 3D data, merge dimensions
        if data_array.ndim == 3:
            data_array = data_array.reshape(n_samples, time_steps * n_features)

        # Log the original shape of the input data.
        logger.debug(f"Input data shape: {data_array.shape}")
        if data_array.ndim == 3:
            logger.debug(f"Reshaped to: {data_array.shape}")

        # Retrieve number of samples and features.
        n_samples, n_features = data_array.shape
        # Ensure there are at least as many samples as clusters.
        if n_samples < config.n_clusters:
            raise ValueError(
                f"Number of samples ({n_samples}) must be >= number of clusters ({config.n_clusters})"
            )

        # Initialize the clustering model (KMeans or MiniBatchKMeans) based on data size and config.
        model = _init_clustering_model(n_samples, config)

        # Fit the clustering model to the data.
        try:
            model.fit(data_array)
        except Exception as e:
            logger.error(f"Clustering failed: {str(e)}")
            raise RuntimeError(f"Clustering failed: {str(e)}")

        # Compute distances from each data point to the cluster centers.
        distances = compute_cluster_distances(
            data_array,
            model,
            config=config.distance_config,
        )

        # For each point, the strangeness is the minimum distance to any cluster center.
        strangeness_scores = distances.min(axis=1)

        # Ensure the output is a 1D array with one strangeness value per sample.
        if strangeness_scores.shape != (n_samples,):
            raise ValueError(
                f"Invalid output shape: {strangeness_scores.shape}, expected ({n_samples},)"
            )

        # Log summary statistics of the computed strangeness values.
        logger.debug(
            f"Computed strangeness values: min={strangeness_scores.min():.4f}, "
            f"max={strangeness_scores.max():.4f}, mean={strangeness_scores.mean():.4f}"
        )
        return strangeness_scores

    except (ValueError, TypeError) as e:
        # Don't wrap ValueError and TypeError in RuntimeError
        raise
    except Exception as e:
        # Log and wrap other errors in RuntimeError
        logger.error(f"Strangeness computation failed: {str(e)}")
        raise RuntimeError(f"Strangeness computation failed: {str(e)}")


@final
def get_pvalue(
    strangeness: Union[List[float], np.ndarray],
    random_state: Optional[int] = None,
) -> float:
    """Compute conformal p-value for the last strangeness value.

    Uses Vovk's tie-breaking rule:
    p = (#{i: α(xᵢ) > α(xₙ)} + θ#{i: α(xᵢ) = α(xₙ)}) / n
    where θ ~ U(0,1) and xₙ is the new point.

    Args:
        strangeness: Sequence of strangeness values, last element is the new point
        random_state: Random seed for reproducibility

    Returns:
        Conformal p-value in [0,1]

    Raises:
        TypeError: If input type is invalid
        ValueError: If input is empty
    """
    try:
        # Ensure the input is a list or numpy array.
        if not isinstance(strangeness, (list, np.ndarray)):
            raise TypeError("strangeness must be a list or numpy array")
        if len(strangeness) == 0:
            raise ValueError("Empty strangeness sequence")

        # Set the random seed if provided to guarantee reproducibility.
        if random_state is not None:
            random.seed(random_state)

        # Convert the sequence to a numpy array for vectorized operations.
        s_array = np.asarray(strangeness)
        # The current (new) point is assumed to be the last element.
        current = s_array[-1]

        # Count how many points have strictly greater strangeness than the current point.
        num_larger = np.sum(s_array > current)
        # Count how many points have equal strangeness (to be broken by randomness).
        num_equal = np.sum(s_array == current)

        # Generate a random number in [0,1] for tie-breaking.
        theta = random.random()

        # Compute the conformal p-value based on Vovk's tie-breaking rule.
        pvalue = (num_larger + theta * num_equal) / len(s_array)

        # Log the details of the computation.
        logger.debug(
            f"Computed p-value: {pvalue:.4f} "
            f"(#larger={num_larger}, #equal={num_equal})"
        )
        return pvalue

    except (TypeError, ValueError) as e:
        # Don't wrap TypeError and ValueError in RuntimeError
        raise
    except Exception as e:
        # Log and wrap other errors in RuntimeError
        logger.error(f"P-value computation failed: {str(e)}")
        raise RuntimeError(f"P-value computation failed: {str(e)}")


def _init_clustering_model(
    n_samples: int,
    config: StrangenessConfig,
) -> Union[KMeans, MiniBatchKMeans]:
    """Initialize appropriate clustering model based on data size and configuration.

    Args:
        n_samples: Number of samples in the data
        config: Strangeness computation configuration

    Returns:
        Initialized clustering model
    """
    # If a batch size is provided and the number of samples exceeds it,
    # use MiniBatchKMeans for efficiency on large datasets.
    if config.batch_size is not None and n_samples > config.batch_size:
        logger.debug(f"Using MiniBatchKMeans with batch_size={config.batch_size}")
        return MiniBatchKMeans(
            n_clusters=config.n_clusters,
            batch_size=config.batch_size,
            random_state=config.random_state,
        )
    else:
        # Otherwise, use the standard KMeans algorithm.
        logger.debug("Using standard KMeans")
        return KMeans(
            n_clusters=config.n_clusters,
            n_init="auto",
            random_state=config.random_state,
        )
