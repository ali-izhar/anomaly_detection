# src/changepoint/martingale.py

import random
import numpy as np
import logging
from typing import List, Dict, Any, Tuple

from ..graph.features import strangeness_point

logger = logging.getLogger(__name__)


def get_pvalue(strangeness: List[float]) -> float:
    """Compute nonparametric p-value using empirical distribution of strangeness measures.
    P(theta) = (count of alpha_i > alpha_n + theta * count of alpha_i = alpha_n) / n, where:
    - alpha_i are strangeness values.
    - alpha_n is current strangeness.
    - theta is a random number uniformly distributed between 0 and 1 for tie-breaking.
    - count() denotes the number of occurrences.

    Args:
        strangeness: Historical sequence of strangeness values [alpha_1, ..., alpha_n]

    Returns:
        p-value in [0,1] for current observation

    Raises:
        ValueError: If strangeness sequence is empty
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


def compute_martingale(
    data: List[Any],
    threshold: float,
    epsilon: float,
    detect: bool = True,
) -> Dict[str, Any]:
    """Compute power martingale sequence for online change detection.
    Implements power martingale: M_n = product of (epsilon * p_i^(epsilon - 1)) for i from 1 to n
    where p_i are p-values and epsilon is a sensitivity parameter in (0,1).
    Change is detected when M_n exceeds the threshold.

    Args:
        data: Sequential observations to monitor
        threshold: Detection threshold tau > 0
        epsilon: Sensitivity epsilon in (0,1), smaller values increase sensitivity
        detect: Whether to reset after detection

    Returns:
        Dictionary containing:
        - change_detected_instant: List of detection times
        - pvalues: Sequence of p-values
        - strangeness: Sequence of strangeness values
        - martingales: Sequence of martingale values
    """
    if not 0 < epsilon < 1:
        logger.error(f"Invalid epsilon value: {epsilon}")
        raise ValueError("Epsilon must be in (0,1)")
    if threshold <= 0:
        logger.error(f"Invalid threshold value: {threshold}")
        raise ValueError("Threshold must be positive")

    logger.info(
        f"Starting martingale computation with epsilon={epsilon}, threshold={threshold}"
    )

    pvalues: List[float] = []
    change_points: List[int] = []
    martingale = [1]  # M_0 = 1
    saved_martingales = [1]
    window: List[Any] = []
    saved_strangeness: List[float] = []

    try:
        for i, point in enumerate(data):
            # Compute strangeness and p-value
            strangeness = strangeness_point(window + [point]) if window else [0]
            saved_strangeness.append(strangeness[-1])

            pvalue = get_pvalue(strangeness)
            pvalues.append(pvalue)

            # Update martingale
            new_martingale = martingale[-1] * epsilon * (pvalue ** (epsilon - 1))
            martingale.append(new_martingale)
            saved_martingales.append(new_martingale)

            logger.debug(f"Time {i}: martingale={new_martingale}, p-value={pvalue}")

            # Check for change point
            if detect and new_martingale > threshold:
                logger.info(
                    f"Change point detected at time {i} (martingale={new_martingale})"
                )
                change_points.append(i)
                window = []
                martingale[-1] = 1
            else:
                window.append(point)

        logger.info(
            f"Martingale computation complete. Found {len(change_points)} change points"
        )
        return {
            "change_detected_instant": change_points,
            "pvalues": pvalues,
            "strangeness": saved_strangeness,
            "martingales": np.array(saved_martingales[1:], dtype=object),
        }
    except Exception as e:
        logger.error(f"Martingale computation failed: {str(e)}")
        raise RuntimeError(f"Martingale computation failed: {str(e)}")


def multiview_martingale_test(
    data: List[List[Any]], threshold: float, epsilon: float
) -> Dict[str, Any]:
    """Compute multivariate martingale test combining evidence across features.
    For d features, computes M_total as the sum of M_jn for each feature j at time n,
    where M_jn is the martingale for feature j at time n. Detects a change when M_total exceeds the threshold.

    Args:
        data: List of d feature sequences [samples x features]
        threshold: Detection threshold tau > 0
        epsilon: Sensitivity epsilon in (0,1)

    Returns:
        Dictionary containing:
        - change_detected_instant: Detection times
        - pvalues: P-values per feature
        - strangeness: Strangeness values per feature
        - martingale_sum: Combined martingale sequence
    """
    if not data or not data[0]:
        logger.error("Empty data sequence provided")
        raise ValueError("Empty data sequence")
    if not 0 < epsilon < 1:
        logger.error(f"Invalid epsilon value: {epsilon}")
        raise ValueError("Epsilon must be in (0,1)")
    if threshold <= 0:
        logger.error(f"Invalid threshold value: {threshold}")
        raise ValueError("Threshold must be positive")

    logger.info(f"Starting multiview martingale test with {len(data)} features")

    num_features = len(data)
    num_samples = len(data[0])
    logger.debug(f"Data dimensions: {num_samples} samples x {num_features} features")

    pvalues = [[] for _ in range(num_features)]
    change_points: List[int] = []
    martingales = [[1] for _ in range(num_features)]
    saved_martingales = [[1] for _ in range(num_features)]
    saved_martingale_sum = [num_features]  # Initial sum M_total = d * M_0 = d * 1
    windows = [[] for _ in range(num_features)]
    saved_strangeness = [[] for _ in range(num_features)]

    try:
        for i in range(num_samples):
            results = process_instant(
                [data[j][i] for j in range(num_features)], windows, martingales, epsilon
            )

            new_martingales, new_pvalues, new_strangeness, martingale_sum, windows = (
                results
            )

            logger.debug(f"Time {i}: martingale_sum={martingale_sum}")

            # Update tracking variables
            for j in range(num_features):
                martingales[j].append(new_martingales[j])
                saved_martingales[j].append(new_martingales[j])
                pvalues[j].append(new_pvalues[j])
                saved_strangeness[j].append(new_strangeness[j])

            saved_martingale_sum.append(martingale_sum)

            if martingale_sum > threshold:
                logger.info(
                    f"Change point detected at time {i} (martingale_sum={martingale_sum})"
                )
                change_points.append(i)
                windows = [[] for _ in range(num_features)]
                for j in range(num_features):
                    martingales[j][-1] = 1
                    windows[j].append(data[j][i])

        logger.info(
            f"Multiview test complete. Found {len(change_points)} change points"
        )
        return {
            "change_detected_instant": change_points,
            "pvalues": pvalues,
            "strangeness": saved_strangeness,
            "martingale_sum": np.array(saved_martingale_sum[1:], dtype=object),
        }
    except Exception as e:
        logger.error(f"Multiview martingale computation failed: {str(e)}")
        raise RuntimeError(f"Multiview martingale computation failed: {str(e)}")


def process_instant(
    data_points: List[Any],
    windows: List[List[Any]],
    martingales: List[List[float]],
    epsilon: float,
) -> Tuple[List[float], List[float], List[float], float, List[List[Any]]]:
    """Process a single time instant for multiview martingale computation.
    For each feature j, computes:
    1. Strangeness alpha_jn
    2. P-value p_jn
    3. Martingale update M_jn = M_j(n-1) * epsilon * p_jn^(epsilon - 1)

    Args:
        data_points: Current observations across features
        windows: Historical windows per feature
        martingales: Current martingale values per feature
        epsilon: Sensitivity parameter

    Returns:
        Tuple of (new_martingales, pvalues, strangeness, martingale_sum, updated_windows)
    """
    new_martingales, pvalues, strangeness_values = [], [], []

    try:
        for j, point in enumerate(data_points):
            window = windows[j]

            # Compute strangeness and p-value
            strangeness = strangeness_point(window + [point]) if window else [0]
            pvalue = get_pvalue(strangeness)

            # Update martingale
            new_martingale = martingales[j][-1] * epsilon * (pvalue ** (epsilon - 1))

            window.append(point)
            new_martingales.append(new_martingale)
            pvalues.append(pvalue)
            strangeness_values.append(strangeness[-1])

            logger.debug(f"Feature {j}: martingale={new_martingale}, p-value={pvalue}")

        martingale_sum = sum(new_martingales)
        return new_martingales, pvalues, strangeness_values, martingale_sum, windows
    except Exception as e:
        logger.error(f"Failed to process instant: {str(e)}")
        raise RuntimeError(f"Failed to process instant: {str(e)}")
