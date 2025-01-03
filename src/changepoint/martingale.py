# src/changepoint/martingale.py

import numpy as np
import logging
from typing import List, Dict, Any, Tuple, Optional

from .strangeness import strangeness_point, get_pvalue

logger = logging.getLogger(__name__)


def compute_martingale(
    data: List[Any],
    threshold: float,
    epsilon: float,
    reset: bool = True,
    window_size: Optional[int] = None,
) -> Dict[str, Any]:
    """Compute a *power martingale* for online change detection over a univariate data stream
    using conformal p-values and a chosen strangeness measure.

    1. Each new data point is assigned a `strangeness` value based on
       how 'unusual' it appears relative to the historical 'window'.
    2. We convert the strangeness values (alpha_1, ..., alpha_n) into a *p-value* via `get_pvalue()`.
    3. We update the power martingale as:

        M_n = M_(n-1) * epsilon * (p_n)^(epsilon - 1),

       where `epsilon` in (0,1) is a *sensitivity parameter*:
       - Smaller epsilon => more sensitive to small p-values (strong anomalies).

    4. A change is declared if M_n > threshold. Optionally, the algorithm can `reset`
       by clearing the window and resetting M_n to 1 after detection.

    Parameters
    ----------
    data : List[Any]
        Sequential observations to monitor (e.g., numeric values).
    threshold : float
        Detection threshold (tau > 0). If M_n > threshold, a change is reported.
    epsilon : float
        Sensitivity parameter in (0,1).
        - Example: epsilon=0.6 (somewhat sensitive); epsilon=0.01 (highly sensitive).
    reset : bool, optional
        Whether to reset the martingale and history window after a detection, by default True.
    window_size : int, optional
        Maximum number of recent observations to keep in the 'window' for
        strangeness computation. If None (default), keeps all.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing:
        - **change_detected_instant** (List[int]): Indices where a change was detected.
        - **pvalues** (List[float]): Sequence of p-values for each observation.
        - **strangeness** (List[float]): Computed strangeness for each observation.
        - **martingales** (np.ndarray): Final sequence of martingale values (size = number of samples).

    Raises
    ------
    ValueError
        If epsilon is not in (0,1) or if threshold <= 0.
    """
    if not 0 < epsilon < 1:
        logger.error(f"Invalid epsilon value: {epsilon}")
        raise ValueError("Epsilon must be in (0,1)")

    if threshold <= 0:
        logger.error(f"Invalid threshold value: {threshold}")
        raise ValueError("Threshold must be positive")

    logger.info(
        f"Starting martingale computation with epsilon={epsilon}, threshold={threshold}, window_size={window_size}"
    )

    pvalues: List[float] = []
    change_points: List[int] = []
    martingale = [1]  # M_0 = 1
    saved_martingales = [1]
    window: List[Any] = []
    saved_strangeness: List[float] = []

    try:
        for i, point in enumerate(data):
            # Keep only the last window_size points if window_size is specified
            if window_size and len(window) >= window_size:
                window = window[-window_size:]

            # ----- 1. Compute strangeness and p-value -----
            # If the window is empty, treat strangeness as [0] (i.e., no history yet)
            strg = strangeness_point(window + [point]) if window else [0]
            # The new strangeness is the last element in the list
            saved_strangeness.append(strg[-1])

            pvalue = get_pvalue(strg)
            pvalues.append(pvalue)

            # ----- 2. Update martingale -----
            new_martingale = martingale[-1] * epsilon * (pvalue ** (epsilon - 1))
            martingale.append(new_martingale)
            saved_martingales.append(new_martingale)

            logger.debug(f"Time {i}: martingale={new_martingale}, p-value={pvalue}")

            # ----- 3. Check for change point -----
            if reset and new_martingale > threshold:
                logger.info(
                    f"Change point detected at time {i} (martingale={new_martingale})"
                )
                change_points.append(i)
                window = []
                martingale[-1] = 1  # Reset the martingale to 1
            else:
                # If no change detected, keep the point in the window
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
    data: List[List[Any]],
    threshold: float,
    epsilon: float,
    batch_size: int = 1000,
    window_size: Optional[int] = None,
    early_stop_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """Compute a *multivariate (multiview) martingale test*, combining evidence across multiple features.

    - Suppose we have d features (views), each generating its own power martingale M_j(n).
    - We define a combined statistic: M_total(n) = sum_{j=1 to d} M_j(n).
    - A change is detected when M_total(n) > threshold.
    - Each feature j has its own 'window' of historical data and its own strangeness measure,
      allowing more fine-grained detection of anomalies in different subspaces.

    Parameters
    ----------
    data : List[List[Any]]
        A list of length d (number of features), where each element is a list of sequential data
        for that feature. So data[j] is the time series for feature j.
    threshold : float
        Detection threshold for the combined martingale sum.
    epsilon : float
        Sensitivity parameter for all features in (0,1).
    batch_size : int, optional
        Number of samples to process at once for efficiency. Default is 1000.
    window_size : int, optional
        Maximum window size for memory efficiency in each feature's history.
    early_stop_threshold : float, optional
        If specified, stops processing early if the combined martingale sum exceeds this value.

    Returns
    -------
    Dict[str, Any]
        - **change_detected_instant** (List[int]): The time indices where a change is detected.
        - **pvalues** (List[List[float]]): p-value sequences for each feature.
        - **strangeness** (List[List[float]]): strangeness sequences for each feature.
        - **martingale_sum** (np.ndarray): The combined martingale sum over time.

    Raises
    ------
    ValueError
        If data is empty, epsilon is invalid, or threshold <= 0.
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

    logger.info(
        f"Starting multiview martingale test with {len(data)} features, "
        f"batch_size={batch_size}, window_size={window_size}"
    )

    num_features = len(data)
    num_samples = len(data[0])
    logger.debug(f"Data dimensions: {num_samples} samples x {num_features} features")

    pvalues = [[] for _ in range(num_features)]
    change_points: List[int] = []
    martingales = [[1] for _ in range(num_features)]
    saved_martingales = [[1] for _ in range(num_features)]
    saved_martingale_sum = [num_features]  # At time 0, sum of M_j(0) = d * 1
    windows = [[] for _ in range(num_features)]
    saved_strangeness = [[] for _ in range(num_features)]

    try:
        # Process data in batches to handle large datasets
        for batch_start in range(0, num_samples, batch_size):
            batch_end = min(batch_start + batch_size, num_samples)
            logger.debug(f"Processing batch {batch_start}:{batch_end}")

            for i in range(batch_start, batch_end):
                # Limit window sizes if specified
                if window_size:
                    windows = [
                        w[-window_size:] if len(w) >= window_size else w
                        for w in windows
                    ]

                results = process_instant(
                    [data[j][i] for j in range(num_features)],
                    windows,
                    martingales,
                    epsilon,
                    early_stop_threshold,
                )

                (
                    new_martingales,
                    new_pvalues,
                    new_strangeness,
                    martingale_sum,
                    windows,
                ) = results

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
                    # Reset windows and martingales for next iteration
                    windows = [[] for _ in range(num_features)]
                    for j in range(num_features):
                        martingales[j][-1] = 1
                        windows[j].append(data[j][i])

                # Early stopping condition
                if early_stop_threshold and martingale_sum > early_stop_threshold:
                    logger.info(
                        f"Early stopping at time {i} (martingale_sum={martingale_sum})"
                    )
                    break

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
    early_stop_threshold: Optional[float] = None,
) -> Tuple[List[float], List[float], List[float], float, List[List[Any]]]:
    """Process a single time instant for all features in a multiview scenario.

    1. For each feature j:
       - Gather its historical 'window' (recent data points).
       - Compute the new point's strangeness (and the entire updated sequence's strangeness).
       - Compute the conformal p-value for the current strangeness.
       - Update the martingale: M_j(n) = M_j(n-1) * epsilon * p_j(n)^(epsilon - 1).
    2. Sum all features' martingales to get M_total(n).
    3. Return updated windows, martingales, p-values, strangeness, and the sum.

    Parameters
    ----------
    data_points : List[Any]
        The observations from all features at the current time instant.
        E.g., if there are d features, data_points = [value_feature_0, ..., value_feature_(d-1)].
    windows : List[List[Any]]
        The historical windows for each feature (list of lists).
    martingales : List[List[float]]
        Current martingale values per feature (each is a list).
    epsilon : float
        Power martingale sensitivity parameter, epsilon in (0,1).
    early_stop_threshold : float, optional
        If specified, stops processing early if any single-feature martingale exceeds this.

    Returns
    -------
    Tuple[List[float], List[float], List[float], float, List[List[Any]]]
        - new_martingales: The updated martingale values for each feature (size = d).
        - pvalues: The new p-values for each feature (size = d).
        - strangeness: The new strangeness values for each feature (size = d).
        - martingale_sum: The sum of new_martingales across features.
        - updated_windows: The updated windows for each feature.
    """
    new_martingales, pvalues, strangeness_values = [], [], []

    try:
        for j, point in enumerate(data_points):
            window = windows[j]

            # If no history, we treat the initial strangeness as [0].
            strg = strangeness_point(window + [point]) if window else [0]
            pvalue = get_pvalue(strg)

            new_martingale = martingales[j][-1] * epsilon * (pvalue ** (epsilon - 1))

            # Early stop check if desired
            if early_stop_threshold and new_martingale > early_stop_threshold:
                logger.debug(
                    f"Early stopping for feature {j}: martingale={new_martingale}"
                )
                # Return partial results with infinite martingale_sum to trigger detection
                return (
                    new_martingales,
                    pvalues,
                    strangeness_values,
                    float("inf"),
                    windows,
                )

            window.append(point)
            new_martingales.append(new_martingale)
            pvalues.append(pvalue)
            strangeness_values.append(strg[-1])

            logger.debug(f"Feature {j}: martingale={new_martingale}, p-value={pvalue}")

        martingale_sum = sum(new_martingales)
        return new_martingales, pvalues, strangeness_values, martingale_sum, windows
    except Exception as e:
        logger.error(f"Failed to process instant: {str(e)}")
        raise RuntimeError(f"Failed to process instant: {str(e)}")
