# src/changepoint/martingale.py

"""Compute a power martingale for online change detection over a univariate data stream
using conformal p-values and a chosen strangeness measure."""

import numpy as np
import logging
from typing import List, Dict, Any, Optional

from .strangeness import strangeness_point, get_pvalue

logger = logging.getLogger(__name__)


def compute_martingale(
    data: List[Any],
    predicted_data: List[np.ndarray],
    threshold: float,
    epsilon: float,
    reset: bool = True,
    window_size: Optional[int] = None,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Compute a *power martingale* for online change detection over a univariate data stream
    using conformal p-values and a chosen strangeness measure.

    Steps:
    1. Each new data point is assigned a `strangeness` value relative to a rolling 'window' of past data.
    2. We convert the strangeness values (alpha_1, ..., alpha_n) into a *p-value* p_n via `get_pvalue()`.
    3. Update the power martingale:
       M_n = M_(n-1) * epsilon * (p_n)^(epsilon - 1).
    4. If M_n > threshold, a change is reported. If `reset=True`, the algorithm clears the window
       and resets M_n to 1 for the next observation.

    Parameters
    ----------
    data : List[Any]
        Sequential observations to monitor (e.g., numeric values).
    predicted_data : List[np.ndarray]
        List of predicted feature vectors for future timesteps.
        Each prediction should have same shape as data rows.
    threshold : float
        Detection threshold (>0). If M_n > threshold, a change is reported.
    epsilon : float

        Sensitivity parameter in (0,1).
        - Smaller epsilon => more sensitive to small p-values.
    reset : bool, optional
        Whether to reset the martingale and history window after a detection, by default True.
    window_size : int, optional
        Maximum number of recent observations to keep in the window.
        If None (default), keeps all historical data.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    Dict[str, Any]
        - change_detected_instant (List[int]): Indices where a change was detected.
        - pvalues (List[float]): Sequence of p-values.
        - strangeness (List[float]): Computed strangeness for each observation.
        - martingales (np.ndarray): Final sequence of martingale values.

    Raises
    ------
    ValueError
        If epsilon is not in (0,1) or threshold <= 0.
    """

    if not 0 < epsilon < 1:
        logger.error(f"Invalid epsilon value: {epsilon}")
        raise ValueError("Epsilon must be in (0,1)")

    if threshold <= 0:
        logger.error(f"Invalid threshold value: {threshold}")
        raise ValueError("Threshold must be positive")

    logger.info(
        f"Starting martingale computation with epsilon={epsilon}, "
        f"threshold={threshold}, window_size={window_size}"
    )

    pvalues: List[float] = []
    change_points: List[int] = []

    # Martingale array: M_0 = 1
    # We'll keep track in parallel: 'martingale' for running updates, 'saved_martingales' for final output
    martingale = [1.0]
    saved_martingales = [1.0]

    # Rolling window for the strangeness baseline
    window: List[Any] = []

    # Keep track of each observation's strangeness
    saved_strangeness: List[float] = []

    try:
        for i, point in enumerate(data):
            # 1. Maintain a rolling window if window_size is set
            if window_size and len(window) >= window_size:
                window = window[-window_size:]

            # 2. Compute strangeness for the new point (last item in window + [point])
            #    If the window is empty, we treat strangeness as 0 for the first point.
            if len(window) == 0:
                # No history yet => strangeness 0
                strangeness_vals = [0.0]
            else:
                # The function returns an array of shape (N,) – the min distance for each row
                # So we pass (window + [point]) to get the strangeness for all, especially the new point
                strangeness_vals = strangeness_point(
                    window + [point], random_state=random_state
                )
            current_strg = strangeness_vals[-1]
            saved_strangeness.append(current_strg)

            # 3. Convert the full sequence of strangeness (strangeness_vals) into a p-value
            #    focusing on the last element. We do get_pvalue(strangeness_vals).
            pvalue = get_pvalue(strangeness_vals, random_state=random_state)
            pvalues.append(pvalue)

            # 4. Update the martingale using the power-martingale formula
            #    M_n = M_{n-1} * epsilon * (p_n)^(epsilon - 1)
            prev_m = martingale[-1]
            new_m = prev_m * epsilon * (pvalue ** (epsilon - 1))

            martingale.append(new_m)
            saved_martingales.append(new_m)

            logger.debug(
                f"Time={i}, point={point}, p-value={pvalue:.4f}, M={new_m:.4f}"
            )

            # 5. Check for change point
            if reset and new_m > threshold:
                logger.info(
                    f"Change point detected at time {i}, M={new_m:.4f} > threshold={threshold}"
                )
                change_points.append(i)

                # (A) Clear the window
                window = []

                # (B) Reset the martingale *internally* to 1 for the *next iteration*,
                #     but keep the crossing value in saved_martingales so we know we exceeded threshold.
                saved_martingales[-1] = new_m  # store the crossing
                martingale[-1] = 1.0  # reset internal running value

            else:
                # If no detection, add the new point to the rolling window
                window.append(point)

        logger.info(
            f"Martingale computation complete. Found {len(change_points)} change points."
        )
        return {
            "change_detected_instant": change_points,
            "pvalues": pvalues,
            "strangeness": saved_strangeness,
            # omit the very first "1.0" if you only want as many martingale values as data points
            "martingales": np.array(saved_martingales[1:], dtype=float),
        }
    except Exception as e:
        logger.error(f"Martingale computation failed: {str(e)}")
        raise RuntimeError(f"Martingale computation failed: {str(e)}")


def multiview_martingale_test(
    data: List[List[Any]],
    predicted_data: List[List[np.ndarray]],
    threshold: float,
    epsilon: float,
    window_size: Optional[int] = None,
    early_stop_threshold: Optional[float] = None,
    batch_size: int = 1000,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Compute a multivariate (multiview) martingale test, combining evidence across multiple features.

    For d features, each feature j maintains its own power martingale M_j(n). We compute:
    M_total(n) = sum_{j=1 to d} M_j(n)
    M_avg(n) = M_total(n) / d
    A change is detected if M_total(n) > threshold.

    Parameters
    ----------
    data : List[List[Any]]
        data[j] is the entire time-series for feature j.
        Must have the same length across features => num_samples.
    predicted_data : List[List[np.ndarray]]
        List of predicted feature vectors for future timesteps.
        Each prediction should have same shape as data rows.
    threshold : float
        Detection threshold for the sum of martingales.
    epsilon : float
        Sensitivity parameter in (0,1) for all features.

    window_size : int, optional
        Rolling window size for each feature. If None, use all past data.
    early_stop_threshold : float, optional
        If specified, stop processing if the sum of martingales > this number.
    batch_size : int, optional
        Process data in chunks for large datasets. Default 1000.
    random_state : int, optional
        Seed for reproducibility.

    Returns
    -------
    Dict[str, Any]
        - change_detected_instant: Indices where M_total(n) first exceeded threshold.
        - pvalues: p-value sequences for each feature (list of lists).
        - strangeness: strangeness sequences for each feature.
        - martingale_sum: time series of M_total(n).
        - martingale_avg: time series of M_avg(n).
        - individual_martingales: individual martingale time series for each feature.
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

    num_features = len(data)
    num_samples = len(data[0])
    logger.info(
        f"Starting multiview martingale test with {num_features} features, "
        f"num_samples={num_samples}, threshold={threshold}, epsilon={epsilon}"
    )

    # Each feature j has:
    #   - window[j]: historical data points
    #   - martingales[j]: the running martingale series
    #   - pvalues[j]: p-value history
    #   - strangeness[j]: strangeness history
    windows = [[] for _ in range(num_features)]
    martingales = [[1.0] for _ in range(num_features)]
    pvalues = [[] for _ in range(num_features)]
    strangeness_vals = [[] for _ in range(num_features)]

    # We'll store both sum and average M_total(n) across time
    # At t=0, M_j(0)=1 => sum is d, avg is 1
    martingale_sum = [float(num_features)]
    martingale_avg = [1.0]
    change_points: List[int] = []

    # Store individual martingales for each feature
    individual_martingales = [[1.0] for _ in range(num_features)]

    # Helper loop to go through data in batches
    idx = 0
    while idx < num_samples:
        batch_end = min(idx + batch_size, num_samples)
        for i in range(idx, batch_end):
            new_martingales = []
            for j in range(num_features):
                # Limit window sizes if specified
                if window_size and len(windows[j]) >= window_size:
                    windows[j] = windows[j][-window_size:]

                # Compute strangeness for the new point on feature j
                if len(windows[j]) == 0:
                    # No history => strangeness is [0]
                    svals = [0.0]
                else:
                    svals = strangeness_point(
                        windows[j] + [data[j][i]], random_state=random_state
                    )

                current_strg = svals[-1]
                strangeness_vals[j].append(current_strg)

                # p-value
                pv = get_pvalue(svals, random_state=random_state)
                pvalues[j].append(pv)

                # power martingale update
                prev_m = martingales[j][-1]
                new_m = prev_m * epsilon * (pv ** (epsilon - 1))
                new_martingales.append(new_m)
                individual_martingales[j].append(new_m)  # Store individual martingale

            # sum and average across features
            total_m = sum(new_martingales)
            avg_m = total_m / num_features
            martingale_sum.append(total_m)
            martingale_avg.append(avg_m)

            # update each feature's memory
            for j in range(num_features):
                # keep the newly computed M_j in the martingale trace
                mj = new_martingales[j]
                martingales[j].append(mj)

            logger.debug(f"Time={i}, M_total={total_m:.4f}, M_avg={avg_m:.4f}")

            # detect a change
            if total_m > threshold:
                logger.info(
                    f"Change detected at time {i}, M_total={total_m:.4f} > threshold={threshold}"
                )
                change_points.append(i)

                # reset each feature's window but keep individual martingales
                for j in range(num_features):
                    windows[j] = []
                    martingales[j][-1] = 1.0  # reset for next iteration
                    # Don't reset individual martingales to preserve the detection signal
                    individual_martingales[j][-1] = new_martingales[
                        j
                    ]  # Keep the actual martingale value

            # add the new data to each feature's window
            for j in range(num_features):
                windows[j].append(data[j][i])

            # optional early stop
            if early_stop_threshold and total_m > early_stop_threshold:
                logger.info(f"Early stopping at time {i} with M_total={total_m:.4f}")
                break
        idx = batch_end

    return {
        "change_detected_instant": change_points,
        "pvalues": pvalues,
        "strangeness": strangeness_vals,
        # skip the initial values if you only want (num_samples) length
        "martingale_sum": np.array(martingale_sum[1:], dtype=float),
        "martingale_avg": np.array(martingale_avg[1:], dtype=float),
        "individual_martingales": [
            np.array(m[1:], dtype=float) for m in individual_martingales
        ],
    }
