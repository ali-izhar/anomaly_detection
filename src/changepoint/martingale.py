# src/changepoint/martingale.py

"""Martingale framework for online change detection using conformal p-values and a chosen strangeness measure.

- Traditional martingale: uses the current observation and previous history.
- Horizon martingale: uses the current observation and multiple predicted future states,
  along with the previous history.

Reset Strategy:
- Traditional martingale resets to 1.0 immediately after detecting a change.
- Horizon martingale only resets when traditional martingale confirms a change.
"""

from typing import List, Dict, Any, Optional, TypedDict, Union, Callable

import logging
import numpy as np

from .strangeness import strangeness_point, get_pvalue
from .betting import (
    BettingFunctionConfig,
    get_betting_function,
)

logger = logging.getLogger(__name__)


def compute_martingale(
    data: List[Any],
    predicted_data: List[np.ndarray],
    threshold: float,
    history_size: int,
    reset: bool = True,
    window_size: Optional[int] = None,
    random_state: Optional[int] = None,
    betting_func_config: BettingFunctionConfig = {
        "name": "power",
        "params": {"epsilon": 0.7},
    },
    distance_measure: str = "euclidean",
    distance_p: float = 2.0,
) -> Dict[str, Any]:
    """
    Compute a martingale for online change detection over a univariate data stream using conformal p-values
    and a chosen strangeness measure.

    Two martingale streams are computed:
      1. Traditional martingale: uses only the current observation with its history.
      2. Horizon martingale: uses the current observation plus all predicted future states (horizon)
         together with the history.

    Parameters
    ----------
    data : List[Any]
        Sequential observations to monitor.
    predicted_data : List[np.ndarray]
        List of predicted feature vectors for future timesteps. Each prediction should have the same shape as a data row.
    threshold : float
        Detection threshold (> 0). A change is reported if the martingale exceeds this value.
    history_size : int
        Minimum number of observations to accumulate before using predictions.
    reset : bool, optional
        Whether to reset the martingale and history window after a detection (default: True).
    window_size : int, optional
        Maximum number of recent observations to keep in the history window.
    random_state : int, optional
        Random seed for reproducibility.
    betting_func_config : BettingFunctionConfig, optional
        Configuration for the betting function.
    distance_measure : str, optional
        Distance metric to use for strangeness computation.
    distance_p : float, optional
        Order parameter for Minkowski distance.

    Returns
    -------
    Dict[str, Any]
        A dictionary with:
         - "traditional_change_points": List[int] indices where the traditional martingale detected a change.
         - "horizon_change_points": List[int] indices where the horizon martingale detected a change.
         - "traditional_martingales": np.ndarray of traditional martingale values over time.
         - "horizon_martingales": np.ndarray of horizon martingale values (if predictions are provided).
    """

    # Get the betting function with its configuration
    betting_function = get_betting_function(betting_func_config)

    logger.debug("Single-view Martingale Input Dimensions:")
    logger.debug(f"  Sequence length: {len(data)}")
    if predicted_data:
        logger.debug(f"  Number of predictions: {len(predicted_data)}")
        logger.debug(f"  Predictions per timestep: {len(predicted_data[0])}")
    logger.debug(f"  History size: {history_size}")
    logger.debug(f"  Window size: {window_size if window_size else 'None'}")
    logger.debug("-" * 50)

    # ------- Input Validation -------
    if threshold <= 0:
        logger.error(f"Invalid threshold value: {threshold}")
        raise ValueError("Threshold must be positive")

    logger.debug(
        f"Starting martingale computation with threshold={threshold}, window_size={window_size}"
    )

    # ------- Initialize Storage -------
    # For traditional martingale detections.
    traditional_change_points: List[int] = []
    horizon_change_points: List[int] = []  # For horizon martingale detections.

    # Initialize both martingale streams with initial value 1.0.
    traditional_martingale = [1.0]  # Running value for traditional updates.
    saved_traditional = [1.0]  # Saved sequence for output.
    horizon_martingale = [1.0]  # Running value for horizon updates.
    saved_horizon = [1.0]  # Saved horizon martingale sequence.

    # Rolling history window for strangeness computation.
    window: List[Any] = []

    # ------- Compute Martingale -------
    try:
        #  For each point in the data stream, compute the martingale.
        for i, point in enumerate(data):

            # Maintain the rolling window if window_size is specified.
            if window_size and len(window) >= window_size:
                window = window[-window_size:]

            # Compute strangeness for the current observation.
            if len(window) == 0:
                # If the window is empty, the strangeness is 0.0.
                s_vals = [0.0]
            else:
                s_vals = strangeness_point(
                    window + [point],
                    random_state=random_state,
                    distance_measure=distance_measure,
                    p=distance_p,
                )

            # Compute conformal p-value from the strangeness values.
            pvalue = get_pvalue(s_vals, random_state=random_state)

            # Retrieve the previous traditional martingale value.
            prev_trad = traditional_martingale[-1]

            # Update the traditional martingale using the betting function.
            new_trad = betting_function(prev_trad, pvalue)

            # Check if a change has been detected in the traditional martingale.
            detected_trad = False
            if reset and new_trad > threshold:
                logger.info(
                    f"Traditional martingale detected change at t={i}: {new_trad:.4f} > {threshold}"
                )
                detected_trad = True
                traditional_change_points.append(i)

            # --- Horizon Martingale Update ---
            new_horizon = None
            if predicted_data is not None and i >= history_size:
                # Use predictions only after accumulating enough history.
                pred_idx = i - history_size

                # Initialize the product factor for the horizon update.
                horizon_update_factor = 1.0

                # Loop over all horizon predictions at the current prediction index.
                for h in range(len(predicted_data[pred_idx])):
                    # Compute strangeness for the predicted state.
                    if len(window) == 0:
                        pred_s_vals = [0.0]
                    else:
                        # Wrap predicted value in a list to mimic an observation.
                        pred_s_vals = strangeness_point(
                            window + [[predicted_data[pred_idx][h]]],
                            random_state=random_state,
                            distance_measure=distance_measure,
                            p=distance_p,
                        )

                    # Compute the conformal p-value for the predicted state.
                    pred_pvalue = get_pvalue(pred_s_vals, random_state=random_state)

                    # Update the product factor.
                    horizon_update_factor *= betting_function(1.0, pred_pvalue)

                # Update the horizon martingale using the same previous traditional value.
                new_horizon = prev_trad * horizon_update_factor
                if new_horizon > threshold:
                    logger.info(
                        f"Horizon martingale detected change at t={i}: {new_horizon:.4f} > {threshold}"
                    )
                    horizon_change_points.append(i)

            elif predicted_data is not None:
                new_horizon = prev_trad

            # --- Reset Logic (Hybrid Approach) ---
            if detected_trad:
                # On traditional detection:
                # 1. Reset traditional martingale to 1.0
                # 2. Reset horizon martingale to 1.0 (hybrid approach)
                # 3. Clear window for fresh start
                window = []  # Clear window after detection
                saved_traditional.append(new_trad)  # Store detection value
                traditional_martingale.append(1.0)  # Reset for next iteration
                if new_horizon is not None:
                    saved_horizon.append(new_horizon)
                    horizon_martingale.append(
                        1.0
                    )  # Reset horizon on traditional detection
            else:
                # No detection: update windows and continue martingale sequences
                window.append(point)
                saved_traditional.append(new_trad)
                traditional_martingale.append(new_trad)
                if new_horizon is not None:
                    saved_horizon.append(new_horizon)
                    horizon_martingale.append(new_horizon)  # Continue horizon sequence

            # Log state every 10 timesteps.
            if i > 0 and i % 10 == 0:
                logger.debug(
                    f"t={i}: window size={len(window)}, traditional M={traditional_martingale[-1]:.4f}"
                )
                if predicted_data is not None and i >= history_size:
                    logger.debug(f"t={i}: horizon M={horizon_martingale[-1]:.4f}")
                logger.debug("-" * 30)

        logger.debug(
            f"Martingale computation complete. Traditional change points: {len(traditional_change_points)}; Horizon change points: {len(horizon_change_points)}."
        )

        return {
            "traditional_change_points": traditional_change_points,
            "horizon_change_points": horizon_change_points,
            "traditional_martingales": np.array(
                saved_traditional[1:], dtype=float
            ),  # Excludes initial value.
            "horizon_martingales": (
                np.array(saved_horizon[1:], dtype=float)
                if predicted_data is not None
                else None
            ),
        }
    except Exception as e:
        logger.error(f"Martingale computation failed: {str(e)}")
        raise RuntimeError(f"Martingale computation failed: {str(e)}")


def multiview_martingale_test(
    data: List[List[Any]],
    predicted_data: List[List[np.ndarray]],
    threshold: float,
    history_size: int,
    window_size: Optional[int] = None,
    batch_size: int = 1000,
    random_state: Optional[int] = None,
    betting_func_config: BettingFunctionConfig = {
        "name": "power",
        "params": {"epsilon": 0.7},
    },
    distance_measure: str = "euclidean",
    distance_p: float = 2.0,
) -> Dict[str, Any]:
    """
    Compute a multivariate (multiview) martingale test by aggregating evidence across multiple features.

    For d features, each feature j maintains its own martingale computed using the traditional update (current observation + history)
    and the horizon update (if predictions are provided). The combined martingale is defined as:
         M_total(n) = sum_{j=1}^{d} M_j(n)
         M_avg(n) = M_total(n) / d
    A change is declared if M_total(n) exceeds the threshold.

    Parameters
    ----------
    data : List[List[Any]]
        List of feature sequences to monitor.
    predicted_data : List[List[np.ndarray]]
        List of predicted feature vectors for each feature.
    threshold : float
        Detection threshold (> 0). A change is reported if the martingale exceeds this value.
    history_size : int
        Minimum number of observations to accumulate before using predictions.
    window_size : int, optional
        Maximum number of recent observations to keep in the history window.
    batch_size : int, optional
        Size of batches for processing (default: 1000).
    random_state : int, optional
        Random seed for reproducibility.
    betting_func_config : BettingFunctionConfig, optional
        Configuration for the betting function.
    distance_measure : str, optional
        Distance metric to use for strangeness computation.
    distance_p : float, optional
        Order parameter for Minkowski distance.

    Returns
    -------
    Dict[str, Any]
        A dictionary with change points and martingale values.
    """

    # Get the betting function with its configuration
    betting_function = get_betting_function(betting_func_config)

    logger.debug("Multiview Martingale Input Dimensions:")
    logger.debug(f"  Number of features: {len(data)}")
    logger.debug(f"  Sequence length per feature: {len(data[0])}")
    if predicted_data:
        logger.debug(f"  Number of prediction timesteps: {len(predicted_data[0])}")
        logger.debug(f"  Predictions per timestep: {len(predicted_data[0][0])}")
    logger.debug(f"  History size: {history_size}")
    logger.debug(f"  Window size: {window_size if window_size else 'None'}")
    logger.debug(f"  Batch size: {batch_size}")
    logger.debug("-" * 50)

    # ------- Input Validation -------
    if not data or not data[0]:
        logger.error("Empty data sequence provided")
        raise ValueError("Empty data sequence")
    if threshold <= 0:
        logger.error(f"Invalid threshold value: {threshold}")
        raise ValueError("Threshold must be positive")

    logger.debug(
        f"Starting multiview martingale test with {len(data)} features, num_samples={len(data[0])}, threshold={threshold}"
    )

    # ------- Initialize Storage -------
    num_features = len(data)
    num_samples = len(data[0])

    # Initialize per-feature history windows and individual traditional martingale series.
    windows = [[] for _ in range(num_features)]
    traditional_martingales = [[1.0] for _ in range(num_features)]
    individual_traditional = [[1.0] for _ in range(num_features)]

    # Aggregated traditional martingale series.
    traditional_sum = [float(num_features)]
    traditional_avg = [1.0]
    traditional_change_points = []

    # Initialize individual horizon martingale series.
    horizon_martingales = [[1.0] for _ in range(num_features)]

    # Aggregated horizon martingale series.
    horizon_sum = [float(num_features)]
    horizon_avg = [1.0]
    horizon_change_points = []

    # Determine prediction horizon length.
    num_horizons = len(predicted_data[0][0]) if predicted_data is not None else 0

    idx = 0
    while idx < num_samples:
        batch_end = min(idx + batch_size, num_samples)
        logger.debug(
            f"Processing batch [{idx}:{batch_end}]: Batch size = {batch_end - idx}"
        )

        for i in range(idx, batch_end):
            new_traditional = (
                []
            )  # New traditional martingale values for current timestep.
            # For each feature, update the traditional martingale.
            for j in range(num_features):

                # Maintain the rolling window if window_size is specified.
                if window_size and len(windows[j]) >= window_size:
                    windows[j] = windows[j][-window_size:]

                # Compute strangeness for the current observation.
                if not windows[j]:
                    s_vals = [0.0]
                else:
                    s_vals = strangeness_point(
                        windows[j] + [data[j][i]],
                        random_state=random_state,
                        distance_measure=distance_measure,
                        p=distance_p,
                    )

                # Compute p-value for update
                pv = get_pvalue(s_vals, random_state=random_state)

                # Update the traditional martingale using the betting function.
                prev_val = traditional_martingales[j][-1]
                new_val = betting_function(prev_val, pv)
                new_traditional.append(new_val)

            # Aggregate the traditional martingale values.
            total_traditional = sum(new_traditional)
            avg_traditional = total_traditional / num_features
            traditional_sum.append(total_traditional)
            traditional_avg.append(avg_traditional)

            # Update the aggregated traditional martingale series.
            for j in range(num_features):
                traditional_martingales[j].append(new_traditional[j])
                individual_traditional[j].append(
                    new_traditional[j]
                )  # Update individual traditional

            # --- Horizon Martingale Update ---
            new_horizon = []
            if predicted_data is not None and i >= history_size:
                pred_idx = i - history_size
                for j in range(num_features):
                    # Start with the same previous value as traditional martingale
                    prev_trad = traditional_martingales[j][
                        -2
                    ]  # Use traditional value before current update
                    horizon_factor = 1.0

                    # Loop over each prediction for feature j.
                    for h in range(num_horizons):
                        if not windows[j]:
                            # Skip horizon updates when window is empty (after reset)
                            continue
                        else:
                            pred_s_val = strangeness_point(
                                windows[j] + [predicted_data[j][pred_idx][h]],
                                random_state=random_state,
                                distance_measure=distance_measure,
                                p=distance_p,
                            )
                            pred_pv = get_pvalue(pred_s_val, random_state=random_state)
                            horizon_factor *= betting_function(1.0, pred_pv)

                    # Update horizon martingale value using traditional previous value
                    new_horizon_val = (
                        prev_trad * horizon_factor
                    )  # Start from traditional value
                    new_horizon.append(new_horizon_val)

                # Compute aggregated horizon martingale
                total_horizon = sum(new_horizon)
                avg_horizon = total_horizon / num_features
                horizon_sum.append(total_horizon)
                horizon_avg.append(avg_horizon)

                # Check for horizon martingale detection
                if total_horizon > threshold:
                    logger.info(
                        f"Horizon martingale detected change at t={i}: Sum={total_horizon:.4f} > {threshold}"
                    )
                    horizon_change_points.append(i)
            else:
                total_horizon = total_traditional
                avg_horizon = avg_traditional
                horizon_sum.append(total_horizon)
                horizon_avg.append(avg_horizon)

            # --- Reset Logic (Hybrid Approach) ---
            if total_traditional > threshold:
                logger.info(
                    f"Traditional martingale detected change at t={i}: Sum={total_traditional:.4f} > {threshold}"
                )
                traditional_change_points.append(i)
                # Reset both traditional and horizon martingales for all features
                for j in range(num_features):
                    windows[j] = []  # Clear windows
                    traditional_martingales[j].append(1.0)  # Reset traditional
                    individual_traditional[j].append(
                        1.0
                    )  # Reset individual traditional
                    horizon_martingales[j].append(
                        1.0
                    )  # Reset horizon (hybrid approach)
            else:
                # No detection: update windows and continue martingale sequences
                for j in range(num_features):
                    windows[j].append(data[j][i])
                    if predicted_data is not None and i >= history_size:
                        horizon_martingales[j].append(new_horizon[j])

            if i > 0 and i % 10 == 0:
                logger.debug(
                    f"t={i}: Avg window size={np.mean([len(w) for w in windows]):.1f}, Traditional Sum={traditional_sum[-1]:.4f}, Horizon Sum={horizon_sum[-1]:.4f}"
                )

        logger.debug(
            f"Completed batch: Traditional Sum={traditional_sum[-1]:.4f}, Change points={len(traditional_change_points)}"
        )
        idx = batch_end

    return {
        "traditional_change_points": traditional_change_points,
        "horizon_change_points": horizon_change_points,
        "traditional_sum_martingales": np.array(traditional_sum[1:], dtype=float),
        "traditional_avg_martingales": np.array(traditional_avg[1:], dtype=float),
        "horizon_sum_martingales": np.array(horizon_sum[1:], dtype=float),
        "horizon_avg_martingales": np.array(horizon_avg[1:], dtype=float),
        "individual_traditional_martingales": [
            np.array(m[1:], dtype=float) for m in traditional_martingales
        ],
        "individual_horizon_martingales": [
            np.array(m[1:], dtype=float) for m in horizon_martingales
        ],
    }
