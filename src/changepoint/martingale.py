# src/changepoint/martingale.py

"""Martingale framework for online change detection using conformal p-values and a chosen strangeness measure.

- Traditional martingale: uses the current observation and previous history.
- Horizon martingale: uses the current observation and multiple predicted future states,
  along with the previous history.
"""

from typing import List, Dict, Any, Optional, Callable
import logging

import numpy as np
from .strangeness import strangeness_point, get_pvalue

logger = logging.getLogger(__name__)


def compute_martingale(
    data: List[Any],
    predicted_data: List[np.ndarray],
    threshold: float,
    epsilon: float,
    history_size: int,
    reset: bool = True,
    window_size: Optional[int] = None,
    random_state: Optional[int] = None,
    bitting_func: Callable[[float, float, float], float] = None,
) -> Dict[str, Any]:
    """
    Compute a martingale for online change detection over a univariate data stream using conformal p-values
    and a chosen strangeness measure.

    There are two martingale streams computed here:
      1. Traditional martingale: uses only the current timestep with its history.
      2. Horizon martingale: uses the current timestep plus all predicted future states (horizon)
         together with the history.

    The martingale update is performed using the provided 'bitting_func', which defaults to the power martingale update.

    Parameters
    ----------
    data : List[Any]
        Sequential observations to monitor (e.g., numeric values).
    predicted_data : List[np.ndarray]
        List of predicted feature vectors for future timesteps.
        Each prediction should have the same shape as a data row.
    threshold : float
        Detection threshold (>0). If the martingale > threshold, a change is reported.
    epsilon : float
        Sensitivity parameter in (0,1); smaller epsilon makes the martingale more sensitive.
    history_size : int
        The required amount of history before using predictions.
    reset : bool, optional
        Whether to reset the martingale and history window after a detection (default is True).
    window_size : int, optional
        Maximum number of recent observations to keep in the history window.
        If None, keeps all past observations.
    random_state : int, optional
        Random seed for reproducibility.
    bitting_func : callable, optional
        Function to update the martingale. It should accept (prev_m, pvalue, epsilon) and return the updated value.
        Defaults to the power martingale update.

    Returns
    -------
    Dict[str, Any]
        A dictionary with:
         - "change_points": List[int] of indices where a change was detected.
         - "horizon_change_points": List[int] of indices where a horizon change was detected.
         - "pvalues": List[float] of p-values for each observation.
         - "strangeness": List[float] of computed strangeness values.
         - "martingales": np.ndarray of traditional martingale values.
         - "prediction_martingales": np.ndarray of horizon martingale values.
         - "prediction_pvalues": p-value sequences for the predictions.
         - "prediction_strangeness": strangeness sequences for the predictions.

    Raises
    ------
    ValueError
        If epsilon is not in (0,1) or threshold <= 0.
    """

    logger.info("Single-view Martingale Input Dimensions:")
    logger.info(f"- Sequence length: {len(data)}")
    if predicted_data:
        logger.info(f"- Number of predictions: {len(predicted_data)}")
        logger.info(f"- Predictions per timestep: {len(predicted_data[0])}")
    logger.info(f"- History size: {history_size}")
    logger.info(f"- Window size: {window_size if window_size else 'None'}")
    logger.info("-" * 50)

    if not 0 < epsilon < 1:
        logger.error(f"Invalid epsilon value: {epsilon}")

        raise ValueError("Epsilon must be in (0,1)")

    if threshold <= 0:
        logger.error(f"Invalid threshold value: {threshold}")
        raise ValueError("Threshold must be positive")

    logger.debug(
        f"Starting martingale computation with epsilon={epsilon}, threshold={threshold}, window_size={window_size}"
    )

    # Lists to store computed values
    pvalues: List[float] = []
    change_points: List[int] = []  # Traditional martingale detections
    horizon_change_points: List[int] = []  # Horizon martingale detections
    saved_strangeness: List[float] = []

    # Traditional martingale: initial value is 1.0.
    martingale = [1.0]  # Running internal value for traditional update.
    saved_martingales = [1.0]  # Saved sequence for output.
    # Horizon martingale: starts with 1.0 and will be computed separately.
    prediction_martingale = [1.0]

    # Rolling history window for computing strangeness.
    window: List[Any] = []

    # Structures for prediction-related values.
    prediction_pvalues: List[List[float]] = (
        []
    )  # List of lists for each horizon step's p-values.
    prediction_strangeness: List[List[float]] = (
        []
    )  # List of lists for each horizon step's strangeness.

    try:
        # Iterate over each data point.
        for i, point in enumerate(data):
            if i == 0:
                logger.info("Processing first data point:")
                logger.info(f"- Point value: {point}")
                logger.info(f"- Window size: {len(window)}")
                logger.info(f"- Initial martingale: {martingale[-1]}")
                logger.info("-" * 30)

            # Maintain a rolling history window if window_size is specified.
            if window_size and len(window) >= window_size:
                window = window[-window_size:]

            # Compute strangeness for the current point.
            if len(window) == 0:
                strangeness_vals = [0.0]
            else:
                strangeness_vals = strangeness_point(
                    window + [point], random_state=random_state
                )
            current_strg = strangeness_vals[-1]
            saved_strangeness.append(current_strg)

            # Convert the computed strangeness to a conformal p-value.
            pvalue = get_pvalue(strangeness_vals, random_state=random_state)
            pvalues.append(pvalue)

            # Get previous martingale value before any updates
            prev_m = martingale[-1]

            # Update the traditional martingale using the provided bitting function.
            new_m = bitting_func(prev_m, pvalue, epsilon)
            traditional_detection = (
                False  # Only set for traditional martingale detection
            )
            detection_value = None

            # Check if the updated traditional martingale exceeds the threshold.
            if reset and new_m > threshold:
                logger.info(
                    f"Change detected at time {i}, traditional M={new_m:.4f} > threshold={threshold}"
                )
                traditional_detection = True  # Only set for traditional martingale
                detection_value = new_m
                change_points.append(i)

            logger.debug(
                f"Time={i}, point={point}, p-value={pvalue:.4f}, traditional M={new_m:.4f}"
            )

            # --- Horizon (Predictive) Martingale Update ---
            new_pred_m = None
            if predicted_data is not None and i >= history_size:
                # Adjust index for predictions.
                pred_idx = i - history_size
                horizon_pvalues = []  # p-values for each horizon step.
                horizon_strangeness = []  # strangeness for each horizon step.
                pred_martingale_factor = 1.0  # Product factor for horizon update.

                for h in range(len(predicted_data[pred_idx])):
                    # Compute strangeness for predicted state.
                    if len(window) == 0:
                        pred_strangeness_vals = [0.0]
                    else:
                        # Wrap the predicted value in an extra list to mimic a single observation.
                        pred_strangeness_vals = strangeness_point(
                            window + [[predicted_data[pred_idx][h]]],
                            random_state=random_state,
                        )
                    current_pred_strg = pred_strangeness_vals[-1]
                    horizon_strangeness.append(current_pred_strg)

                    # Compute the conformal p-value for the predicted state.
                    pred_pvalue = get_pvalue(
                        pred_strangeness_vals, random_state=random_state
                    )
                    horizon_pvalues.append(pred_pvalue)

                    # Update the product factor for this horizon step using the bitting function.
                    pred_martingale_factor *= bitting_func(1.0, pred_pvalue, epsilon)

                # Use the same previous martingale value for horizon update
                new_pred_m = prev_m * pred_martingale_factor

                prediction_pvalues.append(horizon_pvalues)
                prediction_strangeness.append(horizon_strangeness)

                # Store horizon detections
                if new_pred_m > threshold:
                    logger.info(
                        f"Horizon martingale detected change at time {i}, M={new_pred_m:.4f} > threshold={threshold}"
                    )
                    horizon_change_points.append(i)
            elif predicted_data is not None:
                new_pred_m = new_m

            # Now handle reset and update stored values
            # Only reset if traditional martingale detected change
            if traditional_detection:  # Reset ONLY on traditional martingale detection
                window = []  # Clear window after detection
                saved_martingales.append(detection_value)  # Store detection value
                martingale.append(1.0)  # Reset running value for next iteration
            else:
                window.append(point)
                saved_martingales.append(new_m)
                martingale.append(new_m)

            # Store prediction martingale value without resetting
            if new_pred_m is not None:
                prediction_martingale.append(new_pred_m)

            # After prediction processing, log state at key points
            if i > 0 and i % 10 == 0:
                logger.info(f"Processing state at t={i}:")
                logger.info(f"- Current window size: {len(window)}")
                logger.info(f"- Traditional martingale: {martingale[-1]:.4f}")
                if predicted_data is not None and i >= history_size:
                    logger.info(
                        f"- Prediction martingale: {prediction_martingale[-1]:.4f}"
                    )
                logger.info("-" * 30)

        logger.debug(
            f"Martingale computation complete. Found {len(change_points)} traditional change points and {len(horizon_change_points)} horizon change points."
        )

        return {
            "change_points": change_points,
            "horizon_change_points": horizon_change_points,
            "pvalues": pvalues,
            "strangeness": saved_strangeness,
            "martingales": np.array(saved_martingales[1:], dtype=float),
            "prediction_martingales": (
                np.array(prediction_martingale[1:], dtype=float)
                if predicted_data is not None
                else None
            ),
            "prediction_pvalues": (
                prediction_pvalues if predicted_data is not None else None
            ),
            "prediction_strangeness": (
                prediction_strangeness if predicted_data is not None else None
            ),
        }
    except Exception as e:
        logger.error(f"Martingale computation failed: {str(e)}")
        raise RuntimeError(f"Martingale computation failed: {str(e)}")


def multiview_martingale_test(
    data: List[List[Any]],
    predicted_data: List[List[np.ndarray]],
    threshold: float,
    epsilon: float,
    history_size: int,
    window_size: Optional[int] = None,
    early_stop_threshold: Optional[float] = None,
    batch_size: int = 1000,
    random_state: Optional[int] = None,
    bitting_func: Callable[[float, float, float], float] = None,
) -> Dict[str, Any]:
    """
    Compute a multivariate (multiview) martingale test, combining evidence across multiple features.


    For d features, each feature j maintains its own martingale M_j(n) computed using the
    traditional update (current observation + history) and (if provided) the horizon update.
    The combined martingale is defined as:
        M_total(n) = sum_{j=1}^{d} M_j(n)
        M_avg(n) = M_total(n) / d
    A change is detected if M_total(n) > threshold.

    The martingale update is performed using the provided 'bitting_func', which defaults to the
    power martingale update.

    Parameters
    ----------
    data : List[List[Any]]
        data[j] is the entire time-series for feature j.
        All features must have the same number of samples.
    predicted_data : List[List[np.ndarray]]
        List of predicted feature vectors for future timesteps per feature.
        Each prediction should have the same shape as a data row.
    threshold : float
        Detection threshold for the sum of martingales.
    epsilon : float
        Sensitivity parameter in (0,1) for all features.
    history_size : int
        The required amount of history before using predictions.
    window_size : int, optional
        Rolling window size for each feature. If None, use all historical data.
    early_stop_threshold : float, optional
        If specified, processing stops early if the sum of martingales exceeds this value.
    batch_size : int, optional
        Process data in chunks for large datasets.
    random_state : int, optional
        Seed for reproducibility.
    bitting_func : callable, optional
        Function to update the martingale. It should accept (prev_m, pvalue, epsilon) and return the updated value.
        Defaults to the power martingale update.

    Returns
    -------
    Dict[str, Any]
        A dictionary with:
         - "change_points": List[int] where M_total(n) first exceeded the threshold.
         - "horizon_change_points": List[int] where M_total(n) first exceeded the threshold in horizon updates.
         - "pvalues": p-value sequences for each feature (list of lists).
         - "strangeness": strangeness sequences for each feature.
         - "martingale_sum": Time series of M_total(n).
         - "martingale_avg": Time series of M_avg(n).
         - "individual_martingales": Individual martingale time series for each feature.
         - Prediction results (if predicted_data is provided) with the same structure.
    """
    logger.info("Multiview Martingale Input Dimensions:")
    logger.info(f"- Number of features/views: {len(data)}")
    logger.info(f"- Sequence length per view: {len(data[0])}")
    if predicted_data:
        logger.info(f"- Number of prediction timesteps: {len(predicted_data[0])}")
        logger.info(f"- Predictions per timestep: {len(predicted_data[0][0])}")
    logger.info(f"- History size: {history_size}")
    logger.info(f"- Window size: {window_size if window_size else 'None'}")
    logger.info(f"- Batch size: {batch_size}")
    logger.info("-" * 50)

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
    logger.debug(
        f"Starting multiview martingale test with {num_features} features, num_samples={num_samples}, threshold={threshold}, epsilon={epsilon}"
    )

    # Initialize storage structures
    windows = [[] for _ in range(num_features)]
    martingales = [[1.0] for _ in range(num_features)]
    pvalues = []  # List of lists, each inner list has num_features values
    strangeness_vals = []  # List of lists, each inner list has num_features values

    # Setup prediction-related structures
    prediction_pvalues = (
        []
    )  # List of lists, each inner list has num_features * num_horizons values
    prediction_strangeness = (
        []
    )  # List of lists, each inner list has num_features * num_horizons values

    martingale_sum = [float(num_features)]
    martingale_avg = [1.0]
    change_points = []
    horizon_change_points = []

    individual_martingales = [[1.0] for _ in range(num_features)]

    # Add dimension validation
    def validate_dimensions(timestep: int):
        if len(pvalues) != timestep + 1:
            logger.error(
                f"P-values length mismatch at t={timestep}: {len(pvalues)} != {timestep + 1}"
            )
        if len(strangeness_vals) != timestep + 1:
            logger.error(
                f"Strangeness length mismatch at t={timestep}: {len(strangeness_vals)} != {timestep + 1}"
            )
        if predicted_data is not None and timestep >= history_size:
            pred_idx = timestep - history_size
            if len(prediction_pvalues) != pred_idx + 1:
                logger.error(
                    f"Prediction p-values length mismatch at t={timestep}: {len(prediction_pvalues)} != {pred_idx + 1}"
                )
            if len(prediction_strangeness) != pred_idx + 1:
                logger.error(
                    f"Prediction strangeness length mismatch at t={timestep}: {len(prediction_strangeness)} != {pred_idx + 1}"
                )

    # Setup prediction-related structures per feature if predicted data is provided.
    num_horizons = len(predicted_data[0][0]) if predicted_data is not None else 0
    prediction_martingales = [[1.0] for _ in range(num_features * num_horizons)]
    prediction_martingale_sum = [float(num_features)]
    prediction_martingale_avg = [1.0]

    idx = 0
    while idx < num_samples:
        batch_end = min(idx + batch_size, num_samples)

        # Log batch processing
        logger.info(f"Processing batch [{idx}:{batch_end}]:")
        logger.info(f"- Batch size: {batch_end - idx}")

        for i in range(idx, batch_end):
            if i == idx:  # First iteration of batch
                logger.info(f"First point in batch (t={i}):")
                for j in range(num_features):
                    logger.info(f"- Feature {j} window size: {len(windows[j])}")
                    logger.info(
                        f"- Feature {j} current martingale: {martingales[j][-1]:.4f}"
                    )
                logger.info(f"- Total martingale: {martingale_sum[-1]:.4f}")
                logger.info("-" * 30)

            timestep_pvalues = []  # Will store num_features values
            timestep_strangeness = []  # Will store num_features values
            timestep_pred_pvalues = []  # Will store num_features * num_horizons values
            timestep_pred_strangeness = (
                []
            )  # Will store num_features * num_horizons values
            new_martingales = []

            for j in range(num_features):
                if window_size and len(windows[j]) >= window_size:
                    windows[j] = windows[j][-window_size:]

                if len(windows[j]) == 0:
                    svals = [0.0]
                else:
                    logger.debug(f"Traditional martingale data at t={i}:")
                    logger.debug(f"Window for feature {j}: {windows[j]}")
                    logger.debug(f"Current point for feature {j}: {data[j][i]}")
                    logger.debug(f"Combined data: {windows[j] + [data[j][i]]}")

                    svals = strangeness_point(
                        windows[j] + [data[j][i]], random_state=random_state
                    )

                current_strg = svals[-1]
                timestep_strangeness.append(current_strg)

                pv = get_pvalue(svals, random_state=random_state)
                timestep_pvalues.append(pv)

                prev_m = martingales[j][-1]
                new_m = bitting_func(prev_m, pv, epsilon)
                new_martingales.append(new_m)  # Append to list of new martingale values
                individual_martingales[j].append(new_m)

            total_m = sum(new_martingales)
            avg_m = total_m / num_features
            martingale_sum.append(total_m)
            martingale_avg.append(avg_m)

            # Validate shapes before storing
            if len(new_martingales) != num_features:
                logger.error(
                    f"Shape mismatch: new_martingales has {len(new_martingales)} elements, expected {num_features}"
                )
                raise ValueError(f"Shape mismatch in martingale computation")

            # Store original martingale values before potential reset
            for j in range(num_features):
                martingales[j].append(new_martingales[j])

            # Validate after storing
            if any(len(m) != len(martingales[0]) for m in martingales):
                logger.error("Inconsistent martingale sequence lengths")
                raise ValueError("Inconsistent martingale sequence lengths")

            # Check for traditional martingale detection
            traditional_detection = total_m > threshold
            if traditional_detection:
                logger.info(
                    f"Traditional martingale detected change at time {i}, M_total={total_m:.4f} > threshold={threshold}"
                )
                change_points.append(i)

            # --- Horizon Martingale Update for multiview test ---
            if predicted_data is not None and i >= history_size:
                pred_idx = i - history_size
                new_pred_martingales = []
                for j in range(num_features):
                    pred_martingale_factor = 1.0
                    for h in range(num_horizons):
                        logger.debug(
                            f"\nHorizon martingale data at t={i}, horizon={h}:"
                        )
                        logger.debug(f"Window for feature {j}: {windows[j]}")
                        logger.debug(
                            f"Predicted data shape: {np.array(predicted_data[j]).shape}"
                        )

                        logger.debug(
                            f"Predicted value: {predicted_data[j][pred_idx][h]}"
                        )
                        logger.debug(
                            f"Combined data: {windows[j] + [predicted_data[j][pred_idx][h]]}"
                        )

                        if len(windows[j]) == 0:
                            pred_svals = [0.0]
                        else:
                            pred_svals = strangeness_point(
                                windows[j] + [predicted_data[j][pred_idx][h]],
                                random_state=random_state,
                            )

                        current_pred_strg = pred_svals[-1]
                        timestep_pred_strangeness.append(current_pred_strg)

                        pred_pv = get_pvalue(pred_svals, random_state=random_state)
                        timestep_pred_pvalues.append(pred_pv)

                        pred_martingale_factor *= bitting_func(1.0, pred_pv, epsilon)

                    # Use original martingale value (before potential reset)
                    prev_trad_m = new_martingales[j]  # Use value before reset
                    new_pred_m = prev_trad_m * pred_martingale_factor
                    new_pred_martingales.append(new_pred_m)

                    # Store the individual prediction martingale
                    feature_horizon_idx = j * num_horizons + h
                    prediction_martingales[feature_horizon_idx].append(new_pred_m)

                pred_total_m = sum(new_pred_martingales)
                pred_avg_m = pred_total_m / num_features

                # Store horizon detections
                if pred_total_m > threshold:
                    logger.info(
                        f"Horizon martingale detected change at time {i}, M_total={pred_total_m:.4f} > threshold={threshold}"
                    )
                    horizon_change_points.append(i)
            else:
                pred_total_m = total_m
                pred_avg_m = avg_m

            prediction_martingale_sum.append(pred_total_m)
            prediction_martingale_avg.append(pred_avg_m)

            # Handle reset ONLY if traditional martingale detected change
            if traditional_detection:
                for j in range(num_features):
                    windows[j] = []  # Clear windows
                    martingales[j][-1] = 1.0  # Reset traditional martingales
                    individual_martingales[j][-1] = new_martingales[
                        j
                    ]  # Store detection values
            else:
                for j in range(num_features):
                    windows[j].append(data[j][i])

            # Store timestep values
            pvalues.append(
                timestep_pvalues
            )  # Each element is list of num_features values
            strangeness_vals.append(
                timestep_strangeness
            )  # Each element is list of num_features values
            if predicted_data is not None and i >= history_size:
                prediction_pvalues.append(
                    timestep_pred_pvalues
                )  # Each element is list of num_features * num_horizons values
                prediction_strangeness.append(
                    timestep_pred_strangeness
                )  # Each element is list of num_features * num_horizons values

            # Validate dimensions after storing
            validate_dimensions(i)

            # Log state at key points
            if i > 0 and i % 10 == 0:
                logger.info(f"Processing state at t={i}:")
                logger.info(
                    f"- Average window size: {np.mean([len(w) for w in windows]):.1f}"
                )
                logger.info(f"- Total martingale: {martingale_sum[-1]:.4f}")
                logger.info(f"- Average martingale: {martingale_avg[-1]:.4f}")
                if predicted_data is not None and i >= history_size:
                    logger.info(
                        f"- Prediction total martingale: {prediction_martingale_sum[-1]:.4f}"
                    )
                logger.info("-" * 30)

        # Log batch completion
        logger.info(f"Completed batch. Current state:")
        logger.info(f"- Total martingale: {martingale_sum[-1]:.4f}")
        logger.info(f"- Number of change points: {len(change_points)}")
        if change_points:
            logger.info(f"- Last change point: {change_points[-1]}")
        logger.info("-" * 50)

        idx = batch_end

    # Before returning, validate final dimensions
    final_dims = {
        "pvalues": len(pvalues),
        "strangeness": len(strangeness_vals),
        "martingale_sum": len(martingale_sum) - 1,  # -1 for initial value
        "martingale_avg": len(martingale_avg) - 1,
        "individual_martingales": len(individual_martingales),
        "prediction_pvalues": (
            len(prediction_pvalues) if predicted_data is not None else 0
        ),
        "prediction_strangeness": (
            len(prediction_strangeness) if predicted_data is not None else 0
        ),
        "prediction_martingale_sum": len(prediction_martingale_sum) - 1,
        "prediction_martingale_avg": len(prediction_martingale_avg) - 1,
    }

    logger.info("Final dimensions:")
    for key, value in final_dims.items():
        logger.info(f"- {key}: {value}")

    return {
        "change_points": change_points,
        "horizon_change_points": horizon_change_points,
        "pvalues": pvalues,  # List[List[float]], shape: (num_samples, num_features)
        "strangeness": strangeness_vals,  # List[List[float]], shape: (num_samples, num_features)
        "martingale_sum": np.array(
            martingale_sum[1:], dtype=float
        ),  # shape: (num_samples,)
        "martingale_avg": np.array(
            martingale_avg[1:], dtype=float
        ),  # shape: (num_samples,)
        "individual_martingales": [
            np.array(m[1:], dtype=float) for m in individual_martingales
        ],  # List of length num_features
        "prediction_pvalues": (
            prediction_pvalues if predicted_data is not None else None
        ),  # List[List[float]], shape: (num_pred_samples, num_features * num_horizons)
        "prediction_strangeness": (
            prediction_strangeness if predicted_data is not None else None
        ),  # List[List[float]], shape: (num_pred_samples, num_features * num_horizons)
        "prediction_martingale_sum": np.array(
            prediction_martingale_sum[1:], dtype=float
        ),  # shape: (num_samples,)
        "prediction_martingale_avg": np.array(
            prediction_martingale_avg[1:], dtype=float
        ),  # shape: (num_samples,)
        "prediction_individual_martingales": [
            np.array(m[1:], dtype=float) for m in prediction_martingales
        ],  # List of length num_features * num_horizons
    }
