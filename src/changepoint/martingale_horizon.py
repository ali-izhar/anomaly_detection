# src/changepoint/martingale_horizon.py

"""Horizon martingale implementation for change point detection.
Horizon martingale uses the current observation and multiple predicted future states,
along with the previous history."""

import logging
import traceback
import numpy as np
from typing import Dict, List, Any, Optional

from .martingale_base import (
    MartingaleConfig,
    HorizonMartingaleState,
    MultiviewHorizonMartingaleState,
    DataPoint,
    Array,
)

from .martingale_traditional import (
    compute_traditional_martingale,
    multiview_traditional_martingale,
)

from .betting import (
    get_betting_function,
)

from .strangeness import (
    strangeness_point,
    get_pvalue,
)


logger = logging.getLogger(__name__)


def compute_horizon_martingale(
    data: List[DataPoint],
    predicted_data: List[Array],
    config: Optional[MartingaleConfig] = None,
    state: Optional[HorizonMartingaleState] = None,
    traditional_martingale_values: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Compute a horizon martingale for online change detection over a univariate data stream.

    Uses conformal p-values and a chosen strangeness measure to compute horizon martingale
    that uses the current observation plus predicted future states.

    Args:
        data: Sequential observations to monitor.
        predicted_data: List of predicted feature vectors for future timesteps.
        config: Configuration for martingale computation.
        state: Optional state for continuing computation from a previous run.
        traditional_martingale_values: List of traditional martingale values to use as base.

    Returns:
        Dictionary containing:
         - "horizon_change_points": List[int] of indices where horizon martingale detected a change.
         - "horizon_martingales": np.ndarray of horizon martingale values.

    Raises:
        ValueError: If input validation fails.
        RuntimeError: If computation fails.
    """
    if not data:
        raise ValueError("Empty data sequence")

    if not predicted_data:
        raise ValueError("Empty predicted data sequence")

    if config is None:
        raise ValueError("Config is required")

    if state is None:
        state = HorizonMartingaleState()

    # Calculate traditional martingale values if not provided
    if traditional_martingale_values is None:
        # Compute traditional martingale
        trad_result = compute_traditional_martingale(
            data, config, None, printing_from_horizon=True
        )
        traditional_martingale_values = trad_result.get(
            "traditional_martingales", np.ones(len(data))
        ).tolist()

    # Ensure traditional_martingale_values has the right length
    if len(traditional_martingale_values) < len(data):
        padding = [1.0] * (len(data) - len(traditional_martingale_values))
        traditional_martingale_values = padding + traditional_martingale_values

    # Obtain the betting function callable based on the betting_func_config.
    betting_function = get_betting_function(config.betting_func_config)

    # Log input dimensions and configuration details.
    logger.debug("Horizon Martingale Input Dimensions:")
    logger.debug(f"  Sequence length: {len(data)}")
    logger.debug(f"  Number of predictions: {len(predicted_data)}")
    logger.debug(f"  Predictions per timestep: {len(predicted_data[0])}")
    logger.debug(f"  History size: {config.history_size}")
    logger.debug(
        f"  Window size: {config.window_size if config.window_size else 'None'}"
    )
    logger.debug("-" * 50)

    try:
        # Initialize horizon martingale values
        horizon_martingales = []
        horizon_change_points = []

        # Initialize momentum tracking
        if not hasattr(state, "previous_horizon_val"):
            state.previous_horizon_val = 1.0

        # Define threshold-approach ratio (used for momentum-based dampening)
        threshold_approach_ratio = 0.9  # When martingale is above 90% of threshold

        # Initialize horizon-specific martingales
        if not hasattr(state, "horizon_martingales_h"):
            num_horizons = (
                len(predicted_data[0])
                if predicted_data and len(predicted_data) > 0
                else 0
            )
            state.horizon_martingales_h = [1.0] * num_horizons

        # Compute horizon weights with exponential decay (closer horizons get higher weights)
        num_horizons = (
            len(predicted_data[0]) if predicted_data and len(predicted_data) > 0 else 0
        )

        if num_horizons > 0:
            # Use exponential decay to weight horizons - INCREASED DECAY RATE for faster response
            decay_rate = (
                0.8  # Increased from 0.5 to 0.8 to emphasize near-term predictions
            )
            unnormalized_weights = [
                np.exp(-decay_rate * h) for h in range(num_horizons)
            ]
            # Normalize weights to sum to 1
            horizon_weights = [
                w / sum(unnormalized_weights) for w in unnormalized_weights
            ]
        else:
            horizon_weights = []

        # Process each point and compute the horizon component
        for i in range(len(data)):
            # For timesteps before history_size, horizon martingale is not meaningful
            if i < config.history_size:
                horizon_martingales.append(
                    float("nan")
                )  # Use NaN to indicate no prediction available
                state.saved_horizon.append(float("nan"))
                continue

            # Add current point to window for conformity scoring
            if i > 0:
                state.window.append(data[i - 1])

            # Maintain a rolling window if window_size is specified
            if config.window_size and len(state.window) > config.window_size:
                state.window = state.window[-config.window_size :]

            # Use early prediction only if we have enough window data
            if len(state.window) < 2:
                horizon_martingales.append(1.0)
                state.saved_horizon.append(1.0)
                continue

            # Get the traditional martingale value from previous timestep as base
            # Use 1.0 for the first timestep
            trad_base = traditional_martingale_values[i - 1] if i > 0 else 1.0

            # Check predictions at multiple horizons
            horizon_martingales_at_t = []

            # Calculate momentum factor based on recent growth
            momentum_factor = 1.0
            current_momentum = 1.0
            if (
                i > 0
                and len(horizon_martingales) > 0
                and state.previous_horizon_val > 0
            ):
                # Get previous martingale value
                prev_val = state.previous_horizon_val

                # For the first calculation, we don't have a previous value yet
                if (
                    i > config.history_size
                    and isinstance(prev_val, (int, float))
                    and prev_val > 0
                ):
                    # Calculate momentum (rate of increase)
                    if prev_val > 0.01:  # Avoid division by very small numbers
                        current_momentum = max(
                            1.0, min(5.0, horizon_martingales[-1] / prev_val)
                        )

                    # Only apply increased dampening when:
                    # 1. There's momentum (value is increasing)
                    # 2. The value is approaching the threshold
                    if (
                        current_momentum > 1.2
                        and horizon_martingales[-1]
                        > threshold_approach_ratio * config.threshold
                        and horizon_martingales[-1] < config.threshold
                    ):
                        # Stronger dampening when value is close to threshold but not crossing it
                        momentum_factor = (
                            0.5 / current_momentum
                        )  # Smaller values = stronger dampening
                        logger.debug(
                            f"Applying momentum-based dampening at t={i}: momentum={current_momentum:.2f}, "
                            f"factor={momentum_factor:.2f}, current_val={horizon_martingales[-1]:.2f}"
                        )

            # For each horizon h, check if there's evidence of a change in the next h steps
            for h in range(num_horizons):
                # We need predictions made at (i-h) for current time i
                pred_time = i - h

                # Skip if we don't have predictions from that time
                if pred_time < config.history_size or pred_time >= len(data):
                    m_h = trad_base
                    horizon_martingales_at_t.append(m_h)
                    continue

                # The prediction index in our predictions array
                pred_idx = pred_time - config.history_size

                # Skip if prediction is not available
                if (
                    pred_idx < 0
                    or pred_idx >= len(predicted_data)
                    or h >= len(predicted_data[pred_idx])
                ):
                    m_h = trad_base
                    horizon_martingales_at_t.append(m_h)
                    continue

                # Compare the prediction made h steps ago for current time i
                try:
                    # Get prediction that was made at (i-h) for current time i
                    pred_data = np.array(predicted_data[pred_idx][h]).reshape(1, -1)
                    # Current actual data
                    actual_data = np.array(data[i]).reshape(1, -1)

                    # Calculate difference between prediction and actual
                    diff = np.abs(pred_data - actual_data).mean()

                    # Compute conformity score
                    window_data = np.array(state.window).reshape(-1, 1)

                    # Compute conformity: how different is prediction from actual?
                    pred_s_val = strangeness_point(
                        np.vstack([window_data, actual_data]),
                        config=config.strangeness_config,
                    )
                    pred_pv = get_pvalue(pred_s_val, random_state=config.random_state)

                    # Adjust p-value based on prediction error (lower p-value = more surprised) only if enabled
                    if config.enable_pvalue_dampening:
                        # Apply standard dampening
                        scale_factor = max(0.01, min(1.0, 1.0 - diff / 1.0))

                        # Apply additional momentum-based dampening when approaching threshold
                        if momentum_factor < 1.0:
                            scale_factor = max(0.005, scale_factor * momentum_factor)

                        adjusted_pv = pred_pv * scale_factor
                    else:
                        # Use the raw p-value without dampening
                        adjusted_pv = pred_pv

                    # Update martingale for this horizon using traditional as base
                    m_h = trad_base * betting_function(1.0, adjusted_pv)
                except Exception as e:
                    logger.error(f"Error processing horizon {h} at time {i}: {e}")
                    m_h = trad_base

                horizon_martingales_at_t.append(m_h)

            # Ensure we have the right number of horizon martingales
            if len(horizon_martingales_at_t) < num_horizons:
                # Pad with traditional base value
                for h in range(len(horizon_martingales_at_t), num_horizons):
                    horizon_martingales_at_t.append(trad_base)

            # Combine horizons using weighted sum
            if horizon_weights and horizon_martingales_at_t:
                horizon_val = sum(
                    w * m for w, m in zip(horizon_weights, horizon_martingales_at_t)
                )
            else:
                horizon_val = trad_base

            # Store the previous horizon value for momentum calculation
            state.previous_horizon_val = (
                horizon_val if isinstance(horizon_val, (int, float)) else 1.0
            )

            # Update state for horizon-specific martingales (for tracking)
            state.horizon_martingales_h = horizon_martingales_at_t

            # Store the combined horizon martingale value
            state.horizon_martingale = horizon_val
            state.saved_horizon.append(horizon_val)
            horizon_martingales.append(horizon_val)

            # Check for horizon-based change detection
            if horizon_val > config.threshold:
                # Only detect if we're not in a cooldown period from a previous detection
                # For basic horizon martingale, we track this in the state
                detect_change = True

                # Check if we have a last_detection_time attribute, if not add it
                if not hasattr(state, "last_detection_time"):
                    state.last_detection_time = (
                        -40
                    )  # Initialize to effectively no previous detection

                # Use cooldown period from config
                cooldown = config.cooldown_period

                # Skip detection if we're still in cooldown period
                if i - state.last_detection_time < cooldown:
                    detect_change = False
                    logger.debug(
                        f"Suppressing horizon detection at t={i} due to cooldown period ({i - state.last_detection_time} < {cooldown})"
                    )

                if detect_change:
                    logger.debug(
                        f"Horizon martingale detected change at t={i}: {horizon_val:.4f} > {config.threshold}"
                    )
                    horizon_change_points.append(i)
                    state.last_detection_time = i  # Record the detection time

                    # Reset martingale state after detection
                    if config.reset:
                        logger.debug(
                            f"Resetting horizon martingale state after detection at t={i}"
                        )
                        state.horizon_martingales_h = [1.0] * num_horizons
                        state.window = []

        # Create masked array where initial NaN values (before predictions) are removed for display/calculations
        masked_martingales = np.array(horizon_martingales, dtype=float)
        # Replace NaN with 0 for array operations
        masked_martingales = np.nan_to_num(masked_martingales)

        # Return the computed martingale histories and detected change points
        return {
            "horizon_change_points": horizon_change_points,
            "horizon_martingales": masked_martingales,
            "prediction_start_time": config.history_size,  # Add the time when predictions start
        }

    except Exception as e:
        logger.error(f"Horizon martingale computation failed: {str(e)}")
        raise RuntimeError(f"Horizon martingale computation failed: {str(e)}")


def multiview_horizon_martingale(
    data: List[List[DataPoint]],
    predicted_data: List[List[Array]],
    config: Optional[MartingaleConfig] = None,
    state: Optional[MultiviewHorizonMartingaleState] = None,
    batch_size: int = 1000,
    traditional_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compute a multivariate (multiview) horizon martingale test by aggregating evidence across features.

    For d features, each feature maintains its own martingale for each horizon h:
    M_{t,h}^{(k)} = M_{t-1}^{trad,(k)} * g(p_{t,h}^{(k)})

    These are combined across horizons:
    M_{t}^{(k)} = \sum_{h \in \mathcal{H}} w_h M_{t,h}^{(k)}

    And finally combined across features:
    M_t = \sum_{k=1}^K v_k M_t^{(k)}

    A change is declared if M_t exceeds the threshold.

    Args:
        data: List of feature sequences to monitor.
        predicted_data: List of predicted feature vectors for each feature.
        config: Configuration for martingale computation.
        state: Optional state for continuing computation from a previous run.
        batch_size: Size of batches for processing.
        traditional_result: Optional pre-computed traditional martingale results.

    Returns:
        Dictionary containing change points and martingale values.

    Raises:
        ValueError: If input validation fails.
        RuntimeError: If computation fails.
    """
    if not data or not data[0]:
        raise ValueError("Empty data sequence")

    if not predicted_data or not predicted_data[0]:
        raise ValueError("Empty predicted data sequence")

    if config is None:
        raise ValueError("Config is required")

    if state is None:
        state = MultiviewHorizonMartingaleState()
        state.reset(len(data))

    # Get the betting function based on the provided configuration.
    betting_function = get_betting_function(config.betting_func_config)

    # Log input dimensions and configuration details.
    logger.debug("Multiview Horizon Martingale Input Dimensions:")
    logger.debug(f"  Number of features: {len(data)}")
    logger.debug(f"  Sequence length per feature: {len(data[0])}")
    logger.debug(f"  Number of prediction timesteps: {len(predicted_data[0])}")
    logger.debug(f"  Predictions per timestep: {len(predicted_data[0][0])}")
    logger.debug(f"  History size: {config.history_size}")
    logger.debug(
        f"  Window size: {config.window_size if config.window_size else 'None'}"
    )
    logger.debug(f"  Batch size: {batch_size}")
    logger.debug("-" * 50)

    try:
        # Get traditional martingale results
        if traditional_result is None:
            # Compute traditional martingale
            trad_result = multiview_traditional_martingale(
                data, config, None, batch_size, printing_from_horizon=True
            )
        else:
            trad_result = traditional_result

        trad_change_points = trad_result["traditional_change_points"]
        trad_individual_martingales = trad_result["individual_traditional_martingales"]

        num_features = len(data)
        num_samples = len(data[0])
        num_horizons = len(predicted_data[0][0])

        # Initialize results
        horizon_sum_martingales = []
        horizon_avg_martingales = []
        individual_horizon_martingales = [[] for _ in range(num_features)]
        horizon_change_points = []

        # Define threshold-approach ratio (used for momentum-based dampening)
        threshold_approach_ratio = 0.9  # When martingale is above 90% of threshold

        # Initialize martingale state for each feature-horizon combination
        # Ensure we always have proper initialization of feature_horizon_martingales
        if (
            not hasattr(state, "feature_horizon_martingales")
            or len(state.feature_horizon_martingales) != num_features
        ):
            state.feature_horizon_martingales = [
                [1.0] * num_horizons for _ in range(num_features)
            ]
        # Ensure each feature's martingale list has the right number of horizons
        for j in range(num_features):
            if j >= len(state.feature_horizon_martingales):
                state.feature_horizon_martingales.append([1.0] * num_horizons)
            elif len(state.feature_horizon_martingales[j]) != num_horizons:
                state.feature_horizon_martingales[j] = [1.0] * num_horizons

        # Compute horizon weights with exponential decay (closer horizons get higher weights)
        if num_horizons > 0:
            # MODIFIED: Increased decay rate to emphasize near-term predictions
            decay_rate = 0.8  # Increased from 0.5 to 0.8
            unnormalized_weights = [
                np.exp(-decay_rate * h) for h in range(num_horizons)
            ]
            # Normalize weights to sum to 1
            horizon_weights = [
                w / sum(unnormalized_weights) for w in unnormalized_weights
            ]
        else:
            horizon_weights = []

        # Equal weights for features by default - set to 1.0 to match traditional martingale simple sum
        feature_weights = (
            None  # No weighting by default to match traditional martingale sum
        )

        idx = 0
        while idx < num_samples:
            batch_end = min(idx + batch_size, num_samples)

            # Process each sample in the current batch.
            for i in range(idx, batch_end):
                # For timesteps before history_size, horizon martingale is not meaningful
                if i < config.history_size:
                    horizon_sum_martingales.append(float("nan"))
                    horizon_avg_martingales.append(float("nan"))
                    for j in range(num_features):
                        individual_horizon_martingales[j].append(float("nan"))
                    continue

                # Initialize feature-level martingales
                feature_martingales = []

                # Calculate momentum factor based on recent growth
                momentum_factor = 1.0
                current_momentum = 1.0

                # Check if we have previous values to calculate momentum
                if i > 0 and len(horizon_sum_martingales) > 0:
                    # Use the previous sum value
                    prev_val = (
                        state.previous_horizon_sum
                        if hasattr(state, "previous_horizon_sum")
                        else 1.0
                    )

                    # For the first calculation, we don't have a previous value yet
                    if (
                        i > config.history_size
                        and isinstance(prev_val, (int, float))
                        and prev_val > 0
                    ):
                        current_val = horizon_sum_martingales[-1]
                        if not np.isnan(current_val) and current_val > 0.01:
                            # Calculate momentum (rate of increase)
                            current_momentum = max(
                                1.0, min(5.0, current_val / prev_val)
                            )

                            # Only apply increased dampening when:
                            # 1. There's significant momentum (value is increasing rapidly)
                            # 2. The value is approaching the threshold but not crossing it
                            if (
                                current_momentum > 1.2
                                and current_val
                                > threshold_approach_ratio * config.threshold
                                and current_val < config.threshold
                            ):
                                # Stronger dampening when value is close to threshold
                                momentum_factor = (
                                    0.5 / current_momentum
                                )  # Smaller values = stronger dampening
                                logger.debug(
                                    f"Applying momentum-based dampening at t={i}: momentum={current_momentum:.2f}, "
                                    f"factor={momentum_factor:.2f}, current_val={current_val:.2f}"
                                )

                # For each feature, compute horizon martingales
                for j in range(num_features):
                    # Ensure this feature has windows
                    if j >= len(state.windows):
                        state.windows.append([])

                    # Add current point to window
                    if i > 0:
                        state.windows[j].append(data[j][i - 1])

                    # Limit window size if needed
                    if (
                        config.window_size
                        and len(state.windows[j]) > config.window_size
                    ):
                        state.windows[j] = state.windows[j][-config.window_size :]

                    # Get traditional martingale value from previous timestep as base
                    trad_base = (
                        trad_individual_martingales[j][i - 1]
                        if i > 0 and i - 1 < len(trad_individual_martingales[j])
                        else 1.0
                    )

                    # Initialize horizon-specific martingales for this feature
                    horizon_martingales_j = []

                    # Skip feature if window is too small
                    if len(state.windows[j]) < 2:
                        horizon_martingales_j = [trad_base] * num_horizons
                        feature_martingales.append(trad_base)
                        individual_horizon_martingales[j].append(trad_base)
                        continue

                    # For each horizon h, check if there's evidence of a change in the next h steps
                    for h in range(num_horizons):
                        # We need predictions made at (i-h) for current time i
                        pred_time = i - h

                        # Skip if we don't have predictions from that time
                        if pred_time < config.history_size or pred_time >= len(data[0]):
                            horizon_martingales_j.append(trad_base)
                            continue

                        # The prediction index in our predictions array
                        pred_idx = pred_time - config.history_size

                        # Skip if prediction is not available
                        if (
                            j >= len(predicted_data)
                            or pred_idx < 0
                            or pred_idx >= len(predicted_data[j])
                            or h >= len(predicted_data[j][pred_idx])
                        ):
                            horizon_martingales_j.append(trad_base)
                            continue

                        try:
                            # Get prediction that was made at pred_time for current time i
                            pred_data = np.array(
                                predicted_data[j][pred_idx][h]
                            ).reshape(1, -1)
                            # Current actual data
                            actual_data = np.array(data[j][i]).reshape(1, -1)

                            # Calculate difference between prediction and actual
                            diff = np.abs(pred_data - actual_data).mean()

                            # Compute conformity score
                            window_data = np.array(state.windows[j]).reshape(-1, 1)

                            # Compute conformity: how different is prediction from actual?
                            pred_s_val = strangeness_point(
                                np.vstack([window_data, actual_data]),
                                config=config.strangeness_config,
                            )
                            pred_pv = get_pvalue(
                                pred_s_val, random_state=config.random_state
                            )

                            # Adjust p-value based on prediction error only if dampening is enabled
                            if config.enable_pvalue_dampening:
                                # Apply standard dampening
                                scale_factor = max(0.01, min(1.0, 1.0 - diff / 1.0))

                                # Apply additional momentum-based dampening when approaching threshold
                                if momentum_factor < 1.0:
                                    scale_factor = max(
                                        0.005, scale_factor * momentum_factor
                                    )

                                adjusted_pv = pred_pv * scale_factor
                            else:
                                # Use the raw p-value without dampening
                                adjusted_pv = pred_pv

                            # Update martingale for this horizon using the traditional martingale as base
                            m_jh = trad_base * betting_function(1.0, adjusted_pv)
                        except Exception as e:
                            logger.error(
                                f"Error processing feature {j}, horizon {h} at time {i}: {e}"
                            )
                            m_jh = trad_base

                        horizon_martingales_j.append(m_jh)

                    # Ensure we have the right number of horizon martingales
                    if len(horizon_martingales_j) < num_horizons:
                        # Pad with traditional base values
                        for h in range(len(horizon_martingales_j), num_horizons):
                            horizon_martingales_j.append(trad_base)

                    # Update feature horizons state
                    if j < len(state.feature_horizon_martingales):
                        state.feature_horizon_martingales[j] = horizon_martingales_j
                    else:
                        state.feature_horizon_martingales.append(horizon_martingales_j)

                    # Combine horizons for this feature
                    if horizon_weights and horizon_martingales_j:
                        feature_m = sum(
                            w * m
                            for w, m in zip(horizon_weights, horizon_martingales_j)
                        )
                    else:
                        feature_m = trad_base

                    feature_martingales.append(feature_m)
                    individual_horizon_martingales[j].append(feature_m)

                # Combine across features
                if feature_weights and feature_martingales:
                    # Use weighted sum if weights explicitly provided (not default behavior)
                    total_horizon = sum(
                        v * m for v, m in zip(feature_weights, feature_martingales)
                    )
                    # Calculate average separately - exactly matching traditional implementation
                    avg_horizon = total_horizon / num_features
                else:
                    # Simple sum across features (no weighting) to exactly match traditional martingale
                    total_horizon = sum(feature_martingales)
                    # Calculate average separately - exactly matching traditional implementation
                    avg_horizon = total_horizon / num_features

                # Store current sum for next iteration's momentum calculation
                state.previous_horizon_sum = total_horizon

                # Record results directly without using record_values which requires traditional values
                horizon_sum_martingales.append(total_horizon)
                horizon_avg_martingales.append(avg_horizon)

                # Check for horizon-based detection
                if total_horizon > config.threshold:
                    # Check if we're still in cooldown period
                    detect_change = True

                    # Use cooldown period from config
                    cooldown = config.cooldown_period

                    # Skip detection if we're still in cooldown period
                    if i - state.last_detection_time < cooldown:
                        detect_change = False
                        logger.debug(
                            f"Suppressing horizon detection at t={i} due to cooldown period ({i - state.last_detection_time} < {cooldown})"
                        )

                    if detect_change:
                        logger.info(
                            f"Horizon martingale detected change at t={i}: "
                            f"Sum={total_horizon:.4f} > {config.threshold}"
                        )
                        horizon_change_points.append(i)
                        state.last_detection_time = i

                        # Reset martingale state
                        if config.reset:
                            logger.debug(
                                f"Resetting horizon martingale state after detection at t={i}"
                            )
                            state.reset(num_features)
                            state.feature_horizon_martingales = [
                                [1.0] * num_horizons for _ in range(num_features)
                            ]

                # Also reset if traditional martingale detected a change and reset_on_traditional is enabled
                if i in trad_change_points and config.reset_on_traditional:
                    logger.debug(
                        f"Resetting horizon martingale due to traditional detection at t={i}"
                    )
                    state.reset(num_features)
                    state.feature_horizon_martingales = [
                        [1.0] * num_horizons for _ in range(num_features)
                    ]

            idx = batch_end

        # Create masked arrays where initial NaN values (before predictions) are removed
        horizon_sum_martingales_masked = np.array(horizon_sum_martingales, dtype=float)
        horizon_avg_martingales_masked = np.array(horizon_avg_martingales, dtype=float)
        individual_horizon_martingales_masked = [
            np.nan_to_num(np.array(m, dtype=float))
            for m in individual_horizon_martingales
        ]

        # Replace NaN with 0 for array operations
        horizon_sum_martingales_masked = np.nan_to_num(horizon_sum_martingales_masked)
        horizon_avg_martingales_masked = np.nan_to_num(horizon_avg_martingales_masked)

        # Return the aggregated results as numpy arrays
        return {
            "horizon_change_points": horizon_change_points,
            "horizon_sum_martingales": horizon_sum_martingales_masked,
            "horizon_avg_martingales": horizon_avg_martingales_masked,
            "individual_horizon_martingales": individual_horizon_martingales_masked,
            "prediction_start_time": config.history_size,  # Add the time when predictions start
        }

    except Exception as e:
        logger.error(f"Error in multiview horizon martingale computation: {e}")
        logger.error(traceback.format_exc())
        raise RuntimeError(f"Multiview horizon martingale computation failed: {e}")
