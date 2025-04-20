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
) -> Dict[str, Any]:
    """Compute a horizon martingale for online change detection over a univariate data stream.

    Uses conformal p-values and a chosen strangeness measure to compute horizon martingale
    that uses the current observation plus predicted future states.

    Args:
        data: Sequential observations to monitor.
        predicted_data: List of predicted feature vectors for future timesteps.
        config: Configuration for martingale computation.
        state: Optional state for continuing computation from a previous run.

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
        # First compute traditional martingale to use as a foundation
        trad_result = compute_traditional_martingale(data, config, state)
        trad_martingales = trad_result["traditional_martingales"]

        # Initialize horizon martingale values
        horizon_martingales = []
        horizon_change_points = []

        # Process each point and compute the horizon component
        for i in range(len(data)):
            if i >= config.history_size:
                # Get the traditional martingale value for this point
                trad_val = trad_martingales[i] if i < len(trad_martingales) else 1.0

                # Look up predictions for this point
                pred_idx = i - config.history_size
                horizon_update_factor = 1.0

                # Process each prediction
                for j in range(len(predicted_data[pred_idx])):
                    if len(state.window) == 0:
                        pred_s_vals = [0.0]
                    else:
                        # Reshape the window data and prediction to 2D arrays
                        window_data = np.array(state.window).reshape(-1, 1)
                        pred_data = np.array(predicted_data[pred_idx][j]).reshape(1, -1)
                        pred_s_vals = strangeness_point(
                            np.vstack([window_data, pred_data]),
                            config=config.strangeness_config,
                        )
                    pred_pvalue = get_pvalue(
                        pred_s_vals, random_state=config.random_state
                    )
                    # Update the horizon factor with this prediction
                    horizon_update_factor *= betting_function(1.0, pred_pvalue)

                # Final horizon martingale is the traditional value times the horizon factor
                horizon_val = trad_val * horizon_update_factor
                horizon_martingales.append(horizon_val)

                # Check for horizon-based change detection
                if horizon_val > config.threshold:
                    logger.info(
                        f"Horizon martingale detected change at t={i}: {horizon_val:.4f} > {config.threshold}"
                    )
                    horizon_change_points.append(i)
            else:
                # If not enough history, horizon equals traditional
                horizon_val = trad_martingales[i] if i < len(trad_martingales) else 1.0
                horizon_martingales.append(horizon_val)

            # Update window (state is maintained by traditional martingale)
            if i < len(state.window):
                state.horizon_martingale = horizon_val
                state.saved_horizon.append(horizon_val)

        logger.debug(
            f"Horizon martingale computation complete. "
            f"Horizon change points: {len(horizon_change_points)}."
        )

        # Return the computed martingale histories and detected change points
        return {
            "horizon_change_points": horizon_change_points,
            "horizon_martingales": np.array(horizon_martingales, dtype=float),
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
) -> Dict[str, Any]:
    """Compute a multivariate (multiview) horizon martingale test by aggregating evidence across features.

    For d features, each feature uses the traditional martingale as a starting point
    and applies the horizon update (using predictions).
    The combined martingale is defined as:
         M_total(n) = sum_{j=1}^{d} M_j(n)
         M_avg(n) = M_total(n) / d
    A change is declared if M_total(n) exceeds the threshold.

    Args:
        data: List of feature sequences to monitor.
        predicted_data: List of predicted feature vectors for each feature.
        config: Configuration for martingale computation.
        state: Optional state for continuing computation from a previous run.
        batch_size: Size of batches for processing.

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
        # First compute traditional multiview martingale to use as foundation
        trad_result = multiview_traditional_martingale(data, config, None, batch_size)
        trad_change_points = trad_result["traditional_change_points"]

        num_features = len(data)
        num_samples = len(data[0])
        num_horizons = len(predicted_data[0][0])

        # Initialize results
        horizon_sum_martingales = []
        horizon_avg_martingales = []
        individual_horizon_martingales = [[] for _ in range(num_features)]
        horizon_change_points = []
        early_warnings = []

        idx = 0
        while idx < num_samples:
            batch_end = min(idx + batch_size, num_samples)
            logger.debug(
                f"Processing batch [{idx}:{batch_end}]: Batch size = {batch_end - idx}"
            )

            # Process each sample in the current batch.
            for i in range(idx, batch_end):
                # Get the traditional martingale values for this timestep
                if (
                    hasattr(state, "traditional_martingales")
                    and len(state.traditional_martingales) == num_features
                ):
                    prev_traditional_t_minus_1 = state.traditional_martingales
                else:
                    # If traditional values not available, use default values
                    prev_traditional_t_minus_1 = [1.0] * num_features

                # Initialize horizon martingale variables
                new_horizon = []
                total_horizon = 0
                horizon_detection = False

                # Compute horizon martingales if we have enough history
                if i >= config.history_size:
                    pred_idx = i - config.history_size

                    # For each feature, compute the horizon martingale
                    for j in range(num_features):
                        # Use traditional martingale value as starting point
                        prev_trad = prev_traditional_t_minus_1[j]

                        # Initialize horizon factors calculation
                        horizon_factors = []
                        total_weight = 0.0
                        # Reduce dampening: make decay rate less negative to give more weight to later horizons
                        decay_rate = -0.15  # Original was -0.3

                        # Process each horizon prediction
                        for h in range(num_horizons):
                            if not state.windows[j]:
                                # After reset, no history to use for predictions
                                new_horizon_val = 1.0
                                break
                            else:
                                # Get window data and predicted data
                                window_data = np.array(state.windows[j]).reshape(-1, 1)
                                pred_data = np.array(
                                    predicted_data[j][pred_idx][h]
                                ).reshape(1, -1)

                                # Compute strangeness and p-value for prediction
                                pred_s_val = strangeness_point(
                                    np.vstack([window_data, pred_data]),
                                    config=config.strangeness_config,
                                )
                                pred_pv = get_pvalue(
                                    pred_s_val, random_state=config.random_state
                                )

                                # Calculate betting factor with decay weight
                                factor = betting_function(1.0, pred_pv)
                                weight = np.exp(decay_rate * h)
                                horizon_factors.append((factor, weight))
                                total_weight += weight

                        # Compute final horizon martingale value
                        if len(state.windows[j]) > 0 and horizon_factors:
                            # Less dampening: use factors more directly, with less centering effect
                            # Original centered everything around 1.0: [(f - 1.0, w) for f, w in horizon_factors]
                            # Now we only partially adjust toward 1.0, preserving more of the signal
                            centered_factors = [
                                (f * 0.9 + 0.1, w) for f, w in horizon_factors
                            ]

                            # Weighted average of centered factors
                            avg_factor = (
                                sum(f * w for f, w in centered_factors) / total_weight
                            )

                            # Apply mild dampening and consistency bonus
                            horizon_factor = np.exp(avg_factor)

                            # Minimum factor threshold to reduce noise-triggered growth
                            # Only allow growth if the signal is strong enough
                            if (
                                avg_factor < 0.05
                            ):  # Require at least a 5% average signal
                                horizon_factor = 1.0
                            # Lower threshold for consistency bonus (was 1.1, now 1.05)
                            elif all(f > 1.05 for f, _ in horizon_factors):
                                # Increased consistency bonus from 30% to 35%
                                horizon_factor *= 1.35

                            # Check for significant confidence in prediction
                            strong_signal = sum(
                                1 for f, _ in horizon_factors if f > 1.1
                            ) / len(horizon_factors)

                            # If less than 25% of horizons show strong signal, dampen the growth (reduced from 30%)
                            if strong_signal < 0.25:
                                horizon_factor = min(horizon_factor, 1.5)

                            # Increased growth limit from 2.5x to 4.5x
                            horizon_factor = min(horizon_factor, 4.5)

                            # Final horizon value uses previous traditional martingale as starting point
                            new_horizon_val = prev_trad * horizon_factor
                        else:
                            new_horizon_val = 1.0

                        new_horizon.append(new_horizon_val)
                        state.horizon_martingales[j] = new_horizon_val
                        individual_horizon_martingales[j].append(new_horizon_val)

                    # Calculate aggregated horizon values
                    total_horizon = sum(new_horizon)
                    avg_horizon = total_horizon / num_features

                    horizon_sum_martingales.append(total_horizon)
                    horizon_avg_martingales.append(avg_horizon)

                    # Check for cooldown period to reduce false positives
                    in_cooldown = i - state.last_detection_time < state.cooldown_period
                    cooldown_factor = max(
                        0,
                        (state.cooldown_period - (i - state.last_detection_time))
                        / state.cooldown_period,
                    )
                    cooldown_threshold = config.threshold * (
                        1.0 + 0.5 * cooldown_factor
                    )

                    # Calculate growth rate for early warning system
                    horizon_growth_rate = (
                        total_horizon / state.previous_horizon_sum
                        if state.previous_horizon_sum > 0
                        else 1.0
                    )

                    # Early warning detection based on growth rate
                    if (
                        not horizon_detection
                        and not in_cooldown
                        and horizon_growth_rate > 2.0  # Growth doubled
                        and total_horizon > config.threshold * 0.6
                    ):  # At least 60% of threshold
                        logger.info(
                            f"Early warning at t={i}: Horizon martingale growing rapidly "
                            f"({horizon_growth_rate:.2f}x growth) and approaching threshold "
                            f"(Sum={total_horizon:.4f}, {(total_horizon/config.threshold*100):.1f}% of threshold)"
                        )
                        early_warnings.append(i)

                    # Store current value for next iteration's growth calculation
                    state.previous_horizon_sum = total_horizon

                    # Check if horizon martingale crosses threshold
                    # During cooldown period, require a higher threshold
                    # Lower threshold for horizon martingales by 15% to enable earlier detection
                    horizon_threshold = (
                        cooldown_threshold if in_cooldown else config.threshold * 0.85
                    )

                    if total_horizon > horizon_threshold:
                        horizon_detection = True
                        logger.info(
                            f"Horizon martingale detected change at t={i}: "
                            f"Sum={total_horizon:.4f} > {horizon_threshold:.1f}"
                        )
                        horizon_change_points.append(i)
                        state.last_detection_time = i
                else:
                    # Not enough history yet, use 1.0 default values
                    for j in range(num_features):
                        individual_horizon_martingales[j].append(1.0)
                    horizon_sum_martingales.append(num_features)
                    horizon_avg_martingales.append(1.0)

                # If this time step had a traditional detection, apply a reset
                if i in trad_change_points:
                    state.reset(num_features)

            idx = batch_end

        # Return the aggregated results as numpy arrays
        return {
            "horizon_change_points": horizon_change_points,
            "early_warnings": early_warnings,
            "horizon_sum_martingales": np.array(horizon_sum_martingales, dtype=float),
            "horizon_avg_martingales": np.array(horizon_avg_martingales, dtype=float),
            "individual_horizon_martingales": [
                np.array(m, dtype=float) for m in individual_horizon_martingales
            ],
        }

    except Exception as e:
        logger.error(f"Error in multiview horizon martingale computation: {e}")
        logger.error(traceback.format_exc())
        raise RuntimeError(f"Multiview horizon martingale computation failed: {e}")
