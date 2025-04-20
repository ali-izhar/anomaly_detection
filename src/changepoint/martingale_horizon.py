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
        # Initialize horizon martingale values
        horizon_martingales = []
        horizon_change_points = []

        # Initialize horizon-specific martingales
        if not hasattr(state, "horizon_martingales_h"):
            num_horizons = (
                len(predicted_data[0])
                if predicted_data and len(predicted_data) > 0
                else 0
            )
            state.horizon_martingales_h = [1.0] * num_horizons

        # Equal weights for horizons by default
        num_horizons = (
            len(predicted_data[0]) if predicted_data and len(predicted_data) > 0 else 0
        )
        horizon_weights = (
            [1.0 / num_horizons] * num_horizons if num_horizons > 0 else []
        )

        # Process each point and compute the horizon component
        for i in range(len(data)):
            if i >= config.history_size:
                # Look up predictions for this point
                pred_idx = i - config.history_size

                # Initialize martingales for each horizon
                horizon_martingales_at_t = []

                # Process each prediction horizon
                for h in range(len(predicted_data[pred_idx])):
                    if len(state.window) == 0:
                        pred_s_vals = [0.0]
                    else:
                        # Reshape the window data and prediction to 2D arrays
                        window_data = np.array(state.window).reshape(-1, 1)
                        pred_data = np.array(predicted_data[pred_idx][h]).reshape(1, -1)
                        pred_s_vals = strangeness_point(
                            np.vstack([window_data, pred_data]),
                            config=config.strangeness_config,
                        )

                    # Calculate p-value for this prediction
                    pred_pvalue = get_pvalue(
                        pred_s_vals, random_state=config.random_state
                    )

                    # Update horizon-h martingale using the betting function
                    # M_{t,h}^{(k)} = M_{t-1,h}^{(k)} * g(p_{t,h}^{(k)})
                    if h < len(state.horizon_martingales_h):
                        m_h = state.horizon_martingales_h[h] * betting_function(
                            1.0, pred_pvalue
                        )
                    else:
                        m_h = betting_function(1.0, pred_pvalue)

                    horizon_martingales_at_t.append(m_h)

                # Combine horizons using weighted sum: M_{t}^{(k)} = \sum_{h \in \mathcal{H}} w_h M_{t,h}^{(k)}
                if horizon_martingales_at_t:
                    horizon_val = sum(
                        w * m for w, m in zip(horizon_weights, horizon_martingales_at_t)
                    )

                    # Update state with horizon-specific martingales
                    state.horizon_martingales_h = horizon_martingales_at_t

                    # Store combined martingale
                    state.horizon_martingale = horizon_val
                    state.saved_horizon.append(horizon_val)
                    horizon_martingales.append(horizon_val)

                    # Check for horizon-based change detection
                    if horizon_val > config.threshold:
                        logger.info(
                            f"Horizon martingale detected change at t={i}: {horizon_val:.4f} > {config.threshold}"
                        )
                        horizon_change_points.append(i)
                else:
                    # No predictions available
                    horizon_val = 1.0
                    horizon_martingales.append(horizon_val)
            else:
                # Not enough history for predictions, use default
                horizon_val = 1.0
                horizon_martingales.append(horizon_val)
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

    For d features, each feature maintains its own martingale for each horizon h:
    M_{t,h}^{(k)} = \prod_{i=1}^t g(p_{i,h}^{(k)})

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
        # Get change points from traditional martingale for reset logic
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

        # Initialize martingale state for each feature-horizon combination
        if not hasattr(state, "feature_horizon_martingales"):
            state.feature_horizon_martingales = [
                [1.0] * num_horizons for _ in range(num_features)
            ]

        # Equal weights by default
        horizon_weights = (
            [1.0 / num_horizons] * num_horizons if num_horizons > 0 else []
        )
        feature_weights = (
            [1.0 / num_features] * num_features if num_features > 0 else []
        )

        idx = 0
        while idx < num_samples:
            batch_end = min(idx + batch_size, num_samples)
            logger.debug(
                f"Processing batch [{idx}:{batch_end}]: Batch size = {batch_end - idx}"
            )

            # Process each sample in the current batch.
            for i in range(idx, batch_end):
                # Initialize feature-level martingales
                feature_martingales = []

                # Compute horizon martingales if we have enough history
                if i >= config.history_size:
                    pred_idx = i - config.history_size

                    # For each feature, compute horizon martingales
                    for j in range(num_features):
                        # Initialize horizon-specific martingales for this feature
                        horizon_martingales_j = []

                        # Process each horizon prediction
                        for h in range(num_horizons):
                            if not state.windows[j]:
                                # After reset, no history to use for predictions
                                m_jh = 1.0
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

                                # Update martingale for this feature-horizon combination
                                # M_{t,h}^{(j)} = M_{t-1,h}^{(j)} * g(p_{t,h}^{(j)})
                                m_jh = state.feature_horizon_martingales[j][
                                    h
                                ] * betting_function(1.0, pred_pv)

                            # Store this horizon's martingale
                            horizon_martingales_j.append(m_jh)

                        # Update state for this feature's horizon martingales
                        state.feature_horizon_martingales[j] = horizon_martingales_j

                        # Combine horizons for this feature using weighted sum
                        # M_{t}^{(j)} = \sum_{h \in \mathcal{H}} w_h M_{t,h}^{(j)}
                        feature_m = sum(
                            w * m
                            for w, m in zip(horizon_weights, horizon_martingales_j)
                        )
                        feature_martingales.append(feature_m)

                        # Store in individual results
                        individual_horizon_martingales[j].append(feature_m)

                    # Combine across features using weighted sum
                    # M_t = \sum_{j=1}^K v_j M_t^{(j)}
                    total_horizon = sum(
                        v * m for v, m in zip(feature_weights, feature_martingales)
                    )
                    avg_horizon = total_horizon / num_features

                    # Store results
                    horizon_sum_martingales.append(total_horizon)
                    horizon_avg_martingales.append(avg_horizon)

                    # Check if horizon martingale crosses threshold
                    if total_horizon > config.threshold:
                        logger.info(
                            f"Horizon martingale detected change at t={i}: "
                            f"Sum={total_horizon:.4f} > {config.threshold}"
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
                    # Reset feature-horizon martingales
                    state.feature_horizon_martingales = [
                        [1.0] * num_horizons for _ in range(num_features)
                    ]

            idx = batch_end

        # Return the aggregated results as numpy arrays
        return {
            "horizon_change_points": horizon_change_points,
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
