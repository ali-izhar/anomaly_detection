# src/changepoint/martingale_traditional.py

"""Traditional martingale implementation for change point detection. 
Traditional martingale uses the current observation and previous history."""

import logging
import traceback
import numpy as np
from typing import Dict, List, Any, Optional

from .martingale_base import (
    MartingaleConfig,
    MartingaleState,
    MultiviewMartingaleState,
    DataPoint,
)

from .betting import (
    get_betting_function,
)
from .strangeness import (
    strangeness_point,
    get_pvalue,
)


logger = logging.getLogger(__name__)


def compute_traditional_martingale(
    data: List[DataPoint],
    config: Optional[MartingaleConfig] = None,
    state: Optional[MartingaleState] = None,
    printing_from_horizon: bool = False,
) -> Dict[str, Any]:
    """Compute a traditional martingale for online change detection over a univariate data stream.

    Uses conformal p-values and a chosen strangeness measure to compute a traditional martingale
    that uses only the current observation with its history.

    Args:
        data: Sequential observations to monitor.
        config: Configuration for martingale computation.
        state: Optional state for continuing computation from a previous run.
        printing_from_horizon: Whether to print detection events from horizon martingale.

    Returns:
        Dictionary containing:
         - "traditional_change_points": List[int] of indices where martingale detected a change.
         - "traditional_martingales": np.ndarray of martingale values over time.

    Raises:
        ValueError: If input validation fails.
        RuntimeError: If computation fails.
    """
    if not data:
        raise ValueError("Empty data sequence")

    if config is None:
        raise ValueError("Config is required")

    if state is None:
        state = MartingaleState()

    # Obtain the betting function callable based on the betting_func_config.
    betting_function = get_betting_function(config.betting_func_config)

    # Log input dimensions and configuration details.
    logger.debug("Single-view Traditional Martingale Input Dimensions:")
    logger.debug(f"  Sequence length: {len(data)}")
    logger.debug(f"  History size: {config.history_size}")
    logger.debug(
        f"  Window size: {config.window_size if config.window_size else 'None'}"
    )
    logger.debug("-" * 50)

    try:
        # Process each point in the data stream.
        for i, point in enumerate(data):
            # Maintain a rolling window if a window_size is set.
            if config.window_size and len(state.window) >= config.window_size:
                state.window = state.window[-config.window_size :]

            # Compute strangeness for the current observation.
            # If the window is empty, default strangeness is set to 0.
            if len(state.window) == 0:
                s_vals = [0.0]
            else:
                # Reshape the window data to 2D array (n_samples, 1)
                window_data = np.array(state.window + [point]).reshape(-1, 1)
                s_vals = strangeness_point(
                    window_data,
                    config=config.strangeness_config,
                )

            # Compute conformal p-value using the strangeness scores.
            pvalue = get_pvalue(s_vals, random_state=config.random_state)

            # Update traditional martingale using the betting function.
            prev_trad = state.traditional_martingale
            new_trad = betting_function(prev_trad, pvalue)

            # Check if the updated traditional martingale exceeds the threshold.
            detected_trad = False
            if config.reset and new_trad > config.threshold:
                if not printing_from_horizon:
                    logger.info(
                        f"Traditional martingale detected change at t={i}: {new_trad:.4f} > {config.threshold}"
                    )
                detected_trad = True
                state.traditional_change_points.append(i)

            # Reset state if a change is detected; otherwise, update state.
            if detected_trad:
                state.reset()
                state.saved_traditional.append(1.0)
                continue
            else:
                state.window.append(point)
                state.traditional_martingale = new_trad
                state.saved_traditional.append(new_trad)

        # Return the computed martingale histories and detected change points.
        return {
            "traditional_change_points": state.traditional_change_points,
            "traditional_martingales": np.array(
                state.saved_traditional[1:], dtype=float
            ),
        }

    except Exception as e:
        logger.error(f"Traditional martingale computation failed: {str(e)}")
        raise RuntimeError(f"Traditional martingale computation failed: {str(e)}")


def multiview_traditional_martingale(
    data: List[List[DataPoint]],
    config: Optional[MartingaleConfig] = None,
    state: Optional[MultiviewMartingaleState] = None,
    batch_size: int = 1000,
    printing_from_horizon: bool = False,
) -> Dict[str, Any]:
    """Compute a multivariate (multiview) traditional martingale test by aggregating evidence across features.

    For d features, each feature maintains its own martingale computed using the traditional update
    (current observation + history). The combined martingale is defined as:
         M_total(n) = sum_{j=1}^{d} M_j(n)
         M_avg(n) = M_total(n) / d
    A change is declared if M_total(n) exceeds the threshold.

    Args:
        data: List of feature sequences to monitor.
        config: Configuration for martingale computation.
        state: Optional state for continuing computation from a previous run.
        batch_size: Size of batches for processing.
        silent: Whether to suppress logging of detection events.

    Returns:
        Dictionary containing change points and martingale values.

    Raises:
        ValueError: If input validation fails.
        RuntimeError: If computation fails.
    """
    if not data or not data[0]:
        raise ValueError("Empty data sequence")

    if config is None:
        raise ValueError("Config is required")

    if state is None:
        state = MultiviewMartingaleState()
        state.reset(len(data))

    # Get the betting function based on the provided configuration.
    betting_function = get_betting_function(config.betting_func_config)

    # Log input dimensions and configuration details.
    logger.debug("Multiview Traditional Martingale Input Dimensions:")
    logger.debug(f"  Number of features: {len(data)}")
    logger.debug(f"  Sequence length per feature: {len(data[0])}")
    logger.debug(f"  History size: {config.history_size}")
    logger.debug(
        f"  Window size: {config.window_size if config.window_size else 'None'}"
    )
    logger.debug(f"  Batch size: {batch_size}")
    logger.debug("-" * 50)

    try:
        num_features = len(data)
        num_samples = len(data[0])

        idx = 0
        while idx < num_samples:
            batch_end = min(idx + batch_size, num_samples)
            logger.debug(
                f"Processing batch [{idx}:{batch_end}]: Batch size = {batch_end - idx}"
            )

            # Process each sample in the current batch.
            for i in range(idx, batch_end):
                # Store previous traditional values before ANY updates within each timestep
                prev_traditional_t_minus_1 = state.traditional_martingales.copy()

                # Reset new_traditional for each timestep
                new_traditional = []

                # Update traditional martingale for each feature
                for j in range(num_features):
                    # Maintain rolling window for feature j if window_size is specified.
                    if (
                        config.window_size
                        and len(state.windows[j]) >= config.window_size
                    ):
                        state.windows[j] = state.windows[j][-config.window_size :]

                    # Compute strangeness for the current observation for feature j.
                    if not state.windows[j]:
                        s_vals = [0.0]
                    else:
                        window_data = np.array(state.windows[j] + [data[j][i]]).reshape(
                            -1, 1
                        )
                        s_vals = strangeness_point(
                            window_data,
                            config=config.strangeness_config,
                        )
                    # Compute p-value and update traditional martingale for feature j using M_{t-1}
                    pv = get_pvalue(s_vals, random_state=config.random_state)
                    prev_val = prev_traditional_t_minus_1[j]  # Use M_{t-1}
                    new_val = betting_function(prev_val, pv)
                    new_traditional.append(new_val)

                    # Update running values for each feature
                    state.traditional_martingales[j] = new_val

                # Aggregate traditional martingale values across all features
                total_traditional = sum(new_traditional)

                # Check if traditional martingale crosses threshold
                if total_traditional > config.threshold:
                    # Traditional martingale detection at t-1 because we're using t-1 values to update t
                    if not printing_from_horizon:
                        logger.info(
                            f"Traditional martingale detected change at t={i - 1}: Sum={total_traditional:.4f} > {config.threshold}"
                        )
                    state.traditional_change_points.append(i)

                    # Record detection values and reset
                    state.record_traditional_values(i, new_traditional, True)
                    state.reset(num_features)
                else:
                    # No detection - update windows and record values
                    for j in range(num_features):
                        state.windows[j].append(data[j][i])

                    # Record current values in history
                    state.record_traditional_values(i, new_traditional, False)

            idx = batch_end

        # Return the aggregated results as numpy arrays.
        return {
            "traditional_change_points": state.traditional_change_points,
            "traditional_sum_martingales": np.array(
                state.traditional_sum[1:], dtype=float
            ),
            "traditional_avg_martingales": np.array(
                state.traditional_avg[1:], dtype=float
            ),
            "individual_traditional_martingales": [
                np.array(m[1:], dtype=float) for m in state.individual_traditional
            ],
        }

    except Exception as e:
        logger.error(f"Error in multiview traditional martingale computation: {e}")
        logger.error(traceback.format_exc())
        raise RuntimeError(f"Multiview traditional martingale computation failed: {e}")
