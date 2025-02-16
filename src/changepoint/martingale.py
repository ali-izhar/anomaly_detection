# src/changepoint/martingale.py

"""Martingale framework for online change detection using conformal p-values and a chosen strangeness measure.

- Traditional martingale: uses the current observation and previous history.
- Horizon martingale: uses the current observation and multiple predicted future states,
  along with the previous history.

Reset Strategy:
- Traditional martingale resets to 1.0 immediately after detecting a change.
- Horizon martingale only resets when traditional martingale confirms a change.
"""

from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    List,
    Optional,
    TypeVar,
    Union,
    final,
)

import logging
import numpy as np
from numpy import floating, integer

from .betting import (
    BettingFunctionConfig,
    get_betting_function,
)
from .distance import DistanceConfig
from .strangeness import strangeness_point, get_pvalue, StrangenessConfig


logger = logging.getLogger(__name__)

# Type definitions for scalars and arrays
ScalarType = TypeVar("ScalarType", bound=Union[floating, integer])
Array = np.ndarray
DataPoint = Union[List[float], np.ndarray]


@dataclass(frozen=True)
class MartingaleConfig:
    """Configuration for martingale computation.

    Attributes:
        threshold: Detection threshold for martingale values.
        history_size: Minimum number of observations before using predictions.
        reset: Whether to reset after detection.
        window_size: Maximum window size for strangeness computation.
        random_state: Random seed for reproducibility.
        betting_func_config: Configuration for betting function.
        distance_measure: Distance metric for strangeness computation.
        distance_p: Order parameter for Minkowski distance.
        strangeness_config: Configuration for strangeness computation.
    """

    threshold: float
    history_size: int
    reset: bool = True
    window_size: Optional[int] = None
    random_state: Optional[int] = None
    betting_func_config: Optional[BettingFunctionConfig] = None
    distance_measure: str = "euclidean"
    distance_p: float = 2.0
    strangeness_config: Optional[StrangenessConfig] = None

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.threshold <= 0:
            raise ValueError(f"Threshold must be positive, got {self.threshold}")
        if self.history_size < 1:
            raise ValueError(
                f"History size must be at least 1, got {self.history_size}"
            )
        if self.window_size is not None and self.window_size < 1:
            raise ValueError(
                f"Window size must be at least 1 if specified, got {self.window_size}"
            )
        if self.distance_p <= 0:
            raise ValueError(
                f"Distance order parameter must be positive, got {self.distance_p}"
            )


@dataclass
class MartingaleState:
    """State for martingale computation.

    Attributes:
        window: Rolling window of past observations.
        traditional_martingale: Current traditional martingale value.
        horizon_martingale: Current horizon martingale value.
        saved_traditional: History of traditional martingale values.
        saved_horizon: History of horizon martingale values.
        traditional_change_points: Indices where traditional martingale detected changes.
        horizon_change_points: Indices where horizon martingale detected changes.
    """

    window: List[DataPoint] = field(default_factory=list)
    traditional_martingale: float = 1.0
    horizon_martingale: float = 1.0
    saved_traditional: List[float] = field(default_factory=lambda: [1.0])
    saved_horizon: List[float] = field(default_factory=lambda: [1.0])
    traditional_change_points: List[int] = field(default_factory=list)
    horizon_change_points: List[int] = field(default_factory=list)

    def reset(self):
        """Reset martingale state after a detection event."""
        self.window.clear()
        self.traditional_martingale = 1.0
        self.horizon_martingale = 1.0
        # Append reset values to the history for continuity
        self.saved_traditional.append(1.0)
        self.saved_horizon.append(1.0)


@final
def compute_martingale(
    data: List[DataPoint],
    predicted_data: Optional[List[Array]] = None,
    config: Optional[MartingaleConfig] = None,
    state: Optional[MartingaleState] = None,
) -> Dict[str, Any]:
    """Compute a martingale for online change detection over a univariate data stream.

    Uses conformal p-values and a chosen strangeness measure to compute two martingale streams:
      1. Traditional martingale: uses only the current observation with its history.
      2. Horizon martingale: uses the current observation plus predicted future states.

    Args:
        data: Sequential observations to monitor.
        predicted_data: Optional list of predicted feature vectors for future timesteps.
        config: Configuration for martingale computation.
        state: Optional state for continuing computation from a previous run.

    Returns:
        Dictionary containing:
         - "traditional_change_points": List[int] of indices where traditional martingale detected a change.
         - "horizon_change_points": List[int] of indices where horizon martingale detected a change.
         - "traditional_martingales": np.ndarray of traditional martingale values over time.
         - "horizon_martingales": np.ndarray of horizon martingale values (if predictions provided).

    Raises:
        ValueError: If input validation fails.
        RuntimeError: If computation fails.
    """
    if not data:
        raise ValueError("Empty data sequence")

    # Use provided config or set default values.
    if config is None:
        config = MartingaleConfig(
            threshold=1.0,
            history_size=10,
            reset=True,
            window_size=None,
            random_state=None,
            betting_func_config=BettingFunctionConfig(
                name="power",
                params={"epsilon": 0.7},
            ),
            distance_measure="euclidean",
            distance_p=2.0,
            strangeness_config=StrangenessConfig(
                n_clusters=1,
                batch_size=None,
                random_state=None,
                distance_config=DistanceConfig(metric="euclidean"),
            ),
        )

    # Initialize state if not provided.
    if state is None:
        state = MartingaleState()

    # Obtain the betting function callable based on the betting_func_config.
    betting_function = get_betting_function(config.betting_func_config)

    # Log input dimensions and configuration details.
    logger.debug("Single-view Martingale Input Dimensions:")
    logger.debug(f"  Sequence length: {len(data)}")
    if predicted_data:
        logger.debug(f"  Number of predictions: {len(predicted_data)}")
        logger.debug(f"  Predictions per timestep: {len(predicted_data[0])}")
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
                logger.info(
                    f"Traditional martingale detected change at t={i}: {new_trad:.4f} > {config.threshold}"
                )
                detected_trad = True
                state.traditional_change_points.append(i)

            # Update horizon martingale if predictions are available.
            new_horizon = None
            if predicted_data is not None and i >= config.history_size:
                pred_idx = i - config.history_size
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
                    horizon_update_factor = betting_function(1.0, pred_pvalue)

                new_horizon = prev_trad * horizon_update_factor
                if new_horizon > config.threshold:
                    logger.info(
                        f"Horizon martingale detected change at t={i}: {new_horizon:.4f} > {config.threshold}"
                    )
                    state.horizon_change_points.append(i)
            elif predicted_data is not None:
                # If predictions exist but not enough history, default horizon equals traditional.
                new_horizon = prev_trad

            # Reset state if a change is detected; otherwise, update state.
            if detected_trad:
                state.reset()
                # Save the reset value (1.0) before continuing
                state.saved_traditional.append(1.0)
                if new_horizon is not None:
                    state.saved_horizon.append(1.0)
                # Skip updating state for this point since we just reset
                continue
            else:
                state.window.append(point)
                state.traditional_martingale = new_trad
                state.saved_traditional.append(new_trad)
                if new_horizon is not None:
                    state.horizon_martingale = new_horizon
                    state.saved_horizon.append(new_horizon)

            # Periodically log the current state.
            if i > 0 and i % 10 == 0:
                logger.debug(
                    f"t={i}: window size={len(state.window)}, traditional M={state.traditional_martingale:.4f}"
                )
                if predicted_data is not None and i >= config.history_size:
                    logger.debug(f"t={i}: horizon M={state.horizon_martingale:.4f}")
                logger.debug("-" * 30)

        logger.debug(
            f"Martingale computation complete. Traditional change points: {len(state.traditional_change_points)}; "
            f"Horizon change points: {len(state.horizon_change_points)}."
        )

        # Return the computed martingale histories and detected change points.
        return {
            "traditional_change_points": state.traditional_change_points,
            "horizon_change_points": state.horizon_change_points,
            "traditional_martingales": np.array(
                state.saved_traditional[1:], dtype=float
            ),
            "horizon_martingales": (
                np.array(state.saved_horizon[1:], dtype=float)
                if predicted_data is not None
                else None
            ),
        }

    except Exception as e:
        logger.error(f"Martingale computation failed: {str(e)}")
        raise RuntimeError(f"Martingale computation failed: {str(e)}")


@dataclass
class MultiviewMartingaleState:
    """State for multiview martingale computation.

    Attributes:
        windows: List of rolling windows for each feature.
        traditional_martingales: Current traditional martingale values per feature.
        horizon_martingales: Current horizon martingale values per feature.
        traditional_sum: Sum of traditional martingales across features.
        horizon_sum: Sum of horizon martingales across features.
        traditional_avg: Average of traditional martingales.
        horizon_avg: Average of horizon martingales.
        traditional_change_points: Indices where traditional martingale detected changes.
        horizon_change_points: Indices where horizon martingale detected changes.
        individual_traditional: Martingale history for each individual feature.
        individual_horizon: Horizon martingale history for each individual feature.
    """

    windows: List[List[DataPoint]] = field(default_factory=list)
    traditional_martingales: List[float] = field(default_factory=list)
    horizon_martingales: List[float] = field(default_factory=list)
    traditional_sum: List[float] = field(default_factory=lambda: [1.0])
    horizon_sum: List[float] = field(default_factory=lambda: [1.0])
    traditional_avg: List[float] = field(default_factory=lambda: [1.0])
    horizon_avg: List[float] = field(default_factory=lambda: [1.0])
    traditional_change_points: List[int] = field(default_factory=list)
    horizon_change_points: List[int] = field(default_factory=list)
    individual_traditional: List[List[float]] = field(default_factory=list)
    individual_horizon: List[List[float]] = field(default_factory=list)

    def __post_init__(self):
        """Initialize state lists if they are not already set."""
        if not self.windows:
            self.windows = []
        if not self.traditional_martingales:
            self.traditional_martingales = []
        if not self.horizon_martingales:
            self.horizon_martingales = []
        if not self.individual_traditional:
            self.individual_traditional = []
        if not self.individual_horizon:
            self.individual_horizon = []

    def reset(self, num_features: int):
        """Reset state for all features.

        Args:
            num_features: Number of features to initialize.
        """
        # Reset each feature's rolling window.
        self.windows = [[] for _ in range(num_features)]
        # Reset martingale values for each feature to 1.0.
        self.traditional_martingales = [1.0] * num_features
        self.horizon_martingales = [1.0] * num_features
        # Update overall sum and average with the reset values.
        self.traditional_sum.append(float(num_features))
        self.horizon_sum.append(float(num_features))
        self.traditional_avg.append(1.0)
        self.horizon_avg.append(1.0)
        # Reset individual martingale histories per feature.
        for j in range(num_features):
            if len(self.individual_traditional) <= j:
                self.individual_traditional.append([1.0])
            else:
                self.individual_traditional[j].append(1.0)
            if len(self.individual_horizon) <= j:
                self.individual_horizon.append([1.0])
            else:
                self.individual_horizon[j].append(1.0)


@final
def multiview_martingale_test(
    data: List[List[DataPoint]],
    predicted_data: Optional[List[List[Array]]] = None,
    config: Optional[MartingaleConfig] = None,
    state: Optional[MultiviewMartingaleState] = None,
    batch_size: int = 1000,
) -> Dict[str, Any]:
    """Compute a multivariate (multiview) martingale test by aggregating evidence across features.

    For d features, each feature maintains its own martingale computed using the traditional update
    (current observation + history) and the horizon update (if predictions are provided).
    The combined martingale is defined as:
         M_total(n) = sum_{j=1}^{d} M_j(n)
         M_avg(n) = M_total(n) / d
    A change is declared if M_total(n) exceeds the threshold.

    Args:
        data: List of feature sequences to monitor.
        predicted_data: Optional list of predicted feature vectors for each feature.
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

    # Use provided configuration or default settings.
    if config is None:
        config = MartingaleConfig(
            threshold=1.0,
            history_size=10,
            reset=True,
            window_size=None,
            random_state=None,
            betting_func_config=BettingFunctionConfig(
                name="power",
                params={"epsilon": 0.7},
            ),
            distance_measure="euclidean",
            distance_p=2.0,
            strangeness_config=StrangenessConfig(
                n_clusters=1,
                batch_size=None,
                random_state=None,
                distance_config=DistanceConfig(metric="euclidean"),
            ),
        )

    # Initialize state if not provided.
    if state is None:
        state = MultiviewMartingaleState()
        state.reset(len(data))

    # Get the betting function based on the provided configuration.
    betting_function = get_betting_function(config.betting_func_config)

    # Log input dimensions and configuration details.
    logger.debug("Multiview Martingale Input Dimensions:")
    logger.debug(f"  Number of features: {len(data)}")
    logger.debug(f"  Sequence length per feature: {len(data[0])}")
    if predicted_data:
        logger.debug(f"  Number of prediction timesteps: {len(predicted_data[0])}")
        logger.debug(f"  Predictions per timestep: {len(predicted_data[0][0])}")
    logger.debug(f"  History size: {config.history_size}")
    logger.debug(
        f"  Window size: {config.window_size if config.window_size else 'None'}"
    )
    logger.debug(f"  Batch size: {batch_size}")
    logger.debug("-" * 50)

    try:
        num_features = len(data)
        num_samples = len(data[0])
        num_horizons = len(predicted_data[0][0]) if predicted_data is not None else 0

        idx = 0
        while idx < num_samples:
            batch_end = min(idx + batch_size, num_samples)
            logger.debug(
                f"Processing batch [{idx}:{batch_end}]: Batch size = {batch_end - idx}"
            )

            # Process each sample in the current batch.
            for i in range(idx, batch_end):
                # Store previous traditional values before ANY updates
                prev_traditional_t_minus_1 = state.traditional_martingales.copy()
                new_traditional = []

                # Update traditional martingale for each feature using M_{t-1}
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

                # Aggregate traditional martingale values across all features
                total_traditional = sum(new_traditional)
                avg_traditional = total_traditional / num_features
                state.traditional_sum.append(total_traditional)
                state.traditional_avg.append(avg_traditional)

                # Update individual traditional martingales history for each feature
                for j in range(num_features):
                    state.traditional_martingales[j] = new_traditional[j]
                    state.individual_traditional[j].append(new_traditional[j])

                # Update horizon martingale if predictions are available
                new_horizon = []
                if predicted_data is not None and i >= config.history_size:
                    pred_idx = i - config.history_size
                    for j in range(num_features):
                        # Use M_{t-1} for horizon computation
                        prev_trad = prev_traditional_t_minus_1[j]  # Use M_{t-1}

                        # Initialize list to store horizon betting factors
                        horizon_factors = []
                        total_weight = 0.0
                        decay_rate = -0.3  # Exponential decay rate for horizon weights

                        # Process each horizon prediction for feature j
                        for h in range(num_horizons):
                            if not state.windows[j]:
                                # After reset, set horizon value to 1.0
                                new_horizon_val = 1.0
                                break
                            else:
                                window_data = np.array(state.windows[j]).reshape(-1, 1)
                                pred_data = np.array(
                                    predicted_data[j][pred_idx][h]
                                ).reshape(1, -1)
                                pred_s_val = strangeness_point(
                                    np.vstack([window_data, pred_data]),
                                    config=config.strangeness_config,
                                )
                                pred_pv = get_pvalue(
                                    pred_s_val, random_state=config.random_state
                                )
                                # Store individual betting factors
                                factor = betting_function(1.0, pred_pv)
                                weight = np.exp(
                                    decay_rate * h
                                )  # Earlier horizons get more weight
                                horizon_factors.append((factor, weight))
                                total_weight += weight

                        # Compute weighted geometric mean of horizon factors
                        if len(state.windows[j]) > 0 and horizon_factors:
                            # Center factors around 1.0 to prevent aggressive growth
                            centered_factors = [
                                (f - 1.0, w) for f, w in horizon_factors
                            ]

                            # Compute weighted average of centered factors
                            avg_factor = (
                                sum(f * w for f, w in centered_factors) / total_weight
                            )

                            # Map back to multiplicative space and apply mild dampening
                            horizon_factor = np.exp(avg_factor)

                            # Add consistency bonus if all factors suggest change
                            if all(f > 1.1 for f, _ in horizon_factors):
                                horizon_factor *= (
                                    1.15  # 15% boost for consistent signals
                                )

                            # Limit growth to 2.5x the previous value
                            horizon_factor = min(horizon_factor, 2.5)

                            # Compute final horizon value using previous traditional
                            new_horizon_val = prev_trad * horizon_factor
                        else:
                            new_horizon_val = 1.0  # Reset case
                        new_horizon.append(new_horizon_val)

                # Aggregate horizon martingale values
                total_horizon = sum(new_horizon)
                avg_horizon = total_horizon / num_features
                state.horizon_sum.append(total_horizon)
                state.horizon_avg.append(avg_horizon)

                # Record horizon detection (but don't reset)
                if total_horizon > config.threshold:
                    logger.info(
                        f"Horizon martingale detected change at t={i}: Sum={total_horizon:.4f} > {config.threshold}"
                    )
                    state.horizon_change_points.append(i)

                # Handle reset logic - only reset on traditional martingale detection
                if total_traditional > config.threshold:
                    logger.info(
                        f"Traditional martingale detected change at t={i}: Sum={total_traditional:.4f} > {config.threshold}"
                    )
                    state.traditional_change_points.append(i)

                    # Save detection values before reset
                    for j in range(num_features):
                        # Save the detection values
                        state.individual_traditional[j].append(new_traditional[j])
                        if new_horizon:
                            state.individual_horizon[j].append(new_horizon[j])
                        else:
                            state.individual_horizon[j].append(new_traditional[j])

                    # Reset state
                    state.reset(num_features)
                else:
                    # No detection: update windows and continue martingale sequences
                    for j in range(num_features):
                        state.windows[j].append(data[j][i])

                        if predicted_data is not None and i >= config.history_size:
                            # Update running values
                            state.horizon_martingales[j] = new_horizon[j]
                            # Save to history
                            if len(state.individual_horizon) <= j:
                                state.individual_horizon.append([])
                            state.individual_horizon[j].append(new_horizon[j])

                # Periodically log the current state.
                if i > 0 and i % 10 == 0:
                    logger.debug(
                        f"t={i}: Avg window size={np.mean([len(w) for w in state.windows]):.1f}, "
                        f"Traditional Sum={state.traditional_sum[-1]:.4f}, "
                        f"Horizon Sum={state.horizon_sum[-1]:.4f}"
                    )

            logger.debug(
                f"Completed batch: Traditional Sum={state.traditional_sum[-1]:.4f}, "
                f"Change points={len(state.traditional_change_points)}"
            )
            idx = batch_end

        # Return the aggregated results as numpy arrays.
        return {
            "traditional_change_points": state.traditional_change_points,
            "horizon_change_points": state.horizon_change_points,
            "traditional_sum_martingales": np.array(
                state.traditional_sum[1:], dtype=float
            ),
            "traditional_avg_martingales": np.array(
                state.traditional_avg[1:], dtype=float
            ),
            "horizon_sum_martingales": np.array(state.horizon_sum[1:], dtype=float),
            "horizon_avg_martingales": np.array(state.horizon_avg[1:], dtype=float),
            "individual_traditional_martingales": [
                np.array(m[1:], dtype=float) for m in state.individual_traditional
            ],
            "individual_horizon_martingales": [
                np.array(m[1:], dtype=float) for m in state.individual_horizon
            ],
        }

    except Exception as e:
        logger.error(f"Multiview martingale computation failed: {str(e)}")
        raise RuntimeError(f"Multiview martingale computation failed: {str(e)}")
