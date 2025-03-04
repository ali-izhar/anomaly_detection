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
import traceback
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
        current_timestep: The current timestep being processed.
        has_detection: Flag indicating if a detection has occurred at the current timestep.
        last_detection_time: The timestep of the last detection (either horizon or traditional).
        cooldown_period: Number of timesteps to enforce reduced sensitivity after a detection.
        early_warnings: Timesteps where early warnings were triggered before official detection.
        previous_horizon_sum: The horizon sum value from the previous timestep for growth calculation.
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
    current_timestep: int = 0
    has_detection: bool = False
    last_detection_time: int = -40  # Initialize to effectively no previous detection
    cooldown_period: int = 30  # Minimum timesteps between detections
    early_warnings: List[int] = field(default_factory=list)
    previous_horizon_sum: float = 1.0

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

    def record_values(
        self,
        timestep: int,
        traditional_values: List[float],
        horizon_values: List[float] = None,
        is_detection: bool = False,
    ):
        """Record martingale values at specific timestep.

        This method ensures that values are properly recorded at the correct timestep
        and handles padding for missing values.

        Args:
            timestep: The timestep to record values for
            traditional_values: List of traditional martingale values per feature
            horizon_values: Optional list of horizon martingale values per feature
            is_detection: Whether this recording is for a detection event
        """
        self.current_timestep = timestep
        num_features = len(traditional_values)

        # Calculate sums and averages
        total_traditional = sum(traditional_values)
        avg_traditional = total_traditional / num_features

        # Ensure lists are long enough
        while len(self.traditional_sum) <= timestep:
            self.traditional_sum.append(1.0)
        while len(self.traditional_avg) <= timestep:
            self.traditional_avg.append(1.0)

        # Record traditional values
        self.traditional_sum[timestep] = total_traditional
        self.traditional_avg[timestep] = avg_traditional

        # Update individual traditional martingales
        for j in range(num_features):
            while len(self.individual_traditional) <= j:
                self.individual_traditional.append([1.0])
            while len(self.individual_traditional[j]) <= timestep:
                self.individual_traditional[j].append(1.0)
            self.individual_traditional[j][timestep] = traditional_values[j]

        # If horizon values provided, record them too
        if horizon_values:
            total_horizon = sum(horizon_values)
            avg_horizon = total_horizon / num_features

            while len(self.horizon_sum) <= timestep:
                self.horizon_sum.append(1.0)
            while len(self.horizon_avg) <= timestep:
                self.horizon_avg.append(1.0)

            self.horizon_sum[timestep] = total_horizon
            self.horizon_avg[timestep] = avg_horizon

            # Update individual horizon martingales
            for j in range(num_features):
                while len(self.individual_horizon) <= j:
                    self.individual_horizon.append([1.0])
                while len(self.individual_horizon[j]) <= timestep:
                    self.individual_horizon[j].append(1.0)
                self.individual_horizon[j][timestep] = horizon_values[j]

        # If this is a detection event, mark it
        if is_detection:
            self.has_detection = True

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

        # Reset detection flag
        self.has_detection = False

        # Add reset values to history for continuity
        current_t = self.current_timestep

        # Since we're resetting after the current timestep, we need to add reset values
        # for the next timestep
        next_t = current_t + 1

        # Update overall sum and average with the reset values
        while len(self.traditional_sum) <= next_t:
            self.traditional_sum.append(1.0)
        while len(self.traditional_avg) <= next_t:
            self.traditional_avg.append(1.0)
        while len(self.horizon_sum) <= next_t:
            self.horizon_sum.append(1.0)
        while len(self.horizon_avg) <= next_t:
            self.horizon_avg.append(1.0)

        self.traditional_sum[next_t] = float(num_features)
        self.horizon_sum[next_t] = float(num_features)
        self.traditional_avg[next_t] = 1.0
        self.horizon_avg[next_t] = 1.0

        # Reset individual martingale histories per feature
        for j in range(num_features):
            while len(self.individual_traditional) <= j:
                self.individual_traditional.append([1.0])
            while len(self.individual_traditional[j]) <= next_t:
                self.individual_traditional[j].append(1.0)
            self.individual_traditional[j][next_t] = 1.0

            while len(self.individual_horizon) <= j:
                self.individual_horizon.append([1.0])
            while len(self.individual_horizon[j]) <= next_t:
                self.individual_horizon[j].append(1.0)
            self.individual_horizon[j][next_t] = 1.0


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

                    # Update running values for each feature
                    state.traditional_martingales[j] = new_val

                # Aggregate traditional martingale values across all features
                total_traditional = sum(new_traditional)
                # avg_traditional = total_traditional / num_features

                # Initialize horizon martingale variables
                new_horizon = []
                total_horizon = 0
                # avg_horizon = 0
                horizon_detection = False

                # Compute horizon martingales if predictions are available
                if predicted_data is not None and i >= config.history_size:
                    pred_idx = i - config.history_size

                    # For each feature, compute the horizon martingale
                    for j in range(num_features):
                        # As defined in paper: horizon martingale starts from traditional M_{t-1}
                        # and applies betting factors from predicted states
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

                    # Calculate aggregated horizon values
                    total_horizon = sum(new_horizon)
                    # avg_horizon = total_horizon / num_features

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
                        state.early_warnings.append(i)

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
                        state.horizon_change_points.append(i)
                        state.last_detection_time = i

                    # Calculate aggregated traditional values
                    total_traditional = sum(new_traditional)
                    # avg_traditional = total_traditional / num_features

                    # Check if traditional martingale crosses threshold
                    if total_traditional > config.threshold:
                        traditional_detection = True
                        logger.info(
                            f"Traditional martingale detected change at t={i}: "
                            f"Sum={total_traditional:.4f} > {config.threshold:.1f}"
                        )
                        state.traditional_change_points.append(i)
                        state.last_detection_time = i

                # Detection logic and state update
                if total_traditional > config.threshold:
                    # Traditional martingale detection
                    logger.info(
                        f"Traditional martingale detected change at t={i}: Sum={total_traditional:.4f} > {config.threshold}"
                    )
                    state.traditional_change_points.append(i)

                    # Per the paper's logic, if traditional detects a change,
                    # horizon must also detect the change at the same timestep
                    # since horizon builds on traditional
                    if (
                        predicted_data is not None
                        and not horizon_detection
                        and total_horizon > 0
                    ):
                        logger.info(
                            f"Horizon martingale also detected change at t={i}: Sum={total_horizon:.4f} > {config.threshold} (through traditional detection)"
                        )
                        state.horizon_change_points.append(i)
                        horizon_detection = True

                    # Record detection values
                    state.record_values(
                        i, new_traditional, new_horizon if new_horizon else None, True
                    )

                    # Reset state after detection
                    state.reset(num_features)
                elif horizon_detection:
                    # Horizon-only detection (no reset, just record)
                    state.record_values(i, new_traditional, new_horizon, True)

                    # Don't reset on horizon-only detection
                    # just update windows and continue monitoring
                    for j in range(num_features):
                        state.windows[j].append(data[j][i])
                else:
                    # No detection - update windows and record values
                    for j in range(num_features):
                        state.windows[j].append(data[j][i])

                    # Record current values in history
                    state.record_values(
                        i, new_traditional, new_horizon if new_horizon else None, False
                    )

                # Log periodically
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
            "early_warnings": state.early_warnings,
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
        logger.error(f"Error in multiview martingale computation: {e}")
        logger.error(traceback.format_exc())
        raise RuntimeError(f"Multiview martingale computation failed: {e}")
