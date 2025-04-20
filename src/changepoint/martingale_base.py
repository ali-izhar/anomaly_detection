# src/changepoint/martingale_base.py

"""Base components and interfaces for martingale-based change detection."""

from dataclasses import dataclass, field
from typing import (
    List,
    Optional,
    Protocol,
    TypeVar,
    Union,
)

import logging
import numpy as np
from numpy import floating, integer

from .betting import BettingFunctionConfig
from .strangeness import StrangenessConfig


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
        reset_on_traditional: Whether horizon martingales should reset when traditional detects a change.
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
    reset_on_traditional: bool = False
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
    """Base state for single-view martingale computation.

    Attributes:
        window: Rolling window of past observations.
        traditional_martingale: Current traditional martingale value.
        saved_traditional: History of traditional martingale values.
        traditional_change_points: Indices where traditional martingale detected changes.
    """

    window: List[DataPoint] = field(default_factory=list)
    traditional_martingale: float = 1.0
    saved_traditional: List[float] = field(default_factory=lambda: [1.0])
    traditional_change_points: List[int] = field(default_factory=list)

    def reset(self):
        """Reset martingale state after a detection event."""
        self.window.clear()
        self.traditional_martingale = 1.0
        # Append reset values to the history for continuity
        self.saved_traditional.append(1.0)


@dataclass
class HorizonMartingaleState(MartingaleState):
    """State for horizon martingale computation, extending base martingale state.

    Attributes:
        horizon_martingale: Current horizon martingale value.
        saved_horizon: History of horizon martingale values.
        horizon_change_points: Indices where horizon martingale detected changes.
        horizon_martingales_h: List of martingale values, one per horizon.
    """

    horizon_martingale: float = 1.0
    saved_horizon: List[float] = field(default_factory=lambda: [1.0])
    horizon_change_points: List[int] = field(default_factory=list)
    horizon_martingales_h: List[float] = field(default_factory=list)

    def reset(self):
        """Reset martingale state after a detection event."""
        super().reset()
        self.horizon_martingale = 1.0
        # Maintain horizon-specific martingales but reset them
        self.horizon_martingales_h = (
            [1.0] * len(self.horizon_martingales_h)
            if self.horizon_martingales_h
            else []
        )
        # Append reset values to the history for continuity
        self.saved_horizon.append(1.0)


class MartingaleResult(Protocol):
    """Protocol defining the result format for martingale computations."""

    traditional_change_points: List[int]
    traditional_martingales: np.ndarray


class HorizonMartingaleResult(MartingaleResult, Protocol):
    """Protocol defining the result format for horizon martingale computations."""

    horizon_change_points: List[int]
    horizon_martingales: Optional[np.ndarray]


@dataclass
class MultiviewMartingaleState:
    """Base state for multiview martingale computation.

    Attributes:
        windows: List of rolling windows for each feature.
        traditional_martingales: Current traditional martingale values per feature.
        traditional_sum: Sum of traditional martingales across features.
        traditional_avg: Average of traditional martingales.
        traditional_change_points: Indices where traditional martingale detected changes.
        individual_traditional: Martingale history for each individual feature.
        current_timestep: The current timestep being processed.
        has_detection: Flag indicating if a detection has occurred at the current timestep.
    """

    windows: List[List[DataPoint]] = field(default_factory=list)
    traditional_martingales: List[float] = field(default_factory=list)
    traditional_sum: List[float] = field(default_factory=lambda: [1.0])
    traditional_avg: List[float] = field(default_factory=lambda: [1.0])
    traditional_change_points: List[int] = field(default_factory=list)
    individual_traditional: List[List[float]] = field(default_factory=list)
    current_timestep: int = 0
    has_detection: bool = False

    def __post_init__(self):
        """Initialize state lists if they are not already set."""
        if not self.windows:
            self.windows = []
        if not self.traditional_martingales:
            self.traditional_martingales = []
        if not self.individual_traditional:
            self.individual_traditional = []

    def record_traditional_values(
        self,
        timestep: int,
        traditional_values: List[float],
        is_detection: bool = False,
    ):
        """Record traditional martingale values at specific timestep.

        Args:
            timestep: The timestep to record values for
            traditional_values: List of traditional martingale values per feature
            is_detection: Whether this recording is for a detection event
        """
        self.current_timestep = timestep
        num_features = len(traditional_values)

        # Calculate sums and averages
        total_traditional = sum(traditional_values)
        avg_traditional = total_traditional / num_features

        # Ensure lists are long enough using manual extension to match martingale.py
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

        self.traditional_sum[next_t] = float(num_features)
        self.traditional_avg[next_t] = 1.0

        # Reset individual martingale histories per feature
        for j in range(num_features):
            while len(self.individual_traditional) <= j:
                self.individual_traditional.append([1.0])
            while len(self.individual_traditional[j]) <= next_t:
                self.individual_traditional[j].append(1.0)
            self.individual_traditional[j][next_t] = 1.0


@dataclass
class MultiviewHorizonMartingaleState(MultiviewMartingaleState):
    """State for multiview horizon martingale computation, extending the multiview base state.

    Attributes:
        horizon_martingales: Current horizon martingale values per feature.
        horizon_sum: Sum of horizon martingales across features.
        horizon_avg: Average of horizon martingales.
        horizon_change_points: Indices where horizon martingale detected changes.
        individual_horizon: Horizon martingale history for each individual feature.
        last_detection_time: The timestep of the last detection (either horizon or traditional).
        cooldown_period: Number of timesteps to enforce reduced sensitivity after a detection.
        early_warnings: Timesteps where early warnings were triggered before official detection.
        previous_horizon_sum: The horizon sum value from the previous timestep for growth calculation.
        feature_horizon_martingales: 2D list of martingales with values for each feature-horizon combination.
    """

    horizon_martingales: List[float] = field(default_factory=list)
    horizon_sum: List[float] = field(default_factory=lambda: [1.0])
    horizon_avg: List[float] = field(default_factory=lambda: [1.0])
    horizon_change_points: List[int] = field(default_factory=list)
    individual_horizon: List[List[float]] = field(default_factory=list)
    last_detection_time: int = -40  # Initialize to effectively no previous detection
    cooldown_period: int = 30  # Minimum timesteps between detections
    early_warnings: List[int] = field(default_factory=list)
    previous_horizon_sum: float = 1.0
    feature_horizon_martingales: List[List[float]] = field(default_factory=list)

    def __post_init__(self):
        """Initialize state lists if they are not already set."""
        super().__post_init__()
        if not self.horizon_martingales:
            self.horizon_martingales = []
        if not self.individual_horizon:
            self.individual_horizon = []
        if not self.feature_horizon_martingales:
            self.feature_horizon_martingales = []

    def record_values(
        self,
        timestep: int,
        traditional_values: List[float],
        horizon_values: List[float] = None,
        is_detection: bool = False,
    ):
        """Record both traditional and horizon martingale values at specific timestep.

        Args:
            timestep: The timestep to record values for
            traditional_values: List of traditional martingale values per feature
            horizon_values: Optional list of horizon martingale values per feature
            is_detection: Whether this recording is for a detection event
        """
        # First record the traditional values
        self.record_traditional_values(timestep, traditional_values, is_detection)

        # If horizon values provided, record them too
        if horizon_values:
            num_features = len(horizon_values)
            total_horizon = sum(horizon_values)
            avg_horizon = total_horizon / num_features

            # Ensure lists are long enough using manual extension to match martingale.py
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

    def reset(self, num_features: int):
        """Reset state for all features including horizon martingales.

        Args:
            num_features: Number of features to initialize.
        """
        # Call the parent reset method for traditional martingales
        super().reset(num_features)

        # Reset horizon martingale values for each feature to 1.0
        self.horizon_martingales = [1.0] * num_features

        # Reset feature-horizon martingales
        if self.feature_horizon_martingales:
            num_horizons = (
                len(self.feature_horizon_martingales[0])
                if self.feature_horizon_martingales
                else 0
            )
            self.feature_horizon_martingales = [
                [1.0] * num_horizons for _ in range(num_features)
            ]

        # Get the next timestep (already calculated in parent)
        next_t = self.current_timestep + 1

        # Reset horizon sum and average
        while len(self.horizon_sum) <= next_t:
            self.horizon_sum.append(1.0)
        while len(self.horizon_avg) <= next_t:
            self.horizon_avg.append(1.0)

        self.horizon_sum[next_t] = float(num_features)
        self.horizon_avg[next_t] = 1.0

        # Reset individual horizon martingale histories
        for j in range(num_features):
            while len(self.individual_horizon) <= j:
                self.individual_horizon.append([1.0])
            while len(self.individual_horizon[j]) <= next_t:
                self.individual_horizon[j].append(1.0)
            self.individual_horizon[j][next_t] = 1.0
