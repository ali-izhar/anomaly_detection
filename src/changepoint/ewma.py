# src/changepoint/ewma.py

"""Change point detection using the EWMA (Exponentially Weighted Moving Average) method.

This module implements online change point detection using the EWMA algorithm,
which detects changes by monitoring the exponentially weighted moving average
of a time series and identifying when it deviates significantly from expected behavior.

The EWMA algorithm is widely used in statistical process control for detecting
small shifts in the process mean.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Union, final
import logging
import numpy as np
import numpy.typing as npt
from numpy import floating, integer

from .distance import DistanceConfig

logger = logging.getLogger(__name__)

# Type definitions
Array = npt.NDArray[np.float64]
DetectionMethod = Literal["single_view", "multiview"]


@dataclass(frozen=True)
class EWMAConfig:
    """Configuration for the EWMA change point detector.

    Attributes:
        threshold: Detection threshold
        lambda_param: Smoothing parameter (0 < lambda ≤ 1)
        L: Control limit width
        startup_period: Number of samples to establish baseline
        use_var_adjust: Whether to use variance adjustment for startup period
        robust: Whether to use robust EWMA
        method: Detection method ('single_view' or 'multiview')
        batch_size: Batch size for multiview processing
        history_size: Number of historical observations to use
        reset: Whether to reset after detection
        distance_measure: Distance metric for strangeness computation
        distance_p: Order parameter for Minkowski distance
        random_state: Random seed for reproducibility
    """

    threshold: float = 15.0
    lambda_param: float = 0.1
    L: float = 3.0
    startup_period: int = 20
    use_var_adjust: bool = True
    robust: bool = False
    method: DetectionMethod = "multiview"
    batch_size: int = 1000
    history_size: int = 10
    reset: bool = True
    distance_measure: str = "euclidean"
    distance_p: float = 2.0
    random_state: Optional[int] = 42

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.threshold <= 0:
            raise ValueError(f"Threshold must be positive, got {self.threshold}")
        if self.lambda_param <= 0 or self.lambda_param > 1:
            raise ValueError(f"Lambda must be in range (0, 1], got {self.lambda_param}")
        if self.L <= 0:
            raise ValueError(f"Control limit width must be positive, got {self.L}")
        if self.startup_period < 1:
            raise ValueError(
                f"Startup period must be at least 1, got {self.startup_period}"
            )
        if self.history_size < 1:
            raise ValueError(
                f"History size must be at least 1, got {self.history_size}"
            )
        if self.batch_size < 1:
            raise ValueError(f"Batch size must be at least 1, got {self.batch_size}")
        if self.distance_p <= 0:
            raise ValueError(
                f"Distance order parameter must be positive, got {self.distance_p}"
            )


class EWMADetector:
    """Detector class for identifying change points using the EWMA algorithm.

    The detector monitors a sequence of observations and reports changes when the
    exponentially weighted moving average deviates significantly from expected behavior.

    Main steps:
    1. Compute the EWMA for each observation
    2. Calculate control limits based on the standard error
    3. Report a change point when the EWMA exceeds the control limits
    4. Reset the EWMA after a change is detected (if reset=True)

    The detector supports two operating modes:
    - Single-view: Process a single feature sequence
    - Multiview: Process multiple features independently and combine evidence
    """

    def __init__(self, config: Optional[EWMAConfig] = None):
        """Initialize the detector with configuration.

        Args:
            config: Detector configuration. If None, uses default configuration.
        """
        # Use provided configuration or fall back to defaults
        self.config = config or EWMAConfig()

        # Initialize the internal state
        self._reset_state()

        # Set random seed for reproducibility
        if self.config.random_state is not None:
            np.random.seed(self.config.random_state)

        # Log configuration
        logger.debug(f"Initialized {self.config.method} EWMA detector with:")
        logger.debug(f"  Threshold: {self.config.threshold}")
        logger.debug(f"  Lambda: {self.config.lambda_param}")
        logger.debug(f"  L: {self.config.L}")
        logger.debug(f"  Startup period: {self.config.startup_period}")
        logger.debug(f"  Use variance adjustment: {self.config.use_var_adjust}")
        logger.debug(f"  Robust: {self.config.robust}")
        logger.debug(f"  Reset: {self.config.reset}")
        logger.debug(
            f"  Distance: {self.config.distance_measure} (p={self.config.distance_p})"
        )
        logger.debug(f"  Random state: {self.config.random_state}")

    def _reset_state(self) -> None:
        """Reset the detector's internal state."""
        # Initialize state tracking
        self._state = None
        # Initialize the list to store detected change points
        self._change_points: List[int] = []
        self._early_warnings: List[int] = []

    @property
    def change_points(self) -> List[int]:
        """Get the detected change points."""
        # Return a copy to prevent external modifications
        return self._change_points.copy()

    @property
    def early_warnings(self) -> List[int]:
        """Get the detected early warnings."""
        # Return a copy to prevent external modifications
        return self._early_warnings.copy()

    @final
    def run(
        self,
        data: Array,
        predicted_data: Optional[Array] = None,
        reset_state: bool = False,
    ) -> Dict[str, Any]:
        """Run the change point detection algorithm.

        Args:
            data: Input sequence of shape (n_samples, n_features) for multiview
                 or (n_samples,) for single-view.
            predicted_data: Optional predicted values (unused in this detector)
            reset_state: Whether to reset the detector state before running.

        Returns:
            Dictionary containing:
            - change_points: List of indices where changes were detected
            - ewma_values: Array of EWMA values
            - upper_limits: Array of upper control limits
            - lower_limits: Array of lower control limits

        Raises:
            ValueError: If input data is empty or invalid.
            RuntimeError: If detection fails.
        """
        try:
            # Reset state if requested
            if reset_state:
                self._reset_state()

            # Validate input dimensions and types
            self._validate_input(data, predicted_data)
            # Log the dimensions of the input for debugging
            self._log_input_dimensions(data, predicted_data)

            # Run the appropriate detection method based on the configuration
            if self.config.method == "single_view":
                results = self._run_single_view(data, predicted_data)
            else:  # For multiview detection
                results = self._run_multiview(data, predicted_data)

            # Update internal change points tracking
            self._change_points.extend(results.get("change_points", []))

            return results

        except Exception as e:
            logger.error(f"EWMA detection failed: {str(e)}")
            raise

    def _validate_input(self, data: Array, predicted_data: Optional[Array]) -> None:
        """Validate input data dimensions and types.

        Args:
            data: Input sequence.
            predicted_data: Optional predicted values.

        Raises:
            ValueError: If validation fails.
        """
        if data.size == 0:
            raise ValueError("Empty input sequence")

        if self.config.method == "single_view":
            # Single-view data must be 1-dimensional
            if len(data.shape) != 1:
                raise ValueError(
                    f"Single-view data must be 1-dimensional, got shape {data.shape}"
                )
            # Check predictions if provided
            if predicted_data is not None:
                if len(predicted_data.shape) != 2:
                    raise ValueError(
                        f"Single-view predictions must have shape (n_pred, horizon), got {predicted_data.shape}"
                    )
        else:  # Multiview detection
            # Multiview data should be 2D: (n_samples, n_features)
            if len(data.shape) != 2:
                raise ValueError(
                    f"Multiview data must have shape (n_samples, n_features), got {data.shape}"
                )
            # Check predictions if provided
            if predicted_data is not None:
                if len(predicted_data.shape) != 3:
                    raise ValueError(
                        f"Multiview predictions must have shape (n_pred, horizon, n_features), got {predicted_data.shape}"
                    )
                # Ensure feature dimensions match
                if predicted_data.shape[-1] != data.shape[1]:
                    raise ValueError(
                        f"Number of features in predictions ({predicted_data.shape[-1]}) "
                        f"does not match input data ({data.shape[1]})"
                    )

    def _log_input_dimensions(
        self, data: Array, predicted_data: Optional[Array]
    ) -> None:
        """Log input data dimensions for debugging.

        Args:
            data: Input sequence.
            predicted_data: Optional predicted values.
        """
        logger.debug("EWMA Detector Input Dimensions:")
        if self.config.method == "single_view":
            logger.debug(f"  Sequence length: {len(data)}")
            if predicted_data is not None:
                logger.debug(f"  Number of predictions: {len(predicted_data)}")
                logger.debug(f"  Predictions per timestep: {predicted_data.shape[1]}")
        else:
            logger.debug(f"  Main sequence shape (TxF): {data.shape}")
            if predicted_data is not None:
                logger.debug(
                    f"  Predicted sequence shape (T'xHxF): {predicted_data.shape}"
                )
        logger.debug("-" * 50)

    def _run_single_view(
        self, data: Array, predicted_data: Optional[Array]
    ) -> Dict[str, Any]:
        """Run single-view EWMA detection.

        Args:
            data: Input sequence.
            predicted_data: Optional predicted values (can be used for early warnings).

        Returns:
            Detection results dictionary.
        """
        # Initialize or retrieve EWMA state
        if self._state is None:
            ewma = np.zeros(data.shape[0])
            mean = None
            std = None
            last_change = 0
        else:
            ewma = self._state.get("ewma", np.zeros(data.shape[0]))
            mean = self._state.get("mean")
            std = self._state.get("std")
            last_change = self._state.get("last_change", 0)

        # Get EWMA parameters
        lambda_param = self.config.lambda_param
        L = self.config.L

        # Establish baseline statistics during startup period
        if mean is None or std is None:
            # Use startup_period or available data points, whichever is smaller
            startup_end = min(self.config.startup_period, len(data))
            baseline_data = data[:startup_end]

            if self.config.robust:
                # Use median and MAD for robustness
                mean = np.median(baseline_data)
                std = (
                    np.median(np.abs(baseline_data - mean)) * 1.4826
                )  # Approximate conversion to std
            else:
                mean = np.mean(baseline_data)
                std = np.std(baseline_data)

            # Ensure std is not zero
            std = std if std > 0 else 1.0

            # Initialize EWMA with mean for startup period
            ewma[:startup_end] = mean
            start_idx = startup_end
        else:
            start_idx = 0

        # Initialize control limits and change points list
        upper_limits = np.zeros(data.shape[0])
        lower_limits = np.zeros(data.shape[0])
        change_points = []

        # Process each data point
        for i in range(start_idx, len(data)):
            # Update EWMA
            if i == start_idx:
                ewma[i] = mean
            else:
                ewma[i] = lambda_param * data[i] + (1 - lambda_param) * ewma[i - 1]

            # Calculate control limits
            if self.config.use_var_adjust:
                # Variance adjustment for EWMA statistics
                var_factor = (
                    lambda_param
                    / (2 - lambda_param)
                    * (1 - (1 - lambda_param) ** (2 * (i - start_idx + 1)))
                )
            else:
                # Asymptotic variance for steady state
                var_factor = lambda_param / (2 - lambda_param)

            # Control limits
            ucl = mean + L * std * np.sqrt(var_factor)
            lcl = mean - L * std * np.sqrt(var_factor)

            upper_limits[i] = ucl
            lower_limits[i] = lcl

            # Check for change points
            if ewma[i] > ucl or ewma[i] < lcl:
                change_points.append(i)

                # Reset after detection if configured
                if self.config.reset:
                    # If adaptive, update mean and std based on recent data
                    window_start = max(0, i - self.config.history_size)

                    if self.config.robust:
                        # Use median and MAD for robustness
                        mean = np.median(data[window_start:i])
                        std = np.median(np.abs(data[window_start:i] - mean)) * 1.4826
                    else:
                        mean = np.mean(data[window_start:i])
                        std = np.std(data[window_start:i])

                    # Ensure std is not zero
                    std = std if std > 0 else 1.0

                    # Reset EWMA to mean
                    ewma[i] = mean
                    last_change = i

        # Update state for future calls
        self._state = {
            "ewma": ewma,
            "mean": mean,
            "std": std,
            "last_change": last_change,
        }

        # Calculate EWMA statistics for compatibility
        ewma_stats = np.abs((ewma - mean) / std)

        # Prepare results
        results = {
            "change_points": change_points,
            "ewma_values": ewma,
            "upper_limits": upper_limits,
            "lower_limits": lower_limits,
            "ewma_statistics": ewma_stats,
        }

        # Add extra information for compatibility with other detectors
        results["traditional_change_points"] = change_points
        results["traditional_martingales"] = ewma_stats  # For compatibility

        # Use prediction data if available for early warnings
        if predicted_data is not None:
            # Simple approach: use prediction to see if change points are likely in the future
            early_warnings = []

            # Process only where we have predictions
            for t in range(min(len(predicted_data), len(data))):
                if t in change_points:
                    continue

                # Calculate EWMA predictions
                horizon_alerts = False

                # Predict EWMA evolution
                last_ewma = ewma[t]

                for h in range(predicted_data.shape[1]):
                    # Calculate predicted EWMA
                    pred_ewma = (
                        lambda_param * predicted_data[t, h]
                        + (1 - lambda_param) * last_ewma
                    )

                    # Check if prediction would exceed control limits
                    if pred_ewma > upper_limits[t] or pred_ewma < lower_limits[t]:
                        horizon_alerts = True
                        break

                    # Update for next horizon step
                    last_ewma = pred_ewma

                if horizon_alerts and t not in early_warnings:
                    early_warnings.append(t)

            results["early_warnings"] = early_warnings
            results["horizon_change_points"] = []  # Empty list for API compatibility

        return results

    def _run_multiview(
        self, data: Array, predicted_data: Optional[Array]
    ) -> Dict[str, Any]:
        """Run multiview EWMA detection.

        Args:
            data: Input sequence.
            predicted_data: Optional predicted values.

        Returns:
            Detection results dictionary.
        """
        n_samples, n_features = data.shape

        # Initialize arrays for each feature
        ewma_values = np.zeros((n_features, n_samples))
        upper_limits = np.zeros((n_features, n_samples))
        lower_limits = np.zeros((n_features, n_samples))
        ewma_statistics = np.zeros((n_features, n_samples))

        # Initialize or retrieve state
        if self._state is None:
            means = np.zeros(n_features)
            stds = np.zeros(n_features)
            initialized = False
            last_change = 0
        else:
            means = self._state.get("means", np.zeros(n_features))
            stds = self._state.get("stds", np.zeros(n_features))
            initialized = self._state.get("initialized", False)
            last_change = self._state.get("last_change", 0)

        # Get EWMA parameters
        lambda_param = self.config.lambda_param
        L = self.config.L

        # Establish baseline statistics if not initialized
        if not initialized:
            startup_end = min(self.config.startup_period, n_samples)
            for f in range(n_features):
                # Get baseline data
                baseline_data = data[:startup_end, f]

                if self.config.robust:
                    # Use median and MAD for robustness
                    means[f] = np.median(baseline_data)
                    stds[f] = np.median(np.abs(baseline_data - means[f])) * 1.4826
                else:
                    means[f] = np.mean(baseline_data)
                    stds[f] = np.std(baseline_data)

                # Ensure std is not zero
                stds[f] = stds[f] if stds[f] > 0 else 1.0

                # Initialize EWMA with mean for startup period
                ewma_values[f, :startup_end] = means[f]

            initialized = True
            start_idx = startup_end
        else:
            start_idx = 0

        # Track change points
        all_change_points = []

        # Process each feature separately
        for f in range(n_features):
            feature_change_points = []

            # Process each time point
            for i in range(start_idx, n_samples):
                # Update EWMA
                if i == start_idx:
                    ewma_values[f, i] = means[f]
                else:
                    ewma_values[f, i] = (
                        lambda_param * data[i, f]
                        + (1 - lambda_param) * ewma_values[f, i - 1]
                    )

                # Calculate control limits
                if self.config.use_var_adjust:
                    # Variance adjustment for EWMA statistics
                    var_factor = (
                        lambda_param
                        / (2 - lambda_param)
                        * (1 - (1 - lambda_param) ** (2 * (i - start_idx + 1)))
                    )
                else:
                    # Asymptotic variance for steady state
                    var_factor = lambda_param / (2 - lambda_param)

                # Control limits
                ucl = means[f] + L * stds[f] * np.sqrt(var_factor)
                lcl = means[f] - L * stds[f] * np.sqrt(var_factor)

                upper_limits[f, i] = ucl
                lower_limits[f, i] = lcl

                # Calculate normalized statistic
                ewma_statistics[f, i] = np.abs((ewma_values[f, i] - means[f]) / stds[f])

                # Check for change points
                if ewma_values[f, i] > ucl or ewma_values[f, i] < lcl:
                    feature_change_points.append(i)

                    # Reset after detection if configured
                    if self.config.reset:
                        # If adaptive, update mean and std based on recent data
                        window_start = max(0, i - self.config.history_size)

                        if self.config.robust:
                            # Use median and MAD for robustness
                            means[f] = np.median(data[window_start:i, f])
                            stds[f] = (
                                np.median(np.abs(data[window_start:i, f] - means[f]))
                                * 1.4826
                            )
                        else:
                            means[f] = np.mean(data[window_start:i, f])
                            stds[f] = np.std(data[window_start:i, f])

                        # Ensure std is not zero
                        stds[f] = stds[f] if stds[f] > 0 else 1.0

                        # Reset EWMA to mean
                        ewma_values[f, i] = means[f]

            # Add unique change points
            for cp in feature_change_points:
                if cp not in all_change_points:
                    all_change_points.append(cp)

        # Sort change points
        all_change_points.sort()

        # Update state for future calls
        self._state = {
            "means": means,
            "stds": stds,
            "initialized": initialized,
            "last_change": all_change_points[-1] if all_change_points else last_change,
        }

        # Calculate sum and average of EWMA statistics across features
        sum_ewma_stat = np.sum(ewma_statistics, axis=0)
        avg_ewma_stat = np.mean(ewma_statistics, axis=0)

        # Prepare results
        results = {
            "change_points": all_change_points,
            "ewma_values": ewma_values,
            "upper_limits": upper_limits,
            "lower_limits": lower_limits,
            "ewma_statistics": ewma_statistics,
            "sum_ewma_stat": sum_ewma_stat,
            "avg_ewma_stat": avg_ewma_stat,
            "individual_ewma_stats": [ewma_statistics[f] for f in range(n_features)],
        }

        # Add fields for compatibility with martingale detector
        results["traditional_change_points"] = all_change_points
        results["traditional_martingales"] = avg_ewma_stat  # For compatibility
        results["traditional_sum_martingales"] = sum_ewma_stat
        results["traditional_avg_martingales"] = avg_ewma_stat
        results["individual_traditional_martingales"] = [
            ewma_statistics[f] for f in range(n_features)
        ]

        # Handle predictions if available
        if predicted_data is not None:
            early_warnings = []

            # Process only where we have predictions
            for t in range(min(predicted_data.shape[0], n_samples)):
                if t in all_change_points:
                    continue

                # Check each feature's predictions
                horizon_alerts = 0

                for f in range(n_features):
                    # Start with current EWMA
                    last_ewma = ewma_values[f, t]

                    for h in range(predicted_data.shape[1]):
                        # Calculate predicted EWMA
                        pred_ewma = (
                            lambda_param * predicted_data[t, h, f]
                            + (1 - lambda_param) * last_ewma
                        )

                        # Check if prediction would exceed control limits
                        if (
                            pred_ewma > upper_limits[f, t]
                            or pred_ewma < lower_limits[f, t]
                        ):
                            horizon_alerts += 1
                            break

                        # Update for next horizon step
                        last_ewma = pred_ewma

                # If enough features predict a change, add an early warning
                if horizon_alerts >= n_features / 3 and t not in early_warnings:
                    early_warnings.append(t)

            results["early_warnings"] = early_warnings
            results["horizon_change_points"] = []  # Empty list for API compatibility

        return results
