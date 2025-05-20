"""Change point detection using the CUSUM (Cumulative Sum) method.

This module implements online change point detection using the CUSUM algorithm,
which monitors a sequence of observations and reports changes when the cumulative
sum of deviations from a target value exceeds a specified threshold.

The CUSUM algorithm is a sequential analysis technique for detecting changes in
the statistical properties of a time series.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, final
import logging
import numpy as np
import numpy.typing as npt

from .distance import DistanceConfig

logger = logging.getLogger(__name__)

# Type definitions
Array = npt.NDArray[np.float64]
DetectionMethod = Literal["single_view", "multiview"]


@dataclass(frozen=True)
class CUSUMConfig:
    """Configuration for the CUSUM change point detector.

    Attributes:
        threshold: Detection threshold
        drift: Expected drift under normal conditions (target value)
        startup_period: Number of samples to establish baseline
        fixed_threshold: Whether to use a fixed threshold or dynamic
        k: Sensitivity parameter for CUSUM
        h: Control limit for CUSUM detection
        enable_adaptive: Whether to use adaptive CUSUM
        method: Detection method ('single_view' or 'multiview')
        batch_size: Batch size for multiview processing
        history_size: Number of historical observations to use
        reset: Whether to reset after detection
        distance_measure: Distance metric for strangeness computation
        distance_p: Order parameter for Minkowski distance
        random_state: Random seed for reproducibility
    """

    threshold: float = 15.0
    drift: float = 0.0
    startup_period: int = 20
    fixed_threshold: bool = False
    k: float = 0.5
    h: float = 5.0
    enable_adaptive: bool = True
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
        if self.k <= 0:
            raise ValueError(f"Sensitivity parameter k must be positive, got {self.k}")
        if self.h <= 0:
            raise ValueError(f"Control limit h must be positive, got {self.h}")
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


class CUSUMDetector:
    """Detector class for identifying change points using the CUSUM algorithm.

    The detector monitors a sequence of observations and reports changes when the
    cumulative sum of deviations from a target value exceeds a specified threshold.

    Main steps:
    1. Compute the deviation of each observation from the target value
    2. Accumulate positive and negative deviations separately
    3. Report a change point when either sum exceeds the threshold
    4. Reset the sums after a change is detected (if reset=True)

    The detector supports two operating modes:
    - Single-view: Process a single feature sequence
    - Multiview: Process multiple features independently and combine evidence
    """

    def __init__(self, config: Optional[CUSUMConfig] = None):
        """Initialize the detector with configuration.

        Args:
            config: Detector configuration. If None, uses default configuration.
        """
        # Use provided configuration or fall back to defaults
        self.config = config or CUSUMConfig()

        # Initialize the internal state
        self._reset_state()

        # Set random seed for reproducibility
        if self.config.random_state is not None:
            np.random.seed(self.config.random_state)

        # Log configuration
        logger.debug(f"Initialized {self.config.method} CUSUM detector with:")
        logger.debug(f"  Threshold: {self.config.threshold}")
        logger.debug(f"  Drift: {self.config.drift}")
        logger.debug(f"  Startup period: {self.config.startup_period}")
        logger.debug(f"  Fixed threshold: {self.config.fixed_threshold}")
        logger.debug(f"  k: {self.config.k}")
        logger.debug(f"  h: {self.config.h}")
        logger.debug(f"  Enable adaptive: {self.config.enable_adaptive}")
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
            - cusum_positives: Array of positive CUSUM values
            - cusum_negatives: Array of negative CUSUM values
            - combined_cusum: Array of combined CUSUM values

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
            logger.error(f"CUSUM detection failed: {str(e)}")
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
        logger.debug("CUSUM Detector Input Dimensions:")
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
        """Run single-view CUSUM detection.

        Args:
            data: Input sequence.
            predicted_data: Optional predicted values (can be used for early warnings).

        Returns:
            Detection results dictionary.
        """
        # Initialize or retrieve CUSUM state
        if self._state is None:
            cusum_pos = np.zeros(data.shape[0])
            cusum_neg = np.zeros(data.shape[0])
            mean = None
            std = None
            last_change = 0
        else:
            cusum_pos = self._state.get("cusum_pos", np.zeros(data.shape[0]))
            cusum_neg = self._state.get("cusum_neg", np.zeros(data.shape[0]))
            mean = self._state.get("mean")
            std = self._state.get("std")
            last_change = self._state.get("last_change", 0)

        # Establish baseline statistics during startup period or after a change
        if mean is None or std is None:
            # Use startup_period or available data points, whichever is smaller
            startup_end = min(self.config.startup_period, len(data))
            baseline_data = data[:startup_end]
            mean = np.mean(baseline_data)
            std = (
                np.std(baseline_data) or 1.0
            )  # Use 1.0 if std is 0 to avoid division by zero
            # Skip processing for startup period
            cusum_pos[:startup_end] = 0
            cusum_neg[:startup_end] = 0
            start_idx = startup_end
        else:
            start_idx = 0

        # Set control parameters
        k = self.config.k
        h = self.config.h if self.config.fixed_threshold else self.config.threshold

        # Initialize change points list
        change_points = []

        # Process each data point
        for i in range(start_idx, len(data)):
            # Normalize value
            z = (data[i] - mean) / std

            # Update CUSUM statistics
            if i > 0:
                cusum_pos[i] = max(0, cusum_pos[i - 1] + z - k)
                cusum_neg[i] = max(0, cusum_neg[i - 1] - z - k)
            else:
                cusum_pos[i] = max(0, z - k)
                cusum_neg[i] = max(0, -z - k)

            # Check for change points
            if cusum_pos[i] > h or cusum_neg[i] > h:
                change_points.append(i)
                # Reset after detection if configured
                if self.config.reset:
                    cusum_pos[i] = 0
                    cusum_neg[i] = 0
                    # If adaptive, update mean and std
                    if self.config.enable_adaptive:
                        # Use window from last change to current point
                        window_start = last_change
                        mean = np.mean(data[window_start:i])
                        std = np.std(data[window_start:i]) or 1.0
                    last_change = i

        # Combine CUSUM values for visualization
        combined_cusum = np.maximum(cusum_pos, cusum_neg)

        # Update state for future calls
        self._state = {
            "cusum_pos": cusum_pos,
            "cusum_neg": cusum_neg,
            "mean": mean,
            "std": std,
            "last_change": last_change,
        }

        # Prepare results
        results = {
            "change_points": change_points,
            "cusum_positives": cusum_pos,
            "cusum_negatives": cusum_neg,
            "combined_cusum": combined_cusum,
        }

        # Add extra information for compatibility with other detectors
        results["traditional_change_points"] = change_points
        results["traditional_martingales"] = combined_cusum  # For compatibility

        # Use prediction data if available for early warnings
        if predicted_data is not None:
            # Simple approach: use prediction to see if change points are likely in the future
            early_warnings = []

            # Process only where we have predictions
            for t in range(min(len(predicted_data), len(data))):
                # Get current baseline
                current_mean = mean
                current_std = std

                # Check if any predictions would cause a CUSUM alert
                horizon_alerts = False

                for h in range(predicted_data.shape[1]):
                    pred_z = (predicted_data[t, h] - current_mean) / current_std
                    pred_cusum_pos = max(0, cusum_pos[t] + pred_z - k)
                    pred_cusum_neg = max(0, cusum_neg[t] - pred_z - k)

                    if pred_cusum_pos > h or pred_cusum_neg > h:
                        horizon_alerts = True
                        break

                if (
                    horizon_alerts
                    and t not in change_points
                    and t not in early_warnings
                ):
                    early_warnings.append(t)

            results["early_warnings"] = early_warnings
            results["horizon_change_points"] = []  # Empty list for API compatibility

        return results

    def _run_multiview(
        self, data: Array, predicted_data: Optional[Array]
    ) -> Dict[str, Any]:
        """Run multiview CUSUM detection.

        Args:
            data: Input sequence.
            predicted_data: Optional predicted values.

        Returns:
            Detection results dictionary.
        """
        n_samples, n_features = data.shape

        # Initialize arrays for each feature
        cusum_positives = np.zeros((n_features, n_samples))
        cusum_negatives = np.zeros((n_features, n_samples))
        combined_cusums = np.zeros((n_features, n_samples))

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

        # Establish baseline statistics if not initialized
        if not initialized:
            startup_end = min(self.config.startup_period, n_samples)
            for f in range(n_features):
                means[f] = np.mean(data[:startup_end, f])
                stds[f] = np.std(data[:startup_end, f]) or 1.0
            initialized = True
            start_idx = startup_end
        else:
            start_idx = 0

        # Set control parameters
        k = self.config.k
        h = self.config.h if self.config.fixed_threshold else self.config.threshold

        # Track change points
        all_change_points = []

        # Process each feature separately
        for f in range(n_features):
            feature_change_points = []

            # Process each time point
            for i in range(start_idx, n_samples):
                # Normalize value
                z = (data[i, f] - means[f]) / stds[f]

                # Update CUSUM statistics
                if i > 0:
                    cusum_positives[f, i] = max(0, cusum_positives[f, i - 1] + z - k)
                    cusum_negatives[f, i] = max(0, cusum_negatives[f, i - 1] - z - k)
                else:
                    cusum_positives[f, i] = max(0, z - k)
                    cusum_negatives[f, i] = max(0, -z - k)

                # Store combined CUSUM
                combined_cusums[f, i] = max(
                    cusum_positives[f, i], cusum_negatives[f, i]
                )

                # Check for change point
                if combined_cusums[f, i] > h:
                    feature_change_points.append(i)

                    # Reset after detection if configured
                    if self.config.reset:
                        cusum_positives[f, i] = 0
                        cusum_negatives[f, i] = 0

                        # If adaptive, update mean and std for this feature
                        if self.config.enable_adaptive:
                            window_start = last_change
                            means[f] = np.mean(data[window_start:i, f])
                            stds[f] = np.std(data[window_start:i, f]) or 1.0

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

        # Calculate sum and average of CUSUM values across features
        sum_cusum = np.sum(combined_cusums, axis=0)
        avg_cusum = np.mean(combined_cusums, axis=0)

        # Prepare results
        results = {
            "change_points": all_change_points,
            "cusum_positives": cusum_positives,
            "cusum_negatives": cusum_negatives,
            "combined_cusum": combined_cusums,
            "sum_cusum": sum_cusum,
            "avg_cusum": avg_cusum,
            "individual_cusums": [combined_cusums[f] for f in range(n_features)],
        }

        # Add fields for compatibility with martingale detector
        results["traditional_change_points"] = all_change_points
        results["traditional_martingales"] = avg_cusum  # For compatibility
        results["traditional_sum_martingales"] = sum_cusum
        results["traditional_avg_martingales"] = avg_cusum
        results["individual_traditional_martingales"] = [
            combined_cusums[f] for f in range(n_features)
        ]

        # Handle predictions if available
        if predicted_data is not None:
            early_warnings = []

            # Simple approach: check if predictions would trigger alerts
            for t in range(min(predicted_data.shape[0], n_samples)):
                if t in all_change_points:
                    continue

                # Check each feature's predictions
                horizon_alerts = 0

                for f in range(n_features):
                    for h in range(predicted_data.shape[1]):
                        pred_z = (predicted_data[t, h, f] - means[f]) / stds[f]
                        pred_pos = max(0, cusum_positives[f, t] + pred_z - k)
                        pred_neg = max(0, cusum_negatives[f, t] - pred_z - k)

                        if pred_pos > h or pred_neg > h:
                            horizon_alerts += 1
                            break

                # If enough features predict a change, add an early warning
                if horizon_alerts >= n_features / 3 and t not in early_warnings:
                    early_warnings.append(t)

            results["early_warnings"] = early_warnings
            results["horizon_change_points"] = []  # Empty list for API compatibility

        return results
