# src/changepoint/detector.py

"""Change point detection using the martingale framework derived from conformal prediction.

This module implements online change point detection using exchangeability martingales
derived from conformal prediction. The detector monitors a sequence of observations
and reports changes when the martingale value exceeds a specified threshold.

Mathematical Framework:
---------------------
1. Conformal Prediction:
   - For a new observation xₙ, compute a nonconformity score and a corresponding p-value.
2. Martingale Construction:
   - Update the martingale: Mₙ = Mₙ₋₁ * g(pₙ), where g(p) is a valid betting function.
3. Change Detection:
   - Report a change when Mₙ > λ (detection threshold).

The detector supports two methods:
- Single-view: Processes a single feature sequence.
- Multiview: Processes multiple features independently and then combines evidence.
"""

from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    Generic,
    List,
    Literal,
    Optional,
    TypeVar,
    Union,
    final,
)

import logging
import numpy as np
import numpy.typing as npt
from numpy import floating, integer

from .betting import BettingFunctionConfig
from .martingale import (
    compute_martingale,
    multiview_martingale_test,
    MartingaleConfig,
)

logger = logging.getLogger(__name__)

# Type definitions:
# ScalarType can be any floating point or integer numpy type.
ScalarType = TypeVar("ScalarType", bound=Union[floating, integer])
Array = npt.NDArray[np.float64]
DetectionMethod = Literal["single_view", "multiview"]


@dataclass(frozen=True)
class DetectorConfig:
    """Configuration for the change point detector.

    Attributes:
        method: Detection method ('single_view' or 'multiview')
        threshold: Detection threshold for martingale values
        history_size: Minimum number of observations before using predictions
        batch_size: Batch size for multiview processing
        reset: Whether to reset after detection
        max_window: Maximum window size for strangeness computation
        betting_func_config: Configuration for the betting function
        distance_measure: Distance metric for strangeness computation
        distance_p: Order parameter for Minkowski distance
        random_state: Random seed for reproducibility
    """

    method: DetectionMethod = "multiview"
    threshold: float = 60.0
    history_size: int = 10
    batch_size: int = 1000
    reset: bool = True
    max_window: Optional[int] = None
    betting_func_config: Optional[BettingFunctionConfig] = None
    distance_measure: str = "euclidean"
    distance_p: float = 2.0
    random_state: Optional[int] = 42

    def __post_init__(self):
        """Validate configuration parameters."""
        # Ensure that the detection threshold is positive.
        if self.threshold <= 0:
            raise ValueError(f"Threshold must be positive, got {self.threshold}")
        # Ensure that there is at least one historical observation.
        if self.history_size < 1:
            raise ValueError(
                f"History size must be at least 1, got {self.history_size}"
            )
        # Batch size must be at least 1.
        if self.batch_size < 1:
            raise ValueError(f"Batch size must be at least 1, got {self.batch_size}")
        # If specified, the max_window must be at least 1.
        if self.max_window is not None and self.max_window < 1:
            raise ValueError(
                f"Max window must be at least 1 if specified, got {self.max_window}"
            )
        # The distance order parameter must be positive.
        if self.distance_p <= 0:
            raise ValueError(
                f"Distance order parameter must be positive, got {self.distance_p}"
            )


class ChangePointDetector(Generic[ScalarType]):
    """Detector class for identifying change points in sequential data.

    The detector uses the martingale framework derived from conformal prediction
    to monitor a sequence of observations and report changes when the martingale
    value exceeds a specified threshold.

    Main Steps:
    1. Compute a strangeness measure for new points.
    2. Convert strangeness to a p-value via nonparametric ranking.
    3. Update a martingale using a chosen betting function.
    4. Report a change point if the martingale exceeds the threshold.

    Type Parameters:
    ---------------
    ScalarType : Union[floating, integer]
        The numpy dtype of the input data. Must be a floating point or integer type.
    """

    def __init__(self, config: Optional[DetectorConfig] = None):
        """Initialize the detector with configuration.

        Args:
            config: Detector configuration. If None, uses default configuration.
        """
        # Use provided configuration or fall back to the default settings.
        self.config = config or DetectorConfig()

        # If no betting function config is provided, set a default configuration.
        if self.config.betting_func_config is None:
            self.config.betting_func_config = {
                "name": "power",
                "params": {"epsilon": 0.7},
            }
            logger.debug(
                f"No betting function config provided. Using default: {self.config.betting_func_config}"
            )

        # Initialize the internal state of the detector.
        self._reset_state()

        # Log configuration parameters for debugging purposes.
        logger.debug(f"Initialized {self.config.method} detector with:")
        logger.debug(f"  Threshold: {self.config.threshold}")
        logger.debug(f"  History size: {self.config.history_size}")
        logger.debug(f"  Batch size: {self.config.batch_size}")
        logger.debug(f"  Reset: {self.config.reset}")
        logger.debug(f"  Max window: {self.config.max_window}")
        logger.debug(
            f"  Distance: {self.config.distance_measure} (p={self.config.distance_p})"
        )
        logger.debug(f"  Betting function: {self.config.betting_func_config['name']}")

    def _reset_state(self) -> None:
        """Reset the detector's internal state."""
        # Initialize the traditional martingale value.
        self._traditional_martingale = 1.0
        # Initialize the horizon martingale value.
        self._horizon_martingale = 1.0
        # Initialize the window to hold recent strangeness or p-values.
        self._window: List[float] = []
        # Initialize the list to store detected change point indices.
        self._change_points: List[int] = []

    @property
    def change_points(self) -> List[int]:
        """Get the detected change points."""
        # Return a copy to prevent external modifications of the internal list.
        return self._change_points.copy()

    @final
    def run(
        self,
        data: Array,
        predicted_data: Optional[Array] = None,
    ) -> Dict[str, Any]:
        """Run the change point detection algorithm.

        The detector processes the input sequence and returns detected change points
        along with martingale values. For multiview detection, each feature is
        processed separately and evidence is combined.

        Args:
            data: Input sequence of shape (n_samples, n_features) for multiview
                 or (n_samples,) for single-view.
            predicted_data: Optional predicted values of shape
                          (n_predictions, horizon, n_features) for multiview
                          or (n_predictions, horizon) for single-view.

        Returns:
            Dictionary containing:
            - traditional_change_points: List of indices where traditional martingale detected changes
            - horizon_change_points: List of indices where horizon martingale detected changes
            - traditional_martingales: Array of traditional martingale values
            - horizon_martingales: Array of horizon martingale values (if predictions provided)
            For multiview detection, also includes:
            - traditional_sum_martingales: Sum of traditional martingales across features
            - traditional_avg_martingales: Average of traditional martingales
            - horizon_sum_martingales: Sum of horizon martingales
            - horizon_avg_martingales: Average of horizon martingales
            - individual_traditional_martingales: List of martingales for each feature
            - individual_horizon_martingales: List of horizon martingales for each feature

        Raises:
            ValueError: If input data is empty or invalid.
            RuntimeError: If detection fails.
        """
        try:
            # Validate input dimensions and types.
            self._validate_input(data, predicted_data)
            # Log the dimensions of the input for debugging.
            self._log_input_dimensions(data, predicted_data)

            # Run the appropriate detection method based on the configuration.
            if self.config.method == "single_view":
                return self._run_single_view(data, predicted_data)
            else:  # For multiview detection.
                return self._run_multiview(data, predicted_data)

        except Exception as e:
            logger.error(f"Detection failed: {str(e)}")
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
            # Single-view data must be 1-dimensional.
            if len(data.shape) != 1:
                raise ValueError(
                    f"Single-view data must be 1-dimensional, got shape {data.shape}"
                )
            # Predictions for single-view should be 2D: (n_predictions, horizon).
            if predicted_data is not None:
                if len(predicted_data.shape) != 2:
                    raise ValueError(
                        f"Single-view predictions must have shape (n_pred, horizon), got {predicted_data.shape}"
                    )
        else:  # Multiview detection.
            # Multiview data should be 2D: (n_samples, n_features).
            if len(data.shape) != 2:
                raise ValueError(
                    f"Multiview data must have shape (n_samples, n_features), got {data.shape}"
                )
            # Predictions for multiview should be 3D: (n_pred, horizon, n_features).
            if predicted_data is not None:
                if len(predicted_data.shape) != 3:
                    raise ValueError(
                        f"Multiview predictions must have shape (n_pred, horizon, n_features), got {predicted_data.shape}"
                    )
                # Ensure the number of features in the predictions matches the input.
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
        logger.debug("Detector Input Dimensions:")
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
        """Run single-view detection.

        Args:
            data: Input sequence.
            predicted_data: Optional predicted values.

        Returns:
            Detection results dictionary.
        """
        # Convert predicted_data to list-of-lists if provided (required by compute_martingale).
        pred_data_list = predicted_data.tolist() if predicted_data is not None else None

        # Create martingale config from detector config
        martingale_config = MartingaleConfig(
            threshold=self.config.threshold,
            history_size=self.config.history_size,
            reset=self.config.reset,
            window_size=self.config.max_window,
            betting_func_config=self.config.betting_func_config,
        )

        # Call the compute_martingale function from the martingale module.
        results = compute_martingale(
            data=data.tolist(),  # Convert data to a list.
            predicted_data=pred_data_list,
            config=martingale_config,
        )
        return results

    def _run_multiview(
        self, data: Array, predicted_data: Optional[Array]
    ) -> Dict[str, Any]:
        """Run multiview detection.

        Args:
            data: Input sequence.
            predicted_data: Optional predicted values.

        Returns:
            Detection results dictionary.
        """
        # Split each feature (column) into a separate view.
        views = [data[:, i : i + 1] for i in range(data.shape[1])]
        logger.debug("Multiview Processing:")
        logger.debug(f"  Number of views: {len(views)}")
        logger.debug(f"  Each view shape (Tx1): {views[0].shape}")

        # Split predicted data into separate views if provided.
        predicted_views = None
        if predicted_data is not None:
            predicted_views = [
                predicted_data[..., i : i + 1] for i in range(predicted_data.shape[-1])
            ]
            logger.debug(
                f"  Each predicted view shape (T'xHx1): {predicted_views[0].shape}"
            )
        logger.debug("-" * 50)

        # Convert each view (and predicted view) to a list format.
        data_lists = [view.tolist() for view in views]
        pred_data_lists = (
            [view.tolist() for view in predicted_views]
            if predicted_views is not None
            else None
        )

        # Create martingale config from detector config
        martingale_config = MartingaleConfig(
            threshold=self.config.threshold,
            history_size=self.config.history_size,
            reset=self.config.reset,
            window_size=self.config.max_window,
            betting_func_config=self.config.betting_func_config,
        )

        # Call the multiview martingale test function.
        results = multiview_martingale_test(
            data=data_lists,
            predicted_data=pred_data_lists,
            config=martingale_config,
        )
        return results
