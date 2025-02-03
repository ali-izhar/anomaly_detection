# src/changepoint/detector.py

"""Change point detection using the martingale framework derived from conformal prediction."""

from typing import List, Dict, Any, Optional, Callable, Union
import logging
import numpy as np

from .betting import (
    power_martingale,
    exponential_martingale,
    mixture_martingale,
    constant_martingale,
    beta_martingale,
    kernel_density_martingale,
)
from .martingale import compute_martingale, multiview_martingale_test

logger = logging.getLogger(__name__)

# Mapping of betting function names to their implementations.
BETTING_FUNCTIONS = {
    "power": power_martingale,
    "exponential": exponential_martingale,
    "mixture": mixture_martingale,
    "constant": constant_martingale,
    "beta": beta_martingale,
    "kernel": kernel_density_martingale,
}


class ChangePointDetector:
    """Detector class for identifying change points in sequential data using
    the martingale framework derived from conformal prediction.

    Main Steps:
      1. Compute a strangeness measure for new points.
      2. Convert strangeness to a p-value via a nonparametric rank-based method.
      3. Update a martingale using a chosen betting function.
      4. Report a change point if the martingale exceeds a threshold.

    Attributes:
        martingale_method (str): The martingale method to use ('single_view' or 'multiview').
        threshold (float): Detection threshold for the martingale.
        epsilon (float): Sensitivity parameter for martingale updates.
        random_state (int): Seed for reproducibility.
        batch_size (int): Batch size for multiview processing.
        reset (bool): Whether to reset after detection (for single view).
        max_window (int): Maximum window size for strangeness computation.
        distance_measure (str): Distance metric for strangeness computation.
        distance_p (float): Order parameter for Minkowski distance.
    """

    def __init__(
        self,
        martingale_method: str = "multiview",
        history_size: int = 10,
        threshold: float = 60.0,
        epsilon: float = 0.7,
        random_state: Optional[int] = 42,
        batch_size: int = 1000,
        reset: bool = True,
        max_window: Optional[int] = None,
        betting_func: Union[str, Callable] = "power",
        distance_measure: str = "euclidean",
        distance_p: float = 2.0,
    ):
        """Initialize the detector with specified parameters."""
        self.method = martingale_method
        self.history_size = history_size
        self.threshold = threshold
        self.epsilon = epsilon
        self.random_state = random_state
        self.batch_size = batch_size
        self.reset = reset
        self.max_window = max_window
        self.distance_measure = distance_measure
        self.distance_p = distance_p

        # Handle betting function selection.
        if isinstance(betting_func, str):
            if betting_func not in BETTING_FUNCTIONS:
                raise ValueError(
                    f"Unknown betting function '{betting_func}'. "
                    f"Available options are: {list(BETTING_FUNCTIONS.keys())}"
                )
            self.betting_func = BETTING_FUNCTIONS[betting_func]
        else:
            self.betting_func = betting_func

    def run(
        self,
        data: np.ndarray,
        predicted_data: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Run the change point detection pipeline.

        Args:
            data: np.ndarray of shape (n_samples, n_features)
            predicted_data: Optional[np.ndarray] of shape (n_predictions, horizon, n_features)
        """
        logger.debug("Detector Input Dimensions:")
        logger.debug(f"  Main sequence shape (TxF): {data.shape}")
        if predicted_data is not None:
            logger.debug(f"  Predicted sequence shape (T'xHxF): {predicted_data.shape}")
        logger.debug("-" * 50)

        if self.method == "single_view":
            return self.detect_changes(
                data=data,
                predicted_data=predicted_data,
                threshold=self.threshold,
                epsilon=self.epsilon,
                history_size=self.history_size,
                reset=self.reset,
                max_window=self.max_window,
                random_state=self.random_state,
                betting_func=self.betting_func,
                distance_measure=self.distance_measure,
                distance_p=self.distance_p,
            )
        elif self.method == "multiview":
            # Split each feature into a separate view for multiview detection.
            views = [data[:, i : i + 1] for i in range(data.shape[1])]
            logger.debug("Multiview Processing:")
            logger.debug(f"  Number of views: {len(views)}")
            logger.debug(f"  Each view shape (Tx1): {views[0].shape}")

            # Split predicted features into views if available.
            predicted_views = None
            if predicted_data is not None:
                predicted_views = [
                    predicted_data[..., i : i + 1]
                    for i in range(predicted_data.shape[-1])
                ]
                logger.debug(
                    f"  Each predicted view shape (T'xHx1): {predicted_views[0].shape}"
                )
            logger.debug("-" * 50)

            return self.detect_changes_multiview(
                data=views,
                predicted_data=predicted_views,
                threshold=self.threshold,
                epsilon=self.epsilon,
                history_size=self.history_size,
                max_window=self.max_window,
                batch_size=self.batch_size,
                random_state=self.random_state,
                betting_func=self.betting_func,
                distance_measure=self.distance_measure,
                distance_p=self.distance_p,
            )
        else:
            raise ValueError(f"Invalid method: {self.method}")

    def detect_changes(
        self,
        data: np.ndarray,
        predicted_data: Optional[np.ndarray] = None,
        threshold: float = 60.0,
        epsilon: float = 0.7,
        history_size: int = 10,
        reset: bool = True,
        max_window: Optional[int] = None,
        random_state: Optional[int] = None,
        betting_func: Optional[
            Callable[[float, float, float], float]
        ] = power_martingale,
        distance_measure: str = "euclidean",
        distance_p: float = 2.0,
    ) -> Dict[str, Any]:
        """Detect change points in single-view sequential data."""
        logger.debug("Single-view Detection:")
        logger.debug(f"  Input sequence shape: {data.shape}")
        if predicted_data is not None:
            logger.debug(f"  Predicted sequence shape: {predicted_data.shape}")
        logger.debug(f"  History size: {history_size}")
        logger.debug(f"  Window size: {max_window if max_window else 'None'}")
        logger.debug("-" * 50)

        if data.size == 0:
            raise ValueError("Empty data sequence")

        # Convert predicted_data to list-of-lists if provided.
        pred_data_list = predicted_data.tolist() if predicted_data is not None else None

        # Call the compute_martingale function.
        results = compute_martingale(
            data=data.tolist(),
            predicted_data=pred_data_list,
            threshold=threshold,
            epsilon=epsilon,
            history_size=history_size,
            reset=reset,
            window_size=max_window,
            random_state=random_state,
            betting_func=betting_func,
            distance_measure=distance_measure,
            distance_p=distance_p,
        )

        # Return only the keys that were output from the martingale function.
        return {
            "traditional_change_points": results["traditional_change_points"],
            "horizon_change_points": results["horizon_change_points"],
            "traditional_martingales": results["traditional_martingales"],
            "horizon_martingales": results.get("horizon_martingales"),
        }

    def detect_changes_multiview(
        self,
        data: List[np.ndarray],
        predicted_data: Optional[List[np.ndarray]] = None,
        threshold: float = 60.0,
        epsilon: float = 0.7,
        history_size: int = 10,
        max_window: Optional[int] = None,
        batch_size: Optional[int] = None,
        random_state: Optional[int] = None,
        betting_func: Optional[
            Callable[[float, float, float], float]
        ] = power_martingale,
        distance_measure: str = "euclidean",
        distance_p: float = 2.0,
    ) -> Dict[str, Any]:
        """Detect change points in multiview sequential data."""
        logger.debug("Multiview Detection Processing:")
        logger.debug(f"  Number of views: {len(data)}")
        logger.debug(f"  Each view shape: {data[0].shape}")
        if predicted_data:
            logger.debug(f"  Number of predicted views: {len(predicted_data)}")
            logger.debug(f"  Each predicted view shape: {predicted_data[0].shape}")
        logger.debug(f"  History size: {history_size}")
        logger.debug(f"  Window size: {max_window if max_window else 'None'}")
        logger.debug(f"  Batch size: {batch_size}")
        logger.debug("-" * 50)

        if not isinstance(data, list) or len(data) == 0:
            raise ValueError("Data must be a non-empty list of views")

        # Convert each view's data to list format.
        data_lists = [view.tolist() for view in data]

        # Convert predicted data if provided.
        pred_data_lists = (
            [view.tolist() for view in predicted_data]
            if predicted_data is not None
            else None
        )

        # Call the multiview martingale test function.
        results = multiview_martingale_test(
            data=data_lists,
            predicted_data=pred_data_lists,
            threshold=threshold,
            epsilon=epsilon,
            history_size=history_size,
            window_size=max_window,
            batch_size=batch_size,
            random_state=random_state,
            betting_func=betting_func,
            distance_measure=distance_measure,
            distance_p=distance_p,
        )

        # Return only the keys that were output from the multiview martingale function.
        return {
            "traditional_change_points": results["traditional_change_points"],
            "horizon_change_points": results["horizon_change_points"],
            "traditional_sum_martingales": results["traditional_sum_martingales"],
            "traditional_avg_martingales": results["traditional_avg_martingales"],
            "horizon_sum_martingales": results["horizon_sum_martingales"],
            "horizon_avg_martingales": results["horizon_avg_martingales"],
            "individual_traditional_martingales": results[
                "individual_traditional_martingales"
            ],
            "individual_horizon_martingales": results["individual_horizon_martingales"],
        }
