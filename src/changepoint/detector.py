# src/changepoint/detector.py

"""Change point detection using the martingale framework derived from conformal prediction."""

from typing import List, Dict, Any, Optional, Callable, Union
import logging
import numpy as np
import networkx as nx

# Import the modular betting functions; by default, we use the power martingale.
from .betting import power_martingale, exponential_martingale, mixture_martingale
from .martingale import compute_martingale, multiview_martingale_test

logger = logging.getLogger(__name__)


class ChangePointDetector:
    """Detector class for identifying change points in sequential data using
    the martingale framework derived from conformal prediction.

    Main Steps:
    1. Compute a strangeness measure for new points (or sets of points).
    2. Convert strangeness to a p-value using a nonparametric, rank-based method.
    3. Update a martingale using a chosen betting function:
         M_n = M_(n-1) * (update factor based on p-value)
    4. If the martingale exceeds a threshold, report a change point.

    Attributes:
        method (str): The martingale method to use ('single_view' or 'multiview').
        threshold (float): Detection threshold for martingale.
        epsilon (float): Sensitivity parameter for martingale updates.
        random_state (int): Seed for reproducibility.
        feature_set (str): Which feature set to use.
        batch_size (int): Batch size for multiview processing.
        max_martingale (float): Early stopping threshold for multiview.
        reset (bool): Whether to reset after detection in single view.
        max_window (int): Maximum window size for strangeness computation.
    """

    def __init__(
        self,
        martingale_method: str = "multiview",
        history_size: int = 10,
        threshold: float = 60.0,
        epsilon: float = 0.7,
        random_state: Optional[int] = 42,
        feature_set: str = "all",
        batch_size: int = 1000,
        max_martingale: Optional[float] = None,
        reset: bool = True,
        max_window: Optional[int] = None,
        betting_func: Optional[
            Callable[[float, float, float], float]
        ] = power_martingale,
    ):
        """Initialize the detector with specified parameters."""
        self.method = martingale_method
        self.history_size = history_size
        self.threshold = threshold
        self.epsilon = epsilon
        self.random_state = random_state
        self.feature_set = feature_set
        self.batch_size = batch_size
        self.max_martingale = max_martingale
        self.reset = reset
        self.max_window = max_window
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
        logger.info("Detector Input Dimensions:")
        logger.info(f"- Main sequence shape (TxF): {data.shape}")
        if predicted_data is not None:
            logger.info(f"- Predicted sequence shape (T'xHxF): {predicted_data.shape}")
        logger.info("-" * 50)

        if self.method == "single_view":
            return self.detect_changes(
                data,
                predicted_data,
                self.threshold,
                self.epsilon,
                self.history_size,
                self.reset,
                self.max_window,
                self.random_state,
                self.betting_func,
            )
        elif self.method == "multiview":
            # Split each feature into a separate view for multiview detection
            views = [data[:, i : i + 1] for i in range(data.shape[1])]
            logger.info("Multiview Processing:")
            logger.info(f"- Number of views: {len(views)}")
            logger.info(f"- Each view shape (Tx1): {views[0].shape}")

            # Split predicted features into views if available
            predicted_views = None
            if predicted_data is not None:
                predicted_views = [
                    predicted_data[..., i : i + 1]
                    for i in range(predicted_data.shape[-1])
                ]
                logger.info(
                    f"- Each predicted view shape (T'xHx1): {predicted_views[0].shape}"
                )
            logger.info("-" * 50)

            return self.detect_changes_multiview(
                views,
                predicted_views,
                self.threshold,
                self.epsilon,
                self.history_size,
                self.max_window,
                self.max_martingale,
                self.batch_size,
                self.random_state,
                self.betting_func,
            )

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
    ) -> Dict[str, Any]:
        """Detect change points in single-view sequential data."""
        logger.info("Single-view Detection:")
        logger.info(f"- Input sequence shape: {data.shape}")
        if predicted_data is not None:
            logger.info(f"- Predicted sequence shape: {predicted_data.shape}")
        logger.info(f"- History size: {history_size}")
        logger.info(f"- Window size: {max_window if max_window else 'None'}")
        logger.info("-" * 50)

        if data.size == 0:
            raise ValueError("Empty data sequence")

        # Convert predicted_data to list-of-lists if provided
        pred_data_list = None
        if predicted_data is not None:
            pred_data_list = predicted_data.tolist()

        results = compute_martingale(
            data=data.tolist(),  # Convert only at the last moment
            predicted_data=pred_data_list,
            threshold=threshold,
            epsilon=epsilon,
            history_size=history_size,
            reset=reset,
            window_size=max_window,
            random_state=random_state,
            betting_func=betting_func,
        )

        return {
            "change_points": results["change_points"],
            "horizon_change_points": results["horizon_change_points"],
            "pvalues": results["pvalues"],
            "strangeness": results["strangeness"],
            "martingales": results["martingales"],
            "prediction_martingales": results.get("prediction_martingales"),
            "prediction_pvalues": results.get("prediction_pvalues"),
            "prediction_strangeness": results.get("prediction_strangeness"),
        }

    def detect_changes_multiview(
        self,
        data: List[np.ndarray],
        predicted_data: Optional[List[np.ndarray]] = None,
        threshold: float = 60.0,
        epsilon: float = 0.7,
        history_size: int = 10,
        max_window: Optional[int] = None,
        max_martingale: Optional[float] = None,
        batch_size: Optional[int] = None,
        random_state: Optional[int] = None,
        betting_func: Optional[
            Callable[[float, float, float], float]
        ] = power_martingale,
    ) -> Dict[str, Any]:
        """Detect change points in multi-view sequential data."""
        logger.info("Multiview Detection Processing:")
        logger.info(f"- Number of views: {len(data)}")
        logger.info(f"- Each view shape: {data[0].shape}")
        if predicted_data:
            logger.info(f"- Number of predicted views: {len(predicted_data)}")
            logger.info(f"- Each predicted view shape: {predicted_data[0].shape}")
        logger.info(f"- History size: {history_size}")
        logger.info(f"- Window size: {max_window if max_window else 'None'}")
        logger.info(f"- Batch size: {batch_size}")
        logger.info("-" * 50)

        if not isinstance(data, list) or len(data) == 0:
            raise ValueError("Data must be a non-empty list of views")

        # Convert each view's data to list format
        data_lists = [view.tolist() for view in data]

        # Convert predicted data if provided
        pred_data_lists = None
        if predicted_data is not None:
            pred_data_lists = [view.tolist() for view in predicted_data]

        results = multiview_martingale_test(
            data=data_lists,
            predicted_data=pred_data_lists,
            threshold=threshold,
            epsilon=epsilon,
            history_size=history_size,
            window_size=max_window,
            early_stop_threshold=max_martingale,
            batch_size=batch_size,
            random_state=random_state,
            betting_func=betting_func,
        )

        return {
            "change_points": results["change_points"],
            "horizon_change_points": results["horizon_change_points"],
            "pvalues": results["pvalues"],
            "strangeness": results["strangeness"],
            "martingale_sum": results["martingale_sum"],
            "martingale_avg": results["martingale_avg"],
            "individual_martingales": results["individual_martingales"],
            "prediction_pvalues": results.get("prediction_pvalues"),
            "prediction_strangeness": results.get("prediction_strangeness"),
            "prediction_martingale_sum": results.get("prediction_martingale_sum"),
            "prediction_martingale_avg": results.get("prediction_martingale_avg"),
            "prediction_individual_martingales": results.get(
                "prediction_individual_martingales"
            ),
        }
