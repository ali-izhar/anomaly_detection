# src/changepoint/detector.py

"""Change point detection using the martingale framework derived from conformal prediction."""

from typing import Dict, Any, Optional

import logging
import numpy as np

from .martingale import (
    compute_martingale,
    multiview_martingale_test,
)

logger = logging.getLogger(__name__)


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
        random_state (int): Seed for reproducibility.
        batch_size (int): Batch size for multiview processing.
        reset (bool): Whether to reset after detection (for single view).
        max_window (int): Maximum window size for strangeness computation.
        distance_measure (str): Distance metric for strangeness computation.
        distance_p (float): Order parameter for Minkowski distance.
        betting_func_config (BettingFunctionConfig): Configuration for the betting function.
    """

    def __init__(
        self,
        martingale_method: str = "multiview",
        history_size: int = 10,
        threshold: float = 60.0,
        random_state: Optional[int] = 42,
        batch_size: int = 1000,
        reset: bool = True,
        max_window: Optional[int] = None,
        betting_func_config: Optional[Any] = None,
        distance_measure: str = "euclidean",
        distance_p: float = 2.0,
    ):
        """Initialize the detector with specified parameters."""
        self.method = martingale_method
        self.history_size = history_size
        self.threshold = threshold
        self.random_state = random_state
        self.batch_size = batch_size
        self.reset = reset
        self.max_window = max_window
        self.distance_measure = distance_measure
        self.distance_p = distance_p

        # Set default betting function config if none provided
        if betting_func_config is None:
            betting_func_config = {
                "name": "power",
                "params": {"epsilon": 0.7},
            }
            logger.debug(
                f"No betting function config provided. Using default: {betting_func_config}"
            )
        self.betting_func_config = betting_func_config

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
            if data.size == 0:
                raise ValueError("Empty data sequence")

            # Convert predicted_data to list-of-lists if provided
            pred_data_list = (
                predicted_data.tolist() if predicted_data is not None else None
            )

            # Call the compute_martingale function
            results = compute_martingale(
                data=data.tolist(),
                predicted_data=pred_data_list,
                threshold=self.threshold,
                history_size=self.history_size,
                reset=self.reset,
                window_size=self.max_window,
                random_state=self.random_state,
                betting_func_config=self.betting_func_config,
                distance_measure=self.distance_measure,
                distance_p=self.distance_p,
            )

            return {
                "traditional_change_points": results["traditional_change_points"],
                "horizon_change_points": results["horizon_change_points"],
                "traditional_martingales": results["traditional_martingales"],
                "horizon_martingales": results.get("horizon_martingales"),
            }

        elif self.method == "multiview":
            # Split each feature into a separate view for multiview detection
            views = [data[:, i : i + 1] for i in range(data.shape[1])]
            logger.debug("Multiview Processing:")
            logger.debug(f"  Number of views: {len(views)}")
            logger.debug(f"  Each view shape (Tx1): {views[0].shape}")

            if not views:
                raise ValueError("Data must be a non-empty list of views")

            # Split predicted features into views if available
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

            # Convert data to list format
            data_lists = [view.tolist() for view in views]
            pred_data_lists = (
                [view.tolist() for view in predicted_views]
                if predicted_views is not None
                else None
            )

            # Call the multiview martingale test function
            results = multiview_martingale_test(
                data=data_lists,
                predicted_data=pred_data_lists,
                threshold=self.threshold,
                history_size=self.history_size,
                window_size=self.max_window,
                batch_size=self.batch_size,
                random_state=self.random_state,
                betting_func_config=self.betting_func_config,
                distance_measure=self.distance_measure,
                distance_p=self.distance_p,
            )

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
                "individual_horizon_martingales": results[
                    "individual_horizon_martingales"
                ],
            }

        else:
            raise ValueError(f"Invalid method: {self.method}")
