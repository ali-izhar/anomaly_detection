# src/changepoint/detector.py

import numpy as np
import logging
from typing import List, Dict, Any, Optional

from .martingale import compute_martingale, multiview_martingale_test

logger = logging.getLogger(__name__)


class ChangePointDetector:
    """
    Detector class for identifying change points in sequential data
    using the martingale framework derived from conformal prediction.

    Main Steps:
    1. Compute a strangeness measure for new points (or sets of points).
    2. Convert strangeness to a p-value using a nonparametric, rank-based method.
    3. Update a power martingale with the new p-value:
         M_n = M_(n-1) * epsilon * (p_n)^(epsilon - 1)
    4. If the martingale exceeds a threshold, report a change point.
    """

    def detect_changes(
        self,
        data: np.ndarray,
        threshold: float,
        epsilon: float,
        reset: bool = False,
        max_window: Optional[int] = None,
        random_state: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Detect change points in single-view sequential data.

        Parameters
        ----------
        data : np.ndarray
            Shape: (n_samples, n_features). Each row is an observation in time.
            If n_features=1, it's truly univariate. If >1, we treat each row
            as a single "multi-dimensional" observation.
        threshold : float
            Detection threshold (> 0). If the power martingale exceeds this, we flag a change.
        epsilon : float
            Power martingale sensitivity parameter in (0,1).
        reset : bool, optional
            If True, reset the martingale and clear history when a change is detected.
        max_window : int, optional
            Maximum window size for strangeness computation.
            If None, use all past data.
        random_state : int, optional
            Random seed for reproducibility.

        Returns
        -------
        Dict[str, Any]
            {
              "change_points": List[int],
              "martingale_values": np.ndarray,
              "p_values": List[float],
              "strangeness": List[float]
            }
        """
        if len(data) == 0:
            raise ValueError("Empty data sequence")

        logger.info(
            f"Starting change detection (single view) with threshold={threshold}, "
            f"epsilon={epsilon}, reset={reset}, max_window={max_window}, "
            f"random_state={random_state}"
        )

        # Convert to a Python list-of-lists for strangeness routines
        data_list = data.tolist()

        results = compute_martingale(
            data=data_list,
            threshold=threshold,
            epsilon=epsilon,
            reset=reset,
            window_size=max_window,
            random_state=random_state,
        )

        return {
            "change_points": results["change_detected_instant"],
            "martingale_values": results["martingales"],
            "p_values": results["pvalues"],
            "strangeness": results["strangeness"],
        }

    def detect_changes_multiview(
        self,
        data: List[np.ndarray],
        threshold: float,
        epsilon: float,
        max_window: Optional[int] = None,
        max_martingale: Optional[float] = None,
        random_state: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Detect change points in multi-view (multi-feature) sequential data.

        We have d separate feature sets or "views," each a time series:
            data[j]  for j=0..(d-1).
        Each is monitored with its own power martingale. We sum those martingales
        at each time to get M_total(n). If M_total(n) > threshold, we flag a change.

        Parameters
        ----------
        data : List[np.ndarray]
            Each data[j] must be length n_samples along time dimension.
        threshold : float
            Detection threshold for sum of martingales.
        epsilon : float
            Sensitivity parameter in (0,1).
        max_window : int, optional
            Rolling window size for each feature's strangeness history.
        max_martingale : float, optional
            If specified, we stop early if the sum of martingales
            exceeds this "early stop" threshold.
        random_state : int, optional
            Seed for reproducibility.

        Returns
        -------
        Dict[str, Any]
            {
              "change_points": List[int],
              "martingale_values": np.ndarray,
              "p_values": List[List[float]],
              "strangeness": List[List[float]]
            }
        """
        if not data or len(data) == 0:
            raise ValueError("Empty data sequence for multiview detection")

        logger.info(
            f"Starting multiview change detection with threshold={threshold}, "
            f"epsilon={epsilon}, max_window={max_window}, "
            f"max_martingale={max_martingale}, random_state={random_state}"
        )

        # Convert each feature array to list-of-lists
        data_lists = [arr.tolist() for arr in data]

        results = multiview_martingale_test(
            data=data_lists,
            threshold=threshold,
            epsilon=epsilon,
            window_size=max_window,
            early_stop_threshold=max_martingale,
            random_state=random_state,
        )

        return {
            "change_points": results["change_detected_instant"],
            "martingale_values": results["martingale_sum"],
            "p_values": results["pvalues"],
            "strangeness": results["strangeness"],
        }
