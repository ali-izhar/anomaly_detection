# src/changepoint/detector.py

import numpy as np
import logging
from typing import List, Dict, Any, Optional

from .martingale import compute_martingale, multiview_martingale_test

logger = logging.getLogger(__name__)


class ChangePointDetector:
    """Detector class for identifying change points in sequential data
    using the martingale framework derived from conformal prediction.

    The main idea is:
    1. Compute a strangeness measure for new points (or sets of points).
    2. Convert strangeness to a p-value using a nonparametric, rank-based method.
    3. Update a power martingale with the new p-value.
    4. If the martingale exceeds a given threshold, report a change point.
    """

    def detect_changes(
        self,
        data: np.ndarray,
        threshold: float,
        epsilon: float,
        reset: bool = False,
        max_window: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Detect change points in single-view sequential data.

        - We treat each row of `data` as a new observation in time.
        - At time i, we compute a strangeness measure for the i-th observation
          relative to a historical window of previous observations.
        - We then compute a conformal p-value for the new observation,
          and update a power martingale:

            M_n = M_(n-1) * epsilon * (p_n)^(epsilon - 1),

          where epsilon in (0,1) adjusts sensitivity to small p-values
          (high anomalies). When M_n > `threshold`, we declare a change.

        Parameters
        ----------
        data : np.ndarray
            Sequential observations to monitor. Shape: (n_samples, n_features).
            - n_samples is the temporal dimension (time steps).
            - n_features can be 1 for truly univariate, or >1 if each time
              step has multiple attributes (treated as a single "point").
        threshold : float
            Detection threshold tau > 0. If martingale M_n > threshold, we
            flag a change point at time n.
        epsilon : float
            Power martingale sensitivity parameter in (0,1).
            - Smaller epsilon => more weight on low p-values => more
              sensitive to anomalies.
        max_window : int, optional
            Maximum window size for memory efficiency (sliding window).
            If None, it uses all past data for strangeness computation.

        Returns
        -------
        Dict[str, Any]
            A dictionary with the following keys:
            - **change_points**: Indices where a change point was detected.
            - **martingale_values**: The sequence of martingale values over time.
            - **p_values**: The conformal p-value sequence.
            - **strangeness**: The computed strangeness values per observation.

        Raises
        ------
        ValueError
            If the input data is empty.
        """
        if len(data) == 0:
            raise ValueError("Empty data sequence")

        logger.info(
            f"Starting change detection with threshold={threshold}, "
            f"epsilon={epsilon}, max_window={max_window}"
        )

        results = compute_martingale(
            data=data.tolist(),
            threshold=threshold,
            epsilon=epsilon,
            reset=reset,
            window_size=max_window,
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
        reset: bool = False,
        max_window: Optional[int] = None,
        max_martingale: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Detect change points in multi-view (multi-feature) sequential data.

        - We assume we have `d` different views (or separate feature sets),
          each producing a time series: data[j] with j in [0, d-1].
        - For each feature j, we maintain a separate power martingale M_j(n).
        - We sum them at each time n: M_total(n) = sum_j M_j(n).
        - If M_total(n) > threshold, we declare a change point.

        Parameters
        ----------
        data : List[np.ndarray]
            Each element is an array of shape (n_samples, ?), representing
            a feature or view over time. For instance, data[j] is the j-th
            feature dimension over the entire timeseries. The length of each
            data[j] must match n_samples, though the number of columns may
            differ.
        threshold : float
            Detection threshold for the combined martingale sum.
        epsilon : float
            Power martingale sensitivity parameter in (0,1).
        max_window : int, optional
            Maximum window size to keep for each feature's historical data.
        max_martingale : float, optional
            If specified, the algorithm will stop early if the sum of
            martingales exceeds this value (an "early stop" criterion).

        Returns
        -------
        Dict[str, Any]
            - **change_points**: Indices in [0, n_samples-1] where a change
              was detected.
            - **martingale_values**: Combined martingale sum at each step.
            - **p_values**: p-values per feature (list of lists).
            - **strangeness**: strangeness values per feature (list of lists).

        Raises
        ------
        ValueError
            If data is empty or threshold is <= 0.
        """
        if not data or len(data) == 0:
            raise ValueError("Empty data sequence")

        logger.info(
            f"Starting multiview detection with threshold={threshold}, "
            f"epsilon={epsilon}, max_window={max_window}, "
            f"max_martingale={max_martingale}"
        )

        # Convert numpy arrays to lists for processing
        data_lists = [d.tolist() for d in data]

        results = multiview_martingale_test(
            data=data_lists,
            threshold=threshold,
            epsilon=epsilon,
            window_size=max_window,
            early_stop_threshold=max_martingale,
        )

        return {
            "change_points": results["change_detected_instant"],
            "martingale_values": results["martingale_sum"],
            "p_values": results["pvalues"],
            "strangeness": results["strangeness"],
        }
