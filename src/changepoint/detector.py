# src/changepoint/detector.py

"""Change point detection using martingale framework."""

import numpy as np
import logging
from typing import List, Dict, Any, Optional

from .martingale import compute_martingale, multiview_martingale_test

logger = logging.getLogger(__name__)


class ChangePointDetector:
    """Detector class for identifying change points in sequential data using martingale framework."""

    def detect_changes(
        self,
        data: np.ndarray,
        threshold: float,
        epsilon: float,
        max_window: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Detect change points in single-view sequential data.

        Args:
            data: Sequential observations to monitor [n_samples, n_features]
            threshold: Detection threshold tau > 0
            epsilon: Sensitivity epsilon in (0,1)
            max_window: Maximum window size for memory efficiency

        Returns:
            Dictionary containing:
            - change_points: List of detected change points
            - martingale_values: Sequence of martingale values
            - p_values: Sequence of p-values
            - strangeness: Sequence of strangeness values
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
        max_window: Optional[int] = None,
        max_martingale: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Detect change points in multi-view sequential data.

        Args:
            data: List of d feature sequences [n_samples, n_features]
            threshold: Detection threshold tau > 0
            epsilon: Sensitivity epsilon in (0,1)
            max_window: Maximum window size for memory efficiency
            max_martingale: Early stopping threshold

        Returns:
            Dictionary containing:
            - change_points: List of detected change points
            - martingale_values: Combined martingale sequence
            - p_values: P-values per feature
            - strangeness: Strangeness values per feature
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
