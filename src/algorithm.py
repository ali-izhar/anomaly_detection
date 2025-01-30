# src/algorithm.py

"""
Implements Algorithm 1 from Section 5 of the paper:
'Faster Structural Change Detection in Dynamic Networks via Statistical Forecasting'

The algorithm maintains two parallel martingale streams:
1) M_t (observed): computed on current graph features (equation 14)
2) Mhat_t (horizon): computed on predicted future graphs (equation 15)
"""

from collections import deque
from typing import List, Dict, Any, Optional, Callable, Union

import logging
import numpy as np
import networkx as nx

from .changepoint.detector import ChangePointDetector
from .changepoint.pipeline import MartingalePipeline
from .changepoint.strangeness import strangeness_point, get_pvalue
from .graph.features import NetworkFeatureExtractor
from .predictor.factory import PredictorFactory
from .predictor.base import BasePredictor

logger = logging.getLogger(__name__)


class Algorithm:
    pass


class ObservedStream:
    """Handles the observed martingale stream for change detection.

    This class processes the observed network data and maintains a martingale sequence
    for detecting changes in the observed stream. It uses the MartingalePipeline for
    feature extraction and change detection."""

    def __init__(
        self,
        window_size: int = 50,
        threshold: float = 60.0,
        epsilon: float = 0.7,
        martingale_method: str = "multiview",
        feature_set: str = "all",
        batch_size: int = 1000,
        max_martingale: Optional[float] = None,
        reset: bool = True,
        max_window: Optional[int] = None,
        random_state: Optional[int] = 42,
    ):
        """Initialize the observed stream."""
        self.window_size = window_size
        self.current_time = 0

        # Initialize pipeline
        self.pipeline = MartingalePipeline(
            martingale_method=martingale_method,
            threshold=threshold,
            epsilon=epsilon,
            random_state=random_state,
            feature_set=feature_set,
            batch_size=batch_size,
            max_martingale=max_martingale,
            reset=reset,
            max_window=max_window,
        )

        # Initialize buffers
        self.data_buffer = deque(maxlen=window_size)
        self.feature_buffer = deque(maxlen=window_size)

        # Initialize results storage
        self.martingale_values = []
        self.change_points = []
        self.p_values = []
        self.strangeness_values = []
        self.features_raw = []
        self.features_numeric = []

        # For multiview specific results
        self.martingales_sum = []
        self.martingales_avg = []
        self.individual_martingales = []

        # Store parameters
        self.is_multiview = martingale_method == "multiview"
        self.threshold = threshold

    def update(
        self, data: Union[np.ndarray, nx.Graph], data_type: str = "adjacency"
    ) -> Dict[str, Any]:
        """Update the stream with new network data.

        Args:
            data: New network data (adjacency matrix or networkx graph).
            data_type: Type of input data ('adjacency' or 'graph').

        Returns:
            Dict containing detection results including:
                - is_change: Whether a change was detected
                - martingale: Current martingale value
                - p_value: Current p-value
                - change_points: List of all detected change points
                - features: Extracted features
        """
        # Add data to buffer
        self.data_buffer.append(data)
        self.current_time += 1

        # Only start detection once we have enough data
        if len(self.data_buffer) < self.window_size:
            return self._create_result(is_change=False)

        # Run pipeline on current window
        pipeline_result = self.pipeline.run(
            data=list(self.data_buffer), data_type=data_type
        )

        # Store results
        if self.is_multiview:
            self.martingale_values.append(pipeline_result["martingales_sum"][-1])
            self.martingales_sum.append(pipeline_result["martingales_sum"][-1])
            self.martingales_avg.append(pipeline_result["martingales_avg"][-1])
            if pipeline_result["individual_martingales"]:
                self.individual_martingales.append(
                    [m[-1] for m in pipeline_result["individual_martingales"]]
                )
        else:
            self.martingale_values.append(pipeline_result["martingales"][-1])

        self.p_values.append(pipeline_result["p_values"][-1])
        self.strangeness_values.append(pipeline_result["strangeness"][-1])

        # Store features if available
        if "features_raw" in pipeline_result:
            self.features_raw.append(pipeline_result["features_raw"][-1])
        if "features_numeric" in pipeline_result:
            self.features_numeric.append(pipeline_result["features_numeric"][-1])

        # Check for change point
        is_change = False
        current_mart = self.martingale_values[-1]

        if current_mart > self.threshold:
            self.change_points.append(self.current_time)
            is_change = True

        return self._create_result(is_change)

    def _create_result(self, is_change: bool) -> Dict[str, Any]:
        """Create result dictionary with current state.

        Args:
            is_change: Whether a change was detected.

        Returns:
            Dict containing current state and detection results.
        """
        result = {
            "is_change": is_change,
            "current_time": self.current_time,
            "change_points": self.change_points,
        }

        # Add latest values if available
        if self.martingale_values:
            result["martingale"] = self.martingale_values[-1]
            result["p_value"] = self.p_values[-1]
            result["strangeness"] = self.strangeness_values[-1]

            if self.is_multiview:
                result["martingale_sum"] = self.martingales_sum[-1]
                result["martingale_avg"] = self.martingales_avg[-1]
                if self.individual_martingales:
                    result["individual_martingales"] = self.individual_martingales[-1]

        # Add latest features if available
        if self.features_raw:
            result["features_raw"] = self.features_raw[-1]
        if self.features_numeric:
            result["features_numeric"] = self.features_numeric[-1]

        return result

    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the stream.

        Returns:
            Dict containing the complete state of the stream including:
                - All martingale sequences
                - All feature sequences
                - All change points
                - Current parameters
        """
        state = {
            "current_time": self.current_time,
            "change_points": self.change_points,
            "martingale_values": self.martingale_values,
            "p_values": self.p_values,
            "strangeness_values": self.strangeness_values,
            "features_raw": self.features_raw,
            "features_numeric": self.features_numeric,
            "window_size": self.window_size,
            "threshold": self.threshold,
            "is_multiview": self.is_multiview,
        }

        if self.is_multiview:
            state.update(
                {
                    "martingales_sum": self.martingales_sum,
                    "martingales_avg": self.martingales_avg,
                    "individual_martingales": self.individual_martingales,
                }
            )

        return state


class HorizonStream:
    pass
