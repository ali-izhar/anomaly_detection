# src/algorithm.py

"""
Implements Algorithm 1 from Section 5 of the paper:
'Faster Structural Change Detection in Dynamic Networks via Statistical Forecasting'

The algorithm maintains two parallel martingale streams:
1) M_t (observed): computed on current graph features (equation 14)
2) Mhat_t (horizon): computed on predicted future graphs (equation 15)
"""

from collections import deque
from typing import List, Dict, Any, Optional, Callable

import logging
import numpy as np
import networkx as nx

from .changepoint.detector import ChangePointDetector
from .changepoint.strangeness import strangeness_point, get_pvalue
from .graph.features import NetworkFeatureExtractor
from .predictor.factory import PredictorFactory
from .predictor.base import BasePredictor

logger = logging.getLogger(__name__)


class Algorithm:
    pass


class ObservedStream:
    """Handles the observed martingale stream for each feature.

    This class maintains and updates martingales for each network feature
    based on observed graph states, following equation 14 from the paper:
    M_t^(i) = M_{t-1}^(i) × ε(p_t^(i))^(ε-1)
    """

    def __init__(
        self,
        window_size: int = 50,
        threshold: float = 20.0,
        epsilon: float = 0.7,
        reset_after_change: bool = True,
    ):
        """Initialize the observed martingale stream.

        Args:
            window_size: Size of rolling window for strangeness computation
            threshold: Detection threshold for martingale values
            epsilon: Sensitivity parameter for martingale updates (0 < ε < 1)
            reset_after_change: Whether to reset martingales after detecting change
        """
        self.window_size = window_size
        self.threshold = threshold
        self.epsilon = epsilon
        self.reset_after_change = reset_after_change

        # Initialize detector
        self.detector = ChangePointDetector()

        # Initialize feature buffers and results
        self.feature_buffers: Dict[str, List[float]] = {}
        self.feature_results: Dict[str, Dict[str, Any]] = {}
        self.detected_changes: Dict[str, List[int]] = {}

        # Track current timestep
        self.current_time = 0

    def initialize_feature(self, feature_name: str):
        """Initialize tracking for a new feature."""
        if feature_name not in self.feature_buffers:
            self.feature_buffers[feature_name] = []
            self.feature_results[feature_name] = {
                "martingales": [],
                "p_values": [],
                "strangeness": [],
                "change_points": [],
            }
            self.detected_changes[feature_name] = []

    def update(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Update martingales with new feature observations.

        Args:
            features: Dictionary mapping feature names to their current values

        Returns:
            Dict containing updated martingales and any detected changes
        """
        self.current_time += 1
        changes_detected = {}

        for feature_name, value in features.items():
            # Initialize tracking for new features
            self.initialize_feature(feature_name)

            # Add new value to buffer
            self.feature_buffers[feature_name].append(value)

            # Convert buffer to numpy array for detector
            data = np.array(self.feature_buffers[feature_name]).reshape(-1, 1)

            # Run detector on the sequence
            result = self.detector.detect_changes(
                data=data,
                threshold=self.threshold,
                epsilon=self.epsilon,
                reset=self.reset_after_change,
                max_window=self.window_size,
                random_state=42,
            )

            # Update feature results - convert numpy arrays to lists if needed
            self.feature_results[feature_name] = {
                "martingales": (
                    result["martingales"].tolist()
                    if isinstance(result["martingales"], np.ndarray)
                    else result["martingales"]
                ),
                "p_values": (
                    result["p_values"].tolist()
                    if isinstance(result["p_values"], np.ndarray)
                    else result["p_values"]
                ),
                "strangeness": (
                    result["strangeness"].tolist()
                    if isinstance(result["strangeness"], np.ndarray)
                    else result["strangeness"]
                ),
                "change_points": result["change_points"],
            }

            # Check for new change points
            if result["change_points"]:
                latest_change = result["change_points"][-1]
                if latest_change == len(data) - 1:  # Change detected at current time
                    self.detected_changes[feature_name].append(self.current_time)
                    changes_detected[feature_name] = self.current_time

        # Get current martingale values
        current_martingales = {
            feature: results["martingales"][-1] if results["martingales"] else 1.0
            for feature, results in self.feature_results.items()
        }

        # Calculate combined martingales
        all_martingales = []
        for feature, results in self.feature_results.items():
            if results["martingales"]:
                padded_martingales = [1.0] * (
                    self.current_time - len(results["martingales"])
                ) + results["martingales"]
                all_martingales.append(padded_martingales)

        if all_martingales:
            all_martingales = np.array(all_martingales)
            sum_martingales = np.sum(all_martingales, axis=0).tolist()
            avg_martingales = (np.sum(all_martingales, axis=0) / len(features)).tolist()

            # Add combined results
            self.feature_results["combined"] = {
                "martingales": sum_martingales,
                "martingale_sum": sum_martingales,
                "martingale_avg": avg_martingales,
                "p_values": [],  # Not used for combined
                "strangeness": [],  # Not used for combined
                "change_points": [],  # Will be populated based on threshold crossings
            }

            # Check for changes in combined martingale
            if sum_martingales[-1] > self.threshold:
                self.detected_changes["combined"] = self.detected_changes.get(
                    "combined", []
                ) + [self.current_time]
                changes_detected["combined"] = self.current_time

        return {
            "time": self.current_time,
            "martingales": current_martingales,
            "changes": changes_detected,
            "all_changes": self.detected_changes.copy(),
            "feature_results": self.feature_results,
        }

    def get_state(self) -> Dict[str, Any]:
        """Get current state of all martingales and detected changes."""
        current_martingales = {
            feature: results["martingales"][-1] if results["martingales"] else 1.0
            for feature, results in self.feature_results.items()
        }
        return {
            "time": self.current_time,
            "martingales": current_martingales,
            "all_changes": self.detected_changes.copy(),
            "feature_results": self.feature_results,
        }


class HorizonStream:
    pass
