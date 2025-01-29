# src/algorithm.py

"""
Implements Algorithm 1 from Section 5 of the paper:
'Faster Structural Change Detection in Dynamic Networks via Statistical Forecasting'

The algorithm maintains two parallel martingale streams:
1) M_t (observed): computed on current graph features (equation 14)
2) Mhat_t (horizon): computed on predicted future graphs (equation 15)

Key steps (line numbers reference Algorithm 1 in paper):
1. Initialize martingales M_0 = Mhat_0 = 1 [Line 1]
2. For each time t: [Line 2]
   a) Extract features f_t from current graph G_t [Line 3]
   b) Predict h future graphs {Ghat_{t+j}}_{j=1}^h using forecast method [Line 4]
   c) Extract features from predicted graphs {fhat_{t+j}}_{j=1}^h
   d) Update both martingales [Lines 6-7]
   e) Check for change point [Line 8]
   f) Update history and parameters [Lines 9-10]
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


def extract_numeric_features(feature_dict: dict) -> np.ndarray:
    """Extract numeric features from feature dictionary in a consistent order."""
    # Extract basic metrics
    degrees = feature_dict.get("degrees", [])
    avg_degree = np.mean(degrees) if degrees else 0.0
    density = feature_dict.get("density", 0.0)
    clustering = feature_dict.get("clustering", [])
    avg_clustering = np.mean(clustering) if clustering else 0.0

    # Extract centrality metrics
    betweenness = feature_dict.get("betweenness", [])
    avg_betweenness = np.mean(betweenness) if betweenness else 0.0
    eigenvector = feature_dict.get("eigenvector", [])
    avg_eigenvector = np.mean(eigenvector) if eigenvector else 0.0
    closeness = feature_dict.get("closeness", [])
    avg_closeness = np.mean(closeness) if closeness else 0.0

    # Extract spectral metrics
    singular_values = feature_dict.get("singular_values", [])
    largest_sv = max(singular_values) if singular_values else 0.0
    laplacian_eigenvalues = feature_dict.get("laplacian_eigenvalues", [])
    smallest_nonzero_le = (
        min(x for x in laplacian_eigenvalues if x > 1e-10)
        if laplacian_eigenvalues
        else 0.0
    )

    return np.array(
        [
            avg_degree,
            density,
            avg_clustering,
            avg_betweenness,
            avg_eigenvector,
            avg_closeness,
            largest_sv,
            smallest_nonzero_le,
        ]
    )


class TraditionalMartingale:
    """Handles traditional martingale (M_t) computation on current graph features."""

    def __init__(
        self,
        threshold: float,
        epsilon: float,
        window_size: int = 10,
        random_state: Optional[int] = None,
    ):
        """Initialize the traditional martingale detector.

        Args:
            threshold: Detection threshold for martingale (default: 60.0)
            epsilon: Sensitivity parameter for martingale (default: 0.7)
            window_size: Size of sliding window for history (default: 10)
            random_state: Random seed for reproducibility
        """
        self.threshold = threshold
        self.epsilon = epsilon
        self.window_size = window_size
        self.random_state = random_state
        self.feature_extractor = NetworkFeatureExtractor()
        self.detector = ChangePointDetector()

        # Initialize martingales to 1 for each feature
        self.M = {
            feature: 1.0
            for feature in [
                "degree",
                "density",
                "clustering",
                "betweenness",
                "eigenvector",
                "closeness",
                "singular_value",
                "laplacian",
            ]
        }

        # Store detection results
        self.change_points = []
        self.feature_results = {
            feature: {
                "observed_martingales": [
                    1.0
                ],  # Changed from "martingales" to "observed_martingales"
                "p_values": [],
                "strangeness": [],
                "change_points": [],
            }
            for feature in self.M.keys()
        }

        # Initialize history
        self.history = []  # H_0 ← ∅

    def detect_changes(self, data: np.ndarray) -> Dict[str, Any]:
        """Sequential martingale-based change detection following Algorithm 1."""
        combined_martingales = []

        # For t=1 to n
        for t in range(len(data)):
            # Get current feature value
            current_value = data[t]

            # Update history H_t
            self.history.append({"features": current_value})
            if len(self.history) > self.window_size:
                self.history = self.history[-self.window_size :]  # Sliding window

            # For each feature (in this case just one)
            detected_at_t = False
            feature_sum = 0.0

            # Get feature value and history
            feature_value = current_value.item()  # Convert to scalar
            feature_history = [h["features"].item() for h in self.history[:-1]]

            # Compute p-value
            if feature_history:
                p_t = get_pvalue(
                    [*feature_history, feature_value],
                    random_state=self.random_state,
                )
            else:
                p_t = 1.0

            # Update martingale M_t^(i) ← M_{t-1}^(i) × ε(p_t^(i))^(ε-1)
            self.M["degree"] *= self.epsilon * (
                p_t ** (self.epsilon - 1)
            )  # Use any feature name as key

            # Store results
            self.feature_results["degree"]["observed_martingales"].append(
                self.M["degree"]
            )
            self.feature_results["degree"]["p_values"].append(p_t)
            self.feature_results["degree"]["strangeness"].append(0.0)  # Placeholder

            feature_sum = self.M["degree"]

            # Check for change point
            if self.M["degree"] >= self.threshold:
                detected_at_t = True
                self.feature_results["degree"]["change_points"].append(t)

            # Store combined martingale
            combined_martingales.append(feature_sum)

            # Record change point if detected
            if detected_at_t:
                self.change_points.append(t)

        # Store combined results
        self.observed_martingales = {
            "martingales": combined_martingales,
            "martingale_sum": combined_martingales,
            "martingale_avg": combined_martingales,  # Same as sum for single feature
            "p_values": self.feature_results["degree"]["p_values"],
            "strangeness": self.feature_results["degree"]["strangeness"],
        }

        return {
            "change_points": self.change_points,
            "feature_results": self.feature_results,
            "observed_martingales": self.observed_martingales,
        }

    def get_statistics(self) -> Dict[str, float]:
        """Get detection statistics after running detection.

        Returns:
            Dictionary containing martingale statistics
        """
        if self.observed_martingales is None:
            raise RuntimeError("Must run detect_changes() before getting statistics")

        return {
            "final_sum_martingale": self.observed_martingales["martingale_sum"][-1],
            "final_avg_martingale": self.observed_martingales["martingale_avg"][-1],
            "max_sum_martingale": np.max(self.observed_martingales["martingale_sum"]),
            "max_avg_martingale": np.max(self.observed_martingales["martingale_avg"]),
        }


class HorizonMartingale:
    """Handles horizon martingale (Mhat_t) computation on predicted future graph features.

    This implements Algorithm 1 from the paper, maintaining two parallel martingale streams:
    1) M_t (observed): computed on current graph features (equation 14)
    2) Mhat_t (horizon): computed on predicted future graphs (equation 15)
    """

    def __init__(
        self,
        threshold: float,
        epsilon: float,
        horizon: int = 5,
        predictor_type: str = "adaptive",
        predictor_config: Optional[Dict[str, Any]] = None,
        random_state: Optional[int] = None,
    ):
        """Initialize the horizon martingale detector.

        Args:
            threshold: Detection threshold for martingale
            epsilon: Sensitivity parameter for martingale
            horizon: Number of future steps to predict
            predictor_type: Type of predictor to use ("adaptive", "auto", "statistical")
            predictor_config: Optional configuration for the predictor
            random_state: Random seed for reproducibility
        """
        self.threshold = threshold
        self.epsilon = epsilon
        self.horizon = horizon
        self.random_state = random_state

        # Initialize detectors and feature extractors
        self.feature_extractor = NetworkFeatureExtractor()

        # Initialize predictor using factory
        self.predictor = PredictorFactory.create(predictor_type, predictor_config)

        # Initialize martingales to 1 for each feature
        self.M = {
            feature: 1.0
            for feature in [
                "degree",
                "density",
                "clustering",
                "betweenness",
                "eigenvector",
                "closeness",
                "singular_value",
                "laplacian",
            ]
        }
        self.Mhat = self.M.copy()  # Initialize horizon martingales

        # Store detection results
        self.change_points = []
        self.feature_results = {
            feature: {
                "observed_martingales": [1.0],  # Initialize with M_0
                "horizon_martingales": [1.0],  # Initialize with Mhat_0
                "p_values": [],
                "strangeness": [],
                "change_points": [],
            }
            for feature in self.M.keys()
        }

        # Initialize history
        self.history = []  # H_0 ← ∅

    def detect_changes(self, data: np.ndarray) -> Dict[str, Any]:
        """Sequential martingale-based change detection following Algorithm 1."""
        observed_martingales = []
        horizon_martingales = []

        # For t=1 to n
        for t in range(len(data)):
            # Get current feature value
            current_value = data[t]

            # Update history H_t
            self.history.append({"features": current_value})

            # Predict future states if enough history
            future_features = []
            if len(self.history) >= self.predictor.history_size:
                # {Ghat_{t+j}}_{j=1}^h ← Forecast(H_t, G_t)
                future_values = self.predictor.predict(
                    self.history[-self.predictor.history_size :], horizon=self.horizon
                )
                future_features = [fv.reshape(-1, 1) for fv in future_values]

            # For each feature (in this case just one)
            detected_at_t = False
            obs_sum = 0.0
            hor_sum = 0.0

            # Get feature value and history
            feature_value = current_value.item()  # Convert to scalar
            feature_history = [h["features"].item() for h in self.history[:-1]]

            # Compute p-value for observed
            if feature_history:
                p_t = get_pvalue(
                    [*feature_history, feature_value],
                    random_state=self.random_state,
                )
            else:
                p_t = 1.0

            # Update observed martingale M_t^(i) ← M_{t-1}^(i) × ε(p_t^(i))^(ε-1)
            self.M["degree"] *= self.epsilon * (p_t ** (self.epsilon - 1))

            # Update horizon martingale if we have predictions
            if future_features:
                # Compute p-values for predicted features
                future_values = [f.item() for f in future_features]
                future_pvalues = []
                for fv in future_values:
                    p_hat = get_pvalue(
                        [*feature_history, feature_value, fv],
                        random_state=self.random_state,
                    )
                    future_pvalues.append(p_hat)

                # Mhat_t^(i) ← M_{t-1}^(i) × ∏_{j=1}^h ε(phat_{t+j}^(i))^(ε-1)
                self.Mhat["degree"] = self.M["degree"] * np.prod(
                    [self.epsilon * (p ** (self.epsilon - 1)) for p in future_pvalues]
                )
            else:
                self.Mhat["degree"] = self.M["degree"]

            # Store results
            self.feature_results["degree"]["observed_martingales"].append(
                self.M["degree"]
            )
            self.feature_results["degree"]["horizon_martingales"].append(
                self.Mhat["degree"]
            )
            self.feature_results["degree"]["p_values"].append(p_t)
            self.feature_results["degree"]["strangeness"].append(0.0)  # Placeholder

            obs_sum = self.M["degree"]
            hor_sum = self.Mhat["degree"]

            # Check for change point
            if (
                self.M["degree"] >= self.threshold
                or self.Mhat["degree"] >= self.threshold
            ):
                detected_at_t = True
                self.feature_results["degree"]["change_points"].append(t)

            # Store combined martingales
            observed_martingales.append(obs_sum)
            horizon_martingales.append(hor_sum)

            # Record change point if detected
            if detected_at_t:
                self.change_points.append(t)

            # Update predictor state if needed
            if len(self.history) >= self.predictor.history_size:
                self.predictor.update_state(self.history[-1])

        # Store combined results
        self.observed_martingales = {
            "martingales": observed_martingales,
            "martingale_sum": observed_martingales,
            "martingale_avg": observed_martingales,  # Same as sum for single feature
            "p_values": self.feature_results["degree"]["p_values"],
            "strangeness": self.feature_results["degree"]["strangeness"],
        }

        self.horizon_martingales = {
            "martingales": horizon_martingales,
            "martingale_sum": horizon_martingales,
            "martingale_avg": horizon_martingales,  # Same as sum for single feature
            "p_values": [],  # Placeholder
            "strangeness": [],  # Placeholder
        }

        return {
            "change_points": self.change_points,
            "feature_results": self.feature_results,
            "observed_martingales": self.observed_martingales,
            "horizon_martingales": self.horizon_martingales,
        }

    def get_statistics(self) -> Dict[str, float]:
        """Get detection statistics after running detection.

        Returns:
            Dictionary containing martingale statistics for both streams
        """
        if self.observed_martingales is None or self.horizon_martingales is None:
            raise RuntimeError("Must run detect_changes() before getting statistics")

        return {
            # Observed martingale statistics
            "final_sum_martingale": self.observed_martingales["martingale_sum"][-1],
            "final_avg_martingale": self.observed_martingales["martingale_avg"][-1],
            "max_sum_martingale": np.max(self.observed_martingales["martingale_sum"]),
            "max_avg_martingale": np.max(self.observed_martingales["martingale_avg"]),
            # Horizon martingale statistics
            "final_sum_horizon": self.horizon_martingales["martingale_sum"][-1],
            "final_avg_horizon": self.horizon_martingales["martingale_avg"][-1],
            "max_sum_horizon": np.max(self.horizon_martingales["martingale_sum"]),
            "max_avg_horizon": np.max(self.horizon_martingales["martingale_avg"]),
        }


def run_forecast_martingale_detection(
    graph_sequence: List[nx.Graph],
    horizon: int = 5,
    threshold: float = 60.0,
    epsilon: float = 0.7,
    window_size: int = 10,
    predictor_type: str = "adaptive",
    predictor_config: Optional[Dict[str, Any]] = None,
    random_state: Optional[int] = None,
    progress_callback: Optional[Callable[[int], None]] = None,
) -> Dict[str, Any]:
    """Run the forecast-based martingale detection algorithm using dual streams.

    This implements Algorithm 1 from the paper, running both traditional and horizon
    martingales in parallel for faster change detection.
    """
    # Convert graphs to adjacency matrices
    adj_matrices = [nx.to_numpy_array(g) for g in graph_sequence]

    # Extract features from all graphs
    feature_extractor = NetworkFeatureExtractor()
    features_numeric = []
    for adj_matrix in adj_matrices:
        graph = nx.from_numpy_array(adj_matrix)
        feature_dict = feature_extractor.get_features(graph)
        numeric_features = extract_numeric_features(feature_dict)
        features_numeric.append(numeric_features)

    # Convert to numpy array
    data = np.array(features_numeric)

    # Initialize detectors
    traditional = TraditionalMartingale(
        threshold=threshold,
        epsilon=epsilon,
        window_size=window_size,
        random_state=random_state,
    )

    horizon = HorizonMartingale(
        threshold=threshold,
        epsilon=epsilon,
        horizon=horizon,
        predictor_type=predictor_type,
        predictor_config=predictor_config,
        random_state=random_state,
    )

    # Run detectors on each feature dimension
    feature_names = [
        "degree",
        "density",
        "clustering",
        "betweenness",
        "eigenvector",
        "closeness",
        "singular_value",
        "laplacian",
    ]

    individual_results_trad = []
    individual_results_horizon = []
    all_change_points = set()

    for i in range(data.shape[1]):
        # Extract single feature dimension
        feature_data = data[:, i : i + 1]

        # Run traditional detector on this feature
        trad_result = traditional.detect_changes(feature_data)
        individual_results_trad.append(trad_result)
        all_change_points.update(trad_result["change_points"])

        # Run horizon detector on this feature
        horizon_result = horizon.detect_changes(feature_data)
        individual_results_horizon.append(horizon_result)
        all_change_points.update(horizon_result["change_points"])

        if progress_callback:
            progress_callback(i + 1)

    # Combine results
    feature_results = {}
    for i, feature_name in enumerate(feature_names):
        feature_results[feature_name] = {
            "observed_martingales": individual_results_trad[i]["observed_martingales"][
                "martingale_sum"
            ],
            "horizon_martingales": individual_results_horizon[i]["horizon_martingales"][
                "martingale_sum"
            ],
            "change_points": sorted(
                list(
                    set(
                        individual_results_trad[i]["change_points"]
                        + individual_results_horizon[i]["change_points"]
                    )
                )
            ),
            "p_values": individual_results_trad[i]["feature_results"][feature_name][
                "p_values"
            ],
            "strangeness": individual_results_trad[i]["feature_results"][feature_name][
                "strangeness"
            ],
        }

    # Calculate combined martingales
    observed_sum = np.zeros(len(adj_matrices))
    horizon_sum = np.zeros(len(adj_matrices))

    for feature in feature_names:
        observed_sum += np.array(feature_results[feature]["observed_martingales"])
        horizon_sum += np.array(feature_results[feature]["horizon_martingales"])

    return {
        "change_points": sorted(list(all_change_points)),
        "traditional_changes": sorted(
            list(
                set(
                    cp for res in individual_results_trad for cp in res["change_points"]
                )
            )
        ),
        "horizon_changes": sorted(
            list(
                set(
                    cp
                    for res in individual_results_horizon
                    for cp in res["change_points"]
                )
            )
        ),
        "M_observed": observed_sum.tolist(),
        "M_predicted": horizon_sum.tolist(),
        "individual_martingales_obs": [
            feature_results[f]["observed_martingales"] for f in feature_names
        ],
        "individual_martingales_pred": [
            feature_results[f]["horizon_martingales"] for f in feature_names
        ],
        "feature_results": feature_results,
        "statistics": {
            "traditional": {
                "final_sum_martingale": observed_sum[-1],
                "final_avg_martingale": observed_sum[-1] / len(feature_names),
                "max_sum_martingale": np.max(observed_sum),
                "max_avg_martingale": np.max(observed_sum) / len(feature_names),
            },
            "horizon": {
                "final_sum_martingale": horizon_sum[-1],
                "final_avg_martingale": horizon_sum[-1] / len(feature_names),
                "max_sum_martingale": np.max(horizon_sum),
                "max_avg_martingale": np.max(horizon_sum) / len(feature_names),
            },
        },
    }
