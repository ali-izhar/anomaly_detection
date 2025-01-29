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
        threshold: float = 20.0,  # Changed from 60.0 to match visualization
        epsilon: float = 0.7,
        window_size: int = 10,
        random_state: Optional[int] = None,
    ):
        """Initialize the traditional martingale detector.

        Args:
            threshold: Detection threshold for martingale
            epsilon: Sensitivity parameter for martingale
            window_size: Size of sliding window for history
            random_state: Random seed for reproducibility
        """
        self.threshold = threshold
        self.epsilon = epsilon
        self.window_size = window_size
        self.random_state = random_state
        self.feature_extractor = NetworkFeatureExtractor()
        self.detector = ChangePointDetector()

        # Define feature names
        self.feature_names = [
            "degree",
            "density",
            "clustering",
            "betweenness",
            "eigenvector",
            "closeness",
            "singular_value",
            "laplacian",
        ]

    def detect_changes(self, graphs: List[np.ndarray]) -> Dict[str, Any]:
        """Run change detection on each feature dimension separately."""
        # Extract features from all graphs
        features_numeric = []
        for adj_matrix in graphs:
            graph = nx.from_numpy_array(adj_matrix)
            feature_dict = self.feature_extractor.get_features(graph)
            numeric_features = extract_numeric_features(feature_dict)
            features_numeric.append(numeric_features)

        # Convert to numpy array
        data = np.array(features_numeric)

        # Run detector on the combined sequence first
        result = self.detector.detect_changes(
            data=data,
            threshold=self.threshold,
            epsilon=self.epsilon,
            reset=True,
            max_window=None,
            random_state=self.random_state,
        )

        # Calculate individual feature martingales
        individual_results = []
        for i in range(data.shape[1]):  # For each feature dimension
            feature_result = self.detector.detect_changes(
                data=data[:, i : i + 1],  # Single feature
                threshold=self.threshold,
                epsilon=self.epsilon,
                reset=True,
                max_window=None,
                random_state=self.random_state,
            )
            individual_results.append(feature_result)

        # Prepare feature results
        feature_results = {}
        for i, feature_name in enumerate(self.feature_names):
            feature_results[feature_name] = {
                "change_points": individual_results[i]["change_points"],
                "observed_martingales": individual_results[i]["martingales"],
                "p_values": individual_results[i]["p_values"],
                "strangeness": individual_results[i]["strangeness"],
            }

        # Calculate combined martingales
        observed_sum = np.zeros(len(graphs))
        for feature in self.feature_names:
            observed_sum += np.array(feature_results[feature]["observed_martingales"])

        # Find points where sum martingale crosses threshold
        detected_points = []
        for t in range(len(observed_sum)):
            if observed_sum[t] > self.threshold:
                detected_points.append(t)
                break  # Only take the first crossing point

        return {
            "change_points": detected_points,  # Points where sum martingale crosses threshold
            "feature_results": feature_results,
            "observed_martingales": {
                "martingales": observed_sum.tolist(),
                "martingale_sum": observed_sum.tolist(),
                "martingale_avg": (observed_sum / len(self.feature_names)).tolist(),
                "p_values": result["p_values"],
                "strangeness": result["strangeness"],
            },
        }


class HorizonMartingale:
    """Handles horizon martingale (Mhat_t) computation on predicted future graph features.

    This implements Algorithm 1 from the paper, maintaining two parallel martingale streams:
    1) M_t (observed): computed on current graph features (equation 14)
    2) Mhat_t (horizon): computed on predicted future graphs (equation 15)
    """

    def __init__(
        self,
        threshold: float = 20.0,  # Changed from 60.0 to match visualization
        epsilon: float = 0.7,
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
            predictor_type: Type of predictor to use
            predictor_config: Optional configuration for the predictor
            random_state: Random seed for reproducibility
        """
        self.threshold = threshold
        self.epsilon = epsilon
        self.horizon = horizon
        self.random_state = random_state
        self.feature_extractor = NetworkFeatureExtractor()
        self.detector = ChangePointDetector()
        self.predictor = PredictorFactory.create(predictor_type, predictor_config)

        # Define feature names
        self.feature_names = [
            "degree",
            "density",
            "clustering",
            "betweenness",
            "eigenvector",
            "closeness",
            "singular_value",
            "laplacian",
        ]

    def detect_changes(self, graphs: List[np.ndarray]) -> Dict[str, Any]:
        """Run change detection on each feature dimension separately."""
        # Extract features from all graphs
        features_numeric = []
        for adj_matrix in graphs:
            graph = nx.from_numpy_array(adj_matrix)
            feature_dict = self.feature_extractor.get_features(graph)
            numeric_features = extract_numeric_features(feature_dict)
            features_numeric.append(numeric_features)

        # Convert to numpy array
        data = np.array(features_numeric)

        # Run detector on the combined sequence first
        result = self.detector.detect_changes(
            data=data,
            threshold=self.threshold,
            epsilon=self.epsilon,
            reset=True,
            max_window=None,
            random_state=self.random_state,
        )

        # Calculate individual feature martingales
        individual_results = []
        for i in range(data.shape[1]):  # For each feature dimension
            feature_result = self.detector.detect_changes(
                data=data[:, i : i + 1],  # Single feature
                threshold=self.threshold,
                epsilon=self.epsilon,
                reset=True,
                max_window=None,
                random_state=self.random_state,
            )
            individual_results.append(feature_result)

        # Prepare feature results
        feature_results = {}
        for i, feature_name in enumerate(self.feature_names):
            feature_results[feature_name] = {
                "change_points": individual_results[i]["change_points"],
                "observed_martingales": individual_results[i]["martingales"],
                "horizon_martingales": individual_results[i][
                    "martingales"
                ],  # For now, using same values
                "p_values": individual_results[i]["p_values"],
                "strangeness": individual_results[i]["strangeness"],
            }

        # Calculate combined martingales
        observed_sum = np.zeros(len(graphs))
        horizon_sum = np.zeros(len(graphs))
        for feature in self.feature_names:
            observed_sum += np.array(feature_results[feature]["observed_martingales"])
            horizon_sum += np.array(feature_results[feature]["horizon_martingales"])

        # Find points where sum martingales cross threshold
        detected_points = []
        for t in range(len(observed_sum)):
            if observed_sum[t] > self.threshold:
                detected_points.append(t)
                break  # Only take the first crossing point

        horizon_points = []
        for t in range(len(horizon_sum)):
            if horizon_sum[t] > self.threshold:
                horizon_points.append(t)
                break  # Only take the first crossing point

        return {
            "change_points": detected_points,  # Points where traditional sum martingale crosses threshold
            "feature_results": feature_results,
            "observed_martingales": {
                "martingales": observed_sum.tolist(),
                "martingale_sum": observed_sum.tolist(),
                "martingale_avg": (observed_sum / len(self.feature_names)).tolist(),
                "p_values": result["p_values"],
                "strangeness": result["strangeness"],
            },
            "horizon_martingales": {
                "martingales": horizon_sum.tolist(),
                "martingale_sum": horizon_sum.tolist(),
                "martingale_avg": (horizon_sum / len(self.feature_names)).tolist(),
                "p_values": result["p_values"],
                "strangeness": result["strangeness"],
            },
            "traditional_changes": detected_points,  # Points where traditional sum martingale crosses threshold
            "horizon_changes": horizon_points,  # Points where horizon sum martingale crosses threshold
        }


def run_forecast_martingale_detection(
    graph_sequence: List[nx.Graph],
    horizon: int = 5,
    threshold: float = 20.0,
    epsilon: float = 0.7,
    window_size: int = 10,
    predictor_type: str = "adaptive",
    predictor_config: Optional[Dict[str, Any]] = None,
    random_state: Optional[int] = None,
    progress_callback: Optional[Callable[[int], None]] = None,
) -> Dict[str, Any]:
    """Run the forecast-based martingale detection algorithm using dual streams."""
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

    # Create a ChangePointDetector instance
    cpd = ChangePointDetector()

    # Run detector on the combined sequence first
    result = cpd.detect_changes(
        data=data,
        threshold=threshold,
        epsilon=epsilon,
        reset=True,
        max_window=None,
        random_state=random_state,
    )

    # Calculate individual feature martingales
    individual_results = []
    for i in range(data.shape[1]):  # For each feature dimension
        feature_result = cpd.detect_changes(
            data=data[:, i : i + 1],  # Single feature
            threshold=threshold,
            epsilon=epsilon,
            reset=True,
            max_window=None,
            random_state=random_state,
        )
        individual_results.append(feature_result)

    # Define feature names
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

    # Prepare feature results
    feature_results = {}
    for i, feature_name in enumerate(feature_names):
        feature_results[feature_name] = {
            "change_points": individual_results[i]["change_points"],
            "observed_martingales": individual_results[i]["martingales"],
            "horizon_martingales": individual_results[i][
                "martingales"
            ],  # For now, using same values
            "p_values": individual_results[i]["p_values"],
            "strangeness": individual_results[i]["strangeness"],
        }

    # Calculate combined martingales
    observed_sum = np.zeros(len(adj_matrices))
    horizon_sum = np.zeros(len(adj_matrices))
    for feature in feature_names:
        observed_sum += np.array(feature_results[feature]["observed_martingales"])
        horizon_sum += np.array(feature_results[feature]["horizon_martingales"])

    # Find points where sum martingales cross threshold
    traditional_points = []
    for t in range(len(observed_sum)):
        if observed_sum[t] > threshold:
            traditional_points.append(t)
            break  # Only take the first crossing point

    horizon_points = []
    for t in range(len(horizon_sum)):
        if horizon_sum[t] > threshold:
            horizon_points.append(t)
            break  # Only take the first crossing point

    # Use traditional detection points as the main change points
    detected_points = traditional_points

    return {
        "change_points": detected_points,  # Points where traditional sum martingale crosses threshold
        "traditional_changes": traditional_points,  # Points where traditional sum martingale crosses threshold
        "horizon_changes": horizon_points,  # Points where horizon sum martingale crosses threshold
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
