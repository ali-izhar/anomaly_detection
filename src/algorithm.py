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

        # Store detection results
        self.change_points = []
        self.feature_results = {}
        self.combined_martingales = None

        # Initialize history
        self.history = []

        # Initialize martingales for each feature
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
        self.M = {feature: 1.0 for feature in self.feature_names}  # M_0^(i) = 1

    def extract_numeric_features(self, feature_dict: dict) -> np.ndarray:
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

    def detect_changes(self, graphs: List[np.ndarray]) -> Dict[str, Any]:
        """Detect changes in a sequence of graphs using sequential martingale detection.

        Implements equation 14 from the paper:
        M_t^(i) = M_{t-1}^(i) × ε(p_t^(i))^(ε-1)
        """
        # Storage for results
        observed_martingales = {
            feature: [1.0] for feature in self.feature_names
        }  # Include M_0
        change_points = []

        # Process each timestep
        for t, adj_matrix in enumerate(graphs):
            # Extract features from current graph
            graph = nx.from_numpy_array(adj_matrix)
            feature_dict = self.feature_extractor.get_features(graph)
            current_features = self.extract_numeric_features(feature_dict)

            # Update history
            self.history.append({"adjacency": adj_matrix, "features": current_features})
            if len(self.history) > self.window_size:
                self.history = self.history[-self.window_size :]

            # For each feature
            detected_at_t = False
            for i, feature_name in enumerate(self.feature_names):
                # Get feature value and history
                feature_value = current_features[i]
                feature_history = [
                    h["features"][i] for h in self.history[:-1]
                ]  # Exclude current point

                # Compute p-value using conformal score
                if feature_history:
                    p_t = get_pvalue(
                        [*feature_history, feature_value],
                        random_state=self.random_state,
                    )
                else:
                    p_t = 1.0  # No history yet

                # Update martingale using equation 14
                self.M[feature_name] *= self.epsilon * (p_t ** (self.epsilon - 1))
                observed_martingales[feature_name].append(self.M[feature_name])

                # Check for change point
                if self.M[feature_name] >= self.threshold:
                    detected_at_t = True

            # Record change point if detected by any feature
            if detected_at_t:
                change_points.append(t)

        # Store results
        self.feature_results = {
            feature: {
                "martingales": observed_martingales[feature][1:],  # Exclude M_0
                "change_points": change_points,
            }
            for feature in self.feature_names
        }

        # Store combined results
        martingale_sum = [
            sum(m) for m in zip(*[v[1:] for v in observed_martingales.values()])
        ]
        martingale_avg = [
            sum(m) / len(self.feature_names)
            for m in zip(*[v[1:] for v in observed_martingales.values()])
        ]

        self.combined_martingales = {
            "martingales": martingale_sum,
            "martingale_sum": martingale_sum,
            "martingale_avg": martingale_avg,
        }

        self.change_points = change_points

        return {
            "change_points": self.change_points,
            "feature_results": self.feature_results,
            "combined_martingales": self.combined_martingales,
        }

    def get_statistics(self) -> Dict[str, float]:
        """Get detection statistics after running detection.

        Returns:
            Dictionary containing martingale statistics
        """
        if self.combined_martingales is None:
            raise RuntimeError("Must run detect_changes() before getting statistics")

        return {
            "final_sum_martingale": self.combined_martingales["martingale_sum"][-1],
            "final_avg_martingale": self.combined_martingales["martingale_avg"][-1],
            "max_sum_martingale": np.max(self.combined_martingales["martingale_sum"]),
            "max_avg_martingale": np.max(self.combined_martingales["martingale_avg"]),
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
        self.detector = ChangePointDetector()
        self.feature_extractor = NetworkFeatureExtractor()

        # Initialize predictor using factory
        self.predictor = PredictorFactory.create(predictor_type, predictor_config)

        # Store detection results
        self.change_points = []
        self.feature_results = {}
        self.observed_martingales = None
        self.horizon_martingales = None

        # Store history for prediction
        self.history = []

        # Initialize traditional martingale for M_{t-1} values
        self.traditional = TraditionalMartingale(
            threshold=threshold,
            epsilon=epsilon,
            window_size=10,  # Same as predictor history size
            random_state=random_state,
        )

    def extract_numeric_features(self, feature_dict: dict) -> np.ndarray:
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

    def detect_changes(self, graphs: List[np.ndarray]) -> Dict[str, Any]:
        """Detect changes using both observed and horizon martingales.

        Implements equations 14 and 15 from the paper:
        M_t^(i) = M_{t-1}^(i) × ε(p_t^(i))^(ε-1)           [eq. 14]
        Mhat_t^(i) = M_{t-1}^(i) × ∏(j=1 to h) ε(phat_{t+j}^(i))^(ε-1)  [eq. 15]
        where M_{t-1}^(i) comes from the traditional martingale
        """
        # Initialize feature names
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

        # Storage for results
        observed_martingales = {feature: [] for feature in feature_names}
        horizon_martingales = {feature: [] for feature in feature_names}
        change_points = []

        # Process each timestep
        for t, adj_matrix in enumerate(graphs):
            # Extract features from current graph
            graph = nx.from_numpy_array(adj_matrix)
            feature_dict = self.feature_extractor.get_features(graph)
            current_features = self.extract_numeric_features(feature_dict)

            # Update history for prediction
            self.history.append({"adjacency": adj_matrix})
            if len(self.history) > self.predictor.history_size:
                self.history = self.history[-self.predictor.history_size :]

            # For each feature
            detected_at_t = False
            for i, feature_name in enumerate(feature_names):
                # 1. Update traditional martingale to get M_{t-1}^(i)
                M_t = self.traditional.M[feature_name]  # Get M_{t-1} before updating

                # Update traditional martingale for current point
                obs_result = self.detector.detect_changes(
                    data=np.array([current_features[i]]).reshape(-1, 1),
                    threshold=self.threshold,
                    epsilon=self.epsilon,
                    reset=False,
                    max_window=None,
                    random_state=self.random_state,
                )

                # Store observed martingale value
                M_t_new = obs_result["martingales"][-1]
                observed_martingales[feature_name].append(M_t_new)
                self.traditional.M[feature_name] = M_t_new  # Update for next iteration

                # 2. Update horizon martingale if we have enough history
                if len(self.history) == self.predictor.history_size:
                    # Predict future states
                    future_graphs = self.predictor.predict(
                        self.history, horizon=self.horizon
                    )

                    # Extract features from predicted graphs
                    future_features = []
                    for future_adj in future_graphs:
                        future_graph = nx.from_numpy_array(future_adj)
                        future_dict = self.feature_extractor.get_features(future_graph)
                        future_numeric = self.extract_numeric_features(future_dict)
                        future_features.append(future_numeric[i])

                    # Get p-values for predicted features using detector
                    horizon_result = self.detector.detect_changes(
                        data=np.array(future_features).reshape(-1, 1),
                        threshold=self.threshold,
                        epsilon=self.epsilon,
                        reset=False,
                        max_window=None,
                        random_state=self.random_state,
                    )

                    # Use M_{t-1} as base and multiply by product of predicted p-values
                    Mhat_t = M_t * np.prod(
                        [
                            self.epsilon * (p ** (self.epsilon - 1))
                            for p in horizon_result["p_values"]
                        ]
                    )
                    horizon_martingales[feature_name].append(Mhat_t)

                    # Check for change point
                    if M_t_new >= self.threshold or Mhat_t >= self.threshold:
                        detected_at_t = True
                else:
                    # When no prediction possible, use observed martingale
                    horizon_martingales[feature_name].append(M_t_new)

            # Record change point if detected by any feature
            if detected_at_t:
                change_points.append(t)

            # Update predictor state
            if len(self.history) == self.predictor.history_size:
                self.predictor.update_state(self.history[-1])

        # Store results
        self.feature_results = {
            feature: {
                "observed_martingales": observed_martingales[feature],
                "horizon_martingales": horizon_martingales[feature],
                "change_points": change_points,
            }
            for feature in feature_names
        }

        # Store combined results
        self.observed_martingales = {
            "martingale_sum": [sum(m) for m in zip(*observed_martingales.values())],
            "martingale_avg": [
                sum(m) / len(feature_names) for m in zip(*observed_martingales.values())
            ],
        }

        self.horizon_martingales = {
            "martingale_sum": [sum(m) for m in zip(*horizon_martingales.values())],
            "martingale_avg": [
                sum(m) / len(feature_names) for m in zip(*horizon_martingales.values())
            ],
        }

        self.change_points = change_points

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

    Args:
        graph_sequence: List of networkx graphs to analyze
        horizon: Number of future steps to predict
        threshold: Detection threshold for martingales
        epsilon: Sensitivity parameter for martingales
        window_size: Size of sliding window for prediction
        predictor_type: Type of predictor to use ("adaptive", "auto", "statistical")
        predictor_config: Optional configuration for the predictor
        random_state: Random seed for reproducibility
        progress_callback: Optional callback function to report progress

    Returns:
        Dictionary containing:
        - change_points: Combined list of detected change points
        - traditional_changes: Change points from traditional martingale
        - horizon_changes: Change points from horizon martingale
        - M_observed: Traditional martingale values
        - M_predicted: Horizon martingale values
        - individual_martingales_obs: Feature-specific traditional martingales
        - individual_martingales_pred: Feature-specific horizon martingales
    """
    # Convert graphs to adjacency matrices
    adj_matrices = [nx.to_numpy_array(g) for g in graph_sequence]

    # Initialize both martingale detectors
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

    # Run both detectors
    trad_results = traditional.detect_changes(adj_matrices)
    horizon_results = horizon.detect_changes(adj_matrices)

    # Get statistics from both detectors
    trad_stats = traditional.get_statistics()
    horizon_stats = horizon.get_statistics()

    # Combine change points from both streams
    all_change_points = sorted(
        list(set(trad_results["change_points"] + horizon_results["change_points"]))
    )

    # Extract feature-specific martingales
    individual_martingales_obs = []
    individual_martingales_pred = []

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

    for feature in feature_names:
        # Traditional martingales for each feature
        if feature in trad_results["feature_results"]:
            individual_martingales_obs.append(
                trad_results["feature_results"][feature]["martingales"]
            )

        # Horizon martingales for each feature
        if feature in horizon_results["feature_results"]:
            individual_martingales_pred.append(
                horizon_results["feature_results"][feature]["horizon_martingales"]
            )

    # Call progress callback if provided
    if progress_callback:
        progress_callback(len(graph_sequence))

    return {
        "change_points": all_change_points,
        "traditional_changes": trad_results["change_points"],
        "horizon_changes": horizon_results["change_points"],
        "M_observed": trad_results["combined_martingales"]["martingale_sum"],
        "M_predicted": horizon_results["horizon_martingales"]["martingale_sum"],
        "individual_martingales_obs": individual_martingales_obs,
        "individual_martingales_pred": individual_martingales_pred,
        "statistics": {"traditional": trad_stats, "horizon": horizon_stats},
    }
