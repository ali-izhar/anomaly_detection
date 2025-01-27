# src/algorithm.py

"""
Implements Algorithm 1 from Section 5 of the paper:
'Faster Structural Change Detection in Dynamic Networks via Statistical Forecasting'

The algorithm maintains two parallel martingale streams:
1) M_t (observed): computed on current graph features
2) Mhat_t (horizon): computed on predicted future graphs

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

import numpy as np
import networkx as nx
from typing import List, Dict, Any
from collections import deque
import logging

from .predictor.predictor import GraphPredictor
from .graph.features import NetworkFeatureExtractor
from .changepoint.detector import ChangePointDetector

logger = logging.getLogger(__name__)

# Default parameters
DEFAULT_PARAMS = {
    "horizon": 5,  # Number of future steps to predict (h in Algorithm 1)
    "threshold": 60.0,  # Detection threshold λ (from empirical study)
    "epsilon": 0.7,  # Power martingale sensitivity (optimal from experiments)
    "window_size": 10,  # Rolling window size k for feature history
}


def extract_numeric_features(feature_dict: dict) -> np.ndarray:
    """
    Extract numeric features from feature dictionary in a consistent order.
    These features correspond to f(G_t) in Algorithm 1, Line 3.

    Features are defined in Section 4 of the paper:
    1) Local Connectivity (degree, density, clustering)
    2) Information Flow (betweenness, eigenvector, closeness)
    3) Global Structure (singular values, laplacian)
    """
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


def run_forecast_martingale_detection(
    graph_sequence: List[nx.Graph],
    horizon: int = DEFAULT_PARAMS["horizon"],
    threshold: float = DEFAULT_PARAMS["threshold"],
    epsilon: float = DEFAULT_PARAMS["epsilon"],
    window_size: int = DEFAULT_PARAMS["window_size"],
    predictor: GraphPredictor = None,
    random_state: int = None,
    progress_callback=None,  # Add callback parameter
) -> Dict[str, Any]:
    """
    Implementation of Algorithm 1 from Section 5 of the paper.

    Algorithm Steps (with paper line numbers):
    [Line 1] Initialize martingales M_0 = Mhat_0 = 1
    [Line 2] For each time t:
        [Line 3] Extract features f_t from current graph G_t
        [Line 4] Generate predictions {Ghat_{t+j}}_{j=1}^h
        [Line 5] Extract features {fhat_{t+j}}_{j=1}^h from predictions
        [Line 6] Update M_t using current features
        [Line 7] Update Mhat_t using predicted features
        [Line 8] If M_t > λ or Mhat_t > λ: record change
        [Line 9] Update history window
        [Line 10] Adapt parameters (beta, gamma)

    Parameters match those defined in Section 5.2 of the paper.
    """
    # [Line 1] Initialize
    if predictor is None:
        predictor = GraphPredictor()

    n = len(graph_sequence)
    logger.info(f"Starting detection on sequence of length {n}")

    change_points = []
    detector = ChangePointDetector()

    # Initialize feature storage for each view (Section 4)
    features_numeric = []  # Store all features as numpy array

    # History for predictor (Section 5.1)
    adjacency_history = deque(maxlen=window_size)

    # Track martingale values (M_t and Mhat_t from paper)
    M_obs_sums = []
    M_pred_sums = []
    individual_marts_obs = [[] for _ in range(8)]
    individual_marts_pred = [[] for _ in range(8)]

    # [Line 2] Main detection loop
    for t in range(n):
        logger.debug(f"Processing time step {t}/{n-1}")

        # [Line 3] Current graph processing
        current_graph = graph_sequence[t]
        adjacency_history.append(nx.to_numpy_array(current_graph))

        feature_extractor = NetworkFeatureExtractor()
        feature_dict = feature_extractor.get_features(current_graph)
        numeric_features = extract_numeric_features(feature_dict)
        features_numeric.append(numeric_features)

        if t >= window_size:
            logger.debug(f"Running detection at t={t} with window size {window_size}")

            # [Line 6] Update observed martingale M_t
            data = np.array(features_numeric)
            logger.debug(f"Feature matrix shape: {data.shape}")

            obs_result = detector.detect_changes_multiview(
                data=[data[:, i : i + 1] for i in range(data.shape[1])],
                threshold=threshold,
                epsilon=epsilon,
                max_window=None,  # Let detector handle window size
                random_state=random_state,
            )

            if "martingales_sum" in obs_result:
                M_obs_sums.append(obs_result["martingales_sum"][-1])
                logger.debug(
                    f"Observed martingale sum: {obs_result['martingales_sum'][-1]:.3f}"
                )
            else:
                logger.warning("No martingales_sum in observed result")

            for i, marts in enumerate(obs_result["individual_martingales"]):
                individual_marts_obs[i].append(marts[-1])

            # [Lines 4-5] Predict future graphs and extract features
            predicted_graphs = predictor.forecast(
                history_adjs=list(adjacency_history)[:-1],
                current_adj=adjacency_history[-1],
                h=horizon,
            )

            pred_features = []
            for pred_adj in predicted_graphs:
                pred_graph = nx.from_numpy_array(pred_adj)
                pred_dict = feature_extractor.get_features(pred_graph)
                pred_numeric = extract_numeric_features(pred_dict)
                pred_features.append(pred_numeric)

            # [Line 7] Update horizon martingale Mhat_t
            pred_data = np.array(pred_features)
            logger.debug(f"Predicted feature matrix shape: {pred_data.shape}")

            pred_result = detector.detect_changes_multiview(
                data=[pred_data[:, i : i + 1] for i in range(pred_data.shape[1])],
                threshold=threshold,
                epsilon=epsilon,
                max_window=None,  # Let detector handle window size
                random_state=random_state,
            )

            if "martingales_sum" in pred_result:
                M_pred_sums.append(pred_result["martingales_sum"][-1])
                logger.debug(
                    f"Predicted martingale sum: {pred_result['martingales_sum'][-1]:.3f}"
                )
            else:
                logger.warning("No martingales_sum in predicted result")

            for i, marts in enumerate(pred_result["individual_martingales"]):
                individual_marts_pred[i].append(marts[-1])

            # [Line 8] Check for change point
            if (
                obs_result.get("martingales_sum", [0])[-1] > threshold
                or pred_result.get("martingales_sum", [0])[-1] > threshold
            ):
                change_points.append(t)
                logger.info(f"Change point detected at t={t}")

            # [Lines 9-10] Update predictor parameters
            if t < n - horizon:
                actual_next = nx.to_numpy_array(graph_sequence[t + 1])
                predictor.update_beta(actual_next, predicted_graphs[0])

        # Update progress if callback provided
        if progress_callback is not None:
            progress_callback(t)

    logger.info(f"Detection completed. Found {len(change_points)} change points")
    return {
        "change_points": change_points,
        "M_observed": np.array(M_obs_sums),
        "M_predicted": np.array(M_pred_sums),
        "individual_martingales_obs": [np.array(m) for m in individual_marts_obs],
        "individual_martingales_pred": [np.array(m) for m in individual_marts_pred],
    }
