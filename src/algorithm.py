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
from typing import List, Dict, Any, Optional

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
    horizon: int,
    threshold: float,
    epsilon: float,
    window_size: int,
    predictor: Optional[BasePredictor] = None,
    predictor_type: str = "adaptive",
    predictor_config: Optional[Dict[str, Any]] = None,
    random_state: int = None,
    progress_callback=None,
) -> Dict[str, Any]:
    """Implementation of Algorithm 1 from Section 5 of the paper."""
    # [Line 1] Initialize
    if predictor is None:
        predictor = PredictorFactory.create(predictor_type, predictor_config)

    n = len(graph_sequence)
    logger.info(f"Starting detection on sequence of length {n}")

    # Track change points from both streams separately
    traditional_changes = []
    horizon_changes = []
    change_points = []  # Combined changes for backward compatibility

    detector = ChangePointDetector()
    feature_extractor = NetworkFeatureExtractor()

    # History tracking
    adjacency_history = deque(maxlen=window_size)  # Only for predictor
    features_history = []  # Full feature history for martingales

    # Track martingale values (M_t and Mhat_t from paper)
    M_obs_sums = []  # Traditional martingales sum
    M_pred_sums = []  # Horizon martingales sum
    individual_marts_obs = [[] for _ in range(8)]  # Individual traditional martingales
    individual_marts_pred = [[] for _ in range(8)]  # Individual horizon martingales

    # Keep track of previous traditional martingale values (M_{t-1})
    prev_traditional_marts = [1.0] * 8  # Initialize M_{t-1} for each feature

    # Detectors for both streams
    traditional_detector = ChangePointDetector()  # For M_t (equation 14)
    horizon_detector = ChangePointDetector()  # For Mhat_t (equation 15)

    # [Line 2] Main detection loop
    for t in range(n):
        logger.debug(f"Processing time step {t}/{n-1}")

        # [Line 3] Current graph processing
        current_graph = graph_sequence[t]
        current_adj = nx.to_numpy_array(current_graph)
        adjacency_history.append(current_adj)  # For predictor only

        # Extract current features f(G_t)
        current_features = extract_numeric_features(
            feature_extractor.get_features(current_graph)
        )
        features_history.append(current_features)  # Full history for both streams

        # [Line 6] Update traditional martingale (equation 14)
        # Use all historical features for proper conformal scores
        features_since_start = np.array(features_history)

        # Ensure features are 2D array (samples x features)
        if len(features_since_start.shape) == 1:
            features_since_start = features_since_start.reshape(1, -1)

        # Split features into separate views for multiview detection
        feature_views = [features_since_start[:, i : i + 1] for i in range(8)]

        # Use multiview_martingale_test for traditional martingale
        obs_result = traditional_detector.detect_changes_multiview(
            data=feature_views,
            threshold=threshold,
            epsilon=epsilon,
            max_window=None,  # Use all history
            random_state=random_state,
        )

        # Store traditional martingale values and update M_{t-1}
        M_obs_sums.append(obs_result["martingales_sum"][-1])
        for i, marts in enumerate(obs_result["individual_martingales"]):
            mart_value = marts[-1]
            individual_marts_obs[i].append(mart_value)
            prev_traditional_marts[i] = mart_value  # Update M_{t-1} for next iteration

        # [Line 7] Update horizon martingale (equation 15)
        M_pred_sum = 0
        pred_martingales = []

        # Only compute horizon martingale if we have enough data for prediction
        if t >= window_size:
            # [Line 4] Generate h-step ahead predictions
            predicted_graphs = predictor.predict(
                history=[{"adjacency": adj} for adj in list(adjacency_history)],
                horizon=horizon,
            )

            # [Line 5] Extract features from predicted graphs
            predicted_features = []
            for pred_adj in predicted_graphs:
                pred_graph = nx.from_numpy_array(pred_adj)
                pred_features = extract_numeric_features(
                    feature_extractor.get_features(pred_graph)
                )
                predicted_features.append(pred_features)

            # For each feature i
            feature_views_pred = []
            for i in range(8):  # 8 features
                # Start with M_{t-1} from traditional martingale
                horizon_mart = prev_traditional_marts[i]

                # Get all historical data for this feature
                history_vals = features_since_start[:, i : i + 1]

                # Get predicted values for this feature
                feature_preds = np.array([pred[i] for pred in predicted_features])
                feature_preds = feature_preds.reshape(-1, 1)  # Make 2D for strangeness

                # Compute strangeness for all predictions together
                try:
                    all_points = np.vstack([history_vals, feature_preds])
                    strangeness_vals = strangeness_point(
                        all_points, random_state=random_state
                    )

                    # Get p-values for each prediction and multiply into horizon martingale
                    for j in range(len(feature_preds)):
                        # For each prediction, compute its p-value using all previous points
                        current_strange = strangeness_vals[-(horizon - j)]
                        prev_strange = list(strangeness_vals[: -(horizon - j)])
                        prev_strange.append(current_strange)

                        # Get p-value using proper function from strangeness.py
                        p_val = get_pvalue(prev_strange, random_state=random_state)

                        # Multiply into horizon martingale (equation 15)
                        # Mhat_t = M_{t-1} * ∏[ε(p_j)^(ε-1)]
                        horizon_mart *= epsilon * (p_val ** (epsilon - 1))

                except ValueError as e:
                    # On error, keep M_{t-1}
                    horizon_mart = prev_traditional_marts[i]

                pred_martingales.append(horizon_mart)
                feature_views_pred.append(feature_preds)

            # Use multiview_martingale_test for horizon martingale
            pred_result = horizon_detector.detect_changes_multiview(
                data=feature_views_pred,
                threshold=threshold,
                epsilon=epsilon,
                max_window=None,  # Use all history
                random_state=random_state,
            )
            M_pred_sum = pred_result["martingales_sum"][-1]
        else:
            # If not enough history, use M_{t-1} values
            pred_martingales = prev_traditional_marts.copy()
            M_pred_sum = sum(pred_martingales)

        # [Line 8] Check for change point using both streams independently
        # Traditional martingale detection (equation 14)
        if obs_result["martingales_sum"][-1] > threshold:
            traditional_changes.append(t)
            logger.info(
                f"Traditional martingale detection at t={t} (M_t = {obs_result['martingales_sum'][-1]:.4f})"
            )

        # Horizon martingale detection (equation 15)
        if M_pred_sum > threshold:
            horizon_changes.append(t)
            logger.info(
                f"Horizon martingale detection at t={t} (Mhat_t = {M_pred_sum:.4f})"
            )

        # Track combined changes for backward compatibility
        if t in traditional_changes or t in horizon_changes:
            change_points.append(t)

        # Store horizon martingale values
        M_pred_sums.append(M_pred_sum)
        for i, mart in enumerate(pred_martingales):
            individual_marts_pred[i].append(mart)

        # [Line 9-10] Update predictor state
        predictor.update_state({"adjacency": current_adj})

        # Update progress if callback provided
        if progress_callback is not None:
            progress_callback(t)

    logger.info(f"Detection completed.")
    logger.info(f"Traditional martingale detected changes at: {traditional_changes}")
    logger.info(f"Horizon martingale detected changes at: {horizon_changes}")
    logger.info(f"Combined change points: {change_points}")

    return {
        "change_points": change_points,
        "traditional_changes": traditional_changes,
        "horizon_changes": horizon_changes,
        "M_observed": np.array(M_obs_sums),
        "M_predicted": np.array(M_pred_sums),
        "individual_martingales_obs": [np.array(m) for m in individual_marts_obs],
        "individual_martingales_pred": [np.array(m) for m in individual_marts_pred],
    }
