# src/predictor/hybrid/ba_predictor.py

import networkx as nx
import numpy as np
from typing import List, Dict, Any
from scipy import stats

from .base import HybridPredictor


class BAPredictor(HybridPredictor):
    """Hybrid predictor specialized for Barabási-Albert networks."""

    def __init__(self, **kwargs):
        super().__init__(model_type="ba", **kwargs)
        self.model_params = self._initialize_model_params()

    def _initialize_model_params(self) -> Dict:
        """Initialize BA-specific parameters."""
        if not self.config or "params" not in self.config:
            return {}

        params = self.config["params"]
        return {
            "m": getattr(params, "m", 3),
            "min_m": getattr(params, "min_m", 2),
            "max_m": getattr(params, "max_m", 4),
            "preferential_exp": getattr(params, "preferential_exp", 1.0),
        }

    def _estimate_pa_exponent(self, degrees: np.ndarray) -> float:
        """Estimate preferential attachment exponent from degree distribution."""
        degrees = degrees[degrees > 0]
        log_degrees = np.log(degrees)
        log_counts = np.log(stats.itemfreq(degrees)[:, 1])
        slope, _, _, _, _ = stats.linregress(log_degrees, log_counts)
        return abs(slope)

    def _make_single_prediction(self, history: List[Dict[str, Any]]) -> np.ndarray:
        """Make prediction preserving BA structure while matching weighted predictor's edge count."""
        # Get weighted prediction first as baseline
        weighted_pred = self.base_predictor.predict(history, horizon=1)[0]

        # Get exact target metrics from weighted predictor
        G_weighted = nx.from_numpy_array(weighted_pred)
        target_edges = G_weighted.number_of_edges()  # We must match this exactly
        n = weighted_pred.shape[0]

        # Get current degrees for preferential attachment
        G_current = history[-1]["graph"]
        degrees = np.array([G_current.degree(i) for i in range(n)])

        # Create edge scores matrix
        edge_scores = np.zeros((n, n))

        # Score edges based on both weighted predictor and preferential attachment
        for i in range(n):
            for j in range(i + 1, n):  # Avoid self-loops
                # Base score from weighted predictor
                base_score = weighted_pred[i, j]

                # Modify score based on degrees (preferential attachment)
                degree_factor = ((degrees[i] + 1) * (degrees[j] + 1)) ** 0.5
                edge_scores[i, j] = edge_scores[j, i] = base_score * degree_factor

        # Create list of edges with their scores
        edges = []
        for i in range(n):
            for j in range(i + 1, n):  # Upper triangular only
                edges.append((edge_scores[i, j], i, j))

        # Sort edges by score
        edges.sort(reverse=True, key=lambda x: x[0])

        # Take EXACTLY target_edges number of edges
        P_final = np.zeros((n, n))
        for _, i, j in edges[:target_edges]:  # Only take top target_edges edges
            P_final[i, j] = P_final[j, i] = 1

        return P_final
