# src/predictor/hybrid/ws_predictor.py

import networkx as nx
import numpy as np
from typing import List, Dict, Any

from .base import HybridPredictor


class WSPredictor(HybridPredictor):
    """Hybrid predictor specialized for Watts-Strogatz networks."""

    def __init__(self, **kwargs):
        super().__init__(model_type="ws", **kwargs)
        self.model_params = self._initialize_model_params()

    def _initialize_model_params(self) -> Dict:
        """Initialize WS-specific parameters."""
        if not self.config or "params" not in self.config:
            return {}

        params = self.config["params"]
        return {
            "k": getattr(params, "k_nearest", 6),
            "rewire_prob": getattr(params, "rewire_prob", 0.1),
        }

    def _make_single_prediction(self, history: List[Dict[str, Any]]) -> np.ndarray:
        """Make prediction preserving WS structure while matching weighted predictor's edge count."""
        # Get weighted prediction first as baseline
        weighted_pred = self.base_predictor.predict(history, horizon=1)[0]

        # Get exact target metrics from weighted predictor
        G_weighted = nx.from_numpy_array(weighted_pred)
        target_edges = G_weighted.number_of_edges()  # We must match this exactly
        n = weighted_pred.shape[0]

        # Get current graph structure
        G_current = history[-1]["graph"]

        # Create edge scores matrix
        edge_scores = np.zeros((n, n))

        # Score edges based on both weighted predictor and local structure
        for i in range(n):
            for j in range(i + 1, n):  # Avoid self-loops
                # Base score from weighted predictor
                base_score = weighted_pred[i, j]

                # Boost scores for edges that maintain local clustering
                common_neighbors = len(
                    set(G_current.neighbors(i)) & set(G_current.neighbors(j))
                )
                if common_neighbors > 0:
                    edge_scores[i, j] = edge_scores[j, i] = base_score * (
                        1 + 0.1 * common_neighbors
                    )
                else:
                    edge_scores[i, j] = edge_scores[j, i] = base_score

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
