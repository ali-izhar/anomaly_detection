# src/predictor/hybrid/rcp_predictor.py

import networkx as nx
import numpy as np
from typing import List, Dict, Any
from sklearn.cluster import KMeans

from .base import HybridPredictor


class RCPPredictor(HybridPredictor):
    """Hybrid predictor specialized for Rich-Club/Core-Periphery networks."""

    def __init__(self, **kwargs):
        super().__init__(model_type="rcp", **kwargs)
        self.model_params = self._initialize_model_params()

    def _initialize_model_params(self) -> Dict:
        """Initialize RCP-specific parameters."""
        if not self.config or "params" not in self.config:
            return {}

        params = self.config["params"]
        return {
            "core_size": getattr(params, "core_size", params.n // 5),
            "core_prob": getattr(params, "core_prob", 0.8),
            "periph_prob": getattr(params, "periph_prob", 0.05),
            "core_periph_prob": getattr(params, "core_periph_prob", 0.2),
        }

    def _estimate_core_size(self, degrees: np.ndarray) -> int:
        """Estimate core size using degree distribution."""
        X = degrees.reshape(-1, 1)
        kmeans = KMeans(n_clusters=2).fit(X)
        core_mask = kmeans.labels_ == np.argmax(kmeans.cluster_centers_)
        return int(np.sum(core_mask))

    def _make_single_prediction(self, history: List[Dict[str, Any]]) -> np.ndarray:
        """Make prediction preserving RCP structure while matching weighted predictor's edge count."""
        # Get weighted prediction first as baseline
        weighted_pred = self.base_predictor.predict(history, horizon=1)[0]

        # Get exact target metrics from weighted predictor
        G_weighted = nx.from_numpy_array(weighted_pred)
        target_edges = G_weighted.number_of_edges()  # We must match this exactly
        n = weighted_pred.shape[0]

        # Identify core and periphery based on degrees
        G_current = history[-1]["graph"]
        degrees = np.array([G_current.degree(i) for i in range(n)])
        core_size = self.model_params.get("core_size", n // 5)
        core_nodes = set(np.argsort(degrees)[::-1][:core_size])

        # Create edge scores matrix
        edge_scores = np.zeros((n, n))

        # Score edges based on both weighted predictor and core-periphery structure
        for i in range(n):
            for j in range(i + 1, n):  # Avoid self-loops
                # Base score from weighted predictor
                base_score = weighted_pred[i, j]

                # Modify score based on core-periphery membership
                if i in core_nodes and j in core_nodes:
                    edge_scores[i, j] = edge_scores[j, i] = (
                        base_score * 1.2
                    )  # Core-core boost
                elif i not in core_nodes and j not in core_nodes:
                    edge_scores[i, j] = edge_scores[j, i] = (
                        base_score * 0.8
                    )  # Periphery-periphery reduction
                else:
                    edge_scores[i, j] = edge_scores[j, i] = (
                        base_score * 1.0
                    )  # Core-periphery neutral

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
