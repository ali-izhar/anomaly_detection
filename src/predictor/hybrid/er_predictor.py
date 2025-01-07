# src/predictor/hybrid/er_predictor.py

import networkx as nx
import numpy as np
from typing import List, Dict, Any

from .base import HybridPredictor


class ERPredictor(HybridPredictor):
    """Hybrid predictor specialized for Erdős-Rényi networks."""

    def __init__(self, **kwargs):
        super().__init__(model_type="er", **kwargs)
        self.model_params = self._initialize_model_params()

    def _initialize_model_params(self) -> Dict:
        """Initialize ER-specific parameters."""
        if not self.config or "params" not in self.config:
            return {}

        params = self.config["params"]
        return {
            "prob": getattr(params, "prob", 0.15),
            "min_prob": getattr(params, "min_prob", 0.1),
            "max_prob": getattr(params, "max_prob", 0.2),
        }

    def _make_single_prediction(self, history: List[Dict[str, Any]]) -> np.ndarray:
        """Make prediction for ER network while matching weighted predictor's edge count."""
        # For ER model, we can simply use the weighted predictor's output
        # since ER networks are random and don't have special structure to preserve
        weighted_pred = self.base_predictor.predict(history, horizon=1)[0]

        # Get exact target metrics from weighted predictor
        G_weighted = nx.from_numpy_array(weighted_pred)
        target_edges = G_weighted.number_of_edges()
        n = weighted_pred.shape[0]

        # Sort edges by probability
        edges = []
        for i in range(n):
            for j in range(i + 1, n):  # Avoid self-loops
                edges.append((weighted_pred[i, j], i, j))

        edges.sort(reverse=True, key=lambda x: x[0])

        # Take EXACTLY target_edges number of edges
        P_final = np.zeros((n, n))
        for _, i, j in edges[:target_edges]:
            P_final[i, j] = P_final[j, i] = 1

        return P_final
