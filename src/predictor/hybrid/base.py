# src/predictor/hybrid/base.py

import networkx as nx
import numpy as np
from typing import List, Dict, Any, Optional

from predictor.weighted import WeightedPredictor
from graph.features import NetworkFeatureExtractor


class HybridPredictor:
    """Base class for hybrid predictors combining weighted averaging with structural role preservation."""

    def __init__(
        self,
        n_history: int = 5,
        weights: Optional[np.ndarray] = None,
        adaptive: bool = True,
        model_type: str = "ba",
        config: Optional[Dict] = None,
        alpha: float = 0.8,
    ):
        self.n_history = n_history
        self.weights = weights
        self.adaptive = adaptive
        self.model_type = model_type
        self.config = config
        self.alpha = alpha

        # Initialize weighted predictor with more history and adaptivity
        self.base_predictor = WeightedPredictor(
            n_history=n_history,
            weights=weights,
            adaptive=True,
            binary=False,
        )

        # Feature extractor for metrics
        self.feature_extractor = NetworkFeatureExtractor()

    def _make_single_prediction(self, history: List[Dict[str, Any]]) -> np.ndarray:
        """Base prediction method that should be overridden by subclasses."""
        # Get weighted prediction as baseline
        weighted_pred = self.base_predictor.predict(history, horizon=1)[0]

        # Get exact target metrics from weighted predictor
        G_weighted = nx.from_numpy_array(weighted_pred)
        target_edges = G_weighted.number_of_edges()
        n = weighted_pred.shape[0]

        # Default implementation just returns weighted prediction
        return weighted_pred

    def predict(
        self,
        history: List[Dict[str, Any]],
        horizon: int = 1,
    ) -> List[np.ndarray]:
        """Make predictions for the specified horizon."""
        if len(history) < self.n_history:
            raise ValueError(
                f"Not enough history. Need {self.n_history}, got {len(history)}."
            )

        predictions = []
        current_history = list(history)

        for _ in range(horizon):
            pred = self._make_single_prediction(current_history)
            predictions.append(pred)
            current_history.append(
                {"adjacency": pred, "graph": nx.from_numpy_array(pred)}
            )

        return predictions

    def _compute_edge_confidence(
        self, history: List[Dict[str, Any]], i: int, j: int
    ) -> float:
        """Compute confidence score for an edge based on its historical presence."""
        confidence = 0.0
        total_weight = 0.0

        # Look at recent history with exponential decay
        for idx, state in enumerate(reversed(history[-self.n_history :])):
            weight = np.exp(-idx * 0.5)  # Exponential decay
            total_weight += weight
            if state["adjacency"][i, j] > 0:
                confidence += weight

        return confidence / total_weight if total_weight > 0 else 0.0
