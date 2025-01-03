"""Ensemble-based network prediction.

This module implements network prediction using an ensemble of base predictors.
The core idea is to combine multiple prediction strategies to create a more
robust and accurate forecasting system.

Mathematical Foundation:
----------------------
Given K base predictors, we:

1. Individual Predictions:
   For each predictor k, get prediction matrix P_k(t+1)
   where P_k represents the predicted adjacency

2. Weighted Combination:
   A(t+1) = ∑(w_k * P_k(t+1)) for k=1..K
   where:
   - w_k are predictor weights
   - ∑w_k = 1 (convex combination)

3. Weight Optimization:
   Minimize L(w) = ||A_true - ∑(w_k * P_k)||²
   subject to:
   - w_k ≥ 0 (non-negativity)
   - ∑w_k = 1 (sum to unity)

4. Prediction Aggregation:
   - Weighted voting for binary outcomes
   - Threshold-based consensus
   - Confidence-weighted decisions
"""

import numpy as np
import networkx as nx
from typing import List, Dict, Any, Optional, Type
from scipy.optimize import minimize
from .base import BasePredictor
from .weighted import WeightedPredictor
from .spectral import SpectralPredictor
from .embedding import EmbeddingPredictor
from .dynamical import DynamicalPredictor


class EnsemblePredictor(BasePredictor):
    """Predicts network evolution using an ensemble of base predictors.

    Parameters
    ----------
    predictors : List[Type[BasePredictor]], optional
        List of predictor classes to use
    weights : Optional[np.ndarray], optional
        Initial weights for predictors, by default None
    optimize_weights : bool, optional
        Whether to optimize weights using validation data, by default True
    threshold : float, optional
        Threshold for binary predictions, by default 0.5
    """

    def __init__(
        self,
        predictors: Optional[List[Type[BasePredictor]]] = None,
        weights: Optional[np.ndarray] = None,
        optimize_weights: bool = True,
        threshold: float = 0.5,
    ):
        """Initialize the ensemble predictor."""
        if predictors is None:
            predictors = [
                WeightedPredictor,
                SpectralPredictor,
                EmbeddingPredictor,
                DynamicalPredictor,
            ]

        self.predictor_classes = predictors
        self.predictors = [cls() for cls in predictors]
        self.weights = weights
        if weights is None:
            self.weights = np.ones(len(predictors)) / len(predictors)
        self.optimize_weights = optimize_weights
        self.threshold = threshold

    def predict(
        self,
        history: List[Dict[str, Any]],
        horizon: int = 1,
        **kwargs,
    ) -> List[np.ndarray]:
        """Predict future network states using ensemble.

        Parameters
        ----------
        history : List[Dict[str, Any]]
            Historical network states
        horizon : int, optional
            Number of steps to predict forward, by default 1

        Returns
        -------
        List[np.ndarray]
            Predicted adjacency matrices
        """
        # Optimize weights if needed
        if self.optimize_weights and len(history) > 2:
            self._optimize_weights(history[:-1], history[-1])

        # Get predictions from each base predictor
        base_predictions = []
        for predictor in self.predictors:
            pred = predictor.predict(history, horizon=horizon)
            base_predictions.append(pred)

        predictions = []
        # Combine predictions for each timestep
        for step in range(horizon):
            # Get all predictions for this timestep
            step_predictions = [p[step].astype(np.float64) for p in base_predictions]

            # Weighted combination
            combined_adj = np.zeros_like(step_predictions[0], dtype=np.float64)
            for w, pred in zip(self.weights, step_predictions):
                combined_adj += w * pred

            # Threshold and symmetrize
            combined_adj = (combined_adj + combined_adj.T) / 2
            combined_adj = (combined_adj > self.threshold).astype(np.float64)

            # Store prediction
            predictions.append(combined_adj)

        return predictions

    def _optimize_weights(
        self,
        train_history: List[Dict[str, Any]],
        validation_state: Dict[str, Any],
    ) -> None:
        """Optimize ensemble weights using validation data.

        Parameters
        ----------
        train_history : List[Dict[str, Any]]
            Training data for base predictors
        validation_state : Dict[str, Any]
            True state to validate against

        Notes
        -----
        Solves constrained optimization:
        min ||A_true - ∑(w_k * P_k)||²
        s.t. w_k ≥ 0, ∑w_k = 1
        """
        # Get predictions from each base predictor
        base_predictions = []
        for predictor in self.predictors:
            pred = predictor.predict(train_history, horizon=1)
            base_predictions.append(pred[0])  # Get first prediction

        # True adjacency matrix
        true_adj = validation_state["adjacency"]

        # Define objective function
        def objective(w):
            # Combine predictions
            combined = np.zeros_like(true_adj)
            for i, pred in enumerate(base_predictions):
                combined += w[i] * pred
            return np.sum((true_adj - combined) ** 2)

        # Constraints
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},  # sum to 1
        ]
        bounds = [(0, 1) for _ in range(len(self.predictors))]  # non-negative

        # Initial weights
        w0 = np.ones(len(self.predictors)) / len(self.predictors)

        # Optimize
        result = minimize(
            objective,
            w0,
            method="SLSQP",
            constraints=constraints,
            bounds=bounds,
        )

        # Update weights
        self.weights = result.x

    def add_predictor(
        self,
        predictor_class: Type[BasePredictor],
        weight: Optional[float] = None,
    ) -> None:
        """Add a new predictor to the ensemble.

        Parameters
        ----------
        predictor_class : Type[BasePredictor]
            Class of the predictor to add
        weight : Optional[float], optional
            Initial weight for the predictor, by default None
        """
        self.predictor_classes.append(predictor_class)
        self.predictors.append(predictor_class())

        # Update weights
        if weight is None:
            # Equal weighting
            self.weights = np.ones(len(self.predictors)) / len(self.predictors)
        else:
            # Scale existing weights and add new
            self.weights = np.append(self.weights * (1 - weight), weight)

    def remove_predictor(self, index: int) -> None:
        """Remove a predictor from the ensemble.

        Parameters
        ----------
        index : int
            Index of predictor to remove
        """
        if 0 <= index < len(self.predictors):
            self.predictor_classes.pop(index)
            self.predictors.pop(index)
            self.weights = np.delete(self.weights, index)
            # Renormalize weights
            self.weights = self.weights / np.sum(self.weights)
