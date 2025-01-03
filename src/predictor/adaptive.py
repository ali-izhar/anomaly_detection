"""Adaptive ensemble-based network prediction.

This module implements network prediction using an adaptive ensemble that
dynamically adjusts predictor weights based on recent performance. The core
idea is to learn which predictors are most effective in different scenarios.

Mathematical Foundation:
----------------------
Given K base predictors and their predictions P_k, we:

1. Performance Tracking:
   For each predictor k, compute error e_k(t):
   e_k(t) = ||A_true(t) - P_k(t)||²
   where A_true is the true adjacency

2. Weight Adaptation:
   w_k(t+1) = w_k(t) * exp(-η * e_k(t))
   where:
   - η is the learning rate
   - weights are normalized after update

3. Online Learning:
   - Exponential weight updates
   - Multiplicative weight updates
   - Hedge algorithm variants

4. Prediction Combination:
   A_pred(t+1) = ∑(w_k(t) * P_k(t+1))
   with adaptive thresholding
"""

import numpy as np
import networkx as nx
from typing import List, Dict, Any, Optional, Type
from scipy.special import softmax
from .base import BasePredictor
from .weighted import WeightedPredictor
from .spectral import SpectralPredictor
from .embedding import EmbeddingPredictor
from .dynamical import DynamicalPredictor


class AdaptivePredictor(BasePredictor):
    """Predicts network evolution using adaptive ensemble.

    Parameters
    ----------
    predictors : List[Type[BasePredictor]], optional
        List of predictor classes to use
    learning_rate : float, optional
        Rate for weight adaptation, by default 0.1
    window_size : int, optional
        Window for performance tracking, by default 5
    threshold : float, optional
        Threshold for binary predictions, by default 0.5
    min_weight : float, optional
        Minimum weight for any predictor, by default 0.01
    """

    def __init__(
        self,
        predictors: Optional[List[Type[BasePredictor]]] = None,
        learning_rate: float = 0.1,
        window_size: int = 5,
        threshold: float = 0.5,
        min_weight: float = 0.01,
    ):
        """Initialize the adaptive predictor."""
        if predictors is None:
            predictors = [
                WeightedPredictor,
                SpectralPredictor,
                EmbeddingPredictor,
                DynamicalPredictor,
            ]

        self.predictor_classes = predictors
        self.predictors = [cls() for cls in predictors]
        self.learning_rate = learning_rate
        self.window_size = window_size
        self.threshold = threshold
        self.min_weight = min_weight

        # Initialize weights and error history
        self.weights = np.ones(len(predictors)) / len(predictors)
        self.error_history: List[List[float]] = [[] for _ in predictors]

    def predict(
        self,
        history: List[Dict[str, Any]],
        horizon: int = 1,
        **kwargs,
    ) -> List[np.ndarray]:
        """Predict future network states using adaptive ensemble.

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
        # Update weights based on recent performance
        if len(history) > 1:
            self._update_weights(history)

        # Get predictions from each base predictor
        base_predictions = []
        for predictor in self.predictors:
            pred = predictor.predict(history, horizon=horizon)
            base_predictions.append(pred)

        predictions = []
        # Combine predictions for each timestep
        for step in range(horizon):
            # Get all predictions for this timestep and convert to float64
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

    def _update_weights(self, history: List[Dict[str, Any]]) -> None:
        """Update predictor weights based on recent performance.

        Parameters
        ----------
        history : List[Dict[str, Any]]
            Historical network states

        Notes
        -----
        Uses exponential weight update:
        w_k(t+1) = w_k(t) * exp(-η * e_k(t))
        """
        # Get recent true states
        recent_states = history[-self.window_size :]
        if len(recent_states) < 2:
            return

        # Compute errors for each predictor
        errors = np.zeros(len(self.predictors))
        for i, predictor in enumerate(self.predictors):
            # Make predictions for recent history
            for j in range(len(recent_states) - 1):
                pred = predictor.predict(recent_states[: j + 1], horizon=1)[0]
                true_adj = recent_states[j + 1]["adjacency"]
                error = np.mean((true_adj - pred) ** 2)
                self.error_history[i].append(error)
                errors[i] += error

        # Average errors over window
        errors /= len(recent_states) - 1

        # Update weights using exponential update rule
        self.weights *= np.exp(-self.learning_rate * errors)

        # Ensure minimum weight
        self.weights = np.maximum(self.weights, self.min_weight)

        # Normalize weights
        self.weights /= np.sum(self.weights)

    def get_predictor_weights(self) -> Dict[str, float]:
        """Get current predictor weights.

        Returns
        -------
        Dict[str, float]
            Dictionary mapping predictor names to weights
        """
        return {
            pred.__class__.__name__: weight
            for pred, weight in zip(self.predictors, self.weights)
        }

    def get_predictor_errors(self) -> Dict[str, List[float]]:
        """Get error history for each predictor.

        Returns
        -------
        Dict[str, List[float]]
            Dictionary mapping predictor names to error histories
        """
        return {
            pred.__class__.__name__: errors
            for pred, errors in zip(self.predictors, self.error_history)
        }

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
        self.error_history.append([])

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
            self.error_history.pop(index)
            self.weights = np.delete(self.weights, index)
            # Renormalize weights
            self.weights = self.weights / np.sum(self.weights)
