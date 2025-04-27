"""Hybrid ARIMA-GRNN graph state forecaster."""

import numpy as np
from typing import List, Dict, Optional
from .arima import ARIMAGraphForecaster
from .grnn import GRNNGraphForecaster
import logging

logger = logging.getLogger(__name__)


class HybridGraphForecaster:
    """Hybrid ARIMA-GRNN forecaster for graph states.

    This forecaster combines ARIMA and GRNN models to predict future graph states:
    1. ARIMA captures linear patterns and trends
    2. GRNN handles non-linear residuals
    3. Combines predictions for final output
    """

    def __init__(
        self,
        arima_order: tuple = (3, 1, 3),
        arima_seasonal_order: Optional[tuple] = None,
        grnn_hidden_layers: tuple = (100, 50),
        grnn_activation: str = "relu",
        grnn_solver: str = "adam",
        grnn_alpha: float = 0.0001,
        grnn_batch_size: int = 32,
        grnn_learning_rate: str = "constant",
        grnn_max_iter: int = 200,
        random_state: Optional[int] = None,
        enforce_connectivity: bool = True,
        threshold: float = 0.5,
    ):
        """Initialize the hybrid forecaster.

        Args:
            arima_order: The (p,d,q) order of the ARIMA model
            arima_seasonal_order: The (P,D,Q,s) order of the seasonal component
            grnn_hidden_layers: Number of neurons in each hidden layer
            grnn_activation: Activation function for hidden layers
            grnn_solver: The solver for weight optimization
            grnn_alpha: L2 penalty (regularization term) parameter
            grnn_batch_size: Size of minibatches for stochastic optimizers
            grnn_learning_rate: Learning rate schedule for weight updates
            grnn_max_iter: Maximum number of iterations
            random_state: Random seed for reproducibility
            enforce_connectivity: Whether to ensure predicted graphs are connected
            threshold: Threshold for binarizing predicted adjacency matrices
        """
        self.arima_forecaster = ARIMAGraphForecaster(
            order=arima_order,
            seasonal_order=arima_seasonal_order,
            enforce_connectivity=enforce_connectivity,
            threshold=threshold,
        )

        self.grnn_forecaster = GRNNGraphForecaster(
            hidden_layer_sizes=grnn_hidden_layers,
            activation=grnn_activation,
            solver=grnn_solver,
            alpha=grnn_alpha,
            batch_size=grnn_batch_size,
            learning_rate=grnn_learning_rate,
            max_iter=grnn_max_iter,
            random_state=random_state,
            enforce_connectivity=enforce_connectivity,
            threshold=threshold,
        )

        self.enforce_connectivity = enforce_connectivity
        self.threshold = threshold

    def _extract_features(self, history: List[Dict]) -> np.ndarray:
        """Extract features from historical graph states.

        Args:
            history: List of dictionaries containing past adjacency matrices
                    [{"adjacency": adj_matrix}, ...]

        Returns:
            numpy array of shape (n_timesteps, n_features)
        """
        # Use GRNN's feature extraction as it includes more structural features
        return self.grnn_forecaster._extract_features(history)

    def _combine_predictions(
        self,
        arima_preds: List[np.ndarray],
        grnn_preds: List[np.ndarray],
        weights: tuple = (0.5, 0.5),
    ) -> List[np.ndarray]:
        """Combine predictions from ARIMA and GRNN models.

        Args:
            arima_preds: List of ARIMA-predicted adjacency matrices
            grnn_preds: List of GRNN-predicted adjacency matrices
            weights: Weights for combining predictions (arima_weight, grnn_weight)

        Returns:
            List of combined predicted adjacency matrices
        """
        combined_preds = []
        arima_weight, grnn_weight = weights

        for arima_adj, grnn_adj in zip(arima_preds, grnn_preds):
            # Combine adjacency matrices
            combined_adj = arima_weight * arima_adj + grnn_weight * grnn_adj

            # Threshold to binary
            combined_adj = (combined_adj > self.threshold).astype(int)

            # Ensure symmetric with zero diagonal
            combined_adj = (combined_adj + combined_adj.T) / 2
            np.fill_diagonal(combined_adj, 0)

            combined_preds.append(combined_adj)

        return combined_preds

    def predict(self, history: List[Dict], horizon: int = 5) -> List[np.ndarray]:
        """Predict future graph states.

        Args:
            history: List of dictionaries containing past adjacency matrices
                    [{"adjacency": adj_matrix}, ...]
            horizon: Number of future time steps to predict

        Returns:
            List of predicted adjacency matrices
        """
        if len(history) < 2:
            raise ValueError("Need at least 2 historical states for prediction")

        # Get predictions from both models
        arima_preds = self.arima_forecaster.predict(history, horizon)
        grnn_preds = self.grnn_forecaster.predict(history, horizon)

        # Combine predictions
        combined_preds = self._combine_predictions(arima_preds, grnn_preds)

        return combined_preds
