"""Hybrid network prediction.

This module implements network prediction using a hybrid approach that
integrates multiple prediction strategies within a single model. The core
idea is to combine spectral, structural, temporal, and feature-based
predictions in a unified framework.

Mathematical Foundation:
----------------------
Given a network state A(t), we combine:

1. Spectral Component:
   P_s(t+1) = U(t)Σ'(t)V(t)ᵀ
   where Σ' is predicted singular values

2. Structural Component:
   P_g(t+1) = f(Z(t))
   where Z(t) are node embeddings

3. Temporal Component:
   P_t(t+1) = W·A(t) + b
   using learned dynamics

4. Unified Prediction:
   A(t+1) = α·P_s + β·P_g + γ·P_t
   where α, β, γ are learned weights

5. Optimization:
   min L = ||A_true - A_pred||² + R(α,β,γ)
   where R is a regularization term
"""

import numpy as np
import networkx as nx
from typing import List, Dict, Any, Optional, Tuple
from scipy.optimize import minimize
from scipy.sparse.linalg import eigsh
from .base import BasePredictor


class HybridPredictor(BasePredictor):
    """Predicts network evolution using hybrid approach.

    Parameters
    ----------
    n_components : int, optional
        Number of components for spectral part, by default 32
    embedding_dim : int, optional
        Dimension for structural embeddings, by default 32
    learning_rate : float, optional
        Learning rate for parameters, by default 0.01
    regularization : float, optional
        Regularization strength, by default 0.1
    threshold : float, optional
        Threshold for binary predictions, by default 0.5
    """

    def __init__(
        self,
        n_components: int = 32,
        embedding_dim: int = 32,
        learning_rate: float = 0.01,
        regularization: float = 0.1,
        threshold: float = 0.5,
    ):
        """Initialize the hybrid predictor."""
        self.n_components = n_components
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.threshold = threshold

        # Component weights
        self.spectral_weight = 1 / 3
        self.structural_weight = 1 / 3
        self.temporal_weight = 1 / 3

        # Component parameters
        self.singular_values_history: List[np.ndarray] = []
        self.left_vectors_history: List[np.ndarray] = []
        self.right_vectors_history: List[np.ndarray] = []
        self.embedding_history: List[np.ndarray] = []
        self.temporal_matrix: Optional[np.ndarray] = None
        self.temporal_bias: Optional[np.ndarray] = None

    def predict(
        self,
        history: List[Dict[str, Any]],
        horizon: int = 1,
        **kwargs,
    ) -> List[np.ndarray]:
        """Predict future network states using hybrid approach.

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
        # Get historical adjacency matrices
        adj_matrices = [state["adjacency"] for state in history]

        # Update component histories
        for adj in adj_matrices:
            self._update_histories(adj)

        # Learn temporal dynamics if not already learned
        if self.temporal_matrix is None:
            self._learn_temporal_dynamics(adj_matrices)

        # Current state
        current_adj = adj_matrices[-1]
        predictions = []

        # Predict future states
        for step in range(horizon):
            # Get component predictions
            spectral_pred = self._predict_spectral(current_adj)
            structural_pred = self._predict_structural(current_adj)
            temporal_pred = self._predict_temporal(current_adj)

            # Combine predictions
            combined_adj = (
                self.spectral_weight * spectral_pred
                + self.structural_weight * structural_pred
                + self.temporal_weight * temporal_pred
            )

            # Threshold and symmetrize
            combined_adj = (combined_adj + combined_adj.T) / 2
            combined_adj = (combined_adj > self.threshold).astype(float)

            # Store prediction
            predictions.append(combined_adj)

            # Update current state
            current_adj = combined_adj
            self._update_histories(current_adj)

        return predictions

    def _update_histories(self, adj_matrix: np.ndarray) -> None:
        """Update component histories with new state.

        Parameters
        ----------
        adj_matrix : np.ndarray
            New adjacency matrix
        """
        # Spectral decomposition
        U, S, Vt = np.linalg.svd(adj_matrix)
        if self.n_components:
            U = U[:, : self.n_components]
            S = S[: self.n_components]
            Vt = Vt[: self.n_components, :]

        self.singular_values_history.append(S)
        self.left_vectors_history.append(U)
        self.right_vectors_history.append(Vt)

        # Structural embedding
        embedding = self._compute_embedding(adj_matrix)
        self.embedding_history.append(embedding)

    def _compute_embedding(self, adj_matrix: np.ndarray) -> np.ndarray:
        """Compute structural embeddings.

        Parameters
        ----------
        adj_matrix : np.ndarray
            Input adjacency matrix

        Returns
        -------
        np.ndarray
            Node embeddings
        """
        # Normalized Laplacian
        deg = np.sum(adj_matrix, axis=1)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(deg, 1e-12)))
        L_norm = np.eye(adj_matrix.shape[0]) - D_inv_sqrt @ adj_matrix @ D_inv_sqrt

        # Compute smallest eigenvectors
        _, vectors = eigsh(-L_norm, k=self.embedding_dim, which="LA")
        return vectors

    def _learn_temporal_dynamics(self, adj_matrices: List[np.ndarray]) -> None:
        """Learn temporal dynamics from adjacency matrix sequence.

        Parameters
        ----------
        adj_matrices : List[np.ndarray]
            Historical adjacency matrices
        """
        if len(adj_matrices) < 2:
            return

        # Convert to array and flatten matrices
        X = np.array([adj.flatten() for adj in adj_matrices[:-1]], dtype=np.float64)
        Y = np.array([adj.flatten() for adj in adj_matrices[1:]], dtype=np.float64)

        # Initialize parameters if not exists
        n = adj_matrices[0].shape[0]
        if self.temporal_matrix is None:
            self.temporal_matrix = np.random.normal(0, 0.01, size=(n * n, n * n))
            self.temporal_bias = np.zeros(n * n)

        # Scale data to [0, 1] range
        X_max = np.max(np.abs(X))
        Y_max = np.max(np.abs(Y))
        scale = max(X_max, Y_max)
        if scale > 0:
            X = X / scale
            Y = Y / scale

        # Gradient descent iterations
        for _ in range(100):
            # Forward pass
            pred = X @ self.temporal_matrix.T + self.temporal_bias

            # Compute gradients with clipping
            error = Y - pred
            grad_W = np.clip(error.T @ X, -1, 1)
            grad_b = np.clip(np.mean(error, axis=0), -1, 1)

            # Update parameters with gradient clipping
            self.temporal_matrix += self.learning_rate * grad_W
            self.temporal_bias += self.learning_rate * grad_b

            # Add numerical stability through parameter bounds
            self.temporal_matrix = np.clip(self.temporal_matrix, -1, 1)
            self.temporal_bias = np.clip(self.temporal_bias, -1, 1)

    def _predict_spectral(self, current_adj: np.ndarray) -> np.ndarray:
        """Make spectral-based prediction.

        Parameters
        ----------
        current_adj : np.ndarray
            Current adjacency matrix

        Returns
        -------
        np.ndarray
            Predicted adjacency matrix
        """
        if len(self.singular_values_history) < 2:
            return current_adj

        # Predict singular values
        S_future = self.singular_values_history[-1] + np.mean(
            [
                s2 - s1
                for s1, s2 in zip(
                    self.singular_values_history[:-1], self.singular_values_history[1:]
                )
            ],
            axis=0,
        )

        # Use current vectors
        U = self.left_vectors_history[-1]
        Vt = self.right_vectors_history[-1]

        # Reconstruct
        return U @ np.diag(S_future) @ Vt

    def _predict_structural(self, current_adj: np.ndarray) -> np.ndarray:
        """Make structural prediction.

        Parameters
        ----------
        current_adj : np.ndarray
            Current adjacency matrix

        Returns
        -------
        np.ndarray
            Predicted adjacency matrix
        """
        if len(self.embedding_history) < 2:
            return current_adj

        # Predict next embedding
        Z_current = self.embedding_history[-1]
        Z_future = Z_current + np.mean(
            [
                z2 - z1
                for z1, z2 in zip(
                    self.embedding_history[:-1], self.embedding_history[1:]
                )
            ],
            axis=0,
        )

        # Convert to adjacency
        similarity = Z_future @ Z_future.T
        adj_pred = 1 / (1 + np.exp(-similarity))

        return adj_pred

    def _predict_temporal(self, current_adj: np.ndarray) -> np.ndarray:
        """Make temporal prediction.

        Parameters
        ----------
        current_adj : np.ndarray
            Current adjacency matrix

        Returns
        -------
        np.ndarray
            Predicted adjacency matrix
        """
        if self.temporal_matrix is None:
            return current_adj

        # Flatten input
        x = current_adj.flatten()

        # Make prediction
        pred = x @ self.temporal_matrix.T + self.temporal_bias

        # Reshape to matrix
        pred = pred.reshape(current_adj.shape)

        # Ensure symmetry and bounds
        pred = (pred + pred.T) / 2
        pred = np.clip(pred, 0, 1)

        return pred

    def optimize_weights(
        self,
        validation_history: List[Dict[str, Any]],
    ) -> None:
        """Optimize component weights using validation data.

        Parameters
        ----------
        validation_history : List[Dict[str, Any]]
            Validation data for optimization
        """
        if len(validation_history) < 2:
            return

        # Define objective function
        def objective(weights):
            self.spectral_weight = weights[0]
            self.structural_weight = weights[1]
            self.temporal_weight = weights[2]

            # Make predictions
            preds = self.predict(
                validation_history[:-1],
                validation_history[-2]["time"],
                1,
            )
            pred_adj = preds[0]["adjacency"]
            true_adj = validation_history[-1]["adjacency"]

            return np.mean((true_adj - pred_adj) ** 2)

        # Constraints
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},  # sum to 1
        ]
        bounds = [(0, 1) for _ in range(3)]  # non-negative

        # Initial weights
        w0 = np.array([1 / 3, 1 / 3, 1 / 3])

        # Optimize
        result = minimize(
            objective,
            w0,
            method="SLSQP",
            constraints=constraints,
            bounds=bounds,
        )

        # Update weights
        self.spectral_weight = result.x[0]
        self.structural_weight = result.x[1]
        self.temporal_weight = result.x[2]
