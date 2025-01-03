"""Dynamical system-based network prediction.

This module implements network prediction by modeling network evolution
as a dynamical system. The core idea is to learn the underlying dynamics
that govern how network structure changes over time.

Mathematical Foundation:
----------------------
Given a sequence of adjacency matrices A(t), we model:

1. State Space Representation:
   dA/dt = F(A, t) + η(t)
   where:
   - F(A, t) is the drift function
   - η(t) is the diffusion noise
   
2. Linear Approximation:
   A(t+1) = A(t) + ΔtF(A(t)) where:
   F(A) = WA + AW^T + b
   - W is the learned weight matrix
   - b is the bias term
   
3. Non-linear Extensions:
   F(A) = σ(WA + AW^T + b)
   where σ is a non-linear activation

4. Stability Constraints:
   - Eigenvalue regularization for stability
   - Structural preservation via projection
   - Noise reduction through filtering
"""

import numpy as np
import networkx as nx
from typing import List, Dict, Any, Optional, Tuple
from scipy.linalg import expm
from .base import BasePredictor


class DynamicalPredictor(BasePredictor):
    """Predicts network evolution using dynamical systems.

    Parameters
    ----------
    learning_rate : float, optional
        Learning rate for parameter updates, by default 0.01
    regularization : float, optional
        L2 regularization strength, by default 0.1
    nonlinear : bool, optional
        Whether to use non-linear dynamics, by default False
    noise_std : float, optional
        Standard deviation of noise, by default 0.1
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        regularization: float = 0.1,
        nonlinear: bool = False,
        noise_std: float = 0.1,
    ):
        """Initialize the dynamical predictor."""
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.nonlinear = nonlinear
        self.noise_std = noise_std
        self.weight_matrix: Optional[np.ndarray] = None
        self.bias: Optional[np.ndarray] = None

    def predict(
        self,
        history: List[Dict[str, Any]],
        horizon: int = 1,
        **kwargs,
    ) -> List[np.ndarray]:
        """Predict future network states using learned dynamics.

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

        # Learn dynamics if not already learned
        if self.weight_matrix is None:
            self._learn_dynamics(adj_matrices)

        # Current state
        current_adj = adj_matrices[-1]
        predictions = []

        # Predict future states
        for step in range(horizon):
            # Apply dynamics
            next_adj = self._apply_dynamics(current_adj)

            # Add noise
            if self.noise_std > 0:
                noise = np.random.normal(0, self.noise_std, size=next_adj.shape)
                next_adj += noise

            # Threshold and symmetrize
            next_adj = (next_adj + next_adj.T) / 2
            next_adj = (next_adj > 0.5).astype(float)

            # Store prediction
            predictions.append(next_adj)

            # Update current state
            current_adj = next_adj

        return predictions

    def _learn_dynamics(self, adj_matrices: List[np.ndarray]) -> None:
        """Learn dynamical system parameters from data.

        Parameters
        ----------
        adj_matrices : List[np.ndarray]
            Historical adjacency matrices

        Notes
        -----
        Uses gradient descent to learn W and b in:
        dA/dt ≈ WA + AW^T + b
        """
        if len(adj_matrices) < 2:
            return

        n = adj_matrices[0].shape[0]

        # Initialize parameters if not exists
        if self.weight_matrix is None:
            self.weight_matrix = np.random.normal(
                0, 0.01, size=(n, n)
            )  # Smaller initialization
            self.bias = np.zeros((n, n))

        # Prepare data and normalize
        X = np.array(adj_matrices[:-1], dtype=np.float64)
        Y = np.array(adj_matrices[1:], dtype=np.float64)

        # Scale data to [0, 1] range
        X_max = np.max(np.abs(X))
        Y_max = np.max(np.abs(Y))
        scale = max(X_max, Y_max)
        if scale > 0:
            X = X / scale
            Y = Y / scale

        # Gradient descent iterations
        for _ in range(100):  # Fixed number of iterations
            # Forward pass
            pred = self._batch_dynamics(X)

            # Compute gradients with clipping
            error = Y - pred
            grad_W = np.zeros_like(self.weight_matrix)
            grad_b = np.zeros_like(self.bias)

            for i in range(len(X)):
                # Compute gradients with numerical stability
                dW = np.clip(error[i] @ X[i].T + X[i] @ error[i].T, -1, 1)
                grad_W += dW
                grad_b += np.clip(error[i], -1, 1)

            # Average gradients
            grad_W /= len(X)
            grad_b /= len(X)

            # Update parameters with gradient clipping
            self.weight_matrix += self.learning_rate * np.clip(
                grad_W - self.regularization * self.weight_matrix, -1, 1
            )
            self.bias += self.learning_rate * grad_b

            # Add numerical stability through parameter bounds
            self.weight_matrix = np.clip(self.weight_matrix, -1, 1)
            self.bias = np.clip(self.bias, -1, 1)

    def _batch_dynamics(self, adj_matrices: np.ndarray) -> np.ndarray:
        """Apply dynamics to a batch of adjacency matrices.

        Parameters
        ----------
        adj_matrices : np.ndarray
            Batch of adjacency matrices

        Returns
        -------
        np.ndarray
            Predicted next states
        """
        pred = np.zeros_like(adj_matrices)
        for i in range(len(adj_matrices)):
            pred[i] = self._apply_dynamics(adj_matrices[i])
        return pred

    def _apply_dynamics(self, adj_matrix: np.ndarray) -> np.ndarray:
        """Apply learned dynamics to current state.

        Parameters
        ----------
        adj_matrix : np.ndarray
            Current adjacency matrix

        Returns
        -------
        np.ndarray
            Predicted next state

        Notes
        -----
        Implements the update:
        A(t+1) = A(t) + F(A(t))
        where F is linear or non-linear dynamics
        """
        if self.weight_matrix is None:
            return adj_matrix

        # Ensure inputs are float64
        adj_matrix = adj_matrix.astype(np.float64)
        weight_matrix = self.weight_matrix.astype(np.float64)
        bias = self.bias.astype(np.float64)

        # Compute drift term with numerical stability
        drift = np.clip(
            weight_matrix @ adj_matrix + adj_matrix @ weight_matrix.T + bias, -1, 1
        )

        if self.nonlinear:
            # Stable sigmoid implementation
            drift = 1 / (1 + np.exp(-np.clip(drift, -10, 10)))

        # Euler integration with stability
        next_state = np.clip(adj_matrix + drift, 0, 1)

        return next_state
