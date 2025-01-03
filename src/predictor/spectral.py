"""Spectral-based network prediction.

This module implements network prediction using spectral decomposition methods.
The core idea is to track the evolution of eigenvalues and eigenvectors of
the adjacency matrix to predict future network states.

Mathematical Foundation:
----------------------
Given an adjacency matrix A(t) at time t, we perform:

1. Spectral Decomposition:
   A(t) = U(t)Σ(t)V(t)ᵀ
   where:
   - U(t), V(t) are orthogonal matrices containing eigenvectors
   - Σ(t) is a diagonal matrix of singular values

2. Temporal Pattern Extraction:
   - Track evolution of singular values: Σ(t) → Σ(t+1)
   - Model eigenvector dynamics: U(t) → U(t+1)
   
3. Future State Prediction:
   A(t+1) = U(t+1)Σ(t+1)V(t+1)ᵀ
   where future components are estimated using:
   - Linear extrapolation of singular values
   - Rotation-based prediction of eigenvectors
   - Thresholding to maintain binary adjacency

4. Stability Enhancement:
   - Low-rank approximation for noise reduction
   - Eigenvalue smoothing for temporal consistency
   - Structural constraints preservation
"""

import numpy as np
import networkx as nx
from typing import List, Dict, Any, Optional
from .base import BasePredictor


class SpectralPredictor(BasePredictor):
    """Predicts network evolution using spectral decomposition.

    Parameters
    ----------
    n_components : int, optional
        Number of singular values to use, by default None (uses all)
    threshold : float, optional
        Threshold for binarizing predictions, by default 0.5
    smoothing : float, optional
        Temporal smoothing factor (0 to 1), by default 0.2
    """

    def __init__(
        self,
        n_components: Optional[int] = None,
        threshold: float = 0.5,
        smoothing: float = 0.2,
    ):
        """Initialize the spectral predictor."""
        self.n_components = n_components
        self.threshold = threshold
        self.smoothing = smoothing
        self.singular_values_history: List[np.ndarray] = []
        self.left_vectors_history: List[np.ndarray] = []
        self.right_vectors_history: List[np.ndarray] = []

    def predict(
        self,
        history: List[Dict[str, Any]],
        horizon: int = 1,
        **kwargs,
    ) -> List[np.ndarray]:
        """Predict future network states using spectral patterns.

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

        Notes
        -----
        The prediction process follows these steps:
        1. Decompose historical adjacency matrices
        2. Extract temporal patterns in spectral components
        3. Predict future components using extrapolation
        4. Reconstruct future adjacency matrices
        """
        predictions = []

        # Get historical adjacency matrices
        adj_matrices = [state["adjacency"] for state in history]

        # Update spectral component histories
        for adj in adj_matrices:
            U, S, Vt = np.linalg.svd(adj)
            if self.n_components:
                U = U[:, : self.n_components]
                S = S[: self.n_components]
                Vt = Vt[: self.n_components, :]

            self.singular_values_history.append(S)
            self.left_vectors_history.append(U)
            self.right_vectors_history.append(Vt)

        # Predict future states
        for step in range(horizon):
            # Predict singular values using linear extrapolation
            S_future = self._extrapolate_singular_values()

            # Predict eigenvectors using rotation patterns
            U_future = self._predict_eigenvectors(self.left_vectors_history)
            Vt_future = self._predict_eigenvectors(self.right_vectors_history)

            # Reconstruct future adjacency matrix
            adj_pred = U_future @ np.diag(S_future) @ Vt_future

            # Apply temporal smoothing
            if len(adj_matrices) > 0:
                adj_pred = (
                    1 - self.smoothing
                ) * adj_pred + self.smoothing * adj_matrices[-1]

            # Threshold and symmetrize
            adj_pred = (adj_pred + adj_pred.T) / 2  # Ensure symmetry
            adj_pred = (adj_pred > self.threshold).astype(float)

            # Store prediction
            predictions.append(adj_pred)

            # Update histories with prediction
            adj_matrices.append(adj_pred)
            U, S, Vt = np.linalg.svd(adj_pred)
            if self.n_components:
                U = U[:, : self.n_components]
                S = S[: self.n_components]
                Vt = Vt[: self.n_components, :]

            self.singular_values_history.append(S)
            self.left_vectors_history.append(U)
            self.right_vectors_history.append(Vt)

        return predictions

    def _extrapolate_singular_values(self) -> np.ndarray:
        """Extrapolate future singular values using linear regression.

        Returns
        -------
        np.ndarray
            Predicted singular values

        Notes
        -----
        Uses recent trend in singular values to predict next values:
        S(t+1) = S(t) + Δ
        where Δ is estimated from [S(t) - S(t-1)]
        """
        if len(self.singular_values_history) < 2:
            return self.singular_values_history[-1]

        # Get recent changes
        recent_changes = np.diff(self.singular_values_history[-3:], axis=0)
        avg_change = np.mean(recent_changes, axis=0)

        # Predict next values
        S_pred = self.singular_values_history[-1] + avg_change

        # Ensure non-negativity
        S_pred = np.maximum(S_pred, 0)

        return S_pred

    def _predict_eigenvectors(self, vector_history: List[np.ndarray]) -> np.ndarray:
        """Predict future eigenvectors using rotation patterns.

        Parameters
        ----------
        vector_history : List[np.ndarray]
            History of eigenvector matrices

        Returns
        -------
        np.ndarray
            Predicted eigenvector matrix

        Notes
        -----
        Estimates rotation matrix R between consecutive time steps:
        U(t) ≈ U(t-1)R
        Then applies this rotation to predict U(t+1)
        """
        if len(vector_history) < 2:
            return vector_history[-1]

        # Get last two eigenvector matrices
        U_prev = vector_history[-2]
        U_curr = vector_history[-1]

        # Estimate rotation matrix
        R = np.linalg.pinv(U_prev) @ U_curr

        # Predict next eigenvectors
        U_pred = U_curr @ R

        # Ensure orthonormality
        U_pred, _ = np.linalg.qr(U_pred)

        return U_pred
