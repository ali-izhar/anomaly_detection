"""Node embedding-based network prediction.

This module implements network prediction using node embedding techniques.
The core idea is to learn low-dimensional node representations that capture
both structural and temporal patterns in the network evolution.

Mathematical Foundation:
----------------------
Given a sequence of graphs G(t), we:

1. Node Embedding Generation:
   For each graph G(t), compute embeddings Z(t) ∈ ℝⁿˣᵈ where:
   - n is the number of nodes
   - d is the embedding dimension
   - Z(t) captures structural properties using SVD/LSVD:
     - SVD: A(t) = UΣVᵀ → Z(t) = UₖΣₖ
     - LSVD: L(t) = D⁻¹/²AD⁻¹/² = QΛQᵀ → Z(t) = Qₖ

2. Temporal Pattern Learning:
   Model embedding evolution Z(t) → Z(t+1) using:
   - Linear dynamics: Z(t+1) = Z(t)W + b
   - Non-linear transformation: Z(t+1) = σ(Z(t)W + b)
   where W is a learned transition matrix

3. Link Probability Estimation:
   P(A_{ij}(t+1) = 1) = σ(z_i(t+1)ᵀz_j(t+1))
   where:
   - z_i(t+1) is the predicted embedding for node i
   - σ is the sigmoid function

4. Stability Enhancement:
   - Regularization for smooth transitions
   - Structural constraint preservation
   - Ensemble of different embedding types
"""

import numpy as np
import networkx as nx
from typing import List, Dict, Any, Optional, Tuple
from sklearn.preprocessing import normalize
from scipy import sparse
from scipy.sparse.linalg import eigsh
from .base import BasePredictor


class EmbeddingPredictor(BasePredictor):
    """Predicts network evolution using node embeddings.

    Parameters
    ----------
    embedding_dim : int, optional
        Dimension of node embeddings, by default 32
    embedding_type : str, optional
        Type of embedding ('svd' or 'lsvd'), by default 'lsvd'
    learning_rate : float, optional
        Learning rate for transition matrix, by default 0.01
    threshold : float, optional
        Threshold for link prediction, by default 0.5
    regularization : float, optional
        L2 regularization strength, by default 0.1
    """

    def __init__(
        self,
        embedding_dim: int = 32,
        embedding_type: str = "lsvd",
        learning_rate: float = 0.01,
        threshold: float = 0.5,
        regularization: float = 0.1,
    ):
        """Initialize the embedding predictor."""
        self.embedding_dim = embedding_dim
        self.embedding_type = embedding_type
        self.learning_rate = learning_rate
        self.threshold = threshold
        self.regularization = regularization
        self.embedding_history: List[np.ndarray] = []
        self.transition_matrix: Optional[np.ndarray] = None

    def predict(
        self,
        history: List[Dict[str, Any]],
        horizon: int = 1,
        **kwargs,
    ) -> List[np.ndarray]:
        """Predict future network states using embedding patterns.

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

        # Update embedding history
        for adj in adj_matrices:
            embedding = self._compute_embedding(adj)
            self.embedding_history.append(embedding)

        # Learn transition patterns if not already learned
        if self.transition_matrix is None:
            self._learn_transition_matrix()

        predictions = []

        # Predict future states
        for step in range(horizon):
            # Predict next embedding
            Z_current = self.embedding_history[-1]
            Z_pred = self._predict_next_embedding(Z_current)

            # Convert embeddings to adjacency matrix
            adj_pred = self._embeddings_to_adjacency(Z_pred)

            # Store prediction
            predictions.append(adj_pred)

            # Update histories
            self.embedding_history.append(Z_pred)

        return predictions

    def _compute_embedding(self, adj_matrix: np.ndarray) -> np.ndarray:
        """Compute node embeddings from adjacency matrix.

        Parameters
        ----------
        adj_matrix : np.ndarray
            Input adjacency matrix

        Returns
        -------
        np.ndarray
            Node embeddings matrix

        Notes
        -----
        Two embedding types supported:
        1. SVD: Uses singular value decomposition of adjacency
        2. LSVD: Uses Laplacian eigenvectors
        """
        if self.embedding_type == "svd":
            # SVD of adjacency matrix
            U, S, _ = np.linalg.svd(adj_matrix)
            return U[:, : self.embedding_dim] * np.sqrt(S[: self.embedding_dim])

        else:  # LSVD
            # Convert to sparse format
            adj_sparse = sparse.csr_matrix(adj_matrix)

            # Compute degree matrix
            deg = np.array(adj_sparse.sum(axis=1)).flatten()
            D_inv_sqrt = sparse.diags(1.0 / np.sqrt(np.maximum(deg, 1e-12)))

            # Normalized Laplacian
            L_norm = (
                sparse.eye(adj_sparse.shape[0]) - D_inv_sqrt @ adj_sparse @ D_inv_sqrt
            )

            try:
                # Try with increased NCV (number of Lanczos vectors)
                ncv = min(max(2 * self.embedding_dim + 1, 20), adj_matrix.shape[0])
                _, vectors = eigsh(-L_norm, k=self.embedding_dim, which="LA", ncv=ncv)
                return vectors
            except:
                # Fallback to dense eigenvector computation if sparse fails
                L_dense = L_norm.toarray()
                eigvals, eigvecs = np.linalg.eigh(-L_dense)
                # Get largest eigenvalues (which='LA' equivalent)
                idx = np.argsort(eigvals)[::-1][: self.embedding_dim]
                return eigvecs[:, idx]

    def _learn_transition_matrix(self) -> None:
        """Learn transition matrix for embedding evolution.

        Notes
        -----
        Uses ridge regression to learn W in:
        Z(t+1) = Z(t)W + b
        """
        if len(self.embedding_history) < 2:
            return

        # Prepare data
        X = np.vstack(self.embedding_history[:-1])
        Y = np.vstack(self.embedding_history[1:])

        # Ridge regression with regularization
        XtX = X.T @ X + self.regularization * np.eye(X.shape[1])
        XtY = X.T @ Y
        self.transition_matrix = np.linalg.solve(XtX, XtY)

    def _predict_next_embedding(self, current_embedding: np.ndarray) -> np.ndarray:
        """Predict next embedding using learned transition.

        Parameters
        ----------
        current_embedding : np.ndarray
            Current node embeddings

        Returns
        -------
        np.ndarray
            Predicted node embeddings
        """
        if self.transition_matrix is None:
            return current_embedding

        # Apply learned transition
        next_embedding = current_embedding @ self.transition_matrix

        # Normalize to unit length
        return normalize(next_embedding)

    def _embeddings_to_adjacency(self, embeddings: np.ndarray) -> np.ndarray:
        """Convert node embeddings to adjacency matrix.

        Parameters
        ----------
        embeddings : np.ndarray
            Node embedding matrix

        Returns
        -------
        np.ndarray
            Predicted adjacency matrix

        Notes
        -----
        Uses dot product similarity with sigmoid:
        P(A_{ij} = 1) = σ(z_i^T z_j)
        """
        # Compute similarity matrix
        similarity = embeddings @ embeddings.T

        # Apply sigmoid
        adj_pred = 1 / (1 + np.exp(-similarity))

        # Threshold and symmetrize
        adj_pred = (adj_pred + adj_pred.T) / 2
        adj_pred = (adj_pred > self.threshold).astype(float)

        return adj_pred
