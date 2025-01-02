"""Network forecasting models and prediction utilities.

This module provides classes and functions for predicting future states
of evolving networks using various forecasting approaches.
"""

import networkx as nx
import numpy as np
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod


def pad_or_truncate_matrix(matrix: np.ndarray, target_size: int) -> np.ndarray:
    """Pad or truncate a matrix to match the target size.
    
    If matrix is smaller, pad with zeros.
    If matrix is larger, truncate to target size.
    
    Parameters
    ----------
    matrix : np.ndarray
        Input matrix to resize
    target_size : int
        Desired size of output matrix
        
    Returns
    -------
    np.ndarray
        Resized matrix of shape (target_size, target_size)
    """
    current_size = matrix.shape[0]
    if current_size == target_size:
        return matrix
        
    if current_size < target_size:
        # Pad with zeros
        padded = np.zeros((target_size, target_size))
        padded[:current_size, :current_size] = matrix
        return padded
    else:
        # Truncate
        return matrix[:target_size, :target_size]


class BaseNetworkPredictor(ABC):
    """Abstract base class for network prediction models."""

    @abstractmethod
    def predict(
        self, history: List[Dict[str, Any]], horizon: int = 1
    ) -> List[np.ndarray]:
        """Predict future network states.

        Parameters
        ----------
        history : List[Dict[str, Any]]
            Historical network data
        horizon : int, optional
            Number of steps to predict ahead, by default 1

        Returns
        -------
        List[np.ndarray]
            List of predicted adjacency matrices
        """
        pass


class WeightedAveragePredictor(BaseNetworkPredictor):
    """Network predictor using weighted averaging of recent states.

    Parameters
    ----------
    n_history : int, optional
        Number of historical points to use, by default 3
    weights : np.ndarray, optional
        Weights for historical points (newest to oldest), by default None
    """

    def __init__(self, n_history: int = 3, weights: Optional[np.ndarray] = None):
        self.n_history = n_history
        if weights is None:
            # Default weights favor recent history
            weights = np.array([0.5, 0.3, 0.2])
        self.weights = weights / np.sum(weights)  # Normalize weights

    def predict(
        self, history: List[Dict[str, Any]], horizon: int = 1
    ) -> List[np.ndarray]:
        """Predict future network states using weighted averaging.

        Parameters
        ----------
        history : List[Dict[str, Any]]
            Historical network data
        horizon : int, optional
            Number of steps to predict ahead, by default 1

        Returns
        -------
        List[np.ndarray]
            List of predicted adjacency matrices

        Notes
        -----
        The prediction process:
        1. Compute weighted average of recent adjacency matrices
        2. Determine target network properties
        3. Generate new network matching these properties
        4. Ensure network connectivity and degree distribution
        """
        predictions = []
        current_history = history.copy()

        for _ in range(horizon):
            # Get recent networks
            last_networks = current_history[-self.n_history :]

            # Get target properties from most recent network
            latest_network = last_networks[-1]["graph"]
            target_degrees = sorted([d for _, d in latest_network.degree()])
            target_avg_degree = np.mean(target_degrees)

            # Compute weighted average
            avg_adj = self._compute_weighted_average(
                [net["adjacency"] for net in last_networks]
            )

            # Generate predicted network
            predicted_adj = self._generate_network(avg_adj, target_avg_degree)

            # Store prediction
            predictions.append(predicted_adj)

            # Update history for next prediction
            current_history.append(
                {
                    "adjacency": predicted_adj,
                    "graph": nx.from_numpy_array(predicted_adj),
                    "params": current_history[-1]["params"],
                }
            )

        return predictions

    def _compute_weighted_average(
        self, adjacency_matrices: List[np.ndarray]
    ) -> np.ndarray:
        """Compute weighted average of adjacency matrices.

        Parameters
        ----------
        adjacency_matrices : List[np.ndarray]
            List of adjacency matrices

        Returns
        -------
        np.ndarray
            Weighted average matrix
        """
        avg_adj = np.zeros_like(adjacency_matrices[0], dtype=float)
        for adj, weight in zip(adjacency_matrices, self.weights):
            avg_adj += weight * adj.astype(float)

        # Normalize probabilities
        avg_adj = (avg_adj - avg_adj.min()) / (avg_adj.max() - avg_adj.min() + 1e-10)
        return avg_adj

    def _generate_network(
        self, prob_matrix: np.ndarray, target_avg_degree: float
    ) -> np.ndarray:
        """Generate network from probability matrix matching target properties.

        Parameters
        ----------
        prob_matrix : np.ndarray
            Edge probability matrix
        target_avg_degree : float
            Target average degree

        Returns
        -------
        np.ndarray
            Generated adjacency matrix
        """
        n = prob_matrix.shape[0]
        target_edges = int((target_avg_degree * n) / 2)

        # Get upper triangular indices and probabilities
        triu_indices = np.triu_indices(n, k=1)
        edge_probs = prob_matrix[triu_indices]
        edge_indices = list(zip(triu_indices[0], triu_indices[1]))
        sorted_edges = sorted(zip(edge_probs, edge_indices), reverse=True)

        # Create adjacency matrix
        predicted_adj = np.zeros_like(prob_matrix, dtype=int)

        # Add edges based on probability
        for _, (i, j) in sorted_edges[:target_edges]:
            predicted_adj[i, j] = predicted_adj[j, i] = 1

        # Ensure connectivity
        g_temp = nx.from_numpy_array(predicted_adj)
        components = list(nx.connected_components(g_temp))

        if len(components) > 1:
            main_comp = max(components, key=len)
            other_comps = [c for c in components if c != main_comp]

            for comp in other_comps:
                # Find best edge to connect components
                best_edge = None
                best_prob = -1

                for n1 in main_comp:
                    for n2 in comp:
                        prob = prob_matrix[n1, n2]
                        if prob > best_prob:
                            best_prob = prob
                            best_edge = (n1, n2)

                if best_edge:
                    i, j = best_edge
                    predicted_adj[i, j] = predicted_adj[j, i] = 1

        return predicted_adj
