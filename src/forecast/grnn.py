"""GRNN-based graph state forecaster."""

import numpy as np
from sklearn.neural_network import MLPRegressor
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class GRNNGraphForecaster:
    """GRNN-based forecaster for graph states.

    This forecaster uses a Generalized Regression Neural Network to predict
    future graph states by learning patterns in historical graph features.
    """

    def __init__(
        self,
        hidden_layer_sizes: tuple = (100, 50),
        activation: str = "relu",
        solver: str = "adam",
        alpha: float = 0.0001,
        batch_size: int = 32,
        learning_rate: str = "constant",
        max_iter: int = 200,
        random_state: Optional[int] = None,
        enforce_connectivity: bool = True,
        threshold: float = 0.5,
    ):
        """Initialize the GRNN forecaster.

        Args:
            hidden_layer_sizes: Number of neurons in each hidden layer
            activation: Activation function for hidden layers
            solver: The solver for weight optimization
            alpha: L2 penalty (regularization term) parameter
            batch_size: Size of minibatches for stochastic optimizers
            learning_rate: Learning rate schedule for weight updates
            max_iter: Maximum number of iterations
            random_state: Random seed for reproducibility
            enforce_connectivity: Whether to ensure predicted graphs are connected
            threshold: Threshold for binarizing predicted adjacency matrices
        """
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            alpha=alpha,
            batch_size=batch_size,
            learning_rate=learning_rate,
            max_iter=max_iter,
            random_state=random_state,
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
        features = []
        for state in history:
            adj = state["adjacency"]
            if not isinstance(adj, np.ndarray):
                adj = np.array(adj)

            # Extract basic graph features
            n = adj.shape[0]
            density = np.sum(adj) / (n * (n - 1))
            degrees = np.sum(adj, axis=1)
            mean_degree = np.mean(degrees)
            std_degree = np.std(degrees)

            # Extract additional structural features
            clustering = self._compute_clustering(adj)
            assortativity = self._compute_assortativity(adj)

            # Combine features
            feature_vector = [
                density,
                mean_degree,
                std_degree,
                np.max(degrees),
                np.min(degrees),
                clustering,
                assortativity,
            ]
            features.append(feature_vector)

        return np.array(features)

    def _compute_clustering(self, adj: np.ndarray) -> float:
        """Compute average clustering coefficient."""
        n = adj.shape[0]
        clustering = 0
        for i in range(n):
            neighbors = np.where(adj[i] == 1)[0]
            k = len(neighbors)
            if k < 2:
                continue
            # Count triangles
            triangles = 0
            for j in range(len(neighbors)):
                for l in range(j + 1, len(neighbors)):
                    if adj[neighbors[j], neighbors[l]] == 1:
                        triangles += 1
            clustering += 2 * triangles / (k * (k - 1))
        return clustering / n

    def _compute_assortativity(self, adj: np.ndarray) -> float:
        """Compute degree assortativity coefficient."""
        degrees = np.sum(adj, axis=1)
        edges = np.where(np.triu(adj) == 1)
        if len(edges[0]) == 0:
            return 0
        # Compute correlation between degrees of connected nodes
        return np.corrcoef(degrees[edges[0]], degrees[edges[1]])[0, 1]

    def _reconstruct_graph(self, features: np.ndarray, n_nodes: int) -> np.ndarray:
        """Reconstruct a graph from predicted features.

        Args:
            features: Predicted feature vector
            n_nodes: Number of nodes in the graph

        Returns:
            Predicted adjacency matrix
        """
        (
            density,
            mean_degree,
            std_degree,
            max_degree,
            min_degree,
            clustering,
            assortativity,
        ) = features

        # Create initial random graph with target density
        adj = np.random.random((n_nodes, n_nodes))
        adj = (adj + adj.T) / 2  # Make symmetric
        np.fill_diagonal(adj, 0)  # Zero diagonal

        # Threshold to achieve target density
        threshold = np.percentile(adj, (1 - density) * 100)
        adj = (adj > threshold).astype(int)

        # Adjust degrees to match predicted statistics
        current_degrees = np.sum(adj, axis=1)
        target_degrees = np.random.normal(mean_degree, std_degree, n_nodes)
        target_degrees = np.clip(target_degrees, min_degree, max_degree)

        # Adjust edges to match target degrees
        for i in range(n_nodes):
            diff = int(target_degrees[i] - current_degrees[i])
            if diff > 0:
                # Add edges
                non_edges = np.where(adj[i] == 0)[0]
                non_edges = non_edges[non_edges != i]  # Exclude self-loops
                if len(non_edges) > 0:
                    to_add = np.random.choice(non_edges, min(diff, len(non_edges)))
                    adj[i, to_add] = 1
                    adj[to_add, i] = 1
            elif diff < 0:
                # Remove edges
                edges = np.where(adj[i] == 1)[0]
                if len(edges) > 0:
                    to_remove = np.random.choice(edges, min(-diff, len(edges)))
                    adj[i, to_remove] = 0
                    adj[to_remove, i] = 0

        return adj

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

        # Extract features from history
        features = self._extract_features(history)
        n_features = features.shape[1]
        n_nodes = history[0]["adjacency"].shape[0]

        # Prepare training data
        X = features[:-1]  # Input features
        y = features[1:]  # Target features

        # Fit the model
        self.model.fit(X, y)

        # Make predictions
        current_features = features[-1].reshape(1, -1)
        predicted_features = []

        for _ in range(horizon):
            # Predict next step
            next_features = self.model.predict(current_features)
            predicted_features.append(next_features[0])
            # Use prediction as input for next step
            current_features = next_features.reshape(1, -1)

        # Reconstruct graphs from predicted features
        predicted_graphs = []
        for features in predicted_features:
            adj = self._reconstruct_graph(features, n_nodes)
            predicted_graphs.append(adj)

        return predicted_graphs
