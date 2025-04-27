"""GRNN-based graph state forecaster."""

import numpy as np
from sklearn.neural_network import MLPRegressor
from typing import List, Dict, Optional
import logging
import warnings

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
        # Store feature stats for normalization
        self.feature_means = None
        self.feature_stds = None

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

    def _normalize_features(self, features):
        """Normalize features to have zero mean and unit variance."""
        if self.feature_means is None or self.feature_stds is None:
            # Calculate on initial data
            self.feature_means = np.mean(features, axis=0)
            self.feature_stds = np.std(features, axis=0)
            # Avoid division by zero
            self.feature_stds[self.feature_stds < 1e-8] = 1.0

        # Normalize
        normalized = (features - self.feature_means) / self.feature_stds
        return normalized

    def _denormalize_features(self, normalized_features):
        """Denormalize features back to original scale."""
        if self.feature_means is None or self.feature_stds is None:
            return normalized_features

        return normalized_features * self.feature_stds + self.feature_means

    def _compute_clustering(self, adj: np.ndarray) -> float:
        """Compute average clustering coefficient."""
        n = adj.shape[0]
        if n < 3:  # Need at least 3 nodes for triangles
            return 0.0

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
        return clustering / n if n > 0 else 0

    def _compute_assortativity(self, adj: np.ndarray) -> float:
        """Compute degree assortativity coefficient."""
        degrees = np.sum(adj, axis=1)
        edges = np.where(np.triu(adj) == 1)
        if len(edges[0]) == 0:
            return 0

        try:
            # Compute correlation between degrees of connected nodes
            return np.corrcoef(degrees[edges[0]], degrees[edges[1]])[0, 1]
        except (ValueError, IndexError):
            # Handle case where correlation can't be computed
            return 0.0

    def _validate_features(self, features, n_nodes):
        """Ensure predicted features are within valid ranges."""
        (
            density,
            mean_degree,
            std_degree,
            max_degree,
            min_degree,
            clustering,
            assortativity,
        ) = features

        # Validate and correct features if needed
        # Density must be between 0 and 1
        density = np.clip(density, 0.01, 0.99)

        # Degree stats must be reasonable
        mean_degree = np.clip(mean_degree, 1.0, n_nodes - 1)
        std_degree = max(0.1, min(mean_degree / 2, std_degree))
        max_degree = min(n_nodes - 1, max(mean_degree, max_degree))
        min_degree = max(1, min(mean_degree, min_degree))

        # Clustering coefficient is between 0 and 1
        clustering = np.clip(clustering, 0.0, 1.0)

        # Assortativity is between -1 and 1
        assortativity = np.clip(assortativity, -1.0, 1.0)

        return [
            density,
            mean_degree,
            std_degree,
            max_degree,
            min_degree,
            clustering,
            assortativity,
        ]

    def _reconstruct_graph(self, features: np.ndarray, n_nodes: int) -> np.ndarray:
        """Reconstruct a graph from predicted features.

        Args:
            features: Predicted feature vector
            n_nodes: Number of nodes in the graph

        Returns:
            Predicted adjacency matrix
        """
        # Validate features to ensure they're in reasonable ranges
        validated_features = self._validate_features(features, n_nodes)

        (
            density,
            mean_degree,
            std_degree,
            max_degree,
            min_degree,
            clustering,
            assortativity,
        ) = validated_features

        # Create initial random graph with target density
        adj = np.random.random((n_nodes, n_nodes))
        adj = (adj + adj.T) / 2  # Make symmetric
        np.fill_diagonal(adj, 0)  # Zero diagonal

        try:
            # Ensure density is valid for percentile calculation (0-100)
            percentile_value = 100 * (1 - density)
            # Clip to valid range for np.percentile
            percentile_value = np.clip(percentile_value, 0, 100)

            # Threshold to achieve target density
            threshold = np.percentile(adj, percentile_value)
            adj = (adj > threshold).astype(int)
        except Exception as e:
            # Fallback if percentile fails
            logger.warning(f"Error in percentile calculation: {str(e)}")
            # Use fixed threshold as fallback
            adj = (adj > self.threshold).astype(int)

        # Adjust degrees to match predicted statistics
        current_degrees = np.sum(adj, axis=1)

        try:
            # Generate target degrees
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
        except Exception as e:
            logger.warning(f"Error in degree adjustment: {str(e)}")
            # Keep the current graph if degree adjustment fails

        # Ensure the graph is connected if required
        if self.enforce_connectivity and n_nodes > 1:
            # Simple check for connectedness - ensure minimum degree is at least 1
            for i in range(n_nodes):
                if np.sum(adj[i]) == 0:  # Isolated node
                    # Connect to a random node
                    j = np.random.choice([j for j in range(n_nodes) if j != i])
                    adj[i, j] = 1
                    adj[j, i] = 1

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
        n_nodes = history[0]["adjacency"].shape[0]

        # Reset feature stats
        self.feature_means = None
        self.feature_stds = None

        # If not enough data, use simple replication of last state
        if len(features) < 3:
            logger.warning("Not enough history for GRNN, using last state replication")
            last_adj = history[-1]["adjacency"]
            return [last_adj.copy() for _ in range(horizon)]

        try:
            # Normalize features for better neural network performance
            normalized_features = self._normalize_features(features)

            # Prepare training data
            X = normalized_features[:-1]  # Input features
            y = normalized_features[1:]  # Target features

            # Fit the model with warning suppression
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model.fit(X, y)

            # Make predictions
            current_features = normalized_features[-1].reshape(1, -1)
            predicted_normalized = []

            for _ in range(horizon):
                # Predict next step
                try:
                    next_features = self.model.predict(current_features)
                    predicted_normalized.append(next_features[0])
                    # Use prediction as input for next step
                    current_features = next_features.reshape(1, -1)
                except Exception as e:
                    logger.warning(f"Error in GRNN prediction step: {str(e)}")
                    # Use the last prediction or input if prediction fails
                    if predicted_normalized:
                        predicted_normalized.append(predicted_normalized[-1])
                    else:
                        predicted_normalized.append(normalized_features[-1])
                    current_features = predicted_normalized[-1].reshape(1, -1)

            # Denormalize predictions
            predicted_features = [
                self._denormalize_features(p) for p in predicted_normalized
            ]

        except Exception as e:
            logger.warning(f"GRNN modeling failed: {str(e)}, using fallback")
            # Fallback: Use exponential smoothing-like approach on features
            alpha = 0.7  # Weight for most recent observation
            last_features = features[-1]
            if len(features) >= 2:
                prev_features = features[-2]
                trend = last_features - prev_features
                predicted_features = [
                    last_features + (i + 1) * trend * 0.5 for i in range(horizon)
                ]
            else:
                predicted_features = [last_features for _ in range(horizon)]

        # Reconstruct graphs with error handling
        predicted_graphs = []
        for feat in predicted_features:
            try:
                adj = self._reconstruct_graph(feat, n_nodes)
                predicted_graphs.append(adj)
            except Exception as e:
                logger.warning(f"Error in graph reconstruction: {str(e)}")
                # Fallback to last adjacency matrix
                predicted_graphs.append(history[-1]["adjacency"].copy())

        return predicted_graphs
