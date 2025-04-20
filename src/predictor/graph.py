import numpy as np
import networkx as nx
from sklearn.cluster import KMeans
from scipy.optimize import minimize
from scipy.sparse.linalg import eigsh
import warnings
from itertools import combinations

# Handle community detection package import
try:
    from community import community_louvain

    COMMUNITY_DETECTION_AVAILABLE = True
except ImportError:
    COMMUNITY_DETECTION_AVAILABLE = False
    warnings.warn(
        "python-louvain package not found. Using fallback community detection methods."
    )


class GraphPredictor:
    """
    Implements the hybrid graph prediction approach from Chapter 5:
    - Temporal Pattern Prediction
    - Structural Role Preservation
    - Adaptive Integration

    Enhanced with improved optimization approaches to better match theoretical guarantees.
    """

    def __init__(
        self,
        alpha=0.8,
        gamma=0.5,
        omega=None,
        beta_init=0.5,
        enforce_connectivity=True,
        adaptive=True,
        optimization_iterations=3,
        threshold=0.5,
        n_history=10,
    ):
        """
        Initialize the graph predictor

        Args:
            alpha (float): Decay parameter for temporal weights (α ∈ (0,1))
            gamma (float): Regularization parameter for structural constraints
            omega (list): Weights for regularization terms [degree, community, sparsity]
            beta_init (float): Initial mixing parameter between temporal and structural predictions
            enforce_connectivity (bool): Whether to enforce connectivity in predictions
            adaptive (bool): Whether to adaptively update beta based on prediction error
            optimization_iterations (int): Number of iterations for joint constraint optimization
            threshold (float): Threshold for final binarization of adjacency matrix
        """
        self.alpha = alpha
        self.gamma = gamma
        self.omega = omega if omega else [0.4, 0.4, 0.2]  # Default weights
        self.beta = beta_init
        self.enforce_connectivity = enforce_connectivity
        self.adaptive = adaptive
        self.prediction_error_history = []
        self.community_labels = None
        self.optimization_iterations = optimization_iterations
        self.threshold = threshold
        self.n_history = n_history

    def predict(self, history, horizon=5):
        """
        Predict future graph states

        Args:
            history (list): List of dictionaries containing past adjacency matrices
                           [{"adjacency": adj_matrix}, ...]
            horizon (int): Number of future time steps to predict

        Returns:
            list: List of predicted adjacency matrices for the next 'horizon' time steps
        """
        if len(history) < self.n_history:
            raise ValueError(
                f"Need at least {self.n_history} historical graphs for prediction"
            )

        # Extract adjacency matrices and convert to numpy arrays if needed
        adj_matrices = []
        for graph_dict in history:
            adj = graph_dict["adjacency"]
            if not isinstance(adj, np.ndarray):
                adj = np.array(adj)
            adj_matrices.append(adj)

        # Identify communities in the most recent graph for structural preservation
        self._identify_communities(adj_matrices[-1])

        # Calculate historical statistics for degree distribution, etc.
        self._calculate_historical_stats(adj_matrices)

        # Adaptively set parameters based on graph characteristics
        self._set_adaptive_parameters(adj_matrices)

        # Generate predictions for each future time step
        predicted_adjs = []
        for i in range(horizon):
            # 1. Generate temporal prediction
            temporal_pred = self._temporal_prediction(adj_matrices, i + 1)

            # 2. Generate structural prediction (keeping probability information)
            structural_pred = self._structural_prediction(temporal_pred)

            # 3. Combine predictions adaptively
            combined_pred = self._adaptive_integration(temporal_pred, structural_pred)

            # 4. Threshold to binary adjacency matrix only at the final step
            binary_pred = (combined_pred > self.threshold).astype(int)

            # 5. Ensure symmetric matrix with zero diagonal
            binary_pred = self._ensure_valid_adjacency(binary_pred)

            # 6. Optionally ensure connected graph (largest component)
            if self.enforce_connectivity:
                binary_pred = self._ensure_connected(binary_pred)

            # Add to predictions
            predicted_adjs.append(binary_pred)

            # Update history with new prediction for subsequent predictions
            adj_matrices.append(binary_pred)

        return predicted_adjs

    def _set_adaptive_parameters(self, adj_matrices):
        """
        Set adaptive parameters based on graph characteristics

        Args:
            adj_matrices (list): Historical adjacency matrices
        """
        n = adj_matrices[0].shape[0]

        # 1. Set adaptive z-score thresholds based on graph size
        self.high_degree_threshold = 1.5 if n < 100 else 2.0
        self.low_degree_threshold = -1.5 if n < 100 else -2.0

        # 2. Set adaptive density threshold based on historical variance
        densities = [np.sum(adj) / (n * (n - 1)) for adj in adj_matrices]
        density_std = np.std(densities)
        self.density_threshold = max(0.05, min(0.2, 3 * density_std))

        # 3. Set community adjustment probability based on modularity
        if COMMUNITY_DETECTION_AVAILABLE:
            G = nx.from_numpy_array(adj_matrices[-1])
            try:
                partition = community_louvain.best_partition(G)
                modularity = community_louvain.modularity(partition, G)
                self.community_adjustment_prob = 0.2 if modularity < 0.3 else 0.4
            except Exception:
                self.community_adjustment_prob = (
                    0.3  # Default if community detection fails
                )
        else:
            self.community_adjustment_prob = 0.3  # Default if package not available

        # 4. Set weight decay based on temporal stability
        if len(adj_matrices) > 3:
            # Calculate edge stability across time
            edge_stability = self._calculate_edge_stability(adj_matrices)
            self.alpha = max(0.7, min(0.95, edge_stability))

    def _calculate_edge_stability(self, adj_matrices):
        """
        Calculate edge stability across time periods

        Args:
            adj_matrices (list): Historical adjacency matrices

        Returns:
            float: Edge stability metric (higher means more stable)
        """
        n = adj_matrices[0].shape[0]
        stability = 0

        # For each pair of adjacent timestamps
        for t in range(len(adj_matrices) - 1):
            # Calculate consistency ratio (fraction of edges that remain the same)
            consistency = 1 - np.sum(np.abs(adj_matrices[t] - adj_matrices[t + 1])) / (
                n * n
            )
            stability += consistency

        # Average stability
        stability /= len(adj_matrices) - 1
        return stability

    def _temporal_prediction(self, adj_matrices, step):
        """
        Implement temporal pattern prediction using weighted averaging:
        Â_{t+i}^{(T)} = ∑_{j=1}^k w_j A_{t-j}

        Args:
            adj_matrices (list): Historical adjacency matrices
            step (int): Step ahead to predict

        Returns:
            np.ndarray: Temporal prediction of adjacency matrix (probabilities)
        """
        k = len(adj_matrices)

        # Calculate weights: w_j = α^j/∑_{l=1}^k α^l
        # Use time-decaying weights to emphasize recent observations
        unnormalized_weights = np.array([self.alpha**j for j in range(1, k + 1)])
        weights = unnormalized_weights / np.sum(unnormalized_weights)

        # Generate weighted sum of historical adjacencies
        temporal_pred = np.zeros_like(adj_matrices[0], dtype=float)
        for j in range(k):
            temporal_pred += weights[j] * adj_matrices[k - j - 1]

        return temporal_pred

    def _identify_communities(self, adj_matrix):
        """
        Identify communities in the graph using advanced network community detection

        Args:
            adj_matrix (np.ndarray): Adjacency matrix
        """
        # Convert to graph for community detection
        G = nx.from_numpy_array(adj_matrix)

        if COMMUNITY_DETECTION_AVAILABLE:
            try:
                # Use Louvain method for community detection (better for graphs)
                partition = community_louvain.best_partition(G)
                self.community_labels = np.array([partition[i] for i in range(len(G))])
                self.n_communities = len(set(partition.values()))
                return
            except Exception:
                # Fall back to spectral clustering if Louvain fails
                pass

        # Fallback methods if community_louvain is not available or fails
        try:
            # Use eigengap heuristic to estimate number of communities
            laplacian = nx.normalized_laplacian_matrix(G).toarray()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                eigenvalues, _ = eigsh(laplacian, k=min(6, laplacian.shape[0] - 1))

            eigenvalues = sorted(eigenvalues)
            gaps = np.diff(eigenvalues)
            n_clusters = (
                np.argmax(gaps) + 2
            )  # +2 because diff reduces length by 1 and indices start at 0
            n_clusters = max(2, min(n_clusters, 5))  # Keep between 2 and 5 communities

            # Use spectral embedding + KMeans
            spectral_embedding = nx.spectral_layout(
                G, dim=min(n_clusters, G.number_of_nodes() - 1)
            )
            points = np.array(
                [spectral_embedding[i] for i in range(G.number_of_nodes())]
            )
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            self.community_labels = kmeans.fit_predict(points)
            self.n_communities = n_clusters

        except:
            # If all fails, use simple KMeans on adjacency matrix
            self.n_communities = 2
            kmeans = KMeans(n_clusters=self.n_communities, random_state=42)
            feature_matrix = adj_matrix.copy()
            self.community_labels = kmeans.fit_predict(feature_matrix)

    def _calculate_historical_stats(self, adj_matrices):
        """
        Calculate historical statistics for structural constraints

        Args:
            adj_matrices (list): Historical adjacency matrices
        """
        # Calculate mean and std of node degrees
        all_degrees = []
        for adj in adj_matrices:
            degrees = np.sum(adj, axis=1)
            all_degrees.append(degrees)

        self.mean_degree = np.mean(all_degrees, axis=0)
        self.std_degree = np.std(all_degrees, axis=0) + 1e-6  # Avoid division by zero

        # Calculate historical edge density
        n = adj_matrices[0].shape[0]
        self.historical_density = np.mean(
            [np.sum(adj) / (n * (n - 1)) for adj in adj_matrices]
        )

        # Calculate within-community and between-community densities
        if self.community_labels is not None:
            self.historical_community_densities = self._calculate_community_densities(
                adj_matrices[-1]
            )

    def _calculate_community_densities(self, adj_matrix):
        """
        Calculate within-community and between-community densities

        Args:
            adj_matrix (np.ndarray): Adjacency matrix

        Returns:
            tuple: (within_density, between_density) arrays
        """
        n_communities = len(np.unique(self.community_labels))
        within_density = np.zeros(n_communities)
        between_density = np.zeros((n_communities, n_communities))

        for i in range(n_communities):
            community_i = np.where(self.community_labels == i)[0]
            if len(community_i) > 1:
                within_edges = adj_matrix[np.ix_(community_i, community_i)]
                within_density[i] = np.sum(within_edges) / (
                    len(community_i) * (len(community_i) - 1)
                )

            for j in range(i + 1, n_communities):
                community_j = np.where(self.community_labels == j)[0]
                between_edges = adj_matrix[np.ix_(community_i, community_j)]
                if len(community_i) > 0 and len(community_j) > 0:
                    between_density[i, j] = np.sum(between_edges) / (
                        len(community_i) * len(community_j)
                    )
                    between_density[j, i] = between_density[i, j]

        return within_density, between_density

    def _structural_prediction(self, temporal_pred):
        """
        Implement structural role preservation through joint optimization:
        Â_{t+i}^{(S)} = argmin_{A ∈ 𝒜} ||Â_{t+i}^{(T)} - A||_F + γ R(A)

        Uses iterative adjustments to approximate joint optimization,
        preserving probability information until final output.

        Args:
            temporal_pred (np.ndarray): Temporal prediction of adjacency matrix

        Returns:
            np.ndarray: Structurally adjusted prediction (probabilities)
        """
        # Start with temporal prediction (keeping probability values)
        n = temporal_pred.shape[0]
        structural_pred = temporal_pred.copy()

        # Calculate initial structural score
        initial_score = self._calculate_structural_score(structural_pred)

        # Iterative optimization to approximate joint constraint satisfaction
        for iteration in range(self.optimization_iterations):
            # Make small adjustments for each constraint
            adjustment_size = 1.0 / (
                iteration + 1
            )  # Decrease adjustment size in later iterations

            # 1. Adjust for degree distribution
            if self.omega[0] > 0:
                structural_pred = self._adjust_degree_distribution(
                    structural_pred, adjustment_size
                )

            # 2. Adjust for community structure
            if self.omega[1] > 0 and self.community_labels is not None:
                structural_pred = self._adjust_communities(
                    structural_pred, adjustment_size
                )

            # 3. Adjust for sparsity
            if self.omega[2] > 0:
                structural_pred = self._adjust_sparsity(
                    structural_pred, adjustment_size
                )

            # 4. Ensure symmetric with zero diagonal
            structural_pred = self._ensure_valid_adjacency(
                structural_pred, keep_values=True
            )

            # 5. Check if adjustments improved the structural score
            current_score = self._calculate_structural_score(structural_pred)

            # If score is getting worse, reduce the effect of this iteration
            if current_score < initial_score and iteration > 0:
                # Blend with previous prediction to avoid overfitting constraints
                blend_factor = 0.5
                structural_pred = (
                    blend_factor * structural_pred + (1 - blend_factor) * temporal_pred
                )

        return structural_pred

    def _calculate_structural_score(self, adj_prob):
        """
        Calculate a score that measures how well the prediction satisfies structural constraints

        Args:
            adj_prob (np.ndarray): Adjacency matrix with probability values

        Returns:
            float: Structural score (higher is better)
        """
        # Get the binary version for calculating metrics
        adj_binary = (adj_prob > self.threshold).astype(int)

        # 1. Degree distribution score
        current_degrees = np.sum(adj_binary, axis=1)
        z_scores = (current_degrees - self.mean_degree) / self.std_degree
        degree_score = 1.0 / (1.0 + np.mean(np.abs(z_scores)))

        # 2. Community structure score
        if self.community_labels is not None:
            hist_within, hist_between = self.historical_community_densities
            current_within, current_between = self._calculate_community_densities(
                adj_binary
            )

            within_diff = np.mean(np.abs(hist_within - current_within))
            between_diff = np.mean(np.abs(hist_between - current_between))
            community_score = 1.0 / (1.0 + within_diff + between_diff)
        else:
            community_score = 0.5  # Neutral if no community info

        # 3. Sparsity score
        n = adj_binary.shape[0]
        current_density = np.sum(adj_binary) / (n * (n - 1))
        density_diff = abs(current_density - self.historical_density)
        sparsity_score = 1.0 / (1.0 + density_diff)

        # 4. Compute weighted score
        total_score = (
            self.omega[0] * degree_score
            + self.omega[1] * community_score
            + self.omega[2] * sparsity_score
        )

        return total_score

    def _adjust_degree_distribution(self, adj_prob, adjustment_size=1.0):
        """
        Adjust adjacency matrix to preserve historical degree distribution
        while maintaining probabilistic information

        Args:
            adj_prob (np.ndarray): Adjacency matrix with probability values
            adjustment_size (float): Scale of adjustments to make

        Returns:
            np.ndarray: Adjusted adjacency matrix with probability values
        """
        # Binary version for degree calculation
        adj_binary = (adj_prob > self.threshold).astype(int)
        current_degrees = np.sum(adj_binary, axis=1)

        # Identify nodes with degrees too far from historical means
        z_scores = (current_degrees - self.mean_degree) / self.std_degree

        # Create adjustment matrix (start with no adjustments)
        adj_adjusted = adj_prob.copy()

        # Nodes with too high degree - reduce probability of edges
        high_degree_nodes = np.where(z_scores > self.high_degree_threshold)[0]
        for node in high_degree_nodes:
            # Find existing edges
            connected = np.where(adj_binary[node, :] > 0)[0]
            if len(connected) > 0:
                # Sort by temporal prediction probability (adjust least likely edges first)
                edge_probs = adj_prob[node, connected]
                sorted_indices = np.argsort(edge_probs)

                # Reduce probability for a portion of edges
                num_to_adjust = max(1, int(len(connected) * 0.2 * adjustment_size))
                edges_to_adjust = connected[sorted_indices[:num_to_adjust]]

                # Reduce probabilities (proportional to z-score)
                reduction_factor = min(0.8, 0.4 * adjustment_size * abs(z_scores[node]))
                for edge in edges_to_adjust:
                    adj_adjusted[node, edge] *= 1 - reduction_factor
                    adj_adjusted[edge, node] = adj_adjusted[
                        node, edge
                    ]  # Maintain symmetry

        # Nodes with too low degree - increase probability of edges
        low_degree_nodes = np.where(z_scores < self.low_degree_threshold)[0]
        for node in low_degree_nodes:
            # Find non-connections but with non-zero probability
            non_connected = np.where(
                (adj_binary[node, :] == 0) & (adj_prob[node, :] > 0)
            )[0]
            # Remove self-loop possibility
            non_connected = non_connected[non_connected != node]

            if len(non_connected) > 0:
                # Sort by temporal prediction probability (adjust most likely non-edges first)
                edge_probs = adj_prob[node, non_connected]
                sorted_indices = np.argsort(edge_probs)[::-1]  # Descending order

                # Increase probability for a portion of potential edges
                num_to_adjust = max(
                    1,
                    min(
                        int(len(non_connected) * 0.2 * adjustment_size),
                        int(self.mean_degree[node] - current_degrees[node]),
                    ),
                )
                edges_to_adjust = non_connected[sorted_indices[:num_to_adjust]]

                # Increase probabilities (proportional to z-score)
                increase_factor = min(0.8, 0.4 * adjustment_size * abs(z_scores[node]))
                for edge in edges_to_adjust:
                    adj_adjusted[node, edge] = min(
                        0.95, adj_adjusted[node, edge] * (1 + increase_factor)
                    )
                    adj_adjusted[edge, node] = adj_adjusted[
                        node, edge
                    ]  # Maintain symmetry

        return adj_adjusted

    def _adjust_communities(self, adj_prob, adjustment_size=1.0):
        """
        Adjust adjacency matrix to preserve community structure
        while maintaining probabilistic information

        Args:
            adj_prob (np.ndarray): Adjacency matrix with probability values
            adjustment_size (float): Scale of adjustments to make

        Returns:
            np.ndarray: Adjusted adjacency matrix with probability values
        """
        # Binary version for community calculation
        adj_binary = (adj_prob > self.threshold).astype(int)
        adj_adjusted = adj_prob.copy()

        # Calculate current community densities
        current_within, current_between = self._calculate_community_densities(
            adj_binary
        )
        hist_within, hist_between = self.historical_community_densities

        # Define adjustment factors based on historical vs current densities
        within_factor = np.zeros(self.n_communities)
        between_factor = np.zeros((self.n_communities, self.n_communities))

        for i in range(self.n_communities):
            # Within-community adjustment factor
            if hist_within[i] > 0 and current_within[i] > 0:
                within_factor[i] = hist_within[i] / current_within[i]
                within_factor[i] = min(
                    max(within_factor[i], 0.5), 2.0
                )  # Limit adjustment range
            else:
                within_factor[i] = 1.0

            for j in range(i + 1, self.n_communities):
                # Between-community adjustment factor
                if hist_between[i, j] > 0 and current_between[i, j] > 0:
                    between_factor[i, j] = hist_between[i, j] / current_between[i, j]
                    between_factor[i, j] = min(
                        max(between_factor[i, j], 0.5), 2.0
                    )  # Limit adjustment range
                    between_factor[j, i] = between_factor[i, j]
                else:
                    between_factor[i, j] = 1.0
                    between_factor[j, i] = 1.0

        # Apply adaptive adjustments to probabilities
        n = adj_prob.shape[0]
        for i in range(n):
            comm_i = self.community_labels[i]
            for j in range(i + 1, n):  # Only upper triangle to maintain symmetry
                comm_j = self.community_labels[j]

                if comm_i == comm_j:
                    # Within same community
                    if within_factor[comm_i] > 1.0:
                        # Increase connection probability
                        adj_adjusted[i, j] = min(
                            0.95,
                            adj_adjusted[i, j]
                            * (1 + (within_factor[comm_i] - 1) * adjustment_size),
                        )
                    elif within_factor[comm_i] < 1.0:
                        # Decrease connection probability
                        adj_adjusted[i, j] = adj_adjusted[i, j] * (
                            1 - (1 - within_factor[comm_i]) * adjustment_size
                        )
                else:
                    # Between different communities
                    if between_factor[comm_i, comm_j] > 1.0:
                        # Increase connection probability
                        adj_adjusted[i, j] = min(
                            0.95,
                            adj_adjusted[i, j]
                            * (
                                1
                                + (between_factor[comm_i, comm_j] - 1) * adjustment_size
                            ),
                        )
                    elif between_factor[comm_i, comm_j] < 1.0:
                        # Decrease connection probability
                        adj_adjusted[i, j] = adj_adjusted[i, j] * (
                            1 - (1 - between_factor[comm_i, comm_j]) * adjustment_size
                        )

                # Maintain symmetry
                adj_adjusted[j, i] = adj_adjusted[i, j]

        return adj_adjusted

    def _adjust_sparsity(self, adj_prob, adjustment_size=1.0):
        """
        Adjust adjacency matrix to maintain historical sparsity level
        while maintaining probabilistic information

        Args:
            adj_prob (np.ndarray): Adjacency matrix with probability values
            adjustment_size (float): Scale of adjustments to make

        Returns:
            np.ndarray: Adjusted adjacency matrix with probability values
        """
        # Binary version for density calculation
        adj_binary = (adj_prob > self.threshold).astype(int)
        n = adj_binary.shape[0]
        current_density = np.sum(adj_binary) / (n * (n - 1))

        # If density is close to historical, no adjustment needed
        if abs(current_density - self.historical_density) < self.density_threshold:
            return adj_prob

        # Create adjustment matrix
        adj_adjusted = adj_prob.copy()
        density_diff = self.historical_density - current_density
        adjustment_factor = min(0.5, abs(density_diff) * adjustment_size)

        if density_diff < 0:
            # Too dense - reduce edge probabilities globally
            # Get edges in upper triangle (to avoid double counting)
            triu_indices = np.triu_indices(n, k=1)
            # Sort by probability (adjust highest probability edges first)
            edge_probs = adj_prob[triu_indices]
            sorted_indices = np.argsort(edge_probs)[::-1]  # Descending order

            # Determine how many edges to adjust
            edges_to_adjust = int(abs(density_diff) * n * (n - 1) * adjustment_factor)
            if edges_to_adjust > 0:
                # Get coordinates of edges to adjust
                adjust_indices = [
                    (
                        triu_indices[0][sorted_indices[i]],
                        triu_indices[1][sorted_indices[i]],
                    )
                    for i in range(min(edges_to_adjust, len(sorted_indices)))
                ]

                # Reduce probabilities
                for i, j in adjust_indices:
                    adj_adjusted[i, j] *= 1 - adjustment_factor
                    adj_adjusted[j, i] = adj_adjusted[i, j]  # Maintain symmetry

        else:
            # Too sparse - increase edge probabilities globally
            # Get non-edges in upper triangle with some probability
            non_edge_indices = np.where(((adj_binary == 0) & (adj_prob > 0)))
            non_edge_indices = list(zip(non_edge_indices[0], non_edge_indices[1]))
            non_edge_indices = [
                pair for pair in non_edge_indices if pair[0] < pair[1]
            ]  # Upper triangle only

            # Sort by probability (adjust highest probability non-edges first)
            if non_edge_indices:
                non_edge_probs = [adj_prob[i, j] for i, j in non_edge_indices]
                sorted_indices = np.argsort(non_edge_probs)[::-1]  # Descending order

                # Determine how many edges to adjust
                edges_to_adjust = int(density_diff * n * (n - 1) * adjustment_factor)
                if edges_to_adjust > 0:
                    # Get coordinates of non-edges to adjust
                    adjust_indices = [
                        non_edge_indices[sorted_indices[i]]
                        for i in range(min(edges_to_adjust, len(sorted_indices)))
                    ]

                    # Increase probabilities
                    for i, j in adjust_indices:
                        adj_adjusted[i, j] = min(
                            0.95, adj_adjusted[i, j] * (1 + adjustment_factor)
                        )
                        adj_adjusted[j, i] = adj_adjusted[i, j]  # Maintain symmetry

        return adj_adjusted

    def _adaptive_integration(self, temporal_pred, structural_pred):
        """
        Implement adaptive integration:
        Â_{t+i} = β_t Â_{t+i}^{(T)} + (1-β_t)Â_{t+i}^{(S)}

        Args:
            temporal_pred (np.ndarray): Temporal prediction
            structural_pred (np.ndarray): Structural prediction

        Returns:
            np.ndarray: Combined prediction
        """
        # Combine predictions using beta parameter
        combined_pred = self.beta * temporal_pred + (1 - self.beta) * structural_pred

        return combined_pred

    def update_beta(self, actual_adj, predicted_adj):
        """
        Update beta parameter based on prediction error
        β_t = 1/(1 + e^{δ_t})

        Args:
            actual_adj (np.ndarray): Actual adjacency matrix
            predicted_adj (np.ndarray): Predicted adjacency matrix
        """
        if not self.adaptive:
            return

        # Calculate prediction error (MSE)
        error = np.mean((actual_adj - predicted_adj) ** 2)
        self.prediction_error_history.append(error)

        # Use recent errors with exponential weighting
        if len(self.prediction_error_history) > 10:
            self.prediction_error_history = self.prediction_error_history[-10:]

        # Calculate weighted error
        weights = np.array([0.8**i for i in range(len(self.prediction_error_history))])
        weights = weights / np.sum(weights)
        weighted_error = np.sum(weights * np.array(self.prediction_error_history[::-1]))

        # Update beta: β_t = 1/(1 + e^{δ_t})
        self.beta = 1 / (
            1 + np.exp(5 * weighted_error)
        )  # Scale factor 5 for sensitivity

    def _ensure_valid_adjacency(self, adj, keep_values=False):
        """
        Ensure adjacency matrix is symmetric with zero diagonal

        Args:
            adj (np.ndarray): Adjacency matrix
            keep_values (bool): Whether to maintain probability values

        Returns:
            np.ndarray: Valid adjacency matrix
        """
        # Ensure symmetric
        adj = (adj + adj.T) / 2

        # Ensure zero diagonal
        np.fill_diagonal(adj, 0)

        return adj

    def _ensure_connected(self, adj):
        """
        Ensure the graph is connected by finding and connecting components
        using high-probability edges from the original prediction

        Args:
            adj (np.ndarray): Adjacency matrix

        Returns:
            np.ndarray: Connected adjacency matrix
        """
        # Convert to NetworkX graph
        G = nx.from_numpy_array(adj)

        # If already connected, return original
        if nx.is_connected(G):
            return adj

        # Find connected components
        components = list(nx.connected_components(G))

        # If multiple components, connect them
        if len(components) > 1:
            # Sort components by size (descending)
            components.sort(key=len, reverse=True)

            # For each smaller component, find the best edge to connect to largest component
            for i in range(1, len(components)):
                best_edge = None
                best_score = -1

                # Check all possible edges between this component and largest
                for node1 in components[0]:
                    for node2 in components[i]:
                        # Use original prediction score as edge importance
                        score = adj[node1, node2]
                        if score > best_score:
                            best_score = score
                            best_edge = (node1, node2)

                # If we found a reasonable edge, use it
                if best_edge:
                    node1, node2 = best_edge
                    adj[node1, node2] = 1
                    adj[node2, node1] = 1
                else:
                    # Otherwise connect random nodes
                    node1 = np.random.choice(list(components[0]))
                    node2 = np.random.choice(list(components[i]))
                    adj[node1, node2] = 1
                    adj[node2, node1] = 1

        return adj

    def get_state(self):
        return {
            "beta": self.beta,
            "prediction_error_history": self.prediction_error_history,
            "n_history": self.n_history,
        }

    def reset(self):
        self.beta = 0.5
        self.prediction_error_history = []


def predict_graphs(
    history,
    horizon=5,
    alpha=0.8,
    gamma=0.5,
    omega=None,
    beta_init=0.5,
    enforce_connectivity=True,
    adaptive=True,
    optimization_iterations=3,
    threshold=0.5,
):
    """
    Wrapper function to predict future graph states

    Args:
        history (list): List of dictionaries containing past adjacency matrices
                       [{"adjacency": adj_matrix}, ...]
        horizon (int): Number of future time steps to predict
        alpha (float): Decay parameter for temporal weights
        gamma (float): Regularization parameter for structural constraints
        omega (list): Weights for regularization terms [degree, community, sparsity]
        beta_init (float): Initial mixing parameter
        enforce_connectivity (bool): Whether to enforce connectivity
        adaptive (bool): Whether to adaptively update beta
        optimization_iterations (int): Iterations for joint constraint optimization
        threshold (float): Threshold for final binarization

    Returns:
        list: List of predicted adjacency matrices
    """
    predictor = GraphPredictor(
        alpha=alpha,
        gamma=gamma,
        omega=omega,
        beta_init=beta_init,
        enforce_connectivity=enforce_connectivity,
        adaptive=adaptive,
        optimization_iterations=optimization_iterations,
        threshold=threshold,
    )

    return predictor.predict(history, horizon=horizon)
