"""Enhanced weighted average network predictor with structural preservation."""

import networkx as nx
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from scipy import sparse
from scipy.sparse.linalg import eigsh
from sklearn.cluster import SpectralClustering
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


class EnhancedWeightedPredictor:
    """Enhanced predictor that preserves spectral and structural properties.

    Key improvements over base WeightedPredictor:
    1. Spectral regularization to preserve eigenstructure
    2. Community structure preservation
    3. Temporal pattern detection
    4. Feature-based weight adaptation
    """

    def __init__(
        self,
        n_history: int = 3,
        weights: Optional[np.ndarray] = None,
        adaptive: bool = True,
        enforce_connectivity: bool = True,
        binary: bool = True,
        spectral_reg: float = 0.3,
        community_reg: float = 0.3,
        n_communities: int = 2,
        temporal_window: int = 5,
        min_edges_per_component: int = 3,  # Minimum edges to maintain in each component
        degree_reg: float = 0.2,  # Weight for degree distribution regularization
        change_threshold: float = 0.3,  # Threshold for detecting significant changes
    ):
        """Initialize predictor with enhanced parameters.

        Parameters
        ----------
        n_history : int
            Number of recent states to use
        weights : np.ndarray, optional
            Initial weights, will be normalized
        adaptive : bool
            Whether to adapt weights based on prediction accuracy
        enforce_connectivity : bool
            Whether to ensure connected components
        binary : bool
            Whether to output binary adjacency matrices
        spectral_reg : float
            Weight for spectral regularization (0-1)
        community_reg : float
            Weight for community preservation (0-1)
        n_communities : int
            Number of communities to preserve
        temporal_window : int
            Window size for temporal pattern detection
        min_edges_per_component : int
            Minimum edges to maintain in each component
        degree_reg : float
            Weight for degree distribution regularization
        change_threshold : float
            Threshold for detecting significant changes
        """
        self.n_history = n_history

        if weights is None:
            weights = np.array([0.6, 0.3, 0.1])

        # Normalize weights
        weights = np.array(weights, dtype=float)
        self.weights = weights / weights.sum()

        self.adaptive = adaptive
        self.enforce_connectivity = enforce_connectivity
        self.binary = binary
        self.spectral_reg = spectral_reg
        self.community_reg = community_reg
        self.n_communities = n_communities
        self.temporal_window = temporal_window
        self.min_edges_per_component = min_edges_per_component
        self.degree_reg = degree_reg
        self.change_threshold = change_threshold

        # Initialize temporal pattern storage
        self.pattern_history = []
        self.feature_history = []
        self.change_points = []
        self.last_degree_dist = None

    def _compute_degree_distribution(self, adj: np.ndarray) -> np.ndarray:
        """Compute normalized degree distribution."""
        degrees = adj.sum(axis=1)
        unique_degrees = np.unique(degrees)
        dist = np.zeros(int(max(degrees)) + 1)
        for d in degrees:
            dist[int(d)] += 1
        return dist / len(degrees)

    def _compute_spectral_features(
        self, adj: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute leading eigenvalues and eigenvectors with enhanced stability."""
        # Add small diagonal term for numerical stability
        adj_reg = adj + np.eye(adj.shape[0]) * 1e-6
        adj_sparse = sparse.csr_matrix(adj_reg)
        k = min(6, adj.shape[0] - 1)

        try:
            eigenvals, eigenvecs = eigsh(
                adj_sparse, k=k, which="LM", tol=1e-5, maxiter=1000
            )
            idx = np.argsort(np.abs(eigenvals))[::-1]
            return eigenvals[idx], eigenvecs[:, idx]
        except:
            eigenvals, eigenvecs = np.linalg.eigh(adj_reg)
            idx = np.argsort(np.abs(eigenvals))[::-1]
            return eigenvals[idx][:k], eigenvecs[:, idx][:, :k]

    def _detect_communities(self, adj: np.ndarray) -> Tuple[np.ndarray, float]:
        """Detect communities and compute modularity."""
        adj_nn = np.abs(adj)
        np.fill_diagonal(adj_nn, 0)

        try:
            clustering = SpectralClustering(
                n_clusters=self.n_communities,
                affinity="precomputed",
                random_state=42,
                n_init=10,  # Multiple initializations for better stability
            )
            labels = clustering.fit_predict(adj_nn)

            # Compute modularity
            G = nx.from_numpy_array(adj_nn)
            modularity = nx.community.modularity(
                G, [set(np.where(labels == i)[0]) for i in range(self.n_communities)]
            )

            return labels, modularity
        except:
            n = adj.shape[0]
            return np.array([i % self.n_communities for i in range(n)]), 0.0

    def _compute_network_features(self, adj: np.ndarray) -> Dict[str, float]:
        """Compute key network features for adaptation."""
        G = nx.from_numpy_array(adj)

        features = {
            "density": nx.density(G),
            "avg_clustering": nx.average_clustering(G),
            "avg_degree": float(np.mean(list(dict(G.degree()).values()))),
        }

        try:
            features["avg_path_length"] = nx.average_shortest_path_length(G)
        except:
            features["avg_path_length"] = 0.0

        return features

    def _detect_temporal_patterns(
        self, history: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Enhanced temporal pattern detection with change point detection."""
        if len(history) < self.temporal_window:
            return {
                "trend": 0.0,
                "periodicity": 0.0,
                "volatility": 0.0,
                "change_detected": False,
            }

        recent = [h["adjacency"] for h in history[-self.temporal_window :]]
        densities = [np.mean(adj) for adj in recent]

        # Compute trend with robust fitting
        x = np.arange(len(densities))
        trend = np.polyfit(x, densities, 1)[0]

        # Enhanced change detection
        if len(densities) > 2:
            diffs = np.abs(np.diff(densities))
            mean_diff = np.mean(diffs[:-1])
            std_diff = np.std(diffs[:-1]) + 1e-6
            latest_diff = diffs[-1]
            change_detected = (
                latest_diff - mean_diff
            ) / std_diff > self.change_threshold
        else:
            change_detected = False

        # Improved periodicity detection
        if len(densities) > 4:
            ac = np.correlate(densities, densities, mode="full")
            ac = ac[len(ac) // 2 :]
            peaks = [
                i
                for i in range(1, len(ac) - 1)
                if ac[i] > ac[i - 1] and ac[i] > ac[i + 1]
            ]
            periodicity = max([ac[p] / ac[0] for p in peaks]) if peaks else 0
        else:
            periodicity = 0

        volatility = np.std(densities)

        return {
            "trend": trend,
            "periodicity": periodicity,
            "volatility": volatility,
            "change_detected": change_detected,
        }

    def _spectral_regularization(
        self, pred: np.ndarray, target_vals: np.ndarray, target_vecs: np.ndarray
    ) -> np.ndarray:
        """Apply spectral regularization to preserve eigenstructure."""
        # Get current spectral properties
        curr_vals, curr_vecs = self._compute_spectral_features(pred)

        # Compute regularization matrix
        reg_mat = np.zeros_like(pred)
        for i in range(min(len(target_vals), len(curr_vals))):
            v_target = target_vecs[:, i : i + 1]
            v_curr = curr_vecs[:, i : i + 1]

            # Compute regularization term
            reg = (target_vals[i] - curr_vals[i]) * (v_target @ v_target.T)
            reg_mat += reg

        # Apply regularization
        alpha = self.spectral_reg
        pred_reg = (1 - alpha) * pred + alpha * reg_mat

        return pred_reg

    def _preserve_communities(
        self, pred: np.ndarray, comm_labels: np.ndarray, modularity: float
    ) -> np.ndarray:
        """Enhanced community structure preservation."""
        n = pred.shape[0]
        modifier = np.zeros((n, n))

        # Adaptive modification based on modularity
        intra_weight = 0.15 * (1 + modularity)  # Stronger for high modularity
        inter_weight = -0.1 * (1 + modularity)  # Weaker for high modularity

        for i in range(n):
            for j in range(i + 1, n):
                if comm_labels[i] == comm_labels[j]:
                    modifier[i, j] = modifier[j, i] = intra_weight
                else:
                    modifier[i, j] = modifier[j, i] = inter_weight

        # Apply community preservation with adaptive strength
        alpha = self.community_reg * (1 + modularity)
        pred_comm = pred + alpha * modifier

        return np.clip(pred_comm, 0, 1)

    def _enforce_connectivity_enhanced(
        self, pred: np.ndarray, orig_pred: np.ndarray
    ) -> np.ndarray:
        """Enhanced connectivity enforcement with minimum spanning tree."""
        G = nx.from_numpy_array(pred)
        components = list(nx.connected_components(G))

        if len(components) > 1:
            # Sort components by size (descending)
            components.sort(key=len, reverse=True)
            main_comp = components[0]

            # Create MST-based connections
            for comp in components[1:]:
                # Find best edges between components
                best_edges = []
                for i in main_comp:
                    for j in comp:
                        best_edges.append((orig_pred[i, j], i, j))

                # Take top k edges
                best_edges.sort(reverse=True)
                for _, i, j in best_edges[: self.min_edges_per_component]:
                    pred[i, j] = pred[j, i] = 1.0

        return pred

    def _preserve_degree_distribution(
        self, pred: np.ndarray, target_dist: np.ndarray
    ) -> np.ndarray:
        """Preserve degree distribution using smoothed gradient descent."""
        curr_dist = self._compute_degree_distribution(pred)

        # Add smoothing to avoid division by zero
        eps = 1e-8

        # Extend distributions to same length with smoothing
        max_len = max(len(target_dist), len(curr_dist))
        target_ext = np.ones(max_len) * eps
        curr_ext = np.ones(max_len) * eps
        target_ext[: len(target_dist)] += target_dist
        curr_ext[: len(curr_dist)] += curr_dist

        # Normalize to ensure valid distributions
        target_ext = target_ext / target_ext.sum()
        curr_ext = curr_ext / curr_ext.sum()

        # Compute smoothed gradient using ratio of distributions
        grad = np.zeros_like(pred)
        degrees = pred.sum(axis=1)

        for i in range(pred.shape[0]):
            d = int(degrees[i])
            if d < max_len - 1:
                # Compute smoothed gradient effect
                ratio_curr = max(eps, curr_ext[d])
                ratio_next = max(eps, curr_ext[d + 1])
                target_ratio = max(eps, target_ext[d + 1]) / max(eps, target_ext[d])

                # Bound the gradient effect
                grad_effect = np.clip(
                    np.log(target_ratio * ratio_curr / ratio_next), -1.0, 1.0
                )

                # Apply gradient symmetrically
                grad[i, :] = grad_effect
                grad[:, i] = grad_effect

        # Scale gradient by current prediction values to maintain sparsity
        grad = grad * pred

        # Apply gradient with adaptive step size
        step_size = self.degree_reg / (1.0 + np.std(degrees))
        pred_new = pred + step_size * grad

        return np.clip(pred_new, 0, 1)

    def _feature_based_adaptation(
        self, old_adjs: List[np.ndarray], new_adj: np.ndarray
    ) -> np.ndarray:
        """Adapt weights based on feature preservation."""
        feature_diffs = []

        for old_adj in old_adjs:
            old_feats = self._compute_network_features(old_adj)
            new_feats = self._compute_network_features(new_adj)

            # Compute normalized feature differences
            diffs = []
            for k in old_feats:
                if old_feats[k] != 0:
                    diff = abs(old_feats[k] - new_feats[k]) / abs(old_feats[k])
                else:
                    diff = abs(new_feats[k])
                diffs.append(diff)

            feature_diffs.append(np.mean(diffs))

        # Convert differences to weights
        raw_weights = 1.0 / (1.0 + np.array(feature_diffs))
        new_weights = raw_weights / raw_weights.sum()

        return new_weights

    def predict(
        self, history: List[Dict[str, Any]], horizon: int = 1
    ) -> List[np.ndarray]:
        """Enhanced prediction with improved structural preservation."""
        if len(history) < self.n_history:
            raise ValueError(
                f"Not enough history. Need {self.n_history}, got {len(history)}."
            )

        predictions = []
        current_history = list(history)

        # Detect temporal patterns with change points
        patterns = self._detect_temporal_patterns(current_history)

        # Reset weights if change detected
        if patterns["change_detected"]:
            self.weights = np.array([0.6, 0.3, 0.1])
            self.weights = self.weights / self.weights.sum()
            self.change_points.append(len(current_history))

        for _ in range(horizon):
            # Get recent states
            last_states = current_history[-self.n_history :]
            last_adjs = [st["adjacency"] for st in last_states]

            # Base prediction
            pred = np.zeros_like(last_adjs[0], dtype=float)
            for w, adj in zip(self.weights, last_adjs):
                pred += w * adj

            # Store original prediction for connectivity enforcement
            orig_pred = pred.copy()

            # Spectral regularization
            target_vals, target_vecs = self._compute_spectral_features(last_adjs[-1])
            pred = self._spectral_regularization(pred, target_vals, target_vecs)

            # Enhanced community preservation
            comm_labels, modularity = self._detect_communities(last_adjs[-1])
            pred = self._preserve_communities(pred, comm_labels, modularity)

            # Preserve degree distribution
            target_dist = self._compute_degree_distribution(last_adjs[-1])
            pred = self._preserve_degree_distribution(pred, target_dist)

            # Temporal pattern adjustment
            if patterns["periodicity"] > 0.7:
                period_idx = int(len(current_history) % self.temporal_window)
                if period_idx < len(last_adjs):
                    pred = 0.7 * pred + 0.3 * last_adjs[period_idx]

            # Apply trend with volatility-based dampening
            trend_factor = 1.0 / (1.0 + patterns["volatility"])
            pred = pred + patterns["trend"] * trend_factor * np.ones_like(pred)

            # Convert to binary if needed
            if self.binary:
                target_density = np.mean(last_adjs[-1])
                n = pred.shape[0]
                target_edges = int(np.floor(target_density * n * (n - 1) / 2))

                # Sort edges by probability
                triu = np.triu_indices(n, k=1)
                probs = pred[triu]
                edges = list(zip(probs, triu[0], triu[1]))
                edges.sort(key=lambda x: x[0], reverse=True)

                # Create binary matrix
                pred_binary = np.zeros_like(pred)
                for _, i, j in edges[:target_edges]:
                    pred_binary[i, j] = pred_binary[j, i] = 1.0

                pred = pred_binary

                # Enhanced connectivity enforcement
                if self.enforce_connectivity:
                    pred = self._enforce_connectivity_enhanced(pred, orig_pred)

            predictions.append(pred)

            # Update history
            current_history.append(
                {
                    "adjacency": pred,
                    "graph": nx.from_numpy_array(pred) if self.binary else None,
                }
            )

            # Feature-based weight adaptation
            if self.adaptive and not patterns["change_detected"]:
                self.weights = self._feature_based_adaptation(last_adjs, pred)

        return predictions
