# src/predictor/weighted.py

"""Enhanced weighted average network predictor with distribution awareness."""

from typing import List, Dict, Any, Optional, Tuple

import warnings
import networkx as nx
import numpy as np

from scipy import sparse, stats
from scipy.sparse.linalg import eigsh
from sklearn.cluster import SpectralClustering

from .base import BasePredictor

warnings.filterwarnings("ignore", category=UserWarning)


class EnhancedWeightedPredictor(BasePredictor):
    """Enhanced predictor that preserves network distributions and structural properties."""

    def __init__(
        self,
        n_history: int = 5,  # Increased history for better distribution learning
        weights: Optional[np.ndarray] = None,
        adaptive: bool = True,
        enforce_connectivity: bool = True,
        binary: bool = True,
        spectral_reg: float = 0.4,
        community_reg: float = 0.4,
        n_communities: int = 2,
        temporal_window: int = 10,  # Increased for better pattern detection
        min_edges_per_component: int = 4,
        degree_reg: float = 0.3,
        change_threshold: float = 0.25,
        smoothing_window: int = 3,
        min_weight: float = 0.1,
        distribution_memory: int = 20,  # Window for distribution tracking
        phase_length: int = 40,  # Expected minimum phase length
        distribution_reg: float = 0.3,  # Weight for distribution regularization
    ):
        """Initialize predictor with distribution awareness."""
        self.n_history = n_history
        self._history_size = n_history  # Set history size property

        if weights is None:
            weights = np.array([0.5, 0.3, 0.1, 0.05, 0.05])  # More historical context

        weights = np.maximum(np.array(weights, dtype=float), min_weight)
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
        self.smoothing_window = smoothing_window
        self.min_weight = min_weight
        self.distribution_memory = distribution_memory
        self.phase_length = phase_length
        self.distribution_reg = distribution_reg

        # Enhanced state tracking
        self.pattern_history = []
        self.feature_history = []
        self.change_points = []
        self.distribution_history = {
            "degree": [],
            "clustering": [],
            "path_length": [],
            "betweenness": [],
        }
        self.phase_start = 0
        self.current_phase = "unknown"
        self.ema_features = {}

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
        """Enhanced temporal pattern detection with EMA smoothing."""
        if len(history) < self.temporal_window:
            return {
                "trend": 0.0,
                "periodicity": 0.0,
                "volatility": 0.0,
                "change_detected": False,
            }

        recent = [h["adjacency"] for h in history[-self.temporal_window :]]

        # Compute EMAs for multiple features
        alpha = 0.3  # Smoothing factor
        current_density = np.mean(recent[-1])
        current_clustering = nx.average_clustering(nx.from_numpy_array(recent[-1]))
        current_degree = float(
            np.mean([d for _, d in nx.from_numpy_array(recent[-1]).degree()])
        )

        # Initialize or update EMAs
        if self.ema_features is None:
            self.ema_features = {
                "density": current_density,
                "clustering": current_clustering,
                "degree": current_degree,
            }
        else:
            for key in self.ema_features:
                self.ema_features[key] = (
                    alpha * recent[-1][key] + (1 - alpha) * self.ema_features[key]
                )

        # Multi-feature change detection
        density_change = abs(current_density - self.ema_features["density"]) / max(
            self.ema_features["density"], 0.1
        )
        clustering_change = abs(
            current_clustering - self.ema_features["clustering"]
        ) / max(self.ema_features["clustering"], 0.1)
        degree_change = abs(current_degree - self.ema_features["degree"]) / max(
            self.ema_features["degree"], 1.0
        )

        change_detected = (
            density_change > self.change_threshold
            or clustering_change > self.change_threshold
            or degree_change > self.change_threshold
        )

        # Compute other patterns
        densities = [np.mean(adj) for adj in recent]
        x = np.arange(len(densities))
        trend = np.polyfit(x, densities, 1)[0]

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
        self,
        pred: np.ndarray,
        comm_labels: np.ndarray,
        modularity: float,
        strength_factor: float = 1.0,
    ) -> np.ndarray:
        """Enhanced community structure preservation with adaptive strength."""
        n = pred.shape[0]
        modifier = np.zeros((n, n))

        # Stronger adaptive modification
        intra_weight = 0.15 * (1 + modularity) * strength_factor
        inter_weight = -0.1 * (1 + modularity) * strength_factor

        for i in range(n):
            for j in range(i + 1, n):
                if comm_labels[i] == comm_labels[j]:
                    modifier[i, j] = modifier[j, i] = intra_weight
                else:
                    modifier[i, j] = modifier[j, i] = inter_weight

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

    def _detect_model_type(self, adj: np.ndarray) -> str:
        """Detect the likely generative model type based on network properties."""
        G = nx.from_numpy_array(adj)
        n = G.number_of_nodes()
        density = nx.density(G)
        clustering = nx.average_clustering(G)
        degrees = [d for _, d in G.degree()]
        degree_std = np.std(degrees)

        # Check for SBM characteristics
        if self.n_communities > 1:
            _, modularity = self._detect_communities(adj)
            if modularity > 0.3:
                return "sbm"

        # Check for BA characteristics (power law degree distribution)
        if degree_std > 2.0 and max(degrees) > 3 * np.mean(degrees):
            return "ba"

        # Check for WS characteristics (high clustering, low path length)
        if clustering > 0.2 and density < 0.3:
            return "ws"

        # Default to ER if no clear pattern
        return "er"

    def _preserve_model_structure(
        self, pred: np.ndarray, model_type: str, last_adj: np.ndarray
    ) -> np.ndarray:
        """Preserve model-specific structural properties."""
        if model_type == "sbm":
            # Strengthen community structure
            comm_labels, modularity = self._detect_communities(last_adj)
            pred = self._preserve_communities(
                pred, comm_labels, modularity, strength_factor=1.5
            )

        elif model_type == "ba":
            # Preserve hub structure and power law degree distribution
            hub_nodes = np.argsort(np.sum(last_adj, axis=0))[
                -int(0.1 * len(last_adj)) :
            ]
            for hub in hub_nodes:
                hub_weights = last_adj[hub] * 1.5  # Strengthen hub connections
                pred[hub] = (
                    pred[hub] * (1 - self.spectral_reg)
                    + hub_weights * self.spectral_reg
                )
                pred[:, hub] = pred[hub]  # Maintain symmetry

        elif model_type == "ws":
            # Preserve local clustering and rewiring structure
            G_last = nx.from_numpy_array(last_adj)
            clustering_coef = nx.clustering(G_last)
            for node, clust in clustering_coef.items():
                if clust > 0.5:  # Preserve high clustering nodes
                    neighbors = list(G_last.neighbors(node))
                    for n1 in neighbors:
                        for n2 in neighbors:
                            if n1 < n2:
                                pred[n1, n2] = pred[n2, n1] = max(
                                    pred[n1, n2], 0.8 * last_adj[n1, n2]
                                )

        return np.clip(pred, 0, 1)

    def _update_distribution_history(self, adj: np.ndarray):
        """Track distribution statistics over time."""
        G = nx.from_numpy_array(adj)

        # Get distributions
        degrees = [d for _, d in G.degree()]
        clustering = list(nx.clustering(G).values())

        try:
            # Get path length distribution from a sample of nodes
            n = len(G)
            sample_size = min(10, n)  # Limit computational cost
            sampled_nodes = np.random.choice(n, sample_size, replace=False)
            path_lengths = []
            for u in sampled_nodes:
                lengths = nx.single_source_shortest_path_length(G, u)
                path_lengths.extend(lengths.values())
        except:
            path_lengths = []

        try:
            # Sample betweenness for efficiency
            between = list(nx.betweenness_centrality(G, k=min(10, n - 1)).values())
        except:
            between = []

        # Update histories with limited memory
        self.distribution_history["degree"].append(degrees)
        self.distribution_history["clustering"].append(clustering)
        self.distribution_history["path_length"].append(path_lengths)
        self.distribution_history["betweenness"].append(between)

        # Maintain fixed window
        for key in self.distribution_history:
            if len(self.distribution_history[key]) > self.distribution_memory:
                self.distribution_history[key].pop(0)

    def _detect_distribution_type(self) -> Dict[str, str]:
        """Detect likely distribution types for different metrics."""
        if not self.distribution_history["degree"]:
            return {}

        # Analyze recent distributions
        recent_degrees = np.concatenate(self.distribution_history["degree"][-5:])

        types = {}

        # Check degree distribution
        if len(recent_degrees) > 10:
            # Test for power law (BA-like)
            degree_counts = np.bincount(recent_degrees.astype(int))[1:]
            log_degrees = np.log1p(np.arange(1, len(degree_counts) + 1))
            log_counts = np.log1p(degree_counts)
            valid_points = (degree_counts > 0) & np.isfinite(log_counts)

            if np.sum(valid_points) > 3:
                slope, _, r_value, _, _ = stats.linregress(
                    log_degrees[valid_points], log_counts[valid_points]
                )
                if r_value**2 > 0.8 and slope < -1:
                    types["degree"] = "power_law"
                elif np.std(recent_degrees) < 0.5 * np.mean(recent_degrees):
                    types["degree"] = "poisson"
                else:
                    types["degree"] = "unknown"

        return types

    def _estimate_phase_parameters(self) -> Dict[str, float]:
        """Estimate current phase parameters."""
        if not self.distribution_history["degree"]:
            return {}

        recent_degrees = self.distribution_history["degree"][-1]
        recent_clustering = self.distribution_history["clustering"][-1]

        params = {
            "avg_degree": float(np.mean(recent_degrees)),
            "degree_std": float(np.std(recent_degrees)),
            "clustering_coef": float(np.mean(recent_clustering)),
            "density": float(np.mean(recent_degrees)) / (len(recent_degrees) - 1),
        }

        # Estimate additional model-specific parameters
        dist_types = self._detect_distribution_type()

        if dist_types.get("degree") == "power_law":
            # Estimate BA-like parameter
            params["m"] = max(1, int(round(params["avg_degree"] / 2)))

        return params

    def _detect_phase_transition(self, history: List[Dict[str, Any]]) -> bool:
        """Detect phase transitions using multiple indicators."""
        if len(history) < self.temporal_window:
            return False

        # Track multiple features
        features = []
        for state in history[-self.temporal_window :]:
            adj = state["adjacency"]
            G = nx.from_numpy_array(adj)

            feat = {
                "density": nx.density(G),
                "clustering": nx.average_clustering(G),
                "degree_std": float(np.std([d for _, d in G.degree()])),
            }
            features.append(feat)

        # Initialize or update EMAs
        if not self.ema_features:
            self.ema_features = features[-1].copy()
        else:
            alpha = 0.3
            for key in features[-1]:
                current = features[-1][key]
                self.ema_features[key] = (
                    alpha * current + (1 - alpha) * self.ema_features[key]
                )

        # Check for significant changes in multiple features
        changes = []
        for key in features[-1]:
            current = features[-1][key]
            ema = self.ema_features[key]
            if ema > 0:
                change = abs(current - ema) / ema
                changes.append(change > self.change_threshold)

        # Also consider phase length
        time_since_change = len(history) - self.phase_start
        phase_maturity = time_since_change >= self.phase_length

        return any(changes) and phase_maturity

    def _preserve_distribution_properties(
        self, pred: np.ndarray, last_adj: np.ndarray
    ) -> np.ndarray:
        """Preserve detected distribution properties."""
        dist_types = self._detect_distribution_type()
        params = self._estimate_phase_parameters()

        if dist_types.get("degree") == "power_law":
            # Strengthen hub structure for power-law networks
            degrees = np.sum(last_adj, axis=0)
            hub_threshold = np.percentile(degrees, 80)
            hub_mask = degrees > hub_threshold

            # Enhance hub connections
            for i in np.where(hub_mask)[0]:
                neighbors = np.where(last_adj[i] > 0)[0]
                if len(neighbors) > 0:
                    # Strengthen existing hub connections
                    pred[i, neighbors] = pred[neighbors, i] = np.maximum(
                        pred[i, neighbors], 0.8 * last_adj[i, neighbors]
                    )

                    # Promote preferential attachment
                    neighbor_degrees = degrees[neighbors]
                    attach_probs = neighbor_degrees / neighbor_degrees.sum()
                    for j in range(pred.shape[0]):
                        if j not in neighbors:
                            influence = np.sum(attach_probs * (pred[j, neighbors] > 0))
                            pred[i, j] = pred[j, i] = max(pred[i, j], 0.3 * influence)

        elif params.get("clustering_coef", 0) > 0.2:
            # Preserve high clustering structure
            G_last = nx.from_numpy_array(last_adj)
            clustering = nx.clustering(G_last)

            for node, clust in clustering.items():
                if clust > 0.3:
                    neighbors = list(G_last.neighbors(node))
                    for i, n1 in enumerate(neighbors):
                        for n2 in neighbors[i + 1 :]:
                            pred[n1, n2] = pred[n2, n1] = max(
                                pred[n1, n2],
                                0.7 * min(last_adj[n1, node], last_adj[n2, node]),
                            )

        # Regularize towards estimated density
        if "density" in params:
            target_density = params["density"]
            current_density = np.mean(pred)

            if abs(current_density - target_density) > 0.1:
                adjustment = (target_density - current_density) * self.distribution_reg
                pred += adjustment

        return np.clip(pred, 0, 1)

    def predict(
        self, history: List[Dict[str, Any]], horizon: int = 1
    ) -> List[np.ndarray]:
        """
        Predict future network states with distribution awareness.

        Parameters
        ----------
        history : List[Dict[str, Any]]
            List of historical network states, each containing an 'adjacency' key
        horizon : int, optional
            Number of steps to predict ahead, by default 1

        Returns
        -------
        List[np.ndarray]
            List of predicted adjacency matrices
        """
        if len(history) < self.n_history:
            raise ValueError(
                f"Not enough history. Need {self.n_history}, got {len(history)}."
            )

        # Extract adjacency matrices and create state dictionaries
        current_history = []
        for state in history:
            adj = state["adjacency"]
            current_history.append(
                {
                    "adjacency": adj,
                    "graph": nx.from_numpy_array(adj) if self.binary else None,
                }
            )

        # Update distribution tracking
        self._update_distribution_history(current_history[-1]["adjacency"])

        # Check for phase transition
        if self._detect_phase_transition(current_history):
            self.phase_start = len(current_history)
            # Reset weights but maintain minimum constraint
            self.weights = np.array([0.5, 0.3, 0.1, 0.05, 0.05])
            self.weights = np.maximum(self.weights, self.min_weight)
            self.weights = self.weights / self.weights.sum()
            self.change_points.append(len(current_history))

        predictions = []
        for _ in range(horizon):
            # Get recent states
            last_states = current_history[-self.n_history :]
            last_adjs = [st["adjacency"] for st in last_states]

            # Enhanced temporal prediction
            if len(predictions) >= self.smoothing_window:
                smoothed_history = last_adjs + predictions[-self.smoothing_window :]
                weights = np.concatenate(
                    [self.weights, np.ones(self.smoothing_window) * 0.1]
                )
                weights = weights / weights.sum()
                pred = np.zeros_like(last_adjs[0], dtype=float)
                for w, adj in zip(weights, smoothed_history):
                    pred += w * adj
            else:
                pred = np.zeros_like(last_adjs[0], dtype=float)
                for w, adj in zip(self.weights, last_adjs):
                    pred += w * adj

            # Store original prediction
            orig_pred = pred.copy()

            # Distribution-aware preservation
            pred = self._preserve_distribution_properties(pred, last_adjs[-1])

            # Standard structural preservation
            target_vals, target_vecs = self._compute_spectral_features(last_adjs[-1])
            pred = self._spectral_regularization(pred, target_vals, target_vecs)

            # Community preservation if high modularity detected
            comm_labels, modularity = self._detect_communities(last_adjs[-1])
            if modularity > 0.3:
                pred = self._preserve_communities(pred, comm_labels, modularity)

            # Ensure valid probabilities
            pred = np.clip(pred, 0, 1)

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

                # Enforce connectivity if needed
                if self.enforce_connectivity:
                    pred = self._enforce_connectivity_enhanced(pred, orig_pred)

            predictions.append(pred)

            # Update history and distribution tracking
            current_history.append(
                {
                    "adjacency": pred,
                    "graph": nx.from_numpy_array(pred) if self.binary else None,
                }
            )
            self._update_distribution_history(pred)

            # Adaptive weight updates if no recent transition
            time_since_change = len(current_history) - self.phase_start
            if self.adaptive and time_since_change > self.temporal_window:
                new_weights = self._feature_based_adaptation(last_adjs, pred)
                self.weights = np.maximum(new_weights, self.min_weight)
                self.weights = self.weights / self.weights.sum()

        return predictions

    def update_state(self, actual_state: Dict[str, Any]) -> None:
        """
        Update predictor's internal state with new observation.

        Parameters
        ----------
        actual_state : Dict[str, Any]
            The actual observed network state containing 'adjacency' matrix
        """
        actual_adj = actual_state["adjacency"]

        # Update distribution history
        self._update_distribution_history(actual_adj)

        # Update feature history
        features = self._compute_network_features(actual_adj)
        self.feature_history.append(features)

        # Detect and update phase if needed
        if len(self.feature_history) >= self.temporal_window:
            recent_history = [{"adjacency": actual_adj, "features": features}]
            if self._detect_phase_transition(recent_history):
                self.phase_start = len(self.pattern_history)
                self.current_phase = self._detect_model_type(actual_adj)

        # Update pattern history
        self.pattern_history.append(
            {"adjacency": actual_adj, "features": features, "phase": self.current_phase}
        )

    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the predictor.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing current predictor state and metrics
        """
        return {
            "parameters": {
                "weights": self.weights.tolist(),
                "current_phase": self.current_phase,
                "phase_start": self.phase_start,
            },
            "distribution_history": {
                k: list(v) for k, v in self.distribution_history.items()
            },
            "change_points": self.change_points,
            "features": {
                "current": self.feature_history[-1] if self.feature_history else None,
                "ema": self.ema_features,
            },
        }

    def reset(self) -> None:
        """Reset the predictor to its initial state."""
        # Reset weights to initial values
        self.weights = np.array([0.5, 0.3, 0.1, 0.05, 0.05])
        self.weights = np.maximum(self.weights, self.min_weight)
        self.weights = self.weights / self.weights.sum()

        # Clear histories
        self.pattern_history = []
        self.feature_history = []
        self.change_points = []
        self.distribution_history = {
            "degree": [],
            "clustering": [],
            "path_length": [],
            "betweenness": [],
        }

        # Reset phase tracking
        self.phase_start = 0
        self.current_phase = "unknown"
        self.ema_features = {}
