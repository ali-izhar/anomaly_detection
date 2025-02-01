# src/predictor/adaptive.py

"""Adaptive distribution-aware predictor with multi-constraint structural preservation."""

from typing import List, Dict, Any, Optional, Tuple

import logging
import warnings
import networkx as nx
import numpy as np

from scipy import sparse, stats
from scipy.sparse.linalg import eigsh
from sklearn.cluster import SpectralClustering

from .base import BasePredictor

warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)


class AdaptiveDistributionAwarePredictor(BasePredictor):
    """Adaptive distribution-aware predictor with multi-constraint structural preservation."""

    def __init__(
        self,
        n_history: int = 5,
        weights: Optional[np.ndarray] = None,
        adaptive: bool = True,
        enforce_connectivity: bool = True,
        binary: bool = True,
        spectral_reg: float = 0.4,
        community_reg: float = 0.4,
        n_communities: int = 2,
        temporal_window: int = 10,
        min_edges_per_component: int = 4,
        degree_reg: float = 0.3,
        change_threshold: float = 0.25,
        smoothing_window: int = 3,
        min_weight: float = 0.1,
        distribution_memory: int = 20,
        phase_length: int = 40,
        distribution_reg: float = 0.3,
    ):
        """
        Initialize the adaptive predictor.

        Parameters
        ----------
        n_history : int
            Number of past states to incorporate in the weighted average.
        weights : Optional[np.ndarray]
            Custom weights to multiply recent adjacency matrices.
        adaptive : bool
            Whether to adapt weights over time based on feature differences.
        enforce_connectivity : bool
            Whether to enforce connectivity on the final graph prediction.
        binary : bool
            Whether to binarize the adjacency matrix in the final prediction.
        spectral_reg : float
            Regularization weight for preserving eigenvalues/eigenvectors.
        community_reg : float
            Regularization weight for preserving community structure.
        n_communities : int
            Number of communities to detect/preserve in spectral clustering.
        temporal_window : int
            Window size for temporal pattern detection.
        min_edges_per_component : int
            Minimum number of edges to add when connecting components.
        degree_reg : float
            Regularization weight for preserving the degree distribution.
        change_threshold : float
            Threshold for detecting structural changes (e.g., density/clustering).
        smoothing_window : int
            Number of predicted adjacency matrices to include in smoothing.
        min_weight : float
            Minimum allowed weight for any historical adjacency matrix.
        distribution_memory : int
            How many past states to store for distribution-based computations.
        phase_length : int
            Minimum length of a "phase" before we can detect a possible transition.
        distribution_reg : float
            Strength of distribution-based adjustments (e.g., density alignment).
        """
        self.n_history = n_history

        # Default weighting scheme (exponential decay) if no custom weights are provided
        if weights is None:
            # take n_history and compute exponentially decaying weights for each state
            weights = np.exp(-np.arange(n_history) / n_history)

        # Ensure no weight is below min_weight, then normalize
        weights = np.maximum(np.array(weights, dtype=float), min_weight)
        self.weights = weights / weights.sum()

        # Key parameters for controlling the model
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

        # Internal state tracking
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

    # =========================================================================
    #                    DISTRIBUTION & SPECTRAL COMPUTATIONS
    # =========================================================================

    def _compute_degree_distribution(self, adj: np.ndarray) -> np.ndarray:
        """Compute normalized degree distribution for a given adjacency matrix."""
        degrees = adj.sum(axis=1)
        dist = np.zeros(int(max(degrees)) + 1)
        for d in degrees:
            dist[int(d)] += 1
        return dist / len(degrees)

    def _compute_spectral_features(
        self, adj: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the leading eigenvalues and eigenvectors with enhanced stability.
        Uses either eigsh (for sparse matrices) or np.linalg.eigh as a fallback."""
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
        """Detect communities using spectral clustering and compute modularity.

        Returns
        -------
        labels : np.ndarray
            Community labels for each node.
        modularity : float
            The network modularity for the detected partition.
        """
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

            # Compute modularity via networkx
            G = nx.from_numpy_array(adj_nn)
            modularity = nx.community.modularity(
                G, [set(np.where(labels == i)[0]) for i in range(self.n_communities)]
            )

            return labels, modularity
        except:
            n = adj.shape[0]
            return np.array([i % self.n_communities for i in range(n)]), 0.0

    # =========================================================================
    #                       NETWORK FEATURE UTILITIES
    # =========================================================================

    def _compute_network_features(self, adj: np.ndarray) -> Dict[str, float]:
        """Compute key network features (density, clustering, average degree, etc.)
        for adaptation or detecting structural changes."""
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

    # =========================================================================
    #                        TEMPORAL PATTERN DETECTION
    # =========================================================================

    def _detect_temporal_patterns(
        self, history: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Enhanced temporal pattern detection using exponential moving averages
        across a window of past adjacency matrices."""
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
                # NOTE: Keeping the code as-is, even though it references 'recent[-1][key]'
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

        # Compute simple linear trend on densities
        densities = [np.mean(adj) for adj in recent]
        x = np.arange(len(densities))
        trend = np.polyfit(x, densities, 1)[0]

        # Estimate periodicity from autocorrelation peaks
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

    # =========================================================================
    #                       STRUCTURAL PRESERVATION METHODS
    # =========================================================================

    def _spectral_regularization(
        self, pred: np.ndarray, target_vals: np.ndarray, target_vecs: np.ndarray
    ) -> np.ndarray:
        """Apply spectral regularization to preserve eigenvalue/eigenvector structure
        from the last observed adjacency matrix."""
        curr_vals, _ = self._compute_spectral_features(pred)

        # Construct a correction matrix based on differences
        reg_mat = np.zeros_like(pred)
        for i in range(min(len(target_vals), len(curr_vals))):
            v_target = target_vecs[:, i : i + 1]
            reg = (target_vals[i] - curr_vals[i]) * (v_target @ v_target.T)
            reg_mat += reg

        # Blend the predicted adjacency with the regularization matrix
        alpha = self.spectral_reg
        pred_reg = (1 - alpha) * pred + alpha * reg_mat

        return pred_reg

    def _preserve_communities(
        self, pred: np.ndarray, comm_labels: np.ndarray, modularity: float
    ) -> np.ndarray:
        """Enhanced community preservation with explicit block probability estimation."""
        n = pred.shape[0]
        modifier = np.zeros((n, n))

        # Statistical-style block probability estimation
        block0 = comm_labels == 0
        block1 = comm_labels == 1

        # Compute block-wise edge probabilities like statistical predictor
        A_00 = pred[block0][:, block0]
        A_11 = pred[block1][:, block1]
        A_01 = pred[block0][:, block1]

        n0, n1 = block0.sum(), block1.sum()
        e00 = A_00[np.triu_indices(n0, k=1)].sum()
        e11 = A_11[np.triu_indices(n1, k=1)].sum()
        e01 = A_01.sum()

        poss00 = max(1, n0 * (n0 - 1) / 2)
        poss11 = max(1, n1 * (n1 - 1) / 2)
        poss01 = max(1, n0 * n1)

        p_intra0 = e00 / poss00
        p_intra1 = e11 / poss11
        p_inter = e01 / poss01

        # Use these probabilities for more accurate community preservation
        for i in range(n):
            for j in range(i + 1, n):
                if comm_labels[i] == comm_labels[j]:
                    target_p = p_intra0 if comm_labels[i] == 0 else p_intra1
                    modifier[i, j] = modifier[j, i] = target_p - pred[i, j]
                else:
                    modifier[i, j] = modifier[j, i] = p_inter - pred[i, j]

        # Blend with original prediction using modularity-based weight
        alpha = self.community_reg * (1 + modularity)
        pred_comm = pred + alpha * modifier

        return np.clip(pred_comm, 0, 1)

    def _preserve_global_structure(
        self, pred: np.ndarray, last_adj: np.ndarray
    ) -> np.ndarray:
        """Enhance global structure preservation for better closeness centrality."""

        # Compute current closeness centrality
        G_pred = nx.from_numpy_array(pred)
        G_last = nx.from_numpy_array(last_adj)

        try:
            close_pred = nx.closeness_centrality(G_pred)
            close_last = nx.closeness_centrality(G_last)

            # Identify nodes with significant closeness deviation
            deviations = {}
            for node in G_pred.nodes():
                if close_last[node] > 0:  # avoid division by zero
                    dev = abs(close_pred[node] - close_last[node]) / close_last[node]
                    if dev > 0.1:  # threshold for significant deviation
                        deviations[node] = dev

            # For nodes with high deviation, adjust their connections
            if deviations:
                modifier = np.zeros_like(pred)
                for node, dev in deviations.items():
                    # Find optimal paths in last adjacency
                    paths = nx.single_source_shortest_path_length(G_last, node)

                    # Strengthen connections that maintain similar path lengths
                    for target, path_len in paths.items():
                        if path_len <= 2:  # focus on close neighbors
                            weight = (1.0 / (path_len + 1)) * dev
                            modifier[node, target] = modifier[target, node] = weight

                # Apply modifications with adaptive weight
                alpha = 0.3 * (
                    1.0
                    - np.mean(list(close_pred.values()))
                    / np.mean(list(close_last.values()))
                )
                pred = pred + alpha * modifier

            return np.clip(pred, 0, 1)
        except:
            return pred

    def _enforce_connectivity_enhanced(
        self, pred: np.ndarray, orig_pred: np.ndarray
    ) -> np.ndarray:
        """Ensure the graph is connected by linking isolated components with MST-like logic,
        drawing on the original predicted weights for guidance."""
        G = nx.from_numpy_array(pred)
        components = list(nx.connected_components(G))

        if len(components) > 1:
            # Sort components by descending size
            components.sort(key=len, reverse=True)
            main_comp = components[0]

            # Connect each smaller component to the main component
            for comp in components[1:]:
                best_edges = []
                for i in main_comp:
                    for j in comp:
                        best_edges.append((orig_pred[i, j], i, j))

                # Sort potential edges by descending weight
                best_edges.sort(reverse=True)

                # Take top k edges to connect the component
                for _, i, j in best_edges[: self.min_edges_per_component]:
                    pred[i, j] = pred[j, i] = 1.0

        return pred

    def _preserve_degree_distribution(
        self, pred: np.ndarray, target_dist: np.ndarray
    ) -> np.ndarray:
        """Attempt to preserve a target degree distribution via a smoothed gradient
        approach that adjusts edges based on distribution mismatches."""
        curr_dist = self._compute_degree_distribution(pred)

        # Avoid zero divisions by adding tiny eps
        eps = 1e-8

        # Extend distributions to a common length
        max_len = max(len(target_dist), len(curr_dist))
        target_ext = np.ones(max_len) * eps
        curr_ext = np.ones(max_len) * eps
        target_ext[: len(target_dist)] += target_dist
        curr_ext[: len(curr_dist)] += curr_dist

        # Re-normalize
        target_ext = target_ext / target_ext.sum()
        curr_ext = curr_ext / curr_ext.sum()

        # Compute a gradient based on ratio differences
        grad = np.zeros_like(pred)
        degrees = pred.sum(axis=1)

        for i in range(pred.shape[0]):
            d = int(degrees[i])
            if d < max_len - 1:
                ratio_curr = max(eps, curr_ext[d])
                ratio_next = max(eps, curr_ext[d + 1])
                target_ratio = max(eps, target_ext[d + 1]) / max(eps, target_ext[d])
                grad_effect = np.clip(
                    np.log(target_ratio * ratio_curr / ratio_next), -1.0, 1.0
                )
                grad[i, :] = grad_effect
                grad[:, i] = grad_effect

        grad = grad * pred

        # Apply gradient with adaptive step size
        step_size = self.degree_reg / (1.0 + np.std(degrees))
        pred_new = pred + step_size * grad

        return np.clip(pred_new, 0, 1)

    # =========================================================================
    #           Triadic Closure Reinforcement for Higher Clustering
    # =========================================================================
    def _reinforce_triadic_closure(
        self, pred: np.ndarray, factor: float = 0.05
    ) -> np.ndarray:
        """
        Gently boosts the probability of missing edges in existing triads
        to improve local clustering without drastically altering the graph.

        factor : float
            Small increment factor for each potential missing edge in a triad.
        """
        # Work on a copy so we can adjust without interfering mid-iteration
        new_pred = pred.copy()
        G = nx.from_numpy_array(pred)

        for node in range(pred.shape[0]):
            neighbors = list(G.neighbors(node))
            # For each pair of neighbors, try to close the triangle
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    n1, n2 = neighbors[i], neighbors[j]
                    if new_pred[n1, n2] < 1.0:
                        # Increase by a small fraction
                        increment = factor * (pred[node, n1] + pred[node, n2]) / 2.0
                        new_val = new_pred[n1, n2] + increment
                        new_pred[n1, n2] = new_pred[n2, n1] = min(1.0, new_val)

        return np.clip(new_pred, 0, 1)

    # =========================================================================
    #    Slight Path-Length Shaping for Better Closeness Approximation
    # =========================================================================
    def _reduce_path_length(
        self, pred: np.ndarray, last_adj: np.ndarray, alpha: float = 0.02
    ) -> np.ndarray:
        """
        If the newly predicted graph has a higher average path length than the
        last observed adjacency, lightly add bridging edges among the pairs
        that are currently far apart.

        alpha : float
            Probability or fraction for bridging edges among the most distant pairs.
        """
        G_pred = nx.from_numpy_array(pred)
        G_last = nx.from_numpy_array(last_adj)

        try:
            apl_pred = nx.average_shortest_path_length(G_pred)
            apl_last = nx.average_shortest_path_length(G_last)
        except:
            # If the graph is disconnected, skip
            return pred

        # Only if new graph has significantly higher path length, add a few bridging edges
        if apl_pred > apl_last * 1.05:  # 5% tolerance
            new_pred = pred.copy()
            # Identify largest connected component to avoid bridging everything
            components = sorted(nx.connected_components(G_pred), key=len, reverse=True)
            main_comp = components[0] if components else set()
            # We'll add edges from the largest component to smaller ones or far pairs
            # to reduce distance. We'll randomly pick some pairs to link.

            # For each smaller component, pick a random bridging node
            for comp in components[1:]:
                comp = list(comp)
                main_comp_list = list(main_comp)
                # Random bridging
                if comp and main_comp_list:
                    c_node = np.random.choice(comp)
                    m_node = np.random.choice(main_comp_list)
                    # With probability alpha, turn on that edge strongly
                    if np.random.rand() < alpha:
                        new_pred[c_node, m_node] = 1.0
                        new_pred[m_node, c_node] = 1.0

            return np.clip(new_pred, 0, 1)
        else:
            return pred

    # =========================================================================
    #            FEATURE-BASED ADAPTATION & MODEL TYPE DETECTION
    # =========================================================================

    def _feature_based_adaptation(
        self, old_adjs: List[np.ndarray], new_adj: np.ndarray
    ) -> np.ndarray:
        """Adapt the historical weights based on how well the new adjacency matrix
        preserves key network features from each old adjacency."""
        feature_diffs = []

        for old_adj in old_adjs:
            old_feats = self._compute_network_features(old_adj)
            new_feats = self._compute_network_features(new_adj)
            diffs = []
            for k in old_feats:
                if old_feats[k] != 0:
                    diff = abs(old_feats[k] - new_feats[k]) / abs(old_feats[k])
                else:
                    diff = abs(new_feats[k])
                diffs.append(diff)

            feature_diffs.append(np.mean(diffs))

        raw_weights = 1.0 / (1.0 + np.array(feature_diffs))
        new_weights = raw_weights / raw_weights.sum()

        return new_weights

    def _detect_model_type(self, adj: np.ndarray) -> str:
        """Heuristic detection of the likely random graph model (SBM, BA, WS, or ER)
        based on network features (modularity, power-law degrees, clustering, etc.)."""
        G = nx.from_numpy_array(adj)
        n = G.number_of_nodes()
        density = nx.density(G)
        clustering = nx.average_clustering(G)
        degrees = [d for _, d in G.degree()]
        degree_std = np.std(degrees)

        # Check for SBM-like (high modularity)
        if self.n_communities > 1:
            _, modularity = self._detect_communities(adj)
            if modularity > 0.3:
                return "sbm"

        # Check for BA-like (power law / hub-dominated)
        if degree_std > 2.0 and max(degrees) > 3 * np.mean(degrees):
            return "ba"

        # Check for WS-like (reasonable clustering, moderate density)
        if clustering > 0.2 and density < 0.3:
            return "ws"

        # Default to ER
        return "er"

    def _preserve_model_structure(
        self, pred: np.ndarray, model_type: str, last_adj: np.ndarray
    ) -> np.ndarray:
        """Model-specific structural refinements based on the detected model type
        (SBM, BA, WS)."""
        if model_type == "sbm":
            # Strengthen community structure
            comm_labels, modularity = self._detect_communities(last_adj)
            pred = self._preserve_communities(pred, comm_labels, modularity)

        elif model_type == "ba":
            # Enhance hub connections for BA-like networks
            hub_nodes = np.argsort(np.sum(last_adj, axis=0))[
                -int(0.1 * len(last_adj)) :
            ]
            for hub in hub_nodes:
                hub_weights = last_adj[hub] * 1.5
                pred[hub] = (pred[hub] * (1 - self.spectral_reg)) + (
                    hub_weights * self.spectral_reg
                )
                pred[:, hub] = pred[hub]  # Maintain symmetry

        elif model_type == "ws":
            # Preserve local clustering (Watts–Strogatz style)
            G_last = nx.from_numpy_array(last_adj)
            clustering_coef = nx.clustering(G_last)
            for node, clust in clustering_coef.items():
                if clust > 0.5:
                    neighbors = list(G_last.neighbors(node))
                    for n1 in neighbors:
                        for n2 in neighbors:
                            if n1 < n2:
                                pred[n1, n2] = pred[n2, n1] = max(
                                    pred[n1, n2], 0.8 * last_adj[n1, n2]
                                )

        return np.clip(pred, 0, 1)

    # =========================================================================
    #            DISTRIBUTION HISTORY & PHASE TRANSITION DETECTION
    # =========================================================================

    def _update_distribution_history(self, adj: np.ndarray):
        """Track distribution statistics (degree, clustering, path length, betweenness)
        in a rolling window to detect changes over time."""
        G = nx.from_numpy_array(adj)

        # Degree & clustering distribution
        degrees = [d for _, d in G.degree()]
        clustering = list(nx.clustering(G).values())

        # Approximate path-length distribution by sampling
        try:
            n = len(G)
            sample_size = min(10, n)
            sampled_nodes = np.random.choice(n, sample_size, replace=False)
            path_lengths = []
            for u in sampled_nodes:
                lengths = nx.single_source_shortest_path_length(G, u)
                path_lengths.extend(lengths.values())
        except:
            path_lengths = []

        # Approximate betweenness for efficiency
        try:
            between = list(nx.betweenness_centrality(G, k=min(10, n - 1)).values())
        except:
            between = []

        # Append to history and maintain a fixed memory window
        self.distribution_history["degree"].append(degrees)
        self.distribution_history["clustering"].append(clustering)
        self.distribution_history["path_length"].append(path_lengths)
        self.distribution_history["betweenness"].append(between)

        for key in self.distribution_history:
            if len(self.distribution_history[key]) > self.distribution_memory:
                self.distribution_history[key].pop(0)

    def _detect_distribution_type(self) -> Dict[str, str]:
        """Inspect the recent distribution samples (e.g., degrees) to guess
        whether they fit a power law, Poisson, or remain unknown."""
        if not self.distribution_history["degree"]:
            return {}

        # Analyze the most recent sets of degrees
        recent_degrees = np.concatenate(self.distribution_history["degree"][-5:])
        types = {}

        # Check degree distribution for power-law
        if len(recent_degrees) > 10:
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
        """Estimate relevant phase parameters (e.g., average degree, clustering)
        from the most recent distribution snapshots."""
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

        dist_types = self._detect_distribution_type()
        if dist_types.get("degree") == "power_law":
            # Estimate the BA-like parameter m from average degree
            params["m"] = max(1, int(round(params["avg_degree"] / 2)))

        return params

    def _detect_phase_transition(self, history: List[Dict[str, Any]]) -> bool:
        """Detect phase transitions based on exponential moving averages of
        structural features over a temporal window, combined with the
        minimal phase length constraint."""
        if len(history) < self.temporal_window:
            return False

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

        # Check for significant relative changes
        changes = []
        for key in features[-1]:
            current = features[-1][key]
            ema = self.ema_features[key]
            if ema > 0:
                change = abs(current - ema) / ema
                changes.append(change > self.change_threshold)

        # Consider how long we've been in the current phase
        time_since_change = len(history) - self.phase_start
        phase_maturity = time_since_change >= self.phase_length

        return any(changes) and phase_maturity

    # =========================================================================
    #                  DISTRIBUTION PRESERVATION & PREDICTION
    # =========================================================================

    def _preserve_distribution_properties(
        self, pred: np.ndarray, last_adj: np.ndarray
    ) -> np.ndarray:
        """Apply distribution-based heuristics, e.g., power-law boosting,
        high-clustering reinforcement, or density alignment."""
        dist_types = self._detect_distribution_type()
        params = self._estimate_phase_parameters()

        # Power-law strengthening (BA-like)
        if dist_types.get("degree") == "power_law":
            degrees = np.sum(last_adj, axis=0)
            hub_threshold = np.percentile(degrees, 80)
            hub_mask = degrees > hub_threshold

            for i in np.where(hub_mask)[0]:
                neighbors = np.where(last_adj[i] > 0)[0]
                if len(neighbors) > 0:
                    # Strengthen existing hub connections
                    pred[i, neighbors] = pred[neighbors, i] = np.maximum(
                        pred[i, neighbors], 0.8 * last_adj[i, neighbors]
                    )

                    # Preferential attachment effect
                    neighbor_degrees = degrees[neighbors]
                    attach_probs = neighbor_degrees / neighbor_degrees.sum()
                    for j in range(pred.shape[0]):
                        if j not in neighbors:
                            influence = np.sum(attach_probs * (pred[j, neighbors] > 0))
                            pred[i, j] = pred[j, i] = max(pred[i, j], 0.3 * influence)

        # If clustering is relatively high, preserve local triangles
        elif params.get("clustering_coef", 0) > 0.2:
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

        # Adjust density toward the current estimate
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
            Past states, each containing an 'adjacency' key with np.ndarray.
        horizon : int
            Number of steps to predict ahead.

        Returns
        -------
        List[np.ndarray]
            List of predicted adjacency matrices for the specified horizon.
        """
        if len(history) < self.n_history:
            raise ValueError(
                f"Not enough history. Need {self.n_history}, got {len(history)}."
            )

        # (Unchanged) gather current_history
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
        logger.debug(f"Updating distribution history for {len(current_history)} states")
        self._update_distribution_history(current_history[-1]["adjacency"])

        # Check for phase transition
        if self._detect_phase_transition(current_history):
            self.phase_start = len(current_history)
            self.weights = np.array([0.5, 0.3, 0.1, 0.05, 0.05])
            self.weights = np.maximum(self.weights, self.min_weight)
            self.weights = self.weights / self.weights.sum()
            self.change_points.append(len(current_history))

        predictions = []
        logger.debug(f"Starting prediction loop for {horizon} steps")
        for _ in range(horizon):
            last_states = current_history[-self.n_history :]
            last_adjs = [st["adjacency"] for st in last_states]

            # Combine them into a preliminary prediction
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

            orig_pred = pred.copy()

            # 1) Distribution-based adjustments
            pred = self._preserve_distribution_properties(pred, last_adjs[-1])

            # 2) Spectral regularization
            target_vals, target_vecs = self._compute_spectral_features(last_adjs[-1])
            pred = self._spectral_regularization(pred, target_vals, target_vecs)

            # 3) Enhanced community preservation
            comm_labels, modularity = self._detect_communities(last_adjs[-1])
            pred = self._preserve_communities(pred, comm_labels, modularity)

            # 4) Global structure enhancement for closeness
            pred = self._preserve_global_structure(pred, last_adjs[-1])

            # 5) Light triadic closure reinforcement
            pred = self._reinforce_triadic_closure(pred, factor=0.02)

            # 6) Slightly reduce path length if too large
            pred = self._reduce_path_length(pred, last_adjs[-1], alpha=0.02)

            # Ensure valid probabilities
            pred = np.clip(pred, 0, 1)

            # If binary mode is on, threshold edges by target density
            if self.binary:
                target_density = np.mean(last_adjs[-1])
                n = pred.shape[0]
                target_edges = int(np.floor(target_density * n * (n - 1) / 2))

                triu = np.triu_indices(n, k=1)
                probs = pred[triu]
                edges = list(zip(probs, triu[0], triu[1]))
                edges.sort(key=lambda x: x[0], reverse=True)

                pred_binary = np.zeros_like(pred)
                for _, i, j in edges[:target_edges]:
                    pred_binary[i, j] = pred_binary[j, i] = 1.0

                pred = pred_binary

                # Enforce connectivity if desired
                if self.enforce_connectivity:
                    pred = self._enforce_connectivity_enhanced(pred, orig_pred)

            logger.debug(f"{_ + 1} / {horizon}: Predicted adjacency matrix: {pred}")
            predictions.append(pred)

            # Update history and distribution tracking
            current_history.append(
                {
                    "adjacency": pred,
                    "graph": nx.from_numpy_array(pred) if self.binary else None,
                }
            )
            self._update_distribution_history(pred)

            # Adaptive weight updates if enough time since phase change
            time_since_change = len(current_history) - self.phase_start
            if self.adaptive and time_since_change > self.temporal_window:
                new_weights = self._feature_based_adaptation(last_adjs, pred)
                self.weights = np.maximum(new_weights, self.min_weight)
                self.weights = self.weights / self.weights.sum()

        logger.debug(
            f"Finished prediction loop for {horizon} steps. Returning {len(predictions)} predictions"
        )
        return predictions

    # =========================================================================
    #                     EXTERNAL STATE UPDATES & QUERIES
    # =========================================================================

    def update_state(self, actual_state: Dict[str, Any]) -> None:
        """Update the predictor's internal state with a new observed adjacency.
        This should be called after receiving a real "future" state that
        either confirms or contradicts the predictions."""
        actual_adj = actual_state["adjacency"]

        # Update the distribution histories
        self._update_distribution_history(actual_adj)

        # Update feature history
        features = self._compute_network_features(actual_adj)
        self.feature_history.append(features)

        # Detect if a new phase has begun
        if len(self.feature_history) >= self.temporal_window:
            recent_history = [{"adjacency": actual_adj, "features": features}]
            if self._detect_phase_transition(recent_history):
                self.phase_start = len(self.pattern_history)
                self.current_phase = self._detect_model_type(actual_adj)

        # Track overall pattern evolution
        self.pattern_history.append(
            {"adjacency": actual_adj, "features": features, "phase": self.current_phase}
        )

    def get_state(self) -> Dict[str, Any]:
        """Return a dictionary representing the current internal state,
        including weights, distribution history, detected change points,
        and current/EMA features."""
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
        """Reset the predictor to its initial state: clears all histories,
        reinitializes weights, and starts a new "unknown" phase."""
        self.weights = np.array([0.5, 0.3, 0.1, 0.05, 0.05])
        self.weights = np.maximum(self.weights, self.min_weight)
        self.weights = self.weights / self.weights.sum()

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
