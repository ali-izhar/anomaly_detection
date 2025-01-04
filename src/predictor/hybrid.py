# src/predictor/hybrid.py

"""Hybrid (Weighted + Local Structure) predictor."""

import numpy as np
import networkx as nx
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import defaultdict

from graph.features import NetworkFeatureExtractor
from .base import BasePredictor
from .weighted import WeightedPredictor


class HybridPredictor(BasePredictor):
    """Hybrid predictor that preserves both global and local network properties."""

    def __init__(
        self,
        n_history: int = 3,
        weights: Optional[np.ndarray] = None,
        adaptive: bool = True,
        model_type: str = "ba",
    ):
        # Initialize weighted predictor as base
        self.weighted_predictor = WeightedPredictor(
            n_history=n_history, weights=weights, adaptive=adaptive
        )

        # For computing network metrics
        self.feature_extractor = NetworkFeatureExtractor()

        # Model parameters
        self.model_type = model_type
        self.base_m = 3 if model_type == "ba" else None

        # Role identification thresholds
        self.hub_threshold = 0.8  # Top 20% by degree
        self.bridge_threshold = 0.8  # Top 20% by betweenness
        self.cluster_threshold = 0.7  # Top 30% by clustering

        # Cache for performance
        self._cache = {}

    def _identify_node_roles(self, G: nx.Graph) -> Tuple[Set[int], Set[int], Set[int]]:
        """Identify hub nodes, bridge nodes, and cluster nodes."""
        n = G.number_of_nodes()

        # Get node metrics
        degrees = dict(G.degree())
        betweenness = nx.betweenness_centrality(G)
        clustering = nx.clustering(G)

        # Identify hubs (high degree nodes)
        degree_threshold = sorted(degrees.values(), reverse=True)[
            int(n * (1 - self.hub_threshold))
        ]
        hub_nodes = {node for node, deg in degrees.items() if deg >= degree_threshold}

        # Identify bridges (high betweenness nodes)
        betw_threshold = sorted(betweenness.values(), reverse=True)[
            int(n * (1 - self.bridge_threshold))
        ]
        bridge_nodes = {node for node, b in betweenness.items() if b >= betw_threshold}

        # Identify cluster nodes (high clustering coefficient)
        clust_values = [
            v for v in clustering.values() if v > 0
        ]  # Exclude zero clustering
        if clust_values:
            clust_threshold = sorted(clust_values, reverse=True)[
                int(len(clust_values) * (1 - self.cluster_threshold))
            ]
            cluster_nodes = {
                node for node, c in clustering.items() if c >= clust_threshold
            }
        else:
            cluster_nodes = set()

        return hub_nodes, bridge_nodes, cluster_nodes

    def _compute_role_preservation_score(
        self,
        G: nx.Graph,
        node_i: int,
        node_j: int,
        hub_nodes: Set[int],
        bridge_nodes: Set[int],
        cluster_nodes: Set[int],
    ) -> float:
        """Compute score based on node role preservation."""
        score = 0.0
        count = 0

        # Hub role preservation
        if node_i in hub_nodes or node_j in hub_nodes:
            hub_score = 0.8  # High score for hub connections
            if node_i in hub_nodes and node_j in hub_nodes:
                hub_score = 1.0  # Maximum for hub-hub connections
            score += hub_score
            count += 1

        # Bridge role preservation
        if node_i in bridge_nodes or node_j in bridge_nodes:
            bridge_score = 0.7
            if abs(G.degree(node_i) - G.degree(node_j)) > 2:
                bridge_score = 0.9  # Higher score for connecting different degree nodes
            score += bridge_score
            count += 1

        # Cluster role preservation
        if node_i in cluster_nodes or node_j in cluster_nodes:
            common_neighbors = len(set(G.neighbors(node_i)) & set(G.neighbors(node_j)))
            cluster_score = min(
                1.0, common_neighbors / max(G.degree(node_i), G.degree(node_j))
            )
            score += cluster_score
            count += 1

        return score / max(1, count)

    def _compute_local_structure_score(
        self,
        G: nx.Graph,
        node_i: int,
        node_j: int,
        communities: Optional[List[Set[int]]] = None,
    ) -> float:
        """Compute score based on local structure preservation."""
        # Get neighborhoods
        neighbors_i = set(G.neighbors(node_i))
        neighbors_j = set(G.neighbors(node_j))

        # Common neighbors (triangle formation)
        common = neighbors_i & neighbors_j
        triangle_score = len(common) / max(1, min(len(neighbors_i), len(neighbors_j)))

        # Path length score
        try:
            path_length = nx.shortest_path_length(G, node_i, node_j)
            path_score = 1.0 / (1.0 + path_length)
        except:
            path_score = 0.0

        # Community score
        if communities:
            node_comm = {n: idx for idx, comm in enumerate(communities) for n in comm}
            comm_score = 1.0 if node_comm.get(node_i) == node_comm.get(node_j) else 0.3
        else:
            comm_score = 0.5

        # Combine scores
        return 0.4 * triangle_score + 0.3 * path_score + 0.3 * comm_score

    def _compute_ba_evolution_score(
        self, G: nx.Graph, node_i: int, node_j: int, hub_nodes: Set[int]
    ) -> float:
        """Compute score based on BA evolution principles."""
        n = G.number_of_nodes()
        deg_i = G.degree(node_i)
        deg_j = G.degree(node_j)

        # Preferential attachment score
        pa_score = (deg_i + deg_j) / (2 * n * self.base_m)

        # Hub evolution score
        if node_i in hub_nodes or node_j in hub_nodes:
            hub_score = max(deg_i, deg_j) / (n * self.base_m)
        else:
            hub_score = 0.5

        # Power law preservation score
        power_law_score = np.log(1 + max(deg_i, deg_j)) / np.log(n)

        return 0.4 * pa_score + 0.3 * hub_score + 0.3 * power_law_score

    def _compute_temporal_consistency(
        self, history: List[Dict[str, Any]], node_i: int, node_j: int
    ) -> float:
        """Compute temporal consistency score based on edge history."""
        if len(history) < 2:
            return 0.5

        # Check edge stability in recent history
        recent_states = history[-3:]  # Look at last 3 states
        edge_presence = []

        for state in recent_states:
            adj = state["adjacency"]
            edge_presence.append(adj[node_i, node_j])

        # Compute trend
        if len(edge_presence) >= 2:
            changes = sum(
                abs(edge_presence[i] - edge_presence[i - 1])
                for i in range(1, len(edge_presence))
            )
            stability = 1.0 / (1.0 + changes)
        else:
            stability = 0.5

        # Weight recent states more
        weighted_presence = sum(
            w * p for w, p in zip([0.5, 0.3, 0.2], reversed(edge_presence))
        )

        return 0.7 * stability + 0.3 * weighted_presence

    def _compute_spectral_score(self, G: nx.Graph, node_i: int, node_j: int) -> float:
        """Compute score based on spectral properties."""
        try:
            # Get local subgraph
            neighbors = set(G.neighbors(node_i)) | set(G.neighbors(node_j))
            subgraph = G.subgraph(neighbors | {node_i, node_j})

            # Compute Laplacian spectrum
            L = nx.laplacian_matrix(subgraph).todense()
            eigenvals = np.linalg.eigvalsh(L)

            # Fiedler value (algebraic connectivity)
            fiedler = eigenvals[1] if len(eigenvals) > 1 else 0

            # Spectral gap
            gap = eigenvals[-1] - eigenvals[-2] if len(eigenvals) > 2 else 0

            return 0.6 * (1 - np.exp(-fiedler)) + 0.4 * (1 - np.exp(-gap))
        except:
            return 0.5

    def _enhance_prediction(
        self,
        base_pred: np.ndarray,
        current_adj: np.ndarray,
        history: List[Dict[str, Any]],
    ) -> np.ndarray:
        """Enhance base prediction while preserving both global and local properties."""
        G_current = nx.from_numpy_array(current_adj)
        n = len(G_current)

        # Identify node roles with adjusted thresholds
        self.hub_threshold = 0.85  # More selective hubs (top 15%)
        self.bridge_threshold = 0.85  # More selective bridges (top 15%)
        self.cluster_threshold = 0.8  # More selective clusters (top 20%)
        hub_nodes, bridge_nodes, cluster_nodes = self._identify_node_roles(G_current)

        # Get communities and spectral properties
        try:
            communities = nx.community.louvain_communities(G_current)
        except:
            communities = None

        # Initialize edge scores with historical consistency
        edge_scores = []
        current_edges = set(
            (i, j) for i, j in zip(*np.where(current_adj == 1)) if i < j
        )

        # Compute scores for potential edges
        for i in range(n):
            for j in range(i + 1, n):
                # Start with base prediction probability
                score = base_pred[i, j]

                # Skip unlikely edges for efficiency
                if (
                    score < 0.05
                    and (i, j) not in current_edges
                    and i not in hub_nodes
                    and i not in bridge_nodes
                    and j not in hub_nodes
                    and j not in bridge_nodes
                ):
                    continue

                # Compute component scores
                role_score = self._compute_role_preservation_score(
                    G_current, i, j, hub_nodes, bridge_nodes, cluster_nodes
                )

                local_score = self._compute_local_structure_score(
                    G_current, i, j, communities
                )

                ba_score = self._compute_ba_evolution_score(G_current, i, j, hub_nodes)

                temporal_score = self._compute_temporal_consistency(history, i, j)

                spectral_score = self._compute_spectral_score(G_current, i, j)

                # Adaptive score combination based on node roles
                if i in hub_nodes or j in hub_nodes:
                    # Hub connections: favor BA evolution and temporal consistency
                    final_score = (
                        0.2 * score
                        + 0.2 * role_score
                        + 0.1 * local_score
                        + 0.3 * ba_score
                        + 0.1 * spectral_score
                        + 0.1 * temporal_score
                    )
                elif i in bridge_nodes or j in bridge_nodes:
                    # Bridge connections: favor spectral properties and role preservation
                    final_score = (
                        0.2 * score
                        + 0.25 * role_score
                        + 0.15 * local_score
                        + 0.1 * ba_score
                        + 0.2 * spectral_score
                        + 0.1 * temporal_score
                    )
                elif i in cluster_nodes or j in cluster_nodes:
                    # Cluster connections: favor local structure
                    final_score = (
                        0.2 * score
                        + 0.15 * role_score
                        + 0.3 * local_score
                        + 0.1 * ba_score
                        + 0.15 * spectral_score
                        + 0.1 * temporal_score
                    )
                else:
                    # Regular connections: balanced weights
                    final_score = (
                        0.2 * score
                        + 0.2 * role_score
                        + 0.2 * local_score
                        + 0.15 * ba_score
                        + 0.15 * spectral_score
                        + 0.1 * temporal_score
                    )

                # Boost score for existing edges to promote stability
                if (i, j) in current_edges:
                    final_score *= 1.2

                edge_scores.append(((i, j), final_score))

        # Sort edges by score
        edge_scores.sort(key=lambda x: x[1], reverse=True)

        # Create final prediction while preserving global properties
        target_edges = int(base_pred.sum() / 2)
        final_pred = np.zeros_like(base_pred)
        edges_added = 0

        # Track degree changes
        degree_changes = defaultdict(int)

        # Add edges while respecting constraints
        for (i, j), score in edge_scores:
            if edges_added >= target_edges:
                break

            # Compute allowed degree change based on node role
            max_change_i = 3 if i in hub_nodes else (2 if i in bridge_nodes else 1)
            max_change_j = 3 if j in hub_nodes else (2 if j in bridge_nodes else 1)

            # Check degree constraints
            if degree_changes[i] < max_change_i and degree_changes[j] < max_change_j:
                final_pred[i, j] = final_pred[j, i] = 1
                edges_added += 1
                degree_changes[i] += 1
                degree_changes[j] += 1

        return final_pred

    def predict(
        self, history: List[Dict[str, Any]], horizon: int = 1
    ) -> List[np.ndarray]:
        """Generate predictions using weighted predictor with structural enhancements."""
        predictions = []
        current_history = list(history)

        for _ in range(horizon):
            # Get base prediction from weighted predictor
            base_pred = self.weighted_predictor.predict(current_history, horizon=1)[0]

            if len(history) > 1:
                # Enhance prediction while preserving structure
                final_pred = self._enhance_prediction(
                    base_pred, history[-1]["adjacency"], current_history
                )
            else:
                final_pred = base_pred

            # Update for next iteration
            predictions.append(final_pred)
            current_history.append({"adjacency": final_pred})

        return predictions

    def _get_metrics(self, state: Dict[str, Any]) -> Dict[str, float]:
        """Extract relevant metrics from graph state."""
        G = nx.from_numpy_array(state["adjacency"])
        metrics = self.feature_extractor.get_all_metrics(G)

        return {
            "clustering": metrics.clustering,
            "avg_betweenness": metrics.avg_betweenness,
            "spectral_gap": metrics.spectral_gap,
            "algebraic_connectivity": metrics.algebraic_connectivity,
        }
