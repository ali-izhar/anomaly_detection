"""Hybrid network predictor combining weighted averaging with structural role preservation."""

import networkx as nx
import numpy as np
from typing import List, Dict, Any, Set, Optional, Tuple
import community  # python-louvain package

from .weighted import WeightedPredictor
from graph.features import NetworkFeatureExtractor


class HybridPredictor:
    """Predict future network states while preserving both global and local properties.

    This predictor combines:
    1. Weighted historical averaging (from WeightedPredictor)
    2. Node role preservation (hubs, bridges, clusters)
    3. Local structure preservation
    4. Temporal consistency
    5. Spectral properties

    The key improvement over WeightedPredictor is the preservation of:
    - Node roles (hubs maintain high degrees)
    - Local clustering
    - Bridge nodes that connect communities
    - Spectral properties of the graph
    """

    def __init__(
        self,
        n_history: int = 3,
        weights: Optional[np.ndarray] = None,
        adaptive: bool = True,
        model_type: str = "ba",  # ba, er, ws, sbm
    ):
        """Initialize the hybrid predictor.

        Parameters
        ----------
        n_history : int
            Number of historical states to consider
        weights : np.ndarray, optional
            Weights for historical states (newest to oldest)
        adaptive : bool
            Whether to adapt weights based on prediction accuracy
        model_type : str
            Type of network model, affects evolution rules
        """
        # Initialize base weighted predictor
        self.base_predictor = WeightedPredictor(
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
            hub_score = 0.8
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
            comm_score = 1.0 if node_comm.get(node_i) == node_comm.get(node_j) else 0.5
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
        """Compute score based on spectral properties.

        For disconnected graphs, we:
        1. Use Katz centrality instead of eigenvector centrality
        2. Fall back to degree centrality if Katz fails
        3. Consider component membership
        """
        try:
            # Try Katz centrality first (works for disconnected graphs)
            if "katz_centrality" not in self._cache:
                self._cache["katz_centrality"] = nx.katz_centrality_numpy(
                    G, alpha=0.1, beta=1.0
                )
            centrality = self._cache["katz_centrality"]

        except:
            try:
                # Fall back to degree centrality
                if "degree_centrality" not in self._cache:
                    self._cache["degree_centrality"] = nx.degree_centrality(G)
                centrality = self._cache["degree_centrality"]
            except:
                # If all else fails, use normalized degrees
                n = G.number_of_nodes()
                centrality = {node: deg / (n - 1) for node, deg in G.degree()}

        # Get component membership
        if "components" not in self._cache:
            self._cache["components"] = list(nx.connected_components(G))

        # Check if nodes are in the same component
        same_component = False
        for component in self._cache["components"]:
            if node_i in component and node_j in component:
                same_component = True
                break

        # Compute base score from centrality
        base_score = (centrality[node_i] + centrality[node_j]) / 2

        # Adjust score based on component membership
        if same_component:
            return base_score
        else:
            # Penalize connections between components, but still possible
            return 0.3 * base_score

    def predict(
        self, history: List[Dict[str, Any]], horizon: int = 1
    ) -> List[np.ndarray]:
        """Predict future network states.

        Parameters
        ----------
        history : List[Dict[str, Any]]
            Historical network states
        horizon : int
            Number of steps to predict ahead

        Returns
        -------
        List[np.ndarray]
            Predicted adjacency matrices
        """
        if len(history) < self.base_predictor.n_history:
            raise ValueError(
                f"Not enough history. Need {self.base_predictor.n_history}, got {len(history)}."
            )

        predictions = []
        current_history = list(history)  # local copy

        for _ in range(horizon):
            # 1. Get base prediction from weighted predictor
            base_pred = self.base_predictor._compute_weighted_average(
                [
                    st["adjacency"]
                    for st in current_history[-self.base_predictor.n_history :]
                ]
            )

            # 2. Get current network state
            current_adj = current_history[-1]["adjacency"]
            G_current = nx.from_numpy_array(current_adj)
            n = G_current.number_of_nodes()

            # 3. Identify node roles
            hub_nodes, bridge_nodes, cluster_nodes = self._identify_node_roles(
                G_current
            )

            # 4. Get communities if needed
            try:
                partition = community.best_partition(G_current)
                communities = [
                    {n for n, c in partition.items() if c == i}
                    for i in range(max(partition.values()) + 1)
                ]
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

                    ba_score = self._compute_ba_evolution_score(
                        G_current, i, j, hub_nodes
                    )

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

                    edge_scores.append((final_score, i, j))

            # Sort edges by score
            edge_scores.sort(reverse=True)

            # Create new adjacency matrix
            predicted_adj = np.zeros((n, n))

            # Get target properties from latest network
            target_edges = int(np.sum(current_adj) / 2)  # maintain density

            # Add edges in order of scores
            added_edges = 0
            for score, i, j in edge_scores:
                if added_edges >= target_edges:
                    break
                predicted_adj[i, j] = predicted_adj[j, i] = 1
                added_edges += 1

            # Store prediction
            predictions.append(predicted_adj)

            # Update history for next iteration
            current_history.append(
                {
                    "adjacency": predicted_adj,
                    "graph": nx.from_numpy_array(predicted_adj),
                }
            )

            # Clear cache for next iteration
            self._cache.clear()

        return predictions
