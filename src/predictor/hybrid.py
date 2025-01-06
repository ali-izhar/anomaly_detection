"""Hybrid network predictor combining weighted averaging with structural role preservation."""

import networkx as nx
import numpy as np
from typing import List, Dict, Any, Optional
import community

from .weighted import WeightedPredictor
from graph.features import NetworkFeatureExtractor


class HybridPredictor:
    """Hybrid predictor combining weighted averaging with structural role preservation."""

    def __init__(
        self,
        n_history: int = 5,
        weights: Optional[np.ndarray] = None,
        adaptive: bool = True,
        model_type: str = "ba",
        config: Optional[Dict] = None,
    ):
        # Store parameters
        self.n_history = n_history
        self.weights = weights
        self.adaptive = adaptive
        self.model_type = model_type
        self.config = config

        # Initialize weighted predictor for global properties
        self.base_predictor = WeightedPredictor(
            n_history=n_history,
            weights=weights,
            adaptive=adaptive,
            binary=False,  # We want probabilities, not binary
        )

        # Initialize model-specific parameters
        self.model_params = self._initialize_model_params()

    def _initialize_model_params(self) -> Dict:
        """Initialize parameters based on model type."""
        if not self.config or "params" not in self.config:
            return self._get_default_params()

        params = self.config["params"]
        model_params = {}

        if self.model_type == "sbm":
            # SBM-specific parameters
            model_params.update(
                {
                    "num_blocks": getattr(params, "num_blocks", 3),
                    "min_block_size": getattr(params, "min_block_size", 25),
                    "max_block_size": getattr(params, "max_block_size", 50),
                    "intra_prob": getattr(params, "intra_prob", 0.7),
                    "inter_prob": getattr(params, "inter_prob", 0.01),
                    "min_intra_prob": getattr(params, "min_intra_prob", 0.6),
                    "max_intra_prob": getattr(params, "max_intra_prob", 0.8),
                    "min_inter_prob": getattr(params, "min_inter_prob", 0.005),
                    "max_inter_prob": getattr(params, "max_inter_prob", 0.015),
                    "prob_tolerance": getattr(params, "intra_prob_std", 0.005) * 2,
                }
            )
        elif self.model_type == "ba":
            # BA-specific parameters
            model_params.update(
                {
                    "m": getattr(params, "m", 3),
                    "min_m": getattr(params, "min_m", 2),
                    "max_m": getattr(params, "max_m", 4),
                    "preferential_exp": getattr(params, "preferential_exp", 1.0),
                    "prob_tolerance": 0.1,
                }
            )
        elif self.model_type == "er":
            # ER-specific parameters
            model_params.update(
                {
                    "prob": getattr(params, "prob", 0.15),
                    "min_prob": getattr(params, "min_prob", 0.1),
                    "max_prob": getattr(params, "max_prob", 0.2),
                    "prob_tolerance": getattr(params, "prob_std", 0.01) * 2,
                }
            )
        elif self.model_type == "ws":
            # WS-specific parameters
            model_params.update(
                {
                    "k": getattr(params, "k_nearest", 6),
                    "min_k": getattr(params, "min_k", 4),
                    "max_k": getattr(params, "max_k", 8),
                    "rewire_prob": getattr(params, "rewire_prob", 0.1),
                    "min_rewire_prob": getattr(params, "min_prob", 0.05),
                    "max_rewire_prob": getattr(params, "max_prob", 0.15),
                    "prob_tolerance": getattr(params, "prob_std", 0.01) * 2,
                }
            )
        elif self.model_type == "rcp":
            # RCP-specific parameters
            model_params.update(
                {
                    "core_size": getattr(params, "core_size", params.n // 5),
                    "min_core_size": getattr(params, "min_core_size", params.n // 6),
                    "max_core_size": getattr(params, "max_core_size", params.n // 4),
                    "core_prob": getattr(params, "core_prob", 0.8),
                    "periph_prob": getattr(params, "periph_prob", 0.05),
                    "core_periph_prob": getattr(params, "core_periph_prob", 0.2),
                    "prob_tolerance": getattr(params, "core_prob_std", 0.02) * 2,
                }
            )
        else:
            # Default parameters for other models
            model_params = self._get_default_params()

        return model_params

    def _get_default_params(self) -> Dict:
        """Get default parameters for unknown model types."""
        return {
            "prob_tolerance": 0.05,
            "community_resolution": 1.2,
            "min_edge_prob": 0.01,
            "max_edge_prob": 0.9,
        }

    def _detect_communities(self, G: nx.Graph) -> Dict[int, int]:
        """Detect communities using method appropriate for the model type."""
        try:
            if self.model_type == "sbm":
                # Use higher resolution for SBM
                return community.best_partition(G, resolution=1.2)
            elif self.model_type == "rcp":
                # For RCP, try to identify core-periphery structure
                degrees = dict(G.degree())
                sorted_nodes = sorted(
                    degrees.keys(), key=lambda x: degrees[x], reverse=True
                )
                core_size = self.model_params["core_size"]
                return {
                    node: 0 if i < core_size else 1
                    for i, node in enumerate(sorted_nodes)
                }
            elif self.model_type == "ba":
                # For BA, use degree-based communities
                return community.best_partition(G, resolution=0.8)
            else:
                # Default community detection
                return community.best_partition(G, resolution=1.0)
        except:
            # Fallback to connected components
            components = nx.connected_components(G)
            return {node: idx for idx, comp in enumerate(components) for node in comp}

    def _analyze_block_structure(
        self, G: nx.Graph, communities: Dict[int, int]
    ) -> Dict[str, Any]:
        """Analyze network structure based on model type."""
        # Basic structure analysis
        n_blocks = max(communities.values()) + 1
        block_sizes = {}
        intra_edges = {}
        inter_edges = {}
        node_degrees = {}

        # Count nodes and track degrees per block
        for node in G.nodes():
            block = communities[node]
            block_sizes[block] = block_sizes.get(block, 0) + 1
            node_degrees[block] = node_degrees.get(block, [])
            node_degrees[block].append(G.degree(node))

        # Count edges with model-specific tracking
        for i, j in G.edges():
            bi, bj = communities[i], communities[j]
            if bi == bj:
                intra_edges[bi] = intra_edges.get(bi, 0) + 1
            else:
                key = tuple(sorted([bi, bj]))
                inter_edges[key] = inter_edges.get(key, 0) + 1

        # Calculate basic metrics
        intra_densities = {}
        avg_degrees = {}
        degree_stds = {}

        for block, size in block_sizes.items():
            possible = size * (size - 1) / 2
            if possible > 0:
                intra_densities[block] = intra_edges.get(block, 0) / possible
            if block in node_degrees and node_degrees[block]:
                avg_degrees[block] = np.mean(node_degrees[block])
                degree_stds[block] = np.std(node_degrees[block])

        # Calculate inter-block densities
        inter_densities = {}
        for (b1, b2), edges in inter_edges.items():
            possible = block_sizes[b1] * block_sizes[b2]
            if possible > 0:
                inter_densities[(b1, b2)] = edges / possible

        # Model-specific metrics
        model_specific = {}
        if self.model_type == "ba":
            # Add BA-specific metrics (degree distribution, hubs)
            model_specific["degree_distribution"] = np.array(
                list(dict(G.degree()).values())
            )
            model_specific["hub_nodes"] = [
                n
                for n, d in G.degree()
                if d
                > np.mean(list(dict(G.degree()).values()))
                + np.std(list(dict(G.degree()).values()))
            ]
        elif self.model_type == "ws":
            # Add WS-specific metrics (clustering, path length)
            model_specific["clustering"] = nx.average_clustering(G)
            try:
                model_specific["avg_path_length"] = nx.average_shortest_path_length(G)
            except:
                model_specific["avg_path_length"] = float("inf")
        elif self.model_type == "rcp":
            # Add RCP-specific metrics (core density, periphery density)
            core_nodes = [n for n, c in communities.items() if c == 0]
            model_specific["core_density"] = nx.density(G.subgraph(core_nodes))

        return {
            "block_sizes": block_sizes,
            "intra_densities": intra_densities,
            "inter_densities": inter_densities,
            "avg_degrees": avg_degrees,
            "degree_stds": degree_stds,
            "avg_intra_density": (
                np.mean(list(intra_densities.values())) if intra_densities else 0
            ),
            "avg_inter_density": (
                np.mean(list(inter_densities.values())) if inter_densities else 0
            ),
            "model_specific": model_specific,
        }

    def predict(
        self, history: List[Dict[str, Any]], horizon: int = 1
    ) -> List[np.ndarray]:
        """Predict future states using model-aware hybrid approach."""
        if len(history) < self.n_history:
            raise ValueError(
                f"Not enough history. Need {self.n_history}, got {len(history)}."
            )

        predictions = []
        current_history = list(history)

        # Track density evolution
        density_history = [nx.density(h["graph"]) for h in history[-self.n_history :]]
        density_trend = np.mean(np.diff(density_history))

        for step in range(horizon):
            current_G = current_history[-1]["graph"]
            n = current_G.number_of_nodes()

            # Get weighted prediction probabilities
            weighted_pred = self.base_predictor._compute_weighted_average(
                [st["adjacency"] for st in current_history[-self.n_history :]]
            )

            # Detect and analyze structure
            communities = self._detect_communities(current_G)
            structure_analysis = self._analyze_block_structure(current_G, communities)

            # Calculate target density based on model type
            if self.model_type == "ba":
                # BA models tend to become denser over time
                m = self.model_params["m"]
                target_density = min(0.3, (2 * m * (n - m)) / (n * (n - 1)))
            elif self.model_type == "er":
                # ER models maintain relatively stable density
                target_density = self.model_params["prob"]
            elif self.model_type == "ws":
                # WS models maintain average degree
                k = self.model_params["k"]
                target_density = k / (n - 1)
            elif self.model_type == "rcp":
                # RCP models have distinct core/periphery densities
                core_size = self.model_params["core_size"]
                target_density = (
                    core_size * (core_size - 1) * self.model_params["core_prob"]
                    + (n - core_size)
                    * (n - core_size - 1)
                    * self.model_params["periph_prob"]
                    + 2
                    * core_size
                    * (n - core_size)
                    * self.model_params["core_periph_prob"]
                ) / (n * (n - 1))
            else:
                # Default density evolution
                target_density = nx.density(current_G) + density_trend

            target_density = max(0.01, min(0.95, target_density))
            target_edges = int(target_density * (n * (n - 1) / 2))

            # Calculate edge probabilities based on model type
            edge_probs = []
            for i in range(n):
                for j in range(i + 1, n):
                    base_prob = weighted_pred[i, j]
                    ci, cj = communities[i], communities[j]

                    if self.model_type == "ba":
                        # BA: Consider degree preferential attachment
                        deg_i, deg_j = current_G.degree(i), current_G.degree(j)
                        pref_prob = (deg_i * deg_j) / sum(
                            dict(current_G.degree()).values()
                        ) ** 2
                        prob = (base_prob + pref_prob) / 2
                    elif self.model_type == "rcp":
                        # RCP: Consider core-periphery structure
                        if ci == cj == 0:  # Both in core
                            prob = max(base_prob, self.model_params["core_prob"] * 0.8)
                        elif ci == cj == 1:  # Both in periphery
                            prob = min(
                                base_prob, self.model_params["periph_prob"] * 1.2
                            )
                        else:  # Core-periphery connection
                            prob = self.model_params["core_periph_prob"]
                    else:
                        # Default probability calculation
                        if ci == cj:  # Same community
                            prob = max(
                                base_prob, structure_analysis["avg_intra_density"]
                            )
                        else:  # Different communities
                            prob = min(
                                base_prob, structure_analysis["avg_inter_density"]
                            )

                    edge_probs.append((prob, i, j))

            # Create prediction matrix
            predicted = np.zeros((n, n))
            edges_added = 0

            # Add edges based on probabilities
            for prob, i, j in sorted(edge_probs, reverse=True):
                if edges_added >= target_edges:
                    break

                if current_G.has_edge(
                    i, j
                ):  # Maintain existing edges with high probability
                    if prob > 0.5:
                        predicted[i, j] = predicted[j, i] = 1
                        edges_added += 1
                else:  # Add new edges conservatively
                    if prob > 0.7:
                        predicted[i, j] = predicted[j, i] = 1
                        edges_added += 1

            predictions.append(predicted)
            current_history.append(
                {"adjacency": predicted, "graph": nx.from_numpy_array(predicted)}
            )

        return predictions
