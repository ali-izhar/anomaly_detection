"""Hybrid network predictor combining weighted averaging with structural role preservation."""

import networkx as nx
import numpy as np
from typing import List, Dict, Any, Set, Optional, Tuple
import community  # python-louvain package

from .weighted import WeightedPredictor
from graph.features import NetworkFeatureExtractor


class HybridPredictor:
    """Hybrid predictor combining WeightedPredictor with SBM block structure."""

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

        # Initialize default SBM parameters
        self.sbm_params = {
            "num_blocks": 3,
            "min_block_size": 25,
            "max_block_size": 50,
            "intra_prob": 0.7,
            "inter_prob": 0.01,
            "prob_tolerance": 0.05,
        }

        # Update parameters from config if provided
        if config and "params" in config:
            params = config["params"]
            if hasattr(params, "num_blocks"):
                self.sbm_params["num_blocks"] = params.num_blocks
            if hasattr(params, "min_block_size"):
                self.sbm_params["min_block_size"] = params.min_block_size
            if hasattr(params, "max_block_size"):
                self.sbm_params["max_block_size"] = params.max_block_size
            if hasattr(params, "intra_prob"):
                self.sbm_params["intra_prob"] = params.intra_prob
            if hasattr(params, "inter_prob"):
                self.sbm_params["inter_prob"] = params.inter_prob
            if hasattr(params, "min_intra_prob"):
                self.sbm_params["min_intra_prob"] = params.min_intra_prob
            if hasattr(params, "max_intra_prob"):
                self.sbm_params["max_intra_prob"] = params.max_intra_prob
            if hasattr(params, "min_inter_prob"):
                self.sbm_params["min_inter_prob"] = params.min_inter_prob
            if hasattr(params, "max_inter_prob"):
                self.sbm_params["max_inter_prob"] = params.max_inter_prob

            # Set tolerance based on probability ranges
            if hasattr(params, "intra_prob_std"):
                self.sbm_params["prob_tolerance"] = params.intra_prob_std * 2

    def _detect_communities(self, G: nx.Graph) -> Dict[int, int]:
        """Detect communities using Louvain method with higher resolution."""
        try:
            # Use higher resolution to detect more compact communities
            return community.best_partition(G, resolution=1.2)
        except:
            # Fallback to connected components
            components = nx.connected_components(G)
            return {node: idx for idx, comp in enumerate(components) for node in comp}

    def _analyze_block_structure(
        self, G: nx.Graph, communities: Dict[int, int]
    ) -> Dict[str, Any]:
        """Analyze current block structure with enhanced metrics."""
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

        # Count edges with enhanced tracking
        for i, j in G.edges():
            bi, bj = communities[i], communities[j]
            if bi == bj:
                intra_edges[bi] = intra_edges.get(bi, 0) + 1
            else:
                key = tuple(sorted([bi, bj]))
                inter_edges[key] = inter_edges.get(key, 0) + 1

        # Calculate enhanced metrics
        intra_densities = {}
        avg_degrees = {}
        degree_stds = {}

        for block, size in block_sizes.items():
            # Intra-block density
            possible = size * (size - 1) / 2
            if possible > 0:
                intra_densities[block] = intra_edges.get(block, 0) / possible

            # Degree statistics per block
            if block in node_degrees and node_degrees[block]:
                avg_degrees[block] = np.mean(node_degrees[block])
                degree_stds[block] = np.std(node_degrees[block])

        # Inter-block density with enhanced metrics
        inter_densities = {}
        for (b1, b2), edges in inter_edges.items():
            possible = block_sizes[b1] * block_sizes[b2]
            if possible > 0:
                inter_densities[(b1, b2)] = edges / possible

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
        }

    def predict(
        self, history: List[Dict[str, Any]], horizon: int = 1
    ) -> List[np.ndarray]:
        """Predict future states using enhanced hybrid approach."""
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

            # 1. Get weighted prediction probabilities
            weighted_pred = self.base_predictor._compute_weighted_average(
                [st["adjacency"] for st in current_history[-self.n_history :]]
            )

            # 2. Detect and analyze communities
            communities = self._detect_communities(current_G)
            block_analysis = self._analyze_block_structure(current_G, communities)

            # 3. Adaptive thresholds based on block analysis
            intra_threshold = min(
                self.sbm_params.get("max_intra_prob", self.sbm_params["intra_prob"]),
                max(
                    self.sbm_params.get(
                        "min_intra_prob", self.sbm_params["intra_prob"] * 0.8
                    ),
                    block_analysis["avg_intra_density"] * 1.1,
                ),
            )
            inter_threshold = min(
                self.sbm_params.get(
                    "max_inter_prob", self.sbm_params["inter_prob"] * 1.5
                ),
                max(
                    self.sbm_params.get(
                        "min_inter_prob", self.sbm_params["inter_prob"] * 0.5
                    ),
                    block_analysis["avg_inter_density"] * 0.9,
                ),
            )

            # 4. Calculate target edges with density trend and block structure
            current_edges = current_G.number_of_edges()
            base_density = nx.density(current_G)

            # Adjust density based on block structure
            if self.model_type == "sbm":
                num_blocks = self.sbm_params["num_blocks"]
                expected_intra_edges = sum(
                    size * (size - 1) / 2 * self.sbm_params["intra_prob"]
                    for size in block_analysis["block_sizes"].values()
                )
                total_possible = n * (n - 1) / 2
                target_density = (expected_intra_edges / total_possible) + density_trend
            else:
                target_density = base_density + density_trend

            # Ensure density stays within reasonable bounds
            target_density = max(0.01, min(0.95, target_density))
            target_edges = int(target_density * (n * (n - 1) / 2))

            # 5. Enhanced edge probability calculation with configuration awareness
            edge_probs = []
            for i in range(n):
                for j in range(i + 1, n):
                    ci, cj = communities[i], communities[j]
                    base_prob = weighted_pred[i, j]

                    if ci == cj:  # Same community
                        block_density = block_analysis["intra_densities"].get(ci, 0)
                        if current_G.has_edge(i, j):
                            # Use max_intra_prob as ceiling for existing edges
                            prob = min(
                                self.sbm_params.get("max_intra_prob", 0.8),
                                max(base_prob, block_density) * 1.2,
                            )
                        else:
                            # Use min_intra_prob as floor for new edges
                            prob = max(
                                self.sbm_params.get("min_intra_prob", 0.4),
                                min(base_prob * 1.1, block_density * 0.9),
                            )
                    else:  # Different communities
                        block_key = tuple(sorted([ci, cj]))
                        block_density = block_analysis["inter_densities"].get(
                            block_key, 0
                        )
                        if current_G.has_edge(i, j):
                            # Use max_inter_prob as ceiling for existing edges
                            prob = min(
                                self.sbm_params.get("max_inter_prob", 0.2),
                                max(base_prob * 0.8, block_density),
                            )
                        else:
                            # Use min_inter_prob as floor for new edges
                            prob = max(
                                self.sbm_params.get("min_inter_prob", 0.005),
                                min(base_prob * 0.5, block_density * 0.7),
                            )

                    edge_probs.append((prob, i, j))

            # 6. Create prediction matrix with enhanced edge selection
            predicted = np.zeros((n, n))
            edges_added = 0

            # First maintain existing high-density intra-community edges
            for prob, i, j in sorted(edge_probs, reverse=True):
                if edges_added >= target_edges:
                    break

                ci, cj = communities[i], communities[j]
                if ci == cj and current_G.has_edge(i, j):
                    if prob > intra_threshold * 0.8:
                        predicted[i, j] = predicted[j, i] = 1
                        edges_added += 1

            # Then add new high-probability intra-community edges
            if edges_added < target_edges:
                for prob, i, j in sorted(edge_probs, reverse=True):
                    if edges_added >= target_edges:
                        break

                    ci, cj = communities[i], communities[j]
                    if ci == cj and not predicted[i, j]:
                        if prob > intra_threshold:
                            predicted[i, j] = predicted[j, i] = 1
                            edges_added += 1

            # Finally add essential inter-community edges
            if edges_added < target_edges:
                for prob, i, j in sorted(edge_probs, reverse=True):
                    if edges_added >= target_edges:
                        break

                    ci, cj = communities[i], communities[j]
                    if ci != cj and current_G.has_edge(i, j):
                        if prob > inter_threshold * 0.9:
                            predicted[i, j] = predicted[j, i] = 1
                            edges_added += 1

            predictions.append(predicted)
            current_history.append(
                {"adjacency": predicted, "graph": nx.from_numpy_array(predicted)}
            )

        return predictions
