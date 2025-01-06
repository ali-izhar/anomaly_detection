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
        config: Optional[Dict] = None
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
            binary=False  # We want probabilities, not binary
        )
        
        # Initialize SBM parameters with tighter constraints
        self.sbm_params = {
            "num_blocks": 4,
            "min_block_size": 20,
            "max_block_size": 35,
            "intra_prob": 0.3,
            "inter_prob": 0.05,
            "prob_tolerance": 0.1  # Tolerance for probability variations
        }
        
        if config and model_type == "sbm":
            params = config.get("params", {})
            if isinstance(params, dict):
                self.sbm_params.update({
                    "num_blocks": params.get("num_blocks", 4),
                    "min_block_size": params.get("min_block_size", 20),
                    "max_block_size": params.get("max_block_size", 35),
                    "intra_prob": params.get("intra_prob", 0.3),
                    "inter_prob": params.get("inter_prob", 0.05)
                })
    
    def _detect_communities(self, G: nx.Graph) -> Dict[int, int]:
        """Detect communities using Louvain method."""
        try:
            return community.best_partition(G)
        except:
            # Fallback to connected components
            components = nx.connected_components(G)
            return {node: idx for idx, comp in enumerate(components) for node in comp}
    
    def _analyze_block_structure(self, G: nx.Graph, communities: Dict[int, int]) -> Dict[str, Any]:
        """Analyze current block structure to adapt parameters."""
        n_blocks = max(communities.values()) + 1
        block_sizes = {}
        intra_edges = {}
        inter_edges = {}
        
        # Count nodes per block
        for node, block in communities.items():
            block_sizes[block] = block_sizes.get(block, 0) + 1
        
        # Count edges
        for i, j in G.edges():
            bi, bj = communities[i], communities[j]
            if bi == bj:
                intra_edges[bi] = intra_edges.get(bi, 0) + 1
            else:
                key = tuple(sorted([bi, bj]))
                inter_edges[key] = inter_edges.get(key, 0) + 1
        
        # Calculate densities
        intra_densities = {}
        for block, size in block_sizes.items():
            possible = size * (size - 1) / 2
            if possible > 0:
                intra_densities[block] = intra_edges.get(block, 0) / possible
        
        inter_densities = {}
        for (b1, b2), edges in inter_edges.items():
            possible = block_sizes[b1] * block_sizes[b2]
            if possible > 0:
                inter_densities[(b1, b2)] = edges / possible
        
        return {
            "block_sizes": block_sizes,
            "intra_densities": intra_densities,
            "inter_densities": inter_densities,
            "avg_intra_density": np.mean(list(intra_densities.values())) if intra_densities else 0,
            "avg_inter_density": np.mean(list(inter_densities.values())) if inter_densities else 0
        }
    
    def predict(
        self, history: List[Dict[str, Any]], horizon: int = 1
    ) -> List[np.ndarray]:
        """Predict future states using hybrid approach."""
        if len(history) < self.n_history:
            raise ValueError(f"Not enough history. Need {self.n_history}, got {len(history)}.")
        
        predictions = []
        current_history = list(history)
        
        for step in range(horizon):
            current_G = current_history[-1]["graph"]
            n = current_G.number_of_nodes()
            
            # 1. Get weighted prediction probabilities
            weighted_pred = self.base_predictor._compute_weighted_average(
                [st["adjacency"] for st in current_history[-self.n_history:]]
            )
            
            # 2. Detect communities and analyze structure
            communities = self._detect_communities(current_G)
            block_analysis = self._analyze_block_structure(current_G, communities)
            
            # Adapt thresholds based on observed structure
            intra_threshold = min(
                self.sbm_params["intra_prob"] * 1.1,
                block_analysis["avg_intra_density"] * 1.2
            )
            inter_threshold = max(
                self.sbm_params["inter_prob"] * 0.9,
                block_analysis["avg_inter_density"] * 0.8
            )
            
            # 3. Calculate target edges based on current graph
            current_edges = current_G.number_of_edges()
            target_edges = current_edges  # Maintain same density
            
            # 4. Adjust edge probabilities based on block structure
            edge_probs = []
            for i in range(n):
                for j in range(i + 1, n):
                    ci, cj = communities[i], communities[j]
                    
                    # Base probability from weighted predictor
                    base_prob = weighted_pred[i, j]
                    
                    # Get block-specific densities
                    if ci == cj:
                        block_density = block_analysis["intra_densities"].get(ci, 0)
                        target_prob = max(
                            self.sbm_params["intra_prob"],
                            min(block_density * 1.2, 0.8)  # Cap at 0.8
                        )
                    else:
                        block_key = tuple(sorted([ci, cj]))
                        block_density = block_analysis["inter_densities"].get(block_key, 0)
                        target_prob = min(
                            self.sbm_params["inter_prob"],
                            max(block_density * 0.8, 0.02)  # Floor at 0.02
                        )
                    
                    # Adjust probability based on block structure
                    if ci == cj:  # Same community
                        if base_prob >= target_prob * 0.8:
                            block_prob = base_prob * 1.2
                        else:
                            block_prob = base_prob * 0.5
                    else:  # Different communities
                        if base_prob >= target_prob * 2:
                            block_prob = base_prob * 0.8
                        else:
                            block_prob = base_prob * 0.2
                    
                    # Boost existing edges to maintain stability
                    if current_G.has_edge(i, j):
                        block_prob = min(1.0, block_prob * 1.3)
                    
                    edge_probs.append((block_prob, i, j))
            
            # 5. Sort edges by adjusted probability
            edge_probs.sort(reverse=True)
            
            # 6. Create prediction matrix
            predicted = np.zeros((n, n))
            edges_added = 0
            
            # First add high probability intra-community edges
            for prob, i, j in edge_probs:
                if edges_added >= target_edges:
                    break
                    
                ci, cj = communities[i], communities[j]
                if ci == cj:  # Same community
                    block_density = block_analysis["intra_densities"].get(ci, 0)
                    if prob > max(intra_threshold, block_density * 0.8):
                        predicted[i, j] = predicted[j, i] = 1
                        edges_added += 1
            
            # Then add high probability inter-community edges
            if edges_added < target_edges:
                for prob, i, j in edge_probs:
                    if edges_added >= target_edges:
                        break
                        
                    ci, cj = communities[i], communities[j]
                    if ci != cj:  # Different communities
                        block_key = tuple(sorted([ci, cj]))
                        block_density = block_analysis["inter_densities"].get(block_key, 0)
                        if prob > max(inter_threshold, block_density * 1.2):
                            predicted[i, j] = predicted[j, i] = 1
                            edges_added += 1
            
            predictions.append(predicted)
            current_history.append({
                "adjacency": predicted,
                "graph": nx.from_numpy_array(predicted)
            })
        
        return predictions
