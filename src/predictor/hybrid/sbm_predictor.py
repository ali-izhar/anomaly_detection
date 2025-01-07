# src/predictor/hybrid/sbm_predictor.py

import networkx as nx
import numpy as np
from typing import List, Dict, Any
import community

from .base import HybridPredictor


class SBMPredictor(HybridPredictor):
    """Hybrid predictor specialized for Stochastic Block Models."""

    def __init__(
        self,
        intra_threshold=0.1,  # Optimized: stable threshold for intra-block edges
        inter_threshold=0.2,  # Optimized: best balance for inter-block edges
        intra_hist_weight=0.1,  # Optimized: low weight on history for intra-block
        inter_hist_weight=0.0,  # Optimized: no history weight for inter-block
        **kwargs
    ):
        super().__init__(model_type="sbm", **kwargs)
        self.model_params = self._initialize_model_params()
        self.intra_threshold = intra_threshold
        self.inter_threshold = inter_threshold
        self.intra_hist_weight = intra_hist_weight
        self.inter_hist_weight = inter_hist_weight

    def _initialize_model_params(self) -> Dict:
        """Initialize SBM-specific parameters."""
        if not self.config or "params" not in self.config:
            return {}

        params = self.config["params"]
        return {
            "num_blocks": params.num_blocks,
            "min_block_size": params.min_block_size,
            "max_block_size": params.max_block_size,
            "intra_prob": params.intra_prob,
            "inter_prob": params.inter_prob,
            "min_intra_prob": params.min_intra_prob,
            "max_intra_prob": params.max_intra_prob,
            "min_inter_prob": params.min_inter_prob,
            "max_inter_prob": params.max_inter_prob,
            "intra_prob_std": params.intra_prob_std,
            "inter_prob_std": params.inter_prob_std,
        }

    def _detect_communities(self, G: nx.Graph) -> Dict[int, int]:
        """Detect communities for SBM with resolution matching num_blocks."""
        resolution = 1.0
        if self.model_params:
            # Adjust resolution to try to get desired number of blocks
            target_blocks = self.model_params.get("num_blocks", 3)
            resolution = 1.2 if target_blocks > 2 else 0.8
        return community.best_partition(G, resolution=resolution)

    def _compute_block_likelihood(
        self, block1: set, block2: set, adj: np.ndarray
    ) -> float:
        """Compute likelihood between blocks using maximum likelihood estimation."""
        if not block1 or not block2:
            return 0.0

        # Count existing edges between blocks
        l_b1_b2 = 0  # actual edges
        eta_b1_b2 = len(block1) * len(block2)  # possible edges
        if block1 == block2:
            eta_b1_b2 = len(block1) * (len(block1) - 1) // 2

        for i in block1:
            for j in block2:
                if i != j and adj[i, j] > 0:
                    l_b1_b2 += 1

        # Compute block density (ρ)
        rho = l_b1_b2 / eta_b1_b2 if eta_b1_b2 > 0 else 0

        # Maximum likelihood estimation
        if rho > 0 and rho < 1:
            likelihood = (rho**l_b1_b2) * ((1 - rho) ** (eta_b1_b2 - l_b1_b2))
        else:
            likelihood = 0

        return likelihood

    def _post_prune_edges(
        self,
        P_initial: np.ndarray,
        history: List[Dict[str, Any]],
        weighted_pred: np.ndarray,
    ) -> np.ndarray:
        """Post-process prediction by pruning false positive edges using structural features."""
        n = P_initial.shape[0]
        G_prev = history[-1]["graph"]
        communities = self._detect_communities(G_prev)

        # Get EXACT target edge count from weighted predictor
        G_weighted = nx.from_numpy_array(weighted_pred)
        target_edges = G_weighted.number_of_edges()

        # Score each predicted edge
        edge_scores = []
        for i in range(n):
            for j in range(i + 1, n):
                if P_initial[i, j] > 0:  # Only score edges that were predicted
                    score = 0.0

                    # 1. Community membership (40%)
                    same_community = communities[i] == communities[j]
                    if same_community:
                        block = {
                            k for k, c in communities.items() if c == communities[i]
                        }
                        block_density = self._compute_block_likelihood(
                            block, block, nx.to_numpy_array(G_prev)
                        )
                        score += 0.4 * (block_density + 1.0)  # Scale by block density

                    # 2. Historical presence (30%)
                    hist_confidence = self._compute_edge_confidence(history, i, j)
                    score += 0.3 * hist_confidence

                    # 3. Common neighbors (20%)
                    common = set(G_prev.neighbors(i)) & set(G_prev.neighbors(j))
                    possible = set(G_prev.neighbors(i)) | set(G_prev.neighbors(j))
                    if possible:  # Only compute if there are possible neighbors
                        common_ratio = len(common) / len(possible)
                        score += 0.2 * common_ratio

                    # 4. Weighted predictor confidence (10%)
                    if (
                        common
                    ):  # Only compute local density if there are common neighbors
                        common_list = list(common)
                        local_density = np.mean(
                            weighted_pred[common_list, :][:, common_list]
                        )
                        weighted_score = weighted_pred[i, j] * (1 + local_density)
                    else:
                        weighted_score = weighted_pred[
                            i, j
                        ]  # Just use base score if no common neighbors
                    score += 0.1 * weighted_score

                    edge_scores.append((score, i, j))

        # Sort edges by score
        edge_scores.sort(reverse=True)

        # Create final prediction matrix with EXACTLY target_edges
        P_final = np.zeros((n, n))

        # First, add edges that exist in both predictions
        edges_added = 0
        for score, i, j in edge_scores:
            if edges_added >= target_edges:
                break
            if weighted_pred[i, j] > 0:  # Edge exists in weighted prediction
                P_final[i, j] = P_final[j, i] = 1
                edges_added += 1

        # Then add remaining highest scoring edges until we hit target_edges
        if edges_added < target_edges:
            for score, i, j in edge_scores:
                if P_final[i, j] == 0:  # Only if not already added
                    P_final[i, j] = P_final[j, i] = 1
                    edges_added += 1
                    if edges_added >= target_edges:
                        break

        return P_final

    def _make_single_prediction(self, history: List[Dict[str, Any]]) -> np.ndarray:
        """Make prediction with post-pruning of false positives."""
        # Get weighted prediction first as baseline
        weighted_pred = self.base_predictor.predict(history, horizon=1)[0]

        # Make initial prediction with high coverage
        P_initial = super()._make_single_prediction(history)

        # Post-prune edges
        P_final = self._post_prune_edges(P_initial, history, weighted_pred)

        return P_final
