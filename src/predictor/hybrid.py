import networkx as nx
import numpy as np
from typing import List, Dict, Any, Optional
import community

from .weighted import WeightedPredictor
from graph.features import NetworkFeatureExtractor


class HybridPredictor:
    """
    Hybrid predictor combining weighted averaging with structural role preservation.

    Updated to:
      1. Separate Weighted vs. Model-Based probability matrices.
      2. Combine them via a mixing parameter alpha.
      3. Enforce local node-degree constraints.
      4. Enforce community/block-level constraints (SBM, RCP, etc.).
      5. Post-process for clustering/connectivity if needed.
      6. Optionally adapt alpha over time based on prediction error.
    """

    def __init__(
        self,
        n_history: int = 5,
        weights: Optional[np.ndarray] = None,
        adaptive: bool = True,
        model_type: str = "ba",
        config: Optional[Dict] = None,
        alpha: float = 0.5,  # mixing parameter for Weighted vs. Model-based
        enforce_node_degrees: bool = True,
        enforce_block_structure: bool = True,
        enforce_clustering: bool = False,
        target_clustering: float = None,
    ):
        """
        Parameters
        ----------
        n_history : int
            Number of historical snapshots to consider.
        weights : np.ndarray, optional
            Weight vector for WeightedPredictor.
        adaptive : bool
            Whether to adapt weights after each prediction step.
        model_type : str
            One of ['ba', 'er', 'ws', 'sbm', 'rcp', 'lfr', ...].
        config : Dict, optional
            Model-specific configuration dictionary.
        alpha : float
            Mixing parameter, e.g.,
              P_final = alpha * P_model + (1 - alpha) * P_weighted
        enforce_node_degrees : bool
            If True, enforce local node-degree constraints in the probability matrix.
        enforce_block_structure : bool
            If True, adjust probabilities to preserve community or core-periphery structure.
        enforce_clustering : bool
            If True, do a final pass to adjust probabilities for a target clustering coefficient.
        target_clustering : float
            Desired global clustering; used if enforce_clustering=True.
        """
        self.n_history = n_history
        self.weights = weights
        self.adaptive = adaptive
        self.model_type = model_type
        self.config = config

        # Mixing parameter for Weighted vs. Model-based probabilities
        self.alpha = alpha

        self.enforce_node_degrees = enforce_node_degrees
        self.enforce_block_structure = enforce_block_structure
        self.enforce_clustering = enforce_clustering
        self.target_clustering = target_clustering

        # Initialize weighted predictor for partial (historical) probabilities
        self.base_predictor = WeightedPredictor(
            n_history=n_history,
            weights=weights,
            adaptive=adaptive,
            binary=False,  # We want probabilities, not binary
        )

        # Initialize model-specific parameters
        self.model_params = self._initialize_model_params()

        # Feature extractor for local metrics if needed
        self.feature_extractor = NetworkFeatureExtractor()

    def _initialize_model_params(self) -> Dict:
        """Initialize parameters based on model type from config."""
        if not self.config or "params" not in self.config:
            return self._get_default_params()

        params = self.config["params"]
        model_params = {}

        if self.model_type == "sbm":
            # Use exact parameters from config for SBM
            model_params.update({
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
                "inter_prob_std": params.inter_prob_std
            })
        elif self.model_type == "ba":
            model_params.update(
                {
                    "m": getattr(params, "m", 3),
                    "min_m": getattr(params, "min_m", 2),
                    "max_m": getattr(params, "max_m", 4),
                    "preferential_exp": getattr(params, "preferential_exp", 1.0),
                }
            )
        elif self.model_type == "er":
            model_params.update(
                {
                    "prob": getattr(params, "prob", 0.15),
                    "min_prob": getattr(params, "min_prob", 0.1),
                    "max_prob": getattr(params, "max_prob", 0.2),
                }
            )
        elif self.model_type == "ws":
            model_params.update(
                {
                    "k": getattr(params, "k_nearest", 6),
                    "rewire_prob": getattr(params, "rewire_prob", 0.1),
                    # etc. ...
                }
            )
        elif self.model_type == "rcp":
            model_params.update(
                {
                    "core_size": getattr(params, "core_size", params.n // 5),
                    "core_prob": getattr(params, "core_prob", 0.8),
                    "periph_prob": getattr(params, "periph_prob", 0.05),
                    "core_periph_prob": getattr(params, "core_periph_prob", 0.2),
                }
            )
        else:
            model_params = self._get_default_params()

        return model_params

    def _get_default_params(self) -> Dict:
        """Get default parameters for unknown or generic model types."""
        return {
            "prob_tolerance": 0.05,
            "community_resolution": 1.0,
        }

    def _detect_communities(self, G: nx.Graph) -> Dict[int, int]:
        """
        Detect communities using method appropriate for the model type.
        Returns a dict {node: community_id}.
        """
        try:
            if self.model_type == "sbm":
                # Typically want a higher resolution for clearer blocks
                return community.best_partition(G, resolution=1.2)
            elif self.model_type == "rcp":
                # Heuristic: top-degree nodes -> core, others -> periphery
                degrees = dict(G.degree())
                sorted_nodes = sorted(
                    degrees.keys(), key=lambda x: degrees[x], reverse=True
                )
                core_size = self.model_params.get("core_size", len(G) // 5)
                return {
                    node: 0 if i < core_size else 1
                    for i, node in enumerate(sorted_nodes)
                }
            elif self.model_type == "ba":
                # For BA, still can use a standard partition
                return community.best_partition(G, resolution=0.8)
            else:
                # Default
                return community.best_partition(G, resolution=1.0)
        except:
            # Fallback to connected components if partition fails
            components = nx.connected_components(G)
            return {node: idx for idx, comp in enumerate(components) for node in comp}

    def _analyze_block_structure(
        self, G: nx.Graph, communities: Dict[int, int]
    ) -> Dict[str, Any]:
        """
        Analyze network structure based on the detected communities.
        Returns metrics like block sizes, intra/inter densities, etc.
        """
        n_blocks = max(communities.values()) + 1
        block_sizes = {}
        intra_edges = {}
        inter_edges = {}

        # Count nodes in each block
        for node in G.nodes():
            b = communities[node]
            block_sizes[b] = block_sizes.get(b, 0) + 1

        # Edge counting
        for i, j in G.edges():
            bi, bj = communities[i], communities[j]
            if bi == bj:
                intra_edges[bi] = intra_edges.get(bi, 0) + 1
            else:
                key = tuple(sorted([bi, bj]))
                inter_edges[key] = inter_edges.get(key, 0) + 1

        # Intra-block densities
        intra_densities = {}
        for b, size in block_sizes.items():
            possible = size * (size - 1) / 2
            if possible > 0:
                intra_densities[b] = intra_edges.get(b, 0) / possible

        # Inter-block densities
        inter_densities = {}
        for (b1, b2), edge_count in inter_edges.items():
            possible = block_sizes[b1] * block_sizes[b2]
            inter_densities[(b1, b2)] = edge_count / possible if possible else 0

        return {
            "block_sizes": block_sizes,
            "intra_densities": intra_densities,
            "inter_densities": inter_densities,
            "avg_intra_density": (
                np.mean(list(intra_densities.values())) if intra_densities else 0
            ),
            "avg_inter_density": (
                np.mean(list(inter_densities.values())) if inter_densities else 0
            ),
        }

    def _compute_model_probability(self, G: nx.Graph) -> np.ndarray:
        """Compute model-based probability matrix with stricter SBM control."""
        n = G.number_of_nodes()
        model_probs = np.zeros((n, n))
        
        if self.model_type == "sbm":
            communities = self._detect_communities(G)
            
            # Get exact parameters from model config
            intra_prob = self.model_params["intra_prob"]  # 0.7
            inter_prob = self.model_params["inter_prob"]  # 0.01
            
            for i in range(n):
                for j in range(i + 1, n):  # Avoid self-loops
                    if communities[i] == communities[j]:
                        # Intra-block probability
                        model_probs[i, j] = model_probs[j, i] = intra_prob
                    else:
                        # Inter-block probability
                        model_probs[i, j] = model_probs[j, i] = inter_prob
        
        # Ensure no self-loops
        np.fill_diagonal(model_probs, 0)
        
        return model_probs

    def _enforce_node_degree_constraints(
        self, prob_matrix: np.ndarray, target_degs: np.ndarray
    ) -> np.ndarray:
        """
        Adjust probability matrix so that expected degree of each node
        is close to target_degs.
        """
        n = prob_matrix.shape[0]
        # Iterative approach: scale row i by factor = target_degs[i] / current_deg_i
        # current_deg_i = sum(prob_matrix[i,:])
        # Keep repeated passes until convergence or iteration limit
        for _ in range(2):  # a few passes
            row_sums = prob_matrix.sum(axis=1)
            for i in range(n):
                if row_sums[i] > 1e-9:
                    scale = target_degs[i] / row_sums[i]
                    # Scale row i
                    prob_matrix[i, :] *= scale
                    prob_matrix[:, i] *= scale  # keep symmetry
            # Keep probabilities in [0,1]
            np.clip(prob_matrix, 0, 1, out=prob_matrix)
        return prob_matrix

    def _postprocess_clustering(
        self, prob_matrix: np.ndarray, target_clustering: float
    ) -> np.ndarray:
        """
        Example: Attempt to nudge probabilities to match a target global clustering.
        This is a simplistic placeholder. In practice, you'd do a more
        sophisticated triad-based or link re-weighting to adjust local clustering.
        """
        if target_clustering is None:
            return prob_matrix

        # Compare current expected clustering to target, adjust if off
        # We'll do only a tiny nudge for demonstration
        # (In practice, you'd compute an approximate expected clustering from prob_matrix)
        n = prob_matrix.shape[0]
        # A naive approximation of clustering from prob_matrix could be done,
        # but here let's do a simpler approach: if we want more clustering,
        # slightly boost triadic edges, otherwise reduce them.
        return prob_matrix  # Stub: implement if needed

    def _binarize_probability_matrix(
        self,
        prob_matrix: np.ndarray,
        target_metrics: Dict[str, Any],
        max_iterations: int = 5
    ) -> np.ndarray:
        """
        Enhanced binarization that:
        1. Uses target metrics to guide edge selection
        2. Iteratively refines the binary matrix
        3. Preserves multiple network properties
        4. Handles different model types differently
        """
        n = prob_matrix.shape[0]
        
        # Initial binarization using probability ranking
        binary = self._initial_binarization(prob_matrix, target_metrics["density"])
        best_binary = binary.copy()
        best_score = self._evaluate_binary_matrix(best_binary, target_metrics)
        
        for _ in range(max_iterations):
            # 1. Try edge swaps to improve metrics
            binary = self._optimize_edge_swaps(binary, prob_matrix, target_metrics)
            
            # 2. Enforce model-specific constraints
            binary = self._enforce_model_constraints(binary, target_metrics)
            
            # 3. Evaluate current solution
            score = self._evaluate_binary_matrix(binary, target_metrics)
            
            if score < best_score:
                best_score = score
                best_binary = binary.copy()
        
        return best_binary

    def _initial_binarization(
        self,
        prob_matrix: np.ndarray,
        target_density: float
    ) -> np.ndarray:
        """Initial binarization using probability ranking and target density."""
        n = prob_matrix.shape[0]
        target_edges = int(target_density * n * (n - 1) / 2)
        
        # Get upper triangular indices and probabilities
        triu_indices = np.triu_indices(n, k=1)
        probs = prob_matrix[triu_indices]
        
        # Sort edges by probability
        edges = sorted(
            zip(probs, triu_indices[0], triu_indices[1]),
            key=lambda x: x[0],
            reverse=True
        )
        
        # Initialize binary matrix
        binary = np.zeros((n, n), dtype=int)
        
        # Add top edges
        for _, i, j in edges[:target_edges]:
            binary[i, j] = binary[j, i] = 1
        
        return binary

    def _optimize_edge_swaps(
        self,
        binary: np.ndarray,
        prob_matrix: np.ndarray,
        target_metrics: Dict[str, Any]
    ) -> np.ndarray:
        """Optimize binary matrix through edge swaps."""
        n = binary.shape[0]
        improved = binary.copy()
        
        # Get current edges and non-edges
        edges = list(zip(*np.where(np.triu(improved) > 0)))
        non_edges = list(zip(*np.where(np.triu(improved) == 0)))
        
        # Try random edge swaps
        num_attempts = min(len(edges), 100)  # Limit attempts for efficiency
        
        for _ in range(num_attempts):
            # Select random edge to remove
            e1 = edges[np.random.randint(len(edges))]
            # Select random non-edge to add
            e2 = non_edges[np.random.randint(len(non_edges))]
            
            # Try swap
            temp = improved.copy()
            temp[e1[0], e1[1]] = temp[e1[1], e1[0]] = 0
            temp[e2[0], e2[1]] = temp[e2[1], e2[0]] = 1
            
            # Evaluate swap
            if self._evaluate_binary_matrix(temp, target_metrics) < self._evaluate_binary_matrix(improved, target_metrics):
                improved = temp
                # Update edges and non-edges lists
                edges.remove(e1)
                edges.append(e2)
                non_edges.remove(e2)
                non_edges.append(e1)
        
        return improved

    def _enforce_model_constraints(
        self,
        binary: np.ndarray,
        target_metrics: Dict[str, Any]
    ) -> np.ndarray:
        """Enforce model-specific constraints on binary matrix."""
        if self.model_type == "ba":
            return self._enforce_ba_constraints(binary, target_metrics)
        elif self.model_type == "rcp":
            return self._enforce_rcp_constraints(binary, target_metrics)
        elif self.model_type == "sbm":
            return self._enforce_sbm_constraints(binary, target_metrics)
        else:
            return binary

    def _enforce_ba_constraints(
        self,
        binary: np.ndarray,
        target_metrics: Dict[str, Any]
    ) -> np.ndarray:
        """Enforce BA model constraints (power-law degree distribution)."""
        n = binary.shape[0]
        improved = binary.copy()
        
        # Get current degree sequence
        degrees = np.sum(improved, axis=1)
        target_degrees = target_metrics.get("degree_sequence", None)
        
        if target_degrees is not None:
            # Sort nodes by degree difference from target
            nodes = np.argsort(np.abs(degrees - target_degrees))
            
            for i in nodes:
                if degrees[i] > target_degrees[i]:
                    # Remove edges from highest degree neighbors
                    neighbors = np.where(improved[i, :] > 0)[0]
                    neighbor_degrees = degrees[neighbors]
                    for j in neighbors[np.argsort(neighbor_degrees)[::-1]]:
                        if degrees[i] > target_degrees[i]:
                            improved[i, j] = improved[j, i] = 0
                            degrees[i] -= 1
                            degrees[j] -= 1
                elif degrees[i] < target_degrees[i]:
                    # Add edges to highest degree non-neighbors
                    non_neighbors = np.where(improved[i, :] == 0)[0]
                    non_neighbor_degrees = degrees[non_neighbors]
                    for j in non_neighbors[np.argsort(non_neighbor_degrees)[::-1]]:
                        if degrees[i] < target_degrees[i]:
                            improved[i, j] = improved[j, i] = 1
                            degrees[i] += 1
                            degrees[j] += 1
        
        return improved

    def _enforce_rcp_constraints(
        self,
        binary: np.ndarray,
        target_metrics: Dict[str, Any]
    ) -> np.ndarray:
        """Enforce RCP model constraints (core-periphery structure)."""
        n = binary.shape[0]
        improved = binary.copy()
        
        # Identify core and periphery based on degree
        degrees = np.sum(improved, axis=1)
        core_size = self.model_params.get("core_size", n // 5)
        core_nodes = np.argsort(degrees)[::-1][:core_size]
        periph_nodes = np.argsort(degrees)[::-1][core_size:]
        
        # Enforce core density
        core_density = self.model_params.get("core_prob", 0.8)
        periph_density = self.model_params.get("periph_prob", 0.1)
        core_periph_density = self.model_params.get("core_periph_prob", 0.3)
        
        # Adjust core connections
        for i in core_nodes:
            for j in core_nodes:
                if i < j:
                    if np.random.random() < core_density:
                        improved[i, j] = improved[j, i] = 1
                    else:
                        improved[i, j] = improved[j, i] = 0
        
        # Adjust periphery connections
        for i in periph_nodes:
            for j in periph_nodes:
                if i < j:
                    if np.random.random() < periph_density:
                        improved[i, j] = improved[j, i] = 1
                    else:
                        improved[i, j] = improved[j, i] = 0
        
        # Adjust core-periphery connections
        for i in core_nodes:
            for j in periph_nodes:
                if np.random.random() < core_periph_density:
                    improved[i, j] = improved[j, i] = 1
                else:
                    improved[i, j] = improved[j, i] = 0
        
        return improved

    def _enforce_sbm_constraints(
        self,
        binary: np.ndarray,
        target_metrics: Dict[str, Any]
    ) -> np.ndarray:
        """Enforce SBM model constraints (block structure)."""
        n = binary.shape[0]
        improved = binary.copy()
        
        # Get community structure
        G = nx.from_numpy_array(improved)
        communities = self._detect_communities(G)
        block_info = self._analyze_block_structure(G, communities)
        
        # Enforce block densities
        for i in range(n):
            for j in range(i + 1, n):
                bi, bj = communities[i], communities[j]
                if bi == bj:
                    # Intra-block connection
                    target_density = block_info["intra_densities"].get(bi, 0.7)
                else:
                    # Inter-block connection
                    block_pair = tuple(sorted([bi, bj]))
                    target_density = block_info["inter_densities"].get(block_pair, 0.1)
                
                if np.random.random() < target_density:
                    improved[i, j] = improved[j, i] = 1
                else:
                    improved[i, j] = improved[j, i] = 0
        
        return improved

    def _evaluate_binary_matrix(
        self,
        binary: np.ndarray,
        target_metrics: Dict[str, Any]
    ) -> float:
        """Evaluate how well binary matrix matches target metrics."""
        G = nx.from_numpy_array(binary)
        score = 0.0
        
        try:
            # Density difference
            current_density = nx.density(G)
            score += abs(current_density - target_metrics["density"])
            
            # Degree sequence difference
            if "degree_sequence" in target_metrics:
                current_degrees = sorted(dict(G.degree()).values(), reverse=True)
                target_degrees = target_metrics["degree_sequence"]
                score += np.mean(np.abs(np.array(current_degrees) - np.array(target_degrees)))
            
            # Clustering coefficient difference
            if "clustering" in target_metrics:
                current_clustering = nx.average_clustering(G)
                score += abs(current_clustering - target_metrics["clustering"])
            
            # Path length difference
            if "path_length" in target_metrics:
                try:
                    current_path_length = nx.average_shortest_path_length(G)
                    score += abs(current_path_length - target_metrics["path_length"])
                except:
                    score += 1.0  # Penalty for disconnected graph
            
            # Model-specific metrics
            if self.model_type == "rcp":
                score += self._evaluate_rcp_structure(G, target_metrics)
            elif self.model_type == "sbm":
                score += self._evaluate_sbm_structure(G, target_metrics)
            
        except:
            return float('inf')
        
        return score

    def _evaluate_rcp_structure(
        self,
        G: nx.Graph,
        target_metrics: Dict[str, Any]
    ) -> float:
        """Evaluate core-periphery structure."""
        score = 0.0
        degrees = np.array(list(dict(G.degree()).values()))
        core_size = self.model_params.get("core_size", len(G) // 5)
        
        # Check core density
        core_nodes = np.argsort(degrees)[::-1][:core_size]
        core_subgraph = G.subgraph(core_nodes)
        core_density = nx.density(core_subgraph)
        score += abs(core_density - self.model_params.get("core_prob", 0.8))
        
        return score

    def _evaluate_sbm_structure(
        self,
        G: nx.Graph,
        target_metrics: Dict[str, Any]
    ) -> float:
        """Evaluate stochastic block model structure."""
        score = 0.0
        communities = self._detect_communities(G)
        block_info = self._analyze_block_structure(G, communities)
        
        # Compare block densities
        if "block_densities" in target_metrics:
            target_densities = target_metrics["block_densities"]
            current_densities = block_info["intra_densities"]
            for block in target_densities:
                if block in current_densities:
                    score += abs(target_densities[block] - current_densities[block])
        
        return score

    def _fit_model_parameters(self, history: List[Dict[str, Any]]) -> Dict:
        """Fit model parameters to historical data."""
        # Get the most recent graph
        recent_G = history[-1]["graph"]
        n = recent_G.number_of_nodes()
        
        # Calculate key metrics for model selection
        metrics = {
            "density": nx.density(recent_G),
            "clustering": nx.average_clustering(recent_G),
            "degree_dist": np.array(list(dict(recent_G.degree()).values())),
            "assortativity": nx.degree_assortativity_coefficient(recent_G),
        }
        
        # Fit model-specific parameters
        if self.model_type == "ba":
            # Estimate m from average degree
            avg_degree = np.mean(metrics["degree_dist"])
            m_est = int(avg_degree / 2)
            return {
                "m": m_est,
                "preferential_exp": self._estimate_pa_exponent(metrics["degree_dist"])
            }
        
        elif self.model_type == "sbm":
            communities = self._detect_communities(recent_G)
            block_info = self._analyze_block_structure(recent_G, communities)
            return {
                "num_blocks": len(block_info["block_sizes"]),
                "intra_prob": block_info["avg_intra_density"],
                "inter_prob": block_info["avg_inter_density"]
            }
        
        elif self.model_type == "rcp":
            # Estimate core size and probabilities
            degrees = metrics["degree_dist"]
            core_size = self._estimate_core_size(degrees)
            core_periph_structure = self._analyze_core_periphery(recent_G, core_size)
            return {
                "core_size": core_size,
                "core_prob": core_periph_structure["core_density"],
                "periph_prob": core_periph_structure["periph_density"],
                "core_periph_prob": core_periph_structure["core_periph_density"]
            }
        
        return self._get_default_params()

    def _estimate_pa_exponent(self, degrees: np.ndarray) -> float:
        """Estimate preferential attachment exponent from degree distribution."""
        # Use power law fit on degree distribution
        from scipy import stats
        degrees = degrees[degrees > 0]
        log_degrees = np.log(degrees)
        log_counts = np.log(stats.itemfreq(degrees)[:, 1])
        slope, _, _, _, _ = stats.linregress(log_degrees, log_counts)
        return abs(slope)

    def _estimate_core_size(self, degrees: np.ndarray) -> int:
        """Estimate core size using degree distribution."""
        # Use elbow method or natural breaks in degree distribution
        from sklearn.cluster import KMeans
        X = degrees.reshape(-1, 1)
        kmeans = KMeans(n_clusters=2).fit(X)
        core_mask = kmeans.labels_ == np.argmax(kmeans.cluster_centers_)
        return int(np.sum(core_mask))

    def _compute_combined_probability(
        self, 
        weighted_prob: np.ndarray, 
        model_prob: np.ndarray, 
        last_G: nx.Graph
    ) -> np.ndarray:
        """
        Combine weighted historical and model-based probabilities more intelligently.
        Uses adaptive mixing based on local network properties.
        """
        n = weighted_prob.shape[0]
        combined = np.zeros((n, n))
        
        try:
            # Get node-level metrics to guide mixing
            degrees = dict(last_G.degree())
            clustering = nx.clustering(last_G)
            
            # Handle betweenness calculation more carefully
            try:
                betweenness = nx.betweenness_centrality(last_G)
            except:
                # Fallback if betweenness calculation fails
                betweenness = {node: 0.0 for node in last_G.nodes()}
            
            for i in range(n):
                for j in range(i + 1, n):
                    # Compute local mixing parameter based on node properties
                    local_alpha = self._compute_local_alpha(
                        i, j,
                        degrees[i], degrees[j],
                        clustering[i], clustering[j],
                        betweenness[i], betweenness[j]
                    )
                    
                    # Mix probabilities with local parameter
                    prob = local_alpha * model_prob[i, j] + (1 - local_alpha) * weighted_prob[i, j]
                    combined[i, j] = combined[j, i] = prob
        
        except Exception as e:
            # Fallback to simple mixing if metrics calculation fails
            combined = self.alpha * model_prob + (1 - self.alpha) * weighted_prob
        
        return combined

    def _compute_local_alpha(
        self,
        i: int,
        j: int,
        deg_i: int,
        deg_j: int,
        clust_i: float,
        clust_j: float,
        btw_i: float,
        btw_j: float
    ) -> float:
        """
        Compute local mixing parameter based on node properties.
        Returns alpha in [0,1] for mixing model vs. historical probabilities.
        """
        # Base alpha from global parameter
        alpha = self.alpha
        
        try:
            if self.model_type == "ba":
                # For BA, trust model more for high-degree nodes
                max_deg = max(deg_i, deg_j)
                # Use log1p to handle zero degrees
                alpha *= (1 + np.log1p(max_deg) / 10)
            
            elif self.model_type == "rcp":
                # For RCP, trust model more for core-core or core-periphery edges
                if max(btw_i, btw_j) > 0.5:  # likely core node
                    alpha *= 1.2
            
            elif self.model_type == "sbm":
                # For SBM, trust model more for nodes with similar clustering
                if abs(clust_i - clust_j) < 0.1:
                    alpha *= 1.2
        except:
            # Fallback to base alpha if any calculation fails
            pass
            
        # Ensure alpha stays in [0,1]
        return np.clip(alpha, 0, 1)

    def _enforce_local_structure(
        self,
        prob_matrix: np.ndarray,
        G: nx.Graph,
        iterations: int = 3
    ) -> np.ndarray:
        """
        Enforce local structural features including:
        - Node degrees
        - Local clustering coefficients
        - Betweenness centrality
        - Degree assortativity
        """
        n = prob_matrix.shape[0]
        
        # Get target metrics from current graph
        target_degrees = np.array([G.degree(i) for i in range(n)], dtype=float)
        target_clustering = nx.clustering(G)
        target_betweenness = nx.betweenness_centrality(G)
        
        # Create working copy
        P = prob_matrix.copy()
        
        for _ in range(iterations):
            # 1. Degree constraint
            P = self._enforce_degree_sequence(P, target_degrees)
            
            # 2. Local clustering
            P = self._enforce_local_clustering(P, target_clustering)
            
            # 3. Betweenness preservation
            P = self._enforce_betweenness(P, target_betweenness)
            
            # 4. Degree assortativity
            P = self._enforce_assortativity(P, G)
            
            # Ensure symmetry and valid probabilities
            P = 0.5 * (P + P.T)
            np.clip(P, 0, 1, out=P)
        
        return P

    def _enforce_degree_sequence(
        self,
        P: np.ndarray,
        target_degrees: np.ndarray
    ) -> np.ndarray:
        """Enhanced degree sequence enforcement using iterative scaling."""
        n = P.shape[0]
        P_new = P.copy()
        
        # Iterative Proportional Fitting
        for _ in range(2):
            # Row scaling
            row_sums = P_new.sum(axis=1)
            for i in range(n):
                if row_sums[i] > 0:
                    scale = target_degrees[i] / row_sums[i]
                    P_new[i, :] *= scale
            
            # Column scaling (for symmetry)
            col_sums = P_new.sum(axis=0)
            for j in range(n):
                if col_sums[j] > 0:
                    scale = target_degrees[j] / col_sums[j]
                    P_new[:, j] *= scale
                    
            # Symmetrize
            P_new = 0.5 * (P_new + P_new.T)
            
        return P_new

    def _enforce_local_clustering(
        self,
        P: np.ndarray,
        target_clustering: Dict[int, float]
    ) -> np.ndarray:
        """Adjust probabilities to preserve local clustering coefficients."""
        n = P.shape[0]
        P_new = P.copy()
        
        for i in range(n):
            # Get neighbors with high connection probability
            likely_neighbors = np.where(P_new[i, :] > 0.5)[0]
            target_c = target_clustering[i]
            
            # For each pair of likely neighbors
            for j in likely_neighbors:
                for k in likely_neighbors:
                    if j < k:  # avoid duplicates
                        # Increase triangle probability if needed
                        if target_c > 0.5:  # high clustering target
                            P_new[j, k] = P_new[k, j] = max(
                                P_new[j, k],
                                0.5 * (P_new[i, j] + P_new[i, k])
                            )
        
        return P_new

    def _enforce_betweenness(
        self,
        P: np.ndarray,
        target_betweenness: Dict[int, float]
    ) -> np.ndarray:
        """Adjust probabilities to preserve betweenness centrality."""
        n = P.shape[0]
        P_new = P.copy()
        
        # Sort nodes by target betweenness
        sorted_nodes = sorted(
            range(n),
            key=lambda x: target_betweenness[x],
            reverse=True
        )
        
        # High betweenness nodes should connect to many communities
        communities = self._detect_communities(nx.from_numpy_array(P > 0.5))
        
        for node in sorted_nodes[:int(n * 0.2)]:  # focus on top 20% nodes
            comm_connections = {}
            for j in range(n):
                if j != node:
                    comm_connections[communities[j]] = comm_connections.get(communities[j], 0) + P_new[node, j]
            
            # Ensure high betweenness nodes connect to multiple communities
            for comm in set(communities.values()):
                if comm_connections.get(comm, 0) < 0.5:
                    # Find best nodes in this community to connect to
                    comm_nodes = [j for j in range(n) if communities[j] == comm]
                    if comm_nodes:
                        best_node = max(comm_nodes, key=lambda x: target_betweenness[x])
                        P_new[node, best_node] = P_new[best_node, node] = max(
                            P_new[node, best_node],
                            0.5
                        )
        
        return P_new

    def _enforce_assortativity(
        self,
        P: np.ndarray,
        G: nx.Graph
    ) -> np.ndarray:
        """Adjust probabilities to preserve degree assortativity."""
        n = P.shape[0]
        P_new = P.copy()
        
        try:
            # Get target assortativity with robust error handling
            try:
                # Calculate degree correlation coefficient directly to avoid NetworkX's warning
                degrees = np.array([d for _, d in G.degree()])
                if len(degrees) < 2 or np.all(degrees == degrees[0]):
                    # If all degrees are the same or insufficient data
                    target_assort = 0.0
                else:
                    # Manual calculation of degree correlation
                    source_degrees = []
                    target_degrees = []
                    for u, v in G.edges():
                        source_degrees.append(G.degree(u))
                        target_degrees.append(G.degree(v))
                        # Add reverse edge for undirected graph
                        source_degrees.append(G.degree(v))
                        target_degrees.append(G.degree(u))
                    
                    if len(source_degrees) > 0:
                        source_degrees = np.array(source_degrees)
                        target_degrees = np.array(target_degrees)
                        
                        # Calculate correlation avoiding division by zero
                        source_mean = np.mean(source_degrees)
                        target_mean = np.mean(target_degrees)
                        numerator = np.mean((source_degrees - source_mean) * (target_degrees - target_mean))
                        source_std = np.std(source_degrees)
                        target_std = np.std(target_degrees)
                        
                        if source_std > 0 and target_std > 0:
                            target_assort = numerator / (source_std * target_std)
                        else:
                            target_assort = 0.0
                    else:
                        target_assort = 0.0
            except:
                target_assort = 0.0
            
            if abs(target_assort) > 0.1:  # Only enforce if significant assortativity
                degrees = np.array([G.degree(i) for i in range(n)])
                avg_deg = np.mean(degrees)
                
                for i in range(n):
                    for j in range(i + 1, n):
                        if target_assort > 0:  # positive assortativity
                            if (degrees[i] > avg_deg) == (degrees[j] > avg_deg):
                                # Similar degrees: increase probability
                                P_new[i, j] = P_new[j, i] = min(
                                    1.0,
                                    P_new[i, j] * 1.2
                                )
                            else:
                                # Different degrees: decrease probability
                                P_new[i, j] = P_new[j, i] = max(
                                    0.0,
                                    P_new[i, j] * 0.8
                                )
                        else:  # negative assortativity
                            if (degrees[i] > avg_deg) != (degrees[j] > avg_deg):
                                P_new[i, j] = P_new[j, i] = min(
                                    1.0,
                                    P_new[i, j] * 1.2
                                )
                            else:
                                P_new[i, j] = P_new[j, i] = max(
                                    0.0,
                                    P_new[i, j] * 0.8
                                )
            
        except Exception as e:
            # Return original matrix if assortativity enforcement fails
            return P
        
        return P_new

    def _enforce_block_structure(
        self,
        prob_matrix: np.ndarray,
        G: nx.Graph,
        communities: Dict[int, int],
        structure_info: Dict[str, Any]
    ) -> np.ndarray:
        """
        Enhanced block structure enforcement that:
        1. Maintains block-specific density patterns
        2. Preserves hierarchical structure if present
        3. Handles temporal block evolution
        4. Adapts to different network models
        """
        n = prob_matrix.shape[0]
        P = prob_matrix.copy()
        
        if self.model_type == "sbm":
            # Enhanced SBM-specific adjustments
            P = self._enforce_sbm_structure(P, communities, structure_info)
        
        elif self.model_type == "rcp":
            # Enhanced core-periphery structure
            P = self._enforce_rcp_structure(P, communities, structure_info)
        
        elif self.model_type == "hierarchical":
            # Handle hierarchical community structure
            P = self._enforce_hierarchical_structure(P, G, communities)
        
        # Final density adjustments and symmetrization
        P = self._adjust_block_densities(P, communities, structure_info)
        P = 0.5 * (P + P.T)
        np.clip(P, 0, 1, out=P)
        
        return P

    def _enforce_sbm_structure(
        self,
        P: np.ndarray,
        communities: Dict[int, int],
        structure_info: Dict[str, Any]
    ) -> np.ndarray:
        """Enforce stochastic block model structure with strict probability control."""
        n = P.shape[0]
        P_new = P.copy()
        
        # Get exact bounds from model parameters
        max_intra = self.model_params["max_intra_prob"]  # 0.8
        min_intra = self.model_params["min_intra_prob"]  # 0.6
        max_inter = self.model_params["max_inter_prob"]  # 0.015
        min_inter = self.model_params["min_inter_prob"]  # 0.005
        
        # Adjust probabilities block by block with strict bounds
        for i in range(n):
            for j in range(i + 1, n):  # Avoid self-loops
                bi, bj = communities[i], communities[j]
                
                if bi == bj:
                    # Intra-block: use strict bounds
                    P_new[i, j] = P_new[j, i] = np.clip(
                        P_new[i, j],
                        min_intra,
                        max_intra
                    )
                else:
                    # Inter-block: use strict bounds
                    P_new[i, j] = P_new[j, i] = np.clip(
                        P_new[i, j],
                        min_inter,
                        max_inter
                    )
        
        # Ensure no self-loops
        np.fill_diagonal(P_new, 0)
        
        return P_new

    def _enforce_rcp_structure(
        self,
        P: np.ndarray,
        communities: Dict[int, int],
        structure_info: Dict[str, Any]
    ) -> np.ndarray:
        """Enforce rich-club/core-periphery structure."""
        n = P.shape[0]
        P_new = P.copy()
        
        # Get core and periphery nodes
        core_nodes = [node for node, comm in communities.items() if comm == 0]
        periph_nodes = [node for node, comm in communities.items() if comm == 1]
        
        # Enhanced RCP parameters
        core_density = self.model_params["core_prob"]
        periph_density = self.model_params["periph_prob"]
        core_periph_density = self.model_params["core_periph_prob"]
        
        # Adjust core-core connections
        for i in core_nodes:
            for j in core_nodes:
                if i < j:
                    current = P_new[i, j]
                    # Stronger enforcement for core
                    P_new[i, j] = P_new[j, i] = 0.8 * current + 0.2 * core_density
        
        # Adjust periphery-periphery connections
        for i in periph_nodes:
            for j in periph_nodes:
                if i < j:
                    current = P_new[i, j]
                    # Weaker enforcement for periphery
                    P_new[i, j] = P_new[j, i] = 0.6 * current + 0.4 * periph_density
        
        # Adjust core-periphery connections
        for i in core_nodes:
            for j in periph_nodes:
                current = P_new[i, j]
                # Balanced enforcement for core-periphery
                P_new[i, j] = P_new[j, i] = 0.7 * current + 0.3 * core_periph_density
        
        return P_new

    def _enforce_hierarchical_structure(
        self,
        P: np.ndarray,
        G: nx.Graph,
        communities: Dict[int, int]
    ) -> np.ndarray:
        """Enforce hierarchical community structure if present."""
        # Detect hierarchical levels using modularity optimization
        dendrogram = community.generate_dendrogram(G)
        num_levels = len(dendrogram)
        
        P_new = P.copy()
        
        # Adjust probabilities level by level
        for level in range(num_levels):
            level_communities = community.partition_at_level(dendrogram, level)
            level_info = self._analyze_block_structure(G, level_communities)
            
            # Stronger enforcement at lower levels (finer communities)
            alpha = 1.0 - (level / num_levels)
            P_new = self._adjust_level_structure(P_new, level_communities, level_info, alpha)
        
        return P_new

    def _adjust_level_structure(
        self,
        P: np.ndarray,
        communities: Dict[int, int],
        structure_info: Dict[str, Any],
        alpha: float
    ) -> np.ndarray:
        """Adjust structure at a specific hierarchical level."""
        n = P.shape[0]
        P_new = P.copy()
        
        for i in range(n):
            for j in range(i + 1, n):
                ci, cj = communities[i], communities[j]
                if ci == cj:
                    target = structure_info["intra_densities"].get(ci, 0.5)
        else:
            block_pair = tuple(sorted([ci, cj]))
            target = structure_info["inter_densities"].get(block_pair, 0.1)
        
        current = P_new[i, j]
        # Apply level-specific mixing
        P_new[i, j] = P_new[j, i] = (1 - alpha) * current + alpha * target
        
        return P_new

    def _adjust_block_densities(
        self,
        P: np.ndarray,
        communities: Dict[int, int],
        structure_info: Dict[str, Any]
    ) -> np.ndarray:
        """Final adjustment of block densities with temporal smoothing."""
        n = P.shape[0]
        P_new = P.copy()
        
        # Apply final density adjustments using existing structure info
        for i in range(n):
            for j in range(i + 1, n):
                bi, bj = communities[i], communities[j]
                if bi == bj:
                    # Intra-block connection
                    target = structure_info["intra_densities"].get(bi, 0.5)
                else:
                    # Inter-block connection
                    block_pair = tuple(sorted([bi, bj]))
                    target = structure_info["inter_densities"].get(block_pair, 0.1)
                
                # Smooth adjustment
                current = P_new[i, j]
                P_new[i, j] = P_new[j, i] = 0.8 * current + 0.2 * target
        
        return P_new

    def _update_adaptive_parameters(
        self,
        predicted: np.ndarray,
        actual: np.ndarray,
        history: List[Dict[str, Any]]
    ) -> None:
        """
        Update model parameters based on prediction error:
        1. Adjust mixing parameter alpha
        2. Update model-specific parameters
        3. Update weighted predictor weights
        4. Track error patterns for future adjustments
        """
        # Calculate prediction error
        error_metrics = self._compute_prediction_error(predicted, actual)
        
        # 1. Update mixing parameter alpha
        self._update_mixing_parameter(error_metrics)
        
        # 2. Update model parameters
        self._update_model_parameters(error_metrics, history)
        
        # 3. Track error patterns
        self._update_error_history(error_metrics)

    def _compute_prediction_error(
        self,
        predicted: np.ndarray,
        actual: np.ndarray
    ) -> Dict[str, float]:
        """Compute comprehensive error metrics."""
        # Basic error metrics
        error_metrics = {
            "mse": np.mean((predicted - actual) ** 2),
            "mae": np.mean(np.abs(predicted - actual)),
            "density_error": np.abs(np.mean(predicted) - np.mean(actual)),
        }
        
        # Graph-level metrics
        G_pred = nx.from_numpy_array(predicted)
        G_actual = nx.from_numpy_array(actual)
        
        try:
            # Clustering error
            error_metrics["clustering_error"] = abs(
                nx.average_clustering(G_pred) - nx.average_clustering(G_actual)
            )
            
            # Degree correlation with error handling
            pred_degrees = np.array(sorted([d for _, d in G_pred.degree()], reverse=True))
            actual_degrees = np.array(sorted([d for _, d in G_actual.degree()], reverse=True))
            
            # Ensure arrays are of same length
            min_len = min(len(pred_degrees), len(actual_degrees))
            pred_degrees = pred_degrees[:min_len]
            actual_degrees = actual_degrees[:min_len]
            
            if min_len > 0:
                try:
                    corr = np.corrcoef(pred_degrees, actual_degrees)[0, 1]
                    if np.isnan(corr):
                        corr = 0.0
                except:
                    corr = 0.0
                error_metrics["degree_correlation"] = corr
            
            # Modularity error
            try:
                pred_mod = community.modularity(community.best_partition(G_pred), G_pred)
                actual_mod = community.modularity(community.best_partition(G_actual), G_actual)
                error_metrics["modularity_error"] = abs(pred_mod - actual_mod)
            except:
                error_metrics["modularity_error"] = 1.0
            
        except Exception as e:
            # Set default values if metrics calculation fails
            error_metrics.update({
                "clustering_error": 1.0,
                "degree_correlation": 0.0,
                "modularity_error": 1.0
            })
        
        return error_metrics

    def _update_mixing_parameter(self, error_metrics: Dict[str, float]) -> None:
        """Update alpha based on prediction error patterns."""
        # If we have error history
        if hasattr(self, '_error_history'):
            prev_error = np.mean([e["mse"] for e in self._error_history[-3:]])
            current_error = error_metrics["mse"]
            
            # Adjust alpha based on error trend
            if current_error > prev_error:
                # Prediction getting worse, adjust alpha in opposite direction
                if not hasattr(self, '_last_alpha_adjustment'):
                    self._last_alpha_adjustment = 0.1
                else:
                    self._last_alpha_adjustment *= -0.5
                
                self.alpha = np.clip(
                    self.alpha + self._last_alpha_adjustment,
                    0.1,  # min alpha
                    0.9   # max alpha
                )
            else:
                # Prediction improving, continue in same direction but smaller step
                if hasattr(self, '_last_alpha_adjustment'):
                    self._last_alpha_adjustment *= 0.5
                    self.alpha = np.clip(
                        self.alpha + self._last_alpha_adjustment,
                        0.1,
                        0.9
                    )

    def _update_model_parameters(
        self,
        error_metrics: Dict[str, float],
        history: List[Dict[str, Any]]
    ) -> None:
        """Update model-specific parameters based on error patterns."""
        if self.model_type == "ba":
            self._update_ba_parameters(error_metrics)
        elif self.model_type == "rcp":
            self._update_rcp_parameters(error_metrics)
        elif self.model_type == "sbm":
            self._update_sbm_parameters(error_metrics)

    def _update_ba_parameters(self, error_metrics: Dict[str, float]) -> None:
        """Update BA model parameters."""
        if hasattr(self, '_error_history'):
            # If degree correlation is poor, adjust preferential attachment
            if error_metrics["degree_correlation"] < 0.8:
                # Adjust preferential exponent
                current_exp = self.model_params["preferential_exp"]
                self.model_params["preferential_exp"] = current_exp * (
                    1 + 0.1 * np.sign(error_metrics["degree_correlation"] - 0.8)
                )

    def _update_rcp_parameters(self, error_metrics: Dict[str, float]) -> None:
        """Update RCP model parameters."""
        if hasattr(self, '_error_history'):
            # Adjust core size if structure preservation is poor
            if error_metrics["modularity_error"] > 0.1:
                current_size = self.model_params["core_size"]
                self.model_params["core_size"] = int(
                    current_size * (1 + 0.1 * np.sign(error_metrics["modularity_error"] - 0.1))
                )
            
            # Adjust densities based on error
            self.model_params["core_prob"] *= (1 - 0.1 * error_metrics["density_error"])
            self.model_params["periph_prob"] *= (1 - 0.1 * error_metrics["density_error"])
            
            # Ensure valid probabilities
            for param in ["core_prob", "periph_prob", "core_periph_prob"]:
                self.model_params[param] = np.clip(self.model_params[param], 0.01, 0.99)

    def _update_sbm_parameters(self, error_metrics: Dict[str, float]) -> None:
        """Update SBM model parameters."""
        if hasattr(self, '_error_history'):
            # Adjust number of blocks if modularity error is high
            if error_metrics["modularity_error"] > 0.1:
                current_blocks = self.model_params["num_blocks"]
                self.model_params["num_blocks"] = max(2, current_blocks + np.sign(
                    error_metrics["modularity_error"] - 0.1
                ))
            
            # Adjust probabilities based on density error
            self.model_params["intra_prob"] *= (1 - 0.1 * error_metrics["density_error"])
            self.model_params["inter_prob"] *= (1 - 0.1 * error_metrics["density_error"])
            
            # Ensure valid probabilities
            for param in ["intra_prob", "inter_prob"]:
                self.model_params[param] = np.clip(self.model_params[param], 0.01, 0.99)

    def _update_error_history(self, error_metrics: Dict[str, float]) -> None:
        """Track error history for adaptive updates."""
        if not hasattr(self, '_error_history'):
            self._error_history = []
        
        self._error_history.append(error_metrics)
        
        # Keep last 10 error measurements
        if len(self._error_history) > 10:
            self._error_history.pop(0)

    def predict(
        self,
        history: List[Dict[str, Any]],
        horizon: int = 1,
        actual_future: Optional[List[Dict[str, Any]]] = None
    ) -> List[np.ndarray]:
        """Enhanced predict with adaptive parameter updates."""
        if len(history) < self.n_history:
            raise ValueError(f"Not enough history. Need {self.n_history}, got {len(history)}.")
        
        predictions = []
        current_history = list(history)
        
        for step in range(horizon):
            # Make single prediction
            pred = self._make_single_prediction(current_history)
            
            # Store prediction and update history
            predictions.append(pred)
            current_history.append({
                "adjacency": pred,
                "graph": nx.from_numpy_array(pred)
            })
            
            # Update parameters if we have actual future states
            if actual_future is not None and step < len(actual_future):
                self._update_adaptive_parameters(
                    pred,
                    actual_future[step]["adjacency"],
                    current_history[step:step + self.n_history]
                )
        
        return predictions

    def predict_with_correction(
        self,
        history: List[Dict[str, Any]],
        horizon: int = 1,
        correction_interval: int = 3,
        max_drift_threshold: float = 0.2
    ) -> List[np.ndarray]:
        """
        Enhanced prediction with error correction for multi-step forecasts.
        
        Parameters
        ----------
        history : List[Dict[str, Any]]
            Historical network states
        horizon : int
            Number of steps to predict
        correction_interval : int
            How often to check and correct for drift
        max_drift_threshold : float
            Maximum allowed drift before correction
        """
        predictions = []
        current_history = list(history)
        drift_metrics = []
        
        for step in range(horizon):
            # 1. Make base prediction
            pred = self._make_single_prediction(current_history)
            
            # 2. Check for drift if at correction interval
            if step > 0 and step % correction_interval == 0:
                drift = self._compute_drift_metrics(
                    history[-1]["adjacency"],  # original state
                    pred,                      # current prediction
                    drift_metrics             # drift history
                )
                
                # 3. Apply correction if drift exceeds threshold
                if drift["total_drift"] > max_drift_threshold:
                    pred = self._correct_drift(
                        pred,
                        history[-1]["adjacency"],
                        drift
                    )
            
            # Store prediction and update histories
            predictions.append(pred)
            current_history.append({"adjacency": pred, "graph": nx.from_numpy_array(pred)})
            if step > 0:
                drift_metrics.append(self._compute_step_drift(
                    predictions[-2],  # previous prediction
                    pred             # current prediction
                ))

        return predictions

    def _make_single_prediction(
        self,
        history: List[Dict[str, Any]]
    ) -> np.ndarray:
        """Make a single step prediction using weighted predictor as exact baseline."""
        # 1. Get weighted prediction first as baseline
        weighted_pred = self.base_predictor.predict(history, horizon=1)[0]
        
        # 2. Get exact target metrics from weighted predictor
        G_weighted = nx.from_numpy_array(weighted_pred)
        target_edges = G_weighted.number_of_edges()  # We must match this exactly
        n = weighted_pred.shape[0]
        
        # 3. For SBM, preserve block structure while EXACTLY matching weighted predictor's edge count
        if self.model_type == "sbm":
            # Get communities
            communities = self._detect_communities(history[-1]["graph"])
            
            # Create edge scores matrix directly (no intermediate probability adjustments)
            edge_scores = np.zeros((n, n))
            
            # Score edges based on both weighted predictor and block structure
            for i in range(n):
                for j in range(i + 1, n):  # Avoid self-loops
                    # Base score from weighted predictor (this preserves weighted predictor's preferences)
                    base_score = weighted_pred[i, j]
                    
                    # Modify score based on block membership (small adjustments to preserve relative ordering)
                    if communities[i] == communities[j]:
                        edge_scores[i, j] = edge_scores[j, i] = base_score * 1.1  # Only 10% boost
                    else:
                        edge_scores[i, j] = edge_scores[j, i] = base_score * 0.9  # Only 10% reduction
            
            # Create list of edges with their scores
            edges = []
            for i in range(n):
                for j in range(i + 1, n):  # Upper triangular only
                    edges.append((edge_scores[i, j], i, j))
            
            # Sort edges by score
            edges.sort(reverse=True, key=lambda x: x[0])
            
            # Take EXACTLY target_edges number of edges
            P_final = np.zeros((n, n))
            for _, i, j in edges[:target_edges]:  # Only take top target_edges edges
                P_final[i, j] = P_final[j, i] = 1
            
            return P_final
        
        else:
            # For other models, just use weighted predictor's edge count
            edges = []
            for i in range(n):
                for j in range(i + 1, n):  # Avoid self-loops
                    edges.append((weighted_pred[i, j], i, j))
            
            edges.sort(reverse=True, key=lambda x: x[0])
            
            P_final = np.zeros((n, n))
            for _, i, j in edges[:target_edges]:  # Only take top target_edges edges
                P_final[i, j] = P_final[j, i] = 1
            
            return P_final

    def _compute_drift_metrics(
        self,
        original: np.ndarray,
        current: np.ndarray,
        drift_history: List[Dict[str, float]]
    ) -> Dict[str, float]:
        """
        Compute comprehensive drift metrics between original and current state.
        Also considers drift acceleration from history.
        """
        G_orig = nx.from_numpy_array(original)
        G_curr = nx.from_numpy_array(current)
        
        # Basic structural drift
        drift_metrics = {
            "density_drift": abs(nx.density(G_curr) - nx.density(G_orig)),
            "degree_drift": np.mean(np.abs(
                np.array(sorted([d for _, d in G_curr.degree()], reverse=True)) -
                np.array(sorted([d for _, d in G_orig.degree()], reverse=True))
            )),
            "clustering_drift": abs(
                nx.average_clustering(G_curr) - nx.average_clustering(G_orig)
            )
        }
        
        # Model-specific drift
        if self.model_type == "ba":
            drift_metrics["degree_distribution_drift"] = self._compute_degree_dist_drift(
                G_orig, G_curr
            )
        elif self.model_type == "rcp":
            drift_metrics["core_periphery_drift"] = self._compute_rcp_drift(
                G_orig, G_curr
            )
        elif self.model_type == "sbm":
            drift_metrics["block_structure_drift"] = self._compute_block_drift(
                G_orig, G_curr
            )
        
        # Compute drift acceleration if history exists
        if drift_history:
            drift_metrics["acceleration"] = self._compute_drift_acceleration(
                drift_history
            )
        
        # Compute total drift (weighted combination)
        weights = {
            "density_drift": 1.0,
            "degree_drift": 1.0,
            "clustering_drift": 1.0,
            "acceleration": 2.0  # Higher weight for acceleration
        }
        
        drift_metrics["total_drift"] = sum(
            weights.get(k, 1.0) * v 
            for k, v in drift_metrics.items()
            if k != "total_drift"
        ) / len(weights)
        
        return drift_metrics

    def _compute_step_drift(
        self,
        prev: np.ndarray,
        curr: np.ndarray
    ) -> Dict[str, float]:
        """Compute drift metrics between consecutive predictions."""
        return {
            "density_drift": abs(np.mean(curr) - np.mean(prev)),
            "edge_drift": np.mean(np.abs(curr - prev)),
        }

    def _compute_drift_acceleration(
        self,
        drift_history: List[Dict[str, float]]
    ) -> float:
        """Compute drift acceleration from history."""
        if len(drift_history) < 2:
            return 0.0
        
        # Compute rate of change of drift
        recent_drifts = [d["edge_drift"] for d in drift_history[-3:]]
        return np.mean(np.diff(recent_drifts))

    def _correct_drift(
        self,
        prediction: np.ndarray,
        original: np.ndarray,
        drift: Dict[str, float]
    ) -> np.ndarray:
        """
        Apply drift correction to prediction.
        Uses different strategies based on drift type and model.
        """
        corrected = prediction.copy()
        
        # 1. Basic structural corrections
        if drift["density_drift"] > 0.1:
            corrected = self._correct_density_drift(
                corrected,
                original,
                drift["density_drift"]
            )
        
        if drift["degree_drift"] > 0.1:
            corrected = self._correct_degree_drift(
                corrected,
                original,
                drift["degree_drift"]
            )
        
        # 2. Model-specific corrections
        if self.model_type == "ba" and drift.get("degree_distribution_drift", 0) > 0.1:
            corrected = self._correct_ba_drift(corrected, original)
        elif self.model_type == "rcp" and drift.get("core_periphery_drift", 0) > 0.1:
            corrected = self._correct_rcp_drift(corrected, original)
        elif self.model_type == "sbm" and drift.get("block_structure_drift", 0) > 0.1:
            corrected = self._correct_sbm_drift(corrected, original)
        
        # 3. Acceleration-based correction
        if drift.get("acceleration", 0) > 0.05:
            corrected = self._correct_acceleration(
                corrected,
                drift["acceleration"]
            )
        
        return corrected

    def _correct_density_drift(
        self,
        prediction: np.ndarray,
        original: np.ndarray,
        drift: float
    ) -> np.ndarray:
        """Correct global density drift."""
        target_density = np.mean(original)
        current_density = np.mean(prediction)
        
        # Scale probabilities to match target density
        scale = target_density / (current_density + 1e-10)
        corrected = prediction * scale
        
        return np.clip(corrected, 0, 1)

    def _correct_degree_drift(
        self,
        prediction: np.ndarray,
        original: np.ndarray,
        drift: float
    ) -> np.ndarray:
        """Correct degree sequence drift."""
        n = prediction.shape[0]
        corrected = prediction.copy()
        
        # Get target degrees
        G_orig = nx.from_numpy_array(original)
        target_degrees = np.array([G_orig.degree(i) for i in range(n)])
        
        # Apply degree sequence correction
        corrected = self._enforce_degree_sequence(
            corrected,
            target_degrees
        )
        
        return corrected

    def _correct_acceleration(
        self,
        prediction: np.ndarray,
        acceleration: float
    ) -> np.ndarray:
        """
        Correct for drift acceleration by damping changes.
        Stronger damping for higher acceleration.
        """
        # Compute damping factor based on acceleration
        damping = 1.0 / (1.0 + abs(acceleration) * 10)
        
        # Apply damping to deviation from uniform probability
        n = prediction.shape[0]
        uniform_prob = np.mean(prediction)
        deviation = prediction - uniform_prob
        
        corrected = uniform_prob + damping * deviation
        return np.clip(corrected, 0, 1)

    def _correct_ba_drift(
        self,
        prediction: np.ndarray,
        original: np.ndarray
    ) -> np.ndarray:
        """Correct drift in BA network structure."""
        # Focus on preserving degree distribution shape
        G_orig = nx.from_numpy_array(original)
        target_degrees = sorted([d for _, d in G_orig.degree()], reverse=True)
        
        return self._enforce_degree_sequence(
            prediction,
            np.array(target_degrees)
        )

    def _correct_rcp_drift(
        self,
        prediction: np.ndarray,
        original: np.ndarray
    ) -> np.ndarray:
        """Correct drift in core-periphery structure."""
        G_orig = nx.from_numpy_array(original)
        communities = self._detect_communities(G_orig)
        structure_info = self._analyze_block_structure(G_orig, communities)
        
        return self._enforce_rcp_structure(
            prediction,
            communities,
            structure_info
        )

    def _correct_sbm_drift(
        self,
        prediction: np.ndarray,
        original: np.ndarray
    ) -> np.ndarray:
        """Correct drift in block structure."""
        G_orig = nx.from_numpy_array(original)
        communities = self._detect_communities(G_orig)
        structure_info = self._analyze_block_structure(G_orig, communities)
        
        return self._enforce_sbm_structure(
            prediction,
            communities,
            structure_info
        )
