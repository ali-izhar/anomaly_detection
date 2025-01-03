# src/graph/generator.py

"""Graph Generator for Dynamic Graph Sequences.

This module provides a flexible framework for generating sequences of dynamic graphs
using various NetworkX models. It supports both sparse and dense graph types with
configurable parameters that can change over time.

Key Features:
- Generic interface for various NetworkX-compatible graph models
- Generates sequences with controlled parameter changes
- Maintains consistent node labeling across sequences
- Provides metadata about changes and parameters
"""

import logging
from typing import Dict, List, Optional, Callable, Type, Tuple
import numpy as np
import networkx as nx
from dataclasses import asdict

from .params import (
    BaseParams,
    BAParams,
    WSParams,
    ERParams,
    SBMParams,
    RCPParams,
    LFRParams,
)

logger = logging.getLogger(__name__)


def graph_to_adjacency(G: nx.Graph) -> np.ndarray:
    """Convert NetworkX graph to adjacency matrix with consistent node ordering.

    Args:
        G: NetworkX graph

    Returns:
        Adjacency matrix with nodes ordered from 0 to n-1
    """
    # Ensure nodes are labeled 0 to n-1
    G = nx.convert_node_labels_to_integers(G)
    return nx.to_numpy_array(G)


class GraphGenerator:
    """Generator for dynamic graph sequences.

    This class provides a generic interface to generate sequences of graphs with:
    1. Gradual evolution through _std parameters
    2. Sudden changes through min/max parameters at change points
    """

    def __init__(self):
        """Initialize the graph generator with all supported models."""
        self._generators: Dict[str, Dict] = {}

        # Register Barabási-Albert model
        self.register_model(
            name="barabasi_albert",
            generator_func=self.generate_ba_network,
            param_class=BAParams,
            param_mutation_func=self.ba_param_mutation,
            metadata_func=self.ba_metadata,
        )
        # Also register with short name
        self._generators["ba"] = self._generators["barabasi_albert"]

        # Register Watts-Strogatz model
        self.register_model(
            name="watts_strogatz",
            generator_func=self.generate_ws_network,
            param_class=WSParams,
        )
        self._generators["ws"] = self._generators["watts_strogatz"]

        # Register Erdős-Rényi model
        self.register_model(
            name="erdos_renyi",
            generator_func=self.generate_er_network,
            param_class=ERParams,
        )
        self._generators["er"] = self._generators["erdos_renyi"]

        # Register Stochastic Block Model
        self.register_model(
            name="stochastic_block_model",
            generator_func=self.generate_sbm_network,
            param_class=SBMParams,
        )
        self._generators["sbm"] = self._generators["stochastic_block_model"]

        # Register Random Core-Periphery model
        self.register_model(
            name="random_core_periphery",
            generator_func=self.generate_rcp_network,
            param_class=RCPParams,
        )
        self._generators["rcp"] = self._generators["random_core_periphery"]

        # Register LFR Benchmark model
        self.register_model(
            name="lfr_benchmark",
            generator_func=self.generate_lfr_network,
            param_class=LFRParams,
        )
        self._generators["lfr"] = self._generators["lfr_benchmark"]

    def register_model(
        self,
        name: str,
        generator_func: Callable,
        param_class: Type[BaseParams],
        param_mutation_func: Optional[Callable[[Dict], Dict]] = None,
        metadata_func: Optional[Callable[[Dict], Dict]] = None,
    ) -> None:
        """Register a new graph model.

        Args:
            name: Identifier for the model
            generator_func: NetworkX-compatible graph generator function
            param_class: Parameter class for this model type
            param_mutation_func: Function to mutate parameters for sequence generation
            metadata_func: Function to compute model-specific metadata
        """
        if name in self._generators:
            logger.warning(f"Overwriting existing model: {name}")

        self._generators[name] = {
            "generator": generator_func,
            "param_class": param_class,
            "mutate": param_mutation_func or self._default_param_mutation,
            "metadata": metadata_func or self._default_metadata,
        }
        logger.info(f"Registered model: {name}")

    def generate_sequence(
        self,
        model: str,
        params: BaseParams,
        seed: Optional[int] = None,
    ) -> Dict:
        """Generate a sequence of graphs with changing parameters.

        Args:
            model: Name of the registered graph model to use
            params: Parameters for graph generation
            seed: Random seed for reproducibility

        Returns:
            Dictionary containing:
            - graphs: List of adjacency matrices
            - change_points: List of indices where parameters change
            - params: List of parameters used in each segment
            - metadata: Additional model-specific information
        """
        if model not in self._generators:
            raise ValueError(
                f"Model {model} not registered. Use register_model() first."
            )

        if not isinstance(params, self._generators[model]["param_class"]):
            raise TypeError(
                f"Parameters must be instance of {self._generators[model]['param_class'].__name__}"
            )

        if seed is not None:
            np.random.seed(seed)

        try:
            # Generate change points
            change_points, num_changes = self._generate_change_points(params)
            logger.info(f"Generated {num_changes} change points at: {change_points}")

            # Generate parameter sets for each segment
            param_sets = self._generate_parameter_sets(
                model, params, num_changes, self._generators[model]["mutate"]
            )
            logger.debug(f"Generated parameters for {num_changes + 1} segments")

            # Generate graph segments
            all_graphs = []
            metadata = []

            for i in range(len(change_points) + 1):
                start = change_points[i - 1] if i > 0 else 0
                end = change_points[i] if i < len(change_points) else params.seq_len
                length = end - start

                segment_params = param_sets[i]
                logger.debug(f"Generating segment {i} with params: {segment_params}")

                try:
                    segment = self._generate_graph_segment(
                        model, segment_params, length
                    )
                    meta = self._generators[model]["metadata"](segment_params)

                    all_graphs.extend(segment)
                    metadata.append(meta)
                    logger.debug(f"Generated segment {i} with {len(segment)} graphs")
                except Exception as e:
                    msg = f"Failed to generate segment {i}"
                    logger.error(msg, exc_info=True)
                    raise RuntimeError(msg) from e

            result = {
                "graphs": all_graphs,
                "change_points": change_points,
                "parameters": param_sets,
                "metadata": metadata,
                "model": model,
                "num_changes": num_changes,
                "n": params.n,
                "sequence_length": params.seq_len,
            }

            logger.info(
                f"Successfully generated sequence with {len(all_graphs)} graphs"
            )
            return result

        except Exception as e:
            logger.error(f"Failed to generate sequence: {str(e)}", exc_info=True)
            raise

    def _generate_graph_segment(
        self,
        model: str,
        params: Dict,
        length: int,
    ) -> List[np.ndarray]:
        """Generate a sequence of graphs with evolving parameters."""
        graphs = []
        generator = self._generators[model]["generator"]
        current_params = params.copy()

        for _ in range(length):
            # Generate graph with current parameters
            G = generator(**current_params)
            adj_matrix = graph_to_adjacency(G)
            graphs.append(adj_matrix)

            # Evolve parameters for next iteration
            current_params = self._evolve_parameters(current_params)

        return graphs

    def _generate_change_points(
        self,
        params: BaseParams,
    ) -> Tuple[List[int], int]:
        """Generate random change points for the sequence.

        Args:
            params: Graph generation parameters

        Returns:
            Tuple of (change points list, number of changes)
        """
        seq_len = params.seq_len
        min_segment = params.min_segment
        min_changes = params.min_changes
        max_changes = params.max_changes

        # Validate parameters
        max_possible_changes = (seq_len - min_segment) // min_segment
        if max_changes > max_possible_changes:
            max_changes = max_possible_changes
        if min_changes > max_changes:
            min_changes = max_changes

        # Try to generate valid change points
        max_attempts = 100
        for _ in range(max_attempts):
            # Generate random number of changes
            num_changes = np.random.randint(min_changes, max_changes + 1)

            # Create valid positions for change points
            valid_positions = []
            current_pos = min_segment
            while current_pos <= seq_len - min_segment:
                valid_positions.append(current_pos)
                current_pos += min_segment

            # If we don't have enough valid positions, try again with fewer changes
            if len(valid_positions) < num_changes:
                continue

            # Select change points from valid positions
            points = sorted(
                np.random.choice(valid_positions, size=num_changes, replace=False)
            )

            # Verify all segments meet minimum length
            valid = True
            prev_point = 0
            for point in points + [seq_len]:
                if point - prev_point < min_segment:
                    valid = False
                    break
                prev_point = point

            if valid:
                logger.info(f"Generated {num_changes} change points at: {points}")
                return points, num_changes

        msg = f"Failed to generate valid change points after {max_attempts} attempts"
        logger.error(msg)
        raise RuntimeError(msg)

    def _generate_parameter_sets(
        self,
        model: str,
        params: BaseParams,
        num_changes: int,
        mutation_func: Callable[[Dict], Dict],
    ) -> List[Dict]:
        """Generate parameter sets for each segment, including both evolution and anomaly changes."""
        param_dict = asdict(params)
        param_sets = [param_dict]

        for _ in range(num_changes):
            # Apply anomaly changes through mutation function
            new_params = mutation_func(param_dict.copy())
            param_sets.append(new_params)
            param_dict = new_params

        return param_sets

    @staticmethod
    def _default_param_mutation(params: Dict) -> Dict:
        """Default parameter mutation strategy for anomaly injection.

        For each parameter with min/max bounds, generates a random value
        between those bounds. Evolution parameters (_std) are preserved.
        """
        new_params = params.copy()

        # Find all parameter pairs with min/max bounds
        param_bounds = {}
        for key in params:
            if key.startswith("min_"):
                base_key = key[4:]
                max_key = f"max_{base_key}"
                if max_key in params:
                    param_bounds[base_key] = (params[key], params[max_key])

        # Generate new random values within bounds
        for base_key, (min_val, max_val) in param_bounds.items():
            if isinstance(min_val, int):
                new_params[base_key] = np.random.randint(min_val, max_val + 1)
            else:
                new_params[base_key] = np.random.uniform(min_val, max_val)

        return new_params

    @staticmethod
    def _default_metadata(params: Dict) -> Dict:
        """Default metadata generation."""
        return {"params": params.copy()}

    def generate_ba_network(self, n: int, m: int, **kwargs) -> nx.Graph:
        """Generate Barabási-Albert preferential attachment network."""
        return nx.barabasi_albert_graph(n=n, m=m)

    def generate_ws_network(
        self, n: int, k_nearest: int, rewire_prob: float, **kwargs
    ) -> nx.Graph:
        """Generate Watts-Strogatz small-world network."""
        return nx.watts_strogatz_graph(n=n, k=k_nearest, p=rewire_prob)

    def generate_er_network(self, n: int, prob: float, **kwargs) -> nx.Graph:
        """Generate Erdős-Rényi random network."""
        return nx.erdos_renyi_graph(n=n, p=prob)

    def generate_sbm_network(
        self, n: int, num_blocks: int, intra_prob: float, inter_prob: float, **kwargs
    ) -> nx.Graph:
        """Generate Stochastic Block Model network."""
        # Calculate block sizes
        block_size = n // num_blocks
        sizes = [block_size] * (num_blocks - 1)
        sizes.append(n - sum(sizes))  # Last block gets remaining nodes

        # Create probability matrix
        probs = np.full((num_blocks, num_blocks), inter_prob)
        np.fill_diagonal(probs, intra_prob)

        return nx.stochastic_block_model(sizes, probs)

    def generate_rcp_network(
        self,
        n: int,
        core_size: int,
        core_prob: float,
        periph_prob: float,
        core_periph_prob: float,
        **kwargs,
    ) -> nx.Graph:
        """Generate Random Core-Periphery network."""
        G = nx.Graph()
        G.add_nodes_from(range(n))

        # Add core edges
        core_nodes = list(range(core_size))
        for i in range(core_size):
            for j in range(i + 1, core_size):
                if np.random.random() < core_prob:
                    G.add_edge(i, j)

        # Add periphery edges
        periph_nodes = list(range(core_size, n))
        for i in range(core_size, n):
            for j in range(i + 1, n):
                if np.random.random() < periph_prob:
                    G.add_edge(i, j)

        # Add core-periphery edges
        for i in core_nodes:
            for j in periph_nodes:
                if np.random.random() < core_periph_prob:
                    G.add_edge(i, j)

        return G

    def generate_lfr_network(
        self,
        n: int,
        avg_degree: int,
        max_degree: int,
        mu: float,
        tau1: float,
        tau2: float,
        min_community: int,
        max_community: int,
        **kwargs,
    ) -> nx.Graph:
        """Generate LFR benchmark network.

        Parameters
        ----------
        n : int
            Number of nodes
        avg_degree : int
            Average degree
        max_degree : int
            Maximum degree
        mu : float
            Mixing parameter
        tau1 : float
            Node degree distribution power law exponent
        tau2 : float
            Community size distribution power law exponent
        min_community : int
            Minimum community size
        max_community : int
            Maximum community size
        """
        # Generate base scale-free network
        # Use configuration model with power law degree sequence
        degrees = [min(max_degree, int(d)) for d in nx.utils.powerlaw_sequence(n, tau1)]
        # Ensure minimum degree and even sum
        degrees = [max(avg_degree // 2, d) for d in degrees]
        if sum(degrees) % 2 == 1:
            degrees[0] += 1

        G = nx.configuration_model(degrees)

        # Remove self-loops and parallel edges
        G = nx.Graph(G)

        # Generate community sizes following power law
        remaining_nodes = n
        communities = []
        while remaining_nodes > 0:
            size = min(
                max(
                    min_community,
                    int(nx.utils.powerlaw_sequence(1, tau2)[0] * max_community),
                ),
                remaining_nodes,
            )
            communities.append(size)
            remaining_nodes -= size

        # Assign nodes to communities
        node_communities = []
        start_idx = 0
        for size in communities:
            node_communities.extend([start_idx] * size)
            start_idx += 1

        # Rewire edges to match mixing parameter
        edges = list(G.edges())
        np.random.shuffle(edges)

        # Calculate target number of inter-community edges
        target_inter = int(mu * len(edges))
        current_inter = sum(
            1 for u, v in edges if node_communities[u] != node_communities[v]
        )

        # Rewire edges to achieve target mixing
        while current_inter != target_inter:
            if current_inter < target_inter:
                # Need more inter-community edges
                for u, v in edges:
                    if node_communities[u] == node_communities[v]:
                        # Find a node from different community
                        candidates = [
                            w
                            for w in range(n)
                            if node_communities[w] != node_communities[u]
                            and w != u
                            and w != v
                            and not G.has_edge(u, w)
                        ]
                        if candidates:
                            w = np.random.choice(candidates)
                            G.remove_edge(u, v)
                            G.add_edge(u, w)
                            current_inter += 1
                            if current_inter == target_inter:
                                break
            else:
                # Need more intra-community edges
                for u, v in edges:
                    if node_communities[u] != node_communities[v]:
                        # Find a node from same community
                        candidates = [
                            w
                            for w in range(n)
                            if node_communities[w] == node_communities[u]
                            and w != u
                            and w != v
                            and not G.has_edge(u, w)
                        ]
                        if candidates:
                            w = np.random.choice(candidates)
                            G.remove_edge(u, v)
                            G.add_edge(u, w)
                            current_inter -= 1
                            if current_inter == target_inter:
                                break

        # Store community information
        nx.set_node_attributes(
            G,
            {i: {"community": c} for i, c in enumerate(node_communities)},
        )

        return G

    def ba_param_mutation(self, params: Dict) -> Dict:
        """Parameter mutation for BA networks, supporting both evolution and anomalies."""
        # First apply standard anomaly mutation
        new_params = self._default_param_mutation(params)

        # Preserve evolution parameters
        for key in params:
            if key.endswith("_std"):
                new_params[key] = params[key]

        return new_params

    def ba_metadata(self, params: Dict) -> Dict:
        """Metadata for BA networks."""
        meta = {"params": params.copy()}
        evolving_params = [
            k[:-4] for k in params if k.endswith("_std") and params[k] is not None
        ]
        if evolving_params:
            meta["evolving_parameters"] = evolving_params
        return meta

    def _evolve_parameters(self, params: Dict) -> Dict:
        """Evolve parameters based on their standard deviations.

        For each parameter with a corresponding _std suffix, generates a new value
        from a normal distribution with the current value as mean and _std as
        standard deviation. Ensures numeric constraints (e.g., non-negative values).
        """
        evolved_params = params.copy()

        for key, value in params.items():
            std_key = f"{key}_std"
            if std_key in params and params[std_key] is not None:
                std_value = params[std_key]

                # Generate new value from normal distribution
                new_value = np.random.normal(value, std_value)

                # Apply constraints based on parameter type
                if isinstance(value, int):
                    new_value = int(round(new_value))
                    # Ensure non-negative for most parameters
                    if key not in ["min_changes", "max_changes"]:
                        new_value = max(1, new_value)
                elif isinstance(value, float):
                    # Probabilities should be between 0 and 1
                    if "prob" in key:
                        new_value = np.clip(new_value, 0.0, 1.0)
                    # Other float parameters should be non-negative
                    else:
                        new_value = max(0.0, new_value)

                evolved_params[key] = new_value

        return evolved_params
