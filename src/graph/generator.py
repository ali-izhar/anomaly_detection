# src/graph/generator.py

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
    """Convert a NetworkX graph to a NumPy adjacency matrix with nodes labeled from 0 to n-1.

    Parameters
    ----------
    G : nx.Graph
        The input graph.

    Returns
    -------
    np.ndarray
        An (n x n) adjacency matrix (dense) in NumPy format.

    Notes
    -----
    - Relabels nodes to ensure ordering from 0..n-1.
    - Return dtype is float (0.0 or 1.0 entries).
    """
    G = nx.convert_node_labels_to_integers(G)
    return nx.to_numpy_array(G)


class GraphGenerator:
    """Generator for dynamic graph sequences."""

    def __init__(self):
        """Initialize the generator with all supported models registered."""
        self._generators: Dict[str, Dict] = {}

        # Register Barabási-Albert model
        self.register_model(
            name="barabasi_albert",
            generator_func=self.generate_ba_network,
            param_class=BAParams,
            param_mutation_func=self.ba_param_mutation,
            metadata_func=self.ba_metadata,
        )
        # Short alias
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
        """Register a new graph model with associated utilities.

        Parameters
        ----------
        name : str
            Identifier for the model (e.g., "barabasi_albert").
        generator_func : Callable
            A function that, given model-specific parameters, returns a
            NetworkX graph (e.g., `nx.barabasi_albert_graph`).
        param_class : Type[BaseParams]
            A dataclass describing model-specific parameters.
        param_mutation_func : Callable[[Dict], Dict], optional
            Function that modifies parameters during anomaly injection. If not
            provided, `_default_param_mutation` is used.
        metadata_func : Callable[[Dict], Dict], optional
            Function that returns metadata for the model. Defaults to `_default_metadata`.
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
        """Generate a sequence of graph snapshots with optional random change points.

        Parameters
        ----------
        model : str
            Name (or alias) of the registered graph model (e.g., "ba", "er", "ws").
        params : BaseParams
            A dataclass instance describing the generation parameters (seq_len,
            min_segment, min_changes, etc., plus model-specific fields).
        seed : int, optional
            Random seed for reproducibility across runs.

        Returns
        -------
        Dict
            A dictionary containing:
            - **graphs**: List of adjacency matrices (np.ndarray).
            - **change_points**: List of indices where parameters changed.
            - **parameters**: List of parameter dictionaries for each segment.
            - **metadata**: List of metadata objects from each segment.
            - **model**: The model name used.
            - **num_changes**: How many change points were inserted.
            - **n**: Number of nodes in each graph (as per `params.n`).
            - **sequence_length**: The total `seq_len`.

        Raises
        ------
        ValueError
            If the requested model is not registered or if `params` is the wrong type.

        Notes
        -----
        1. We generate up to `max_changes` abrupt shifts in parameters, each
           ensuring at least `min_segment` steps per segment.
        2. Between steps, parameters can evolve gradually if there are fields
           ending in `_std` (e.g., `m_std` for Barabási-Albert).
        """
        if model not in self._generators:
            raise ValueError(f"Model {model} not registered.")

        if not isinstance(params, self._generators[model]["param_class"]):
            raise TypeError(
                f"Parameters must be instance of "
                f"{self._generators[model]['param_class'].__name__}"
            )

        if seed is not None:
            np.random.seed(seed)

        try:
            # 1. Generate random change points
            change_points, num_changes = self._generate_change_points(params)
            logger.info(f"Generated {num_changes} change points at: {change_points}")

            # 2. Generate parameter sets for each segment
            param_sets = self._generate_parameter_sets(
                model, params, num_changes, self._generators[model]["mutate"]
            )
            logger.debug(f"Generated parameters for {num_changes + 1} segments")

            # 3. Generate graphs for each segment
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
                    logger.debug(f"Segment {i} => {len(segment)} graphs")
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

            logger.info(f"Generated sequence with {len(all_graphs)} total graphs")
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
        """Produce a list of adjacency matrices by calling the model's generator
        repeatedly, evolving parameters after each graph if `_std` fields
        are present.

        Parameters
        ----------
        model : str
            Model name or alias.
        params : Dict
            Model-specific parameter dictionary (keys/values).
        length : int
            Number of graphs to generate in this segment.

        Returns
        -------
        List[np.ndarray]
            A list of adjacency matrices for the segment.
        """
        graphs = []
        generator = self._generators[model]["generator"]
        current_params = params.copy()

        for _ in range(length):
            # Generate graph
            G = generator(**current_params)
            adj_matrix = graph_to_adjacency(G)
            graphs.append(adj_matrix)

            # Evolve parameters for the next step
            current_params = self._evolve_parameters(current_params)

        return graphs

    def _generate_change_points(
        self,
        params: BaseParams,
    ) -> Tuple[List[int], int]:
        """Randomly generate valid change points for the sequence.

        Parameters
        ----------
        params : BaseParams
            A dataclass specifying `seq_len`, `min_segment`, `min_changes`, `max_changes`.

        Returns
        -------
        (List[int], int)
            A tuple of (the sorted list of change point indices, number_of_changes).

        Raises
        ------
        RuntimeError
            If valid change points cannot be found in 100 attempts.

        Notes
        -----
        - Each segment must be at least `min_segment` in length.
        - We sample `num_changes` from [min_changes, max_changes].
        - Then we pick that many points from the valid positions that keep
          every segment >= min_segment in size.
        """
        seq_len = params.seq_len
        min_segment = params.min_segment
        min_changes = params.min_changes
        max_changes = params.max_changes

        # Max possible changes if each segment is min_segment length
        max_possible_changes = (seq_len - min_segment) // min_segment
        if max_changes > max_possible_changes:
            max_changes = max_possible_changes
        if min_changes > max_changes:
            min_changes = max_changes

        max_attempts = 100
        for _ in range(max_attempts):
            num_changes = np.random.randint(min_changes, max_changes + 1)

            # All valid positions spaced by min_segment
            valid_positions = []
            current_pos = min_segment
            while current_pos <= seq_len - min_segment:
                valid_positions.append(current_pos)
                current_pos += min_segment

            # If not enough valid positions, reduce changes
            if len(valid_positions) < num_changes:
                continue

            # Randomly select a sorted subset
            points = sorted(
                np.random.choice(valid_positions, size=num_changes, replace=False)
            )

            # Verify each segment >= min_segment
            valid = True
            prev_point = 0
            for point in points + [seq_len]:
                if point - prev_point < min_segment:
                    valid = False
                    break
                prev_point = point

            if valid:
                return points, num_changes

        msg = f"Failed to generate valid change points after {max_attempts} tries"
        logger.error(msg)
        raise RuntimeError(msg)

    def _generate_parameter_sets(
        self,
        model: str,
        params: BaseParams,
        num_changes: int,
        mutation_func: Callable[[Dict], Dict],
    ) -> List[Dict]:
        """Build a list of parameter dictionaries, one per segment,
        by applying anomaly (mutation) between segments.

        Parameters
        ----------
        model : str
            The model name (unused here, but reserved for future special handling).
        params : BaseParams
            The original dataclass with the base parameter values.
        num_changes : int
            How many segments transitions (and hence how many anomaly injections).
        mutation_func : Callable[[Dict], Dict]
            A function that randomizes or shifts the parameters for an anomaly.

        Returns
        -------
        List[Dict]
            A list of parameter dictionaries, of length num_changes+1.
        """
        param_dict = asdict(params)
        param_sets = [param_dict]

        for _ in range(num_changes):
            new_params = mutation_func(param_dict.copy())
            param_sets.append(new_params)
            param_dict = new_params

        return param_sets

    @staticmethod
    def _default_param_mutation(params: Dict) -> Dict:
        """Default anomaly injection: for each param with `min_foo` and `max_foo`,
        pick a random new value in [min_foo, max_foo].

        Parameters
        ----------
        params : Dict
            Current parameter dictionary.

        Returns
        -------
        Dict
            A new parameter dictionary with potential random jumps in relevant fields.
        """
        new_params = params.copy()

        # Identify all param pairs: min_foo, max_foo
        param_bounds = {}
        for key in params:
            if key.startswith("min_"):
                base_key = key[4:]
                max_key = f"max_{base_key}"
                if max_key in params:
                    param_bounds[base_key] = (params[key], params[max_key])

        # Jump each base_key into [min, max]
        for base_key, (mn, mx) in param_bounds.items():
            if isinstance(mn, int):
                new_params[base_key] = np.random.randint(mn, mx + 1)
            else:
                new_params[base_key] = np.random.uniform(mn, mx)

        return new_params

    @staticmethod
    def _default_metadata(params: Dict) -> Dict:
        """Default metadata is simply a copy of the parameter dictionary.

        Parameters
        ----------
        params : Dict
            Current parameter dictionary.

        Returns
        -------
        Dict
            A dictionary with key "params" storing the original params.
        """
        return {"params": params.copy()}

    def generate_ba_network(self, n: int, m: int, **kwargs) -> nx.Graph:
        """Generate a Barabási-Albert (BA) scale-free network.

        Parameters
        ----------
        n : int
            Number of nodes (>= m+1 recommended).
        m : int
            Number of edges each new node brings.
        """
        return nx.barabasi_albert_graph(n=n, m=m)

    def generate_ws_network(
        self, n: int, k_nearest: int, rewire_prob: float, **kwargs
    ) -> nx.Graph:
        """Generate a Watts-Strogatz small-world network.

        Parameters
        ----------
        n : int
            Number of nodes.
        k_nearest : int
            Each node is connected to k_nearest neighbors on each side.
        rewire_prob : float
            Probability to rewire each edge to a random node (in [0,1]).
        """
        return nx.watts_strogatz_graph(n=n, k=k_nearest, p=rewire_prob)

    def generate_er_network(self, n: int, prob: float, **kwargs) -> nx.Graph:
        """Generate an Erdős-Rényi random network G(n, p).

        Parameters
        ----------
        n : int
            Number of nodes.
        prob : float
            Edge probability in [0,1].
        """
        return nx.erdos_renyi_graph(n=n, p=prob)

    def generate_sbm_network(
        self, n: int, num_blocks: int, intra_prob: float, inter_prob: float, **kwargs
    ) -> nx.Graph:
        """Generate a Stochastic Block Model (SBM) network.

        Parameters
        ----------
        n : int
            Total number of nodes.
        num_blocks : int
            Number of communities.
        intra_prob : float
            Probability of edges within the same block.
        inter_prob : float
            Probability of edges across blocks.
        """
        # Derive block sizes
        block_size = n // num_blocks
        sizes = [block_size] * (num_blocks - 1)
        sizes.append(n - sum(sizes))  # last block gets remainder

        # Probability matrix
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
        """Generate a Random Core-Periphery (RCP) network.

        Parameters
        ----------
        n : int
            Total number of nodes.
        core_size : int
            Number of nodes in the core.
        core_prob : float
            Probability of edges among core nodes.
        periph_prob : float
            Probability of edges among periphery nodes.
        core_periph_prob : float
            Probability of edges across core and periphery.
        """
        G = nx.Graph()
        G.add_nodes_from(range(n))

        core_nodes = list(range(core_size))
        periph_nodes = list(range(core_size, n))

        # Core subgraph
        for i in range(core_size):
            for j in range(i + 1, core_size):
                if np.random.random() < core_prob:
                    G.add_edge(i, j)

        # Periphery subgraph
        for i in range(core_size, n):
            for j in range(i + 1, n):
                if np.random.random() < periph_prob:
                    G.add_edge(i, j)

        # Cross core-periphery
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
        """Generate an LFR (Lancichinetti-Fortunato-Radicchi) benchmark network.

        Parameters
        ----------
        n : int
            Total number of nodes.
        avg_degree : int
            Average node degree.
        max_degree : int
            Maximum node degree.
        mu : float
            Mixing parameter in [0,1]. Higher => more inter-community edges.
        tau1 : float
            Exponent for degree distribution.
        tau2 : float
            Exponent for community size distribution.
        min_community : int
            Minimum size of any community.
        max_community : int
            Maximum size of any community.
        """
        # 1. Generate a power-law degree sequence
        degrees = [min(max_degree, int(d)) for d in nx.utils.powerlaw_sequence(n, tau1)]
        # Ensure min average degree / even sum
        degrees = [max(avg_degree // 2, d) for d in degrees]
        if sum(degrees) % 2 == 1:
            degrees[0] += 1

        # Construct via configuration model
        G = nx.configuration_model(degrees)
        G = nx.Graph(G)  # remove parallel edges/self loops

        # 2. Generate community sizes via power-law
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

        # 3. Assign nodes to communities
        node_communities = []
        start_idx = 0
        for size in communities:
            node_communities.extend([start_idx] * size)
            start_idx += 1

        edges = list(G.edges())
        np.random.shuffle(edges)

        # Current fraction of inter-community edges
        current_inter = sum(
            1 for u, v in edges if node_communities[u] != node_communities[v]
        )
        target_inter = int(mu * len(edges))

        # 4. Rewire edges to reach target mixing
        while current_inter != target_inter:
            if current_inter < target_inter:
                # Need more inter edges
                for u, v in edges:
                    if node_communities[u] == node_communities[v]:
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
                # Need more intra edges
                for u, v in edges:
                    if node_communities[u] != node_communities[v]:
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

        # Store community info
        nx.set_node_attributes(
            G, {i: {"community": c} for i, c in enumerate(node_communities)}
        )

        return G

    def ba_param_mutation(self, params: Dict) -> Dict:
        """Specialized parameter mutation for Barabási-Albert networks.
        In addition to the default jumps in [min_m, max_m], we preserve
        the standard deviation fields for gradual evolution.

        Parameters
        ----------
        params : Dict
            Current parameter dictionary (e.g., containing m, min_m, max_m, m_std).

        Returns
        -------
        Dict
            A new parameter dictionary after random jumps in `[min_m, max_m]`.
        """
        # Default injection
        new_params = self._default_param_mutation(params)

        # Keep _std fields same as old
        for key in params:
            if key.endswith("_std"):
                new_params[key] = params[key]

        return new_params

    def ba_metadata(self, params: Dict) -> Dict:
        """Create metadata for Barabási-Albert segments, listing evolving parameters.

        Parameters
        ----------
        params : Dict
            Current BA parameter dictionary.

        Returns
        -------
        Dict
            A metadata dictionary with keys:
            - "params": a copy of all params
            - "evolving_parameters": list of param names that have _std set
        """
        meta = {"params": params.copy()}
        evolving_params = [
            k[:-4] for k in params if k.endswith("_std") and params[k] is not None
        ]
        if evolving_params:
            meta["evolving_parameters"] = evolving_params
        return meta

    def _evolve_parameters(self, params: Dict) -> Dict:
        """Evolve parameters by applying a Gaussian step for each param that has
        a corresponding `_std` field.

        - If key = X and key_std = X_std in params:
          new_value ~ Normal( old_value, X_std )
        - For integer fields, the result is rounded and clipped to >= 1.
        - For float fields that contain "prob", we clip to [0,1].
        - Otherwise, we clip to >= 0 if that makes sense (e.g., for degrees).

        Parameters
        ----------
        params : Dict
            Parameter dictionary containing both values and optional `_std` fields.

        Returns
        -------
        Dict
            Updated parameter dictionary after small random evolutions.
        """
        evolved_params = params.copy()

        for key, value in params.items():
            std_key = f"{key}_std"
            if std_key in params and params[std_key] is not None:
                std_value = params[std_key]
                new_value = np.random.normal(value, std_value)

                if isinstance(value, int):
                    # Round and ensure positive
                    new_value = int(round(new_value))
                    if key not in ["min_changes", "max_changes"]:
                        new_value = max(1, new_value)
                elif isinstance(value, float):
                    # If param name has "prob", clip to [0,1]
                    if "prob" in key:
                        new_value = float(np.clip(new_value, 0.0, 1.0))
                    else:
                        new_value = max(0.0, new_value)

                evolved_params[key] = new_value

        return evolved_params
