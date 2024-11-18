# src/graph/graph_generator.py

import logging
import networkx as nx
import numpy as np
from typing import List, Callable, Dict, Optional, Any

from .features import graph_to_adjacency

logger = logging.getLogger(__name__)


class GraphGenerator:
    """Generates temporal graph sequences with structural changes.
    - Barabasi-Albert (BA): P(k) ~ k^(-3)
    - Erdos-Renyi (ER): P(edge) = p
    - Newman-Watts (NW): Small-world networks
    """

    def __init__(
        self, graph_types: Optional[Dict[str, List[np.ndarray]]] = None
    ) -> None:
        """Initialize generator with storage for different graph types.

        Args:
            graph_types: Optional pre-initialized graph storage
        """
        self._graphs: Dict[str, List[np.ndarray]] = graph_types or {
            "BA": [],
            "BA_I": [],
            "ER": [],
            "NW": [],
        }
        logger.debug(
            f"Initialized GraphGenerator with types: {list(self._graphs.keys())}"
        )

    @property
    def graphs(self) -> Dict[str, List[np.ndarray]]:
        """Get generated graph sequences by type."""
        return self._graphs

    def _generate_graphs(
        self,
        graph_type: str,
        n: int,
        set1: int,
        set2: int,
        graph_func: Callable,
        params1: Dict[str, Any],
        params2: Dict[str, Any],
    ) -> List[np.ndarray]:
        """Generate graph sequence with parameter change.
        Creates sequence [G1, ..., Gk] where:
        - G1, ..., Gi use params1 (i = set1)
        - Gi+1, ..., Gk use params2 (k = set1 + set2)

        Args:
            graph_type: Model identifier
            n: Number of nodes
            set1: Length of first sequence
            set2: Length of second sequence
            graph_func: NetworkX generator function
            params1: Parameters before change
            params2: Parameters after change

        Returns:
            List of adjacency matrices [n x n]

        Raises:
            ValueError: For invalid parameters
            RuntimeError: For generation failures
        """
        if n <= 0 or set1 < 0 or set2 < 0:
            logger.error(f"Invalid parameters: n={n}, set1={set1}, set2={set2}")
            raise ValueError("Node count and sequence lengths must be positive")

        logger.info(
            f"Generating {graph_type} graphs: {set1} with params1, {set2} with params2"
        )
        logger.debug(f"Parameters before change: {params1}")
        logger.debug(f"Parameters after change: {params2}")

        try:
            graphs: List[np.ndarray] = []

            # Generate first sequence
            logger.info(f"Generating first sequence of {set1} graphs")
            for i in range(set1):
                logger.debug(f"Generating graph {i+1}/{set1} with params1")
                graph = graph_func(n=n, seed=i, **params1)
                graphs.append(graph_to_adjacency(graph))

            # Generate second sequence
            logger.info(f"Generating second sequence of {set2} graphs")
            for i in range(set1, set1 + set2):
                logger.debug(f"Generating graph {i-set1+1}/{set2} with params2")
                graph = graph_func(n=n, seed=i, **params2)
                graphs.append(graph_to_adjacency(graph))

            self._graphs[graph_type] = graphs
            logger.info(f"Successfully generated {len(graphs)} {graph_type} graphs")
            return graphs

        except Exception as e:
            logger.error(f"Failed to generate {graph_type} graphs: {str(e)}")
            raise RuntimeError(f"Failed to generate {graph_type} graphs: {str(e)}")

    def barabasi_albert(
        self, n: int, m1: int, m2: int, set1: int, set2: int
    ) -> List[np.ndarray]:
        """Generate Barabasi-Albert (BA) graphs with varying attachment parameter.
        BA model: P(k) ~ k^(-3), where k is node degree
        Changes m edges per new node: m1 -> m2 at t = set1

        Args:
            n: Number of nodes
            m1: Initial edges per node
            m2: Changed edges per node
            set1: Graphs before change
            set2: Graphs after change

        Returns:
            List of adjacency matrices
        """
        if not 0 < m1 < n or not 0 < m2 < n:
            logger.error(f"Invalid edge parameters: m1={m1}, m2={m2}, n={n}")
            raise ValueError("Edge parameters must be in range (0, n)")

        logger.info(f"Generating BA graphs with n={n}, m1={m1}, m2={m2}")
        return self._generate_graphs(
            graph_type="BA",
            n=n,
            set1=set1,
            set2=set2,
            graph_func=nx.barabasi_albert_graph,
            params1={"m": m1},
            params2={"m": m2},
        )

    def barabasi_albert_internet(
        self,
        n: int,
        m1: int,
        m2: int,
        set1: int,
        set2: int,
        base_seed1: int = 1,
        base_seed2: int = 100,
    ) -> List[np.ndarray]:
        """Generate Barabasi-Albert (BA) graphs with Internet-like initial topology.
        Uses random AS graphs as seeds, then applies BA growth:
        G0 -> BA(m1) -> BA(m2)

        Args:
            n: Number of nodes
            m1, m2: Edge parameters
            set1, set2: Sequence lengths
            base_seed1, base_seed2: Seeds for initial graphs

        Returns:
            List of adjacency matrices
        """
        logger.info(f"Generating Internet-like BA graphs with n={n}, m1={m1}, m2={m2}")
        logger.debug(f"Using base seeds: {base_seed1}, {base_seed2}")

        base1 = nx.random_internet_as_graph(n, seed=base_seed1)
        base2 = nx.random_internet_as_graph(n, seed=base_seed2)
        logger.debug(
            f"Created base graphs: {base1.number_of_edges()}, {base2.number_of_edges()} edges"
        )

        return self._generate_graphs(
            graph_type="BA_I",
            n=n,
            set1=set1,
            set2=set2,
            graph_func=nx.barabasi_albert_graph,
            params1={"m": m1, "initial_graph": base1},
            params2={"m": m2, "initial_graph": base2},
        )

    def erdos_renyi(
        self, n: int, p1: float, p2: float, set1: int, set2: int
    ) -> List[np.ndarray]:
        """Generate Erdos-Renyi (ER) graphs with varying edge probability.
        ER model: P(edge) = p for all node pairs
        Changes probability: p1 -> p2 at t = set1

        Args:
            n: Number of nodes
            p1: Initial edge probability
            p2: Changed edge probability
            set1, set2: Sequence lengths

        Returns:
            List of adjacency matrices
        """
        if not 0 <= p1 <= 1 or not 0 <= p2 <= 1:
            logger.error(f"Invalid probabilities: p1={p1}, p2={p2}")
            raise ValueError("Probabilities must be in [0, 1]")

        logger.info(f"Generating ER graphs with n={n}, p1={p1}, p2={p2}")
        return self._generate_graphs(
            graph_type="ER",
            n=n,
            set1=set1,
            set2=set2,
            graph_func=nx.erdos_renyi_graph,
            params1={"p": p1},
            params2={"p": p2},
        )

    def newman_watts(
        self, n: int, p1: float, p2: float, k1: int, k2: int, set1: int, set2: int
    ) -> List[np.ndarray]:
        """Generate Newman-Watts (NW) small-world graphs with varying parameters.
        NW model: Ring lattice + random edges
        Changes (k, p): (k1, p1) -> (k2, p2) at t = set1

        Args:
            n: Number of nodes
            p1, p2: Rewiring probabilities
            k1, k2: Initial neighbor counts
            set1, set2: Sequence lengths

        Returns:
            List of adjacency matrices
        """
        if not 0 <= p1 <= 1 or not 0 <= p2 <= 1:
            logger.error(f"Invalid probabilities: p1={p1}, p2={p2}")
            raise ValueError("Probabilities must be in [0, 1]")
        if k1 >= n or k2 >= n:
            logger.error(f"Invalid k values: k1={k1}, k2={k2}, n={n}")
            raise ValueError("k must be less than n")

        logger.info(
            f"Generating NW graphs with n={n}, k1={k1}, k2={k2}, p1={p1}, p2={p2}"
        )
        return self._generate_graphs(
            graph_type="NW",
            n=n,
            set1=set1,
            set2=set2,
            graph_func=nx.newman_watts_strogatz_graph,
            params1={"k": k1, "p": p1},
            params2={"k": k2, "p": p2},
        )

    def register_graph_type(
        self,
        graph_type: str,
        graph_func: Callable,
        params1: Dict[str, Any],
        params2: Dict[str, Any],
    ) -> None:
        """Register custom graph generator with parameters.

        Args:
            graph_type: Model identifier
            graph_func: NetworkX-compatible generator
            params1: Initial parameters
            params2: Changed parameters
        """
        if graph_type in self._graphs:
            logger.warning(f"Overwriting existing graph type: {graph_type}")
        else:
            logger.info(f"Registering new graph type: {graph_type}")

        logger.debug(
            f"Parameters for {graph_type}: params1={params1}, params2={params2}"
        )
        self._graphs[graph_type] = []
        setattr(
            self,
            graph_type.lower(),
            lambda n, set1, set2: self._generate_graphs(
                graph_type, n, set1, set2, graph_func, params1, params2
            ),
        )
        logger.info(f"Successfully registered graph type: {graph_type}")
