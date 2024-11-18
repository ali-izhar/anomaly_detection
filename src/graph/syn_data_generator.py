# src/graph/syn_data_generator.py

import logging
from typing import Dict, List, Any
import numpy as np
from dataclasses import dataclass

from .graph_generator import GraphGenerator

logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """Configuration parameters for synthetic data generation.
    Attributes:
        graph_type: Model identifier (BA, ER, NW)
        n: Number of nodes
        params_before: Parameters before change point
        params_after: Parameters after change point
        n_graphs_before: Number of graphs before change
        n_graphs_after: Number of graphs after change
    """

    graph_type: str
    n: int
    params_before: Dict[str, Any]
    params_after: Dict[str, Any]
    n_graphs_before: int
    n_graphs_after: int


class SyntheticDataGenerator:
    """Generates synthetic graph sequences with controlled structural changes.
    - Barabasi-Albert (BA): Scale-free networks
    - Erdos-Renyi (ER): Random graphs
    - Newman-Watts (NW): Small-world networks
    """

    def __init__(self, graph_generator: GraphGenerator) -> None:
        """Initialize generator with graph model backend.

        Args:
            graph_generator: Instance handling low-level graph generation
        """
        self._generator = graph_generator
        self._generated_graphs: Dict[str, List[np.ndarray]] = {}
        logger.debug("Initialized SyntheticDataGenerator with empty graph storage")

    @property
    def graphs(self) -> Dict[str, List[np.ndarray]]:
        """Get generated graph sequences by type."""
        return self._generated_graphs

    def generate(self, config: Any) -> Dict[str, List[np.ndarray]]:
        """Generate graph sequences based on configuration.
        For each graph type, generates sequence [G1, ..., Gk] where:
        - G1, ..., Gi follow distribution P1(theta1)
        - Gi+1, ..., Gk follow distribution P2(theta2)
        where i is the change point and theta are model parameters.

        Args:
            config: Configuration object with generation parameters

        Returns:
            Dictionary mapping graph types to sequences of adjacency matrices

        Raises:
            ValueError: For invalid configuration
            RuntimeError: For generation failures
        """
        if not hasattr(config, "graph"):
            logger.error("Configuration missing 'graph' section")
            raise ValueError("Config must have 'graph' section")

        try:
            graph_gen_config = config.graph
            self._generated_graphs = {}
            logger.info("Starting synthetic graph generation")

            for graph_type in graph_gen_config.__dict__:
                logger.info(f"Processing {graph_type} graph generation")

                # Extract and validate parameters
                logger.debug(f"Extracting parameters for {graph_type}")
                params = self._extract_params(getattr(graph_gen_config, graph_type))
                logger.debug(f"Parameters for {graph_type}: {params}")

                # Get corresponding generator method
                logger.debug(f"Getting generator method for {graph_type}")
                generator = self._get_generator_method(graph_type)

                # Generate graph sequence
                logger.info(f"Generating {graph_type} graph sequence")
                graphs = generator(**params)
                self._generated_graphs[graph_type] = graphs

                logger.info(f"Successfully generated {len(graphs)} {graph_type} graphs")
                logger.debug(
                    f"Memory usage for {graph_type}: {sum(g.nbytes for g in graphs) / 1e6:.2f} MB"
                )

            logger.info(
                f"Completed generation of all graph types: {list(self._generated_graphs.keys())}"
            )
            return self._generated_graphs

        except Exception as e:
            logger.error(f"Graph generation failed: {str(e)}")
            raise RuntimeError(f"Graph generation failed: {str(e)}")

    def _extract_params(self, config: Any) -> Dict[str, Any]:
        """Extract generation parameters from config object.

        Args:
            config: Configuration section for specific graph type

        Returns:
            Dictionary of validated parameters

        Raises:
            ValueError: If required parameters are missing
        """
        logger.debug("Extracting parameters from config")
        params = {}
        for key, value in config.__dict__.items():
            if not key.startswith("_"):
                params[key] = value

        required = {"n", "set1", "set2"}
        missing = required - set(params.keys())
        if missing:
            logger.error(f"Missing required parameters: {missing}")
            raise ValueError(f"Missing required parameters: {missing}")

        logger.debug(f"Extracted parameters: {params}")
        return params

    def _get_generator_method(self, graph_type: str) -> Any:
        """Get corresponding generator method for graph type.

        Args:
            graph_type: Model identifier

        Returns:
            Generator method from backend

        Raises:
            ValueError: If graph type is not supported
        """
        try:
            generator = getattr(self._generator, graph_type.lower())
            logger.debug(f"Found generator method for {graph_type}")
            return generator
        except AttributeError:
            logger.error(f"Unsupported graph type requested: {graph_type}")
            raise ValueError(f"Unsupported graph type: {graph_type}")

    def add_custom_model(self, config: GenerationConfig) -> None:
        """Register custom graph model for generation.

        Args:
            config: Model configuration parameters

        Raises:
            ValueError: If configuration is invalid
        """
        if not isinstance(config, GenerationConfig):
            logger.error("Invalid configuration format provided for custom model")
            raise ValueError("Invalid configuration format")

        logger.info(f"Adding custom model: {config.graph_type}")
        logger.debug(
            f"Custom model parameters - before: {config.params_before}, after: {config.params_after}"
        )

        self._generator.register_graph_type(
            graph_type=config.graph_type,
            graph_func=lambda **kwargs: np.zeros((config.n, config.n)),
            params1=config.params_before,
            params2=config.params_after,
        )
        logger.info(f"Successfully registered custom model: {config.graph_type}")
