# synthetic_data/create_graphs.py

"""
Graph Generator for Dynamic Graph Sequences

Generates sequences of graphs with changing parameters:
- Supports multiple graph types (BA, ER, NW, SBM)
- Generates sequences with controlled change points
- Provides metadata about changes and parameters
"""

import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
from enum import Enum, auto
import numpy as np

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.graph.graph_generator import GraphGenerator
from src.graph.params import BAParams, ERParams, NWParams, SBMParams

logger = logging.getLogger(__name__)


class GraphType(Enum):
    """Supported graph types."""

    BA = auto()  # Barabási-Albert
    ER = auto()  # Erdős-Rényi
    NW = auto()  # Newman-Watts
    SBM = auto()  # Stochastic Block Model


def generate_graph_sequence(
    graph_type: GraphType,
    params: Union[BAParams, ERParams, NWParams, SBMParams],
) -> Dict:
    """Generate a sequence of graphs with changing parameters.

    Args:
        graph_type: Type of graph to generate
        params: Parameters for graph generation

    Returns:
        Dictionary containing:
        - graphs: List of adjacency matrices
        - change_points: List of indices where parameters change
        - params: List of parameters used in each segment
        - n: Number of nodes
        - sequence_length: Length of sequence
    """
    try:
        generator = GraphGenerator()

        if graph_type == GraphType.BA:
            sequence = generator.barabasi_albert(params)
        elif graph_type == GraphType.ER:
            sequence = generator.erdos_renyi(params)
        elif graph_type == GraphType.NW:
            sequence = generator.newman_watts(params)
        elif graph_type == GraphType.SBM:
            sequence = generator.stochastic_block_model(params)
        else:
            raise ValueError(f"Unsupported graph type: {graph_type}")

        return sequence

    except Exception as e:
        logger.error(f"Failed to generate graph sequence: {str(e)}")
        raise
