"""Network generation functions for graph forecasting.

This module provides functions to generate various types of networks
with controlled properties for testing and analysis.
"""

import networkx as nx
import numpy as np
from typing import Dict, Any
import random


def generate_ba_network(N: int = 100, m: int = 3, seed: int = None) -> Dict[str, Any]:
    """Generate Barabási-Albert preferential attachment network.

    Parameters
    ----------
    N : int, optional
        Number of nodes, by default 100
    m : int, optional
        Number of edges to attach from a new node to existing nodes, by default 3
    seed : int, optional
        Random seed for reproducibility, by default None

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - adjacency: np.ndarray
            Adjacency matrix of the network
        - graph: nx.Graph
            NetworkX graph object
        - params: Dict
            Generation parameters used

    Notes
    -----
    The Barabási-Albert model generates scale-free networks using a preferential
    attachment mechanism where new nodes are more likely to connect to existing
    nodes with higher degrees.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    g = nx.barabasi_albert_graph(n=N, m=m)
    adjacency = nx.to_numpy_array(g)

    return {"adjacency": adjacency, "graph": g, "params": {"N": N, "m": m}}


def generate_evolving_ba_network(
    N: int = 100, m_mean: float = 3, m_std: float = 0.5, seed: int = None
) -> Dict[str, Any]:
    """Generate Barabási-Albert network with evolving parameters.

    Parameters
    ----------
    N : int, optional
        Number of nodes, by default 100
    m_mean : float, optional
        Mean number of edges for new nodes, by default 3
    m_std : float, optional
        Standard deviation of number of edges, by default 0.5
    seed : int, optional
        Random seed for reproducibility, by default None

    Returns
    -------
    Dict[str, Any]
        Dictionary containing network information (same as generate_ba_network)

    Notes
    -----
    This variant adds randomness to the attachment process by varying the
    number of edges (m) according to a normal distribution.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    m = max(1, int(np.random.normal(m_mean, m_std)))
    return generate_ba_network(N=N, m=m, seed=seed)
