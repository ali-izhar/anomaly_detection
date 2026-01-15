"""Utility functions for graph operations."""

import networkx as nx
import numpy as np


def graph_to_adjacency(G: nx.Graph, node_labels: bool = True) -> np.ndarray:
    """Convert NetworkX graph to NumPy adjacency matrix with 0-based node labels.

    Args:
        G: NetworkX graph
        node_labels: If True, convert node labels to integers
    Returns:
        NumPy adjacency matrix
    """
    if node_labels:
        G = nx.convert_node_labels_to_integers(G)
    return nx.to_numpy_array(G)


def adjacency_to_graph(A: np.ndarray, node_labels: bool = True) -> nx.Graph:
    """Convert NumPy adjacency matrix to NetworkX graph.

    Args:
        A: NumPy adjacency matrix
        node_labels: If True, convert node labels to integers
    Returns:
        NetworkX graph
    """
    if node_labels:
        G = nx.from_numpy_array(A)
        G = nx.convert_node_labels_to_integers(G)

    else:
        G = nx.from_numpy_array(A)
    return G
