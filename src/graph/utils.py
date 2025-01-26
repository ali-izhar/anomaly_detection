# src/graph/utils.py

import networkx as nx
import numpy as np


def graph_to_adjacency(G: nx.Graph) -> np.ndarray:
    """Convert NetworkX graph to NumPy adjacency matrix with 0-based node labels."""
    G = nx.convert_node_labels_to_integers(G)
    return nx.to_numpy_array(G)
