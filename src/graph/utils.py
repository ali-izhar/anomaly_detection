# src/graph/utils.py

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


def extract_numeric_features(
    feature_dict: dict, feature_set: str = "all"
) -> np.ndarray:
    """Extract numeric features from feature dictionary based on specified feature set.

    Args:
        feature_dict (dict): Dictionary of raw features.
        feature_set (str): Which feature set to use.

    Returns:
        np.ndarray: Array of numeric features.
    """
    features = []

    # Basic metrics
    if feature_set in ["all", "basic"]:
        degrees = feature_dict.get("degrees", [])
        features.append(np.mean(degrees) if degrees else 0.0)
        features.append(feature_dict.get("density", 0.0))
        clustering = feature_dict.get("clustering", [])
        features.append(np.mean(clustering) if clustering else 0.0)

    # Centrality metrics
    if feature_set in ["all", "centrality"]:
        betweenness = feature_dict.get("betweenness", [])
        features.append(np.mean(betweenness) if betweenness else 0.0)
        eigenvector = feature_dict.get("eigenvector", [])
        features.append(np.mean(eigenvector) if eigenvector else 0.0)
        closeness = feature_dict.get("closeness", [])
        features.append(np.mean(closeness) if closeness else 0.0)

    # Spectral metrics
    if feature_set in ["all", "spectral"]:
        singular_values = feature_dict.get("singular_values", [])
        features.append(max(singular_values) if singular_values else 0.0)
        laplacian_eigenvalues = feature_dict.get("laplacian_eigenvalues", [])
        features.append(
            min(x for x in laplacian_eigenvalues if x > 1e-10)
            if laplacian_eigenvalues
            else 0.0
        )

    return np.array(features)
