"""Network metric calculations for graph analysis.

This module provides functions to calculate various network metrics including centrality
measures, spectral properties, and basic graph statistics.
"""

import networkx as nx
import numpy as np
from typing import Dict, Any


def get_network_metrics(graph: nx.Graph) -> Dict[str, float]:
    """Calculate comprehensive network metrics.

    Parameters
    ----------
    graph : nx.Graph
        Input network graph

    Returns
    -------
    Dict[str, float]
        Dictionary containing the following metrics:
        - avg_degree (μ): Average node degree
        - density (ρ): Network density
        - clustering (C): Average clustering coefficient
        - max_degree (Δ): Maximum node degree
        - avg_betweenness (β̄): Average betweenness centrality
        - max_betweenness (β*): Maximum betweenness centrality
        - avg_eigenvector (ē): Average eigenvector centrality
        - max_eigenvector (e*): Maximum eigenvector centrality
        - avg_closeness (c̄): Average closeness centrality
        - spectral_gap (λ₁-λ₂): Difference between largest eigenvalues
        - algebraic_connectivity (λ₂): Second smallest Laplacian eigenvalue
    """
    # Basic network metrics
    avg_degree = np.mean([d for _, d in graph.degree()])
    density = nx.density(graph)
    clustering = nx.average_clustering(graph)

    # Calculate centrality metrics
    degree_centrality = dict(graph.degree())
    betweenness_centrality = nx.betweenness_centrality(graph)
    eigenvector_centrality = nx.eigenvector_centrality(graph, max_iter=1000)
    closeness_centrality = nx.closeness_centrality(graph)

    # Calculate SVD of adjacency matrix
    adj_matrix = nx.to_numpy_array(graph)
    _, S, _ = np.linalg.svd(adj_matrix)

    # Calculate Laplacian SVD
    laplacian = nx.laplacian_matrix(graph).toarray()
    _, L_S, _ = np.linalg.svd(laplacian)

    return {
        "avg_degree": avg_degree,
        "density": density,
        "clustering": clustering,
        "max_degree": max(degree_centrality.values()),
        "avg_betweenness": np.mean(list(betweenness_centrality.values())),
        "max_betweenness": max(betweenness_centrality.values()),
        "avg_eigenvector": np.mean(list(eigenvector_centrality.values())),
        "max_eigenvector": max(eigenvector_centrality.values()),
        "avg_closeness": np.mean(list(closeness_centrality.values())),
        "spectral_gap": S[0] - S[1] if len(S) > 1 else S[0],
        "algebraic_connectivity": L_S[1],
    }


def calculate_error_metrics(
    actual_metrics: Dict[str, float], predicted_metrics: Dict[str, float]
) -> Dict[str, float]:
    """Calculate error metrics between actual and predicted network properties.

    Parameters
    ----------
    actual_metrics : Dict[str, float]
        Dictionary of actual network metrics
    predicted_metrics : Dict[str, float]
        Dictionary of predicted network metrics

    Returns
    -------
    Dict[str, float]
        Dictionary containing absolute errors for each metric
    """
    return {
        key: abs(actual_metrics[key] - predicted_metrics[key])
        for key in actual_metrics.keys()
        if key in predicted_metrics
    }
