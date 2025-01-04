# src/graph/features.py

import networkx as nx
import numpy as np
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


@dataclass
class NetworkMetrics:
    """Container for a set of network metrics.

    Attributes
    ----------
    avg_degree : float
        The mean node degree: sum of all degrees / number of nodes.
    max_degree : float
        The largest node degree in the graph.
    density : float
        Ratio of actual edges to the maximum possible edges.
    clustering : float
        Average clustering coefficient.
    avg_betweenness : float
        Mean of node betweenness centrality values.
    max_betweenness : float
        Maximum node betweenness centrality value.
    avg_eigenvector : float
        Mean of node eigenvector centrality values.
    max_eigenvector : float
        Maximum node eigenvector centrality value.
    avg_closeness : float
        Mean of node closeness centrality values.
    spectral_gap : float
        Difference between the largest and second-largest singular values
        of the adjacency matrix.
    algebraic_connectivity : float
        The Fiedler value (second-smallest eigenvalue) of the Laplacian matrix.
    """

    avg_degree: float
    max_degree: float
    density: float
    clustering: float
    avg_betweenness: float
    max_betweenness: float
    avg_eigenvector: float
    max_eigenvector: float
    avg_closeness: float
    spectral_gap: float
    algebraic_connectivity: float


class BaseFeatureExtractor(ABC):
    """Abstract base class for feature extractors, each returning a dict of metrics."""

    @abstractmethod
    def extract(self, graph: nx.Graph) -> Dict[str, float]:
        """Extract features from the given graph."""
        pass


class BasicMetricsExtractor(BaseFeatureExtractor):
    """Extracts basic network metrics (avg_degree, density, clustering, max_degree)."""

    def extract(self, graph: nx.Graph) -> Dict[str, float]:
        """Extract basic network metrics."""
        degrees = [deg for _, deg in graph.degree()]
        return {
            "avg_degree": float(np.mean(degrees)) if degrees else 0.0,
            "density": nx.density(graph),
            "clustering": nx.average_clustering(graph),
            "max_degree": float(max(degrees)) if degrees else 0.0,
        }


class CentralityMetricsExtractor(BaseFeatureExtractor):
    """Extracts centrality-based metrics (avg_betweenness, max_betweenness, avg_eigenvector, max_eigenvector, avg_closeness)."""

    def extract(self, graph: nx.Graph) -> Dict[str, float]:
        """Extract centrality metrics."""
        if graph.number_of_nodes() == 0:
            # Return zeros if graph is empty
            return {
                "avg_betweenness": 0.0,
                "max_betweenness": 0.0,
                "avg_eigenvector": 0.0,
                "max_eigenvector": 0.0,
                "avg_closeness": 0.0,
            }

        betweenness = nx.betweenness_centrality(graph)  # dict: node -> centrality

        # Compute eigenvector centrality using numpy's eigenvalue solver (more robust than power iteration)
        adj_matrix = nx.to_numpy_array(graph, nodelist=sorted(graph.nodes()))
        eigenvalues, eigenvectors = np.linalg.eigh(adj_matrix)
        # Get the eigenvector corresponding to the largest eigenvalue
        largest_eigenvalue_idx = np.argmax(eigenvalues)
        eigenvector_values = np.abs(eigenvectors[:, largest_eigenvalue_idx])
        # Normalize to sum to 1 for consistency with networkx
        eigenvector_values = eigenvector_values / eigenvector_values.sum()
        # Convert to dict format like other centrality measures
        eigenvector = {
            node: float(value) for node, value in enumerate(eigenvector_values)
        }

        closeness = nx.closeness_centrality(graph)

        return {
            "avg_betweenness": float(np.mean(list(betweenness.values()))),
            "max_betweenness": float(np.max(list(betweenness.values()))),
            "avg_eigenvector": float(np.mean(list(eigenvector.values()))),
            "max_eigenvector": float(np.max(list(eigenvector.values()))),
            "avg_closeness": float(np.mean(list(closeness.values()))),
        }


class SpectralMetricsExtractor(BaseFeatureExtractor):
    """Extracts spectral metrics of the graph:
    - spectral_gap: difference between the first two largest singular values of adjacency.
    - algebraic_connectivity: second-smallest eigenvalue of Laplacian (Fiedler value).
    """

    def extract(self, graph: nx.Graph) -> Dict[str, float]:
        """Extract spectral metrics from adjacency and Laplacian.

        - spectral_gap = S[0] - S[1], where S are the singular values of the adjacency matrix
                        in descending order.
        - algebraic_connectivity = second-smallest eigenvalue of the Laplacian
                                  (also called the Fiedler value).

        Complexity
        ----------
        - SVD of adjacency: O(n^3) for an n x n dense matrix.
        - eigen-decomposition for Laplacian also ~O(n^3).

        Returns
        -------
        Dict[str, float]
            Keys: "spectral_gap", "algebraic_connectivity".
        """
        n = graph.number_of_nodes()
        if n == 0:
            return {"spectral_gap": 0.0, "algebraic_connectivity": 0.0}
        if n == 1:
            # Single node => adjacency is [ [0] ], no edges => SVD => singular values = [0]
            # Laplacian => also zero => second-smallest doesn't exist => 0.0
            return {"spectral_gap": 0.0, "algebraic_connectivity": 0.0}

        # 1. Adjacency-based spectral gap
        A = nx.to_numpy_array(graph, nodelist=sorted(graph.nodes()))
        U, S, Vt = np.linalg.svd(A, full_matrices=False)
        # S is sorted descending by convention. If length>1, gap = S[0]-S[1], else S[0].
        spectral_gap = S[0] - S[1] if len(S) > 1 else S[0]

        # 2. Algebraic connectivity (Fiedler value)
        #   If the graph is unconnected, the second-smallest eigenvalue is 0.0.
        L = nx.laplacian_matrix(graph, nodelist=sorted(graph.nodes())).todense()
        # We use eigh() or eigvalsh() for real symmetric
        lap_eigs = np.linalg.eigvalsh(L)  # sorted ascending by default
        # For an n x n Laplacian, lap_eigs[0] should be ~0, second-smallest is lap_eigs[1]
        # If graph is disconnected, lap_eigs[1] can be 0.0 as well.
        if len(lap_eigs) > 1:
            algebraic_connectivity = float(lap_eigs[1])
        else:
            # If n=1, or something degenerate
            algebraic_connectivity = 0.0

        return {
            "spectral_gap": float(spectral_gap),
            "algebraic_connectivity": algebraic_connectivity,
        }


class NetworkFeatureExtractor:
    """Main class for extracting network features from a given graph."""

    def __init__(self):
        """Initialize with default extractors: basic, centrality, spectral."""
        self.extractors = {
            "basic": BasicMetricsExtractor(),
            "centrality": CentralityMetricsExtractor(),
            "spectral": SpectralMetricsExtractor(),
        }

    def get_all_metrics(self, graph: nx.Graph) -> NetworkMetrics:
        """Get all network metrics in a structured format as a NetworkMetrics instance."""
        # Gather all metrics from each extractor
        combined = {}
        for extractor in self.extractors.values():
            combined.update(extractor.extract(graph))

        # Convert to a typed NetworkMetrics object
        return NetworkMetrics(**combined)

    def get_metrics(
        self, graph: nx.Graph, metric_types: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Get a selected subset of metrics by specifying which extractors to run."""
        if metric_types is None:
            metric_types = list(self.extractors.keys())

        metrics = {}
        for mtype in metric_types:
            if mtype in self.extractors:
                metrics.update(self.extractors[mtype].extract(graph))
            else:
                logger.warning(f"Unknown metric type requested: {mtype}")

        return metrics


def calculate_error_metrics(
    actual_metrics: Dict[str, float], predicted_metrics: Dict[str, float]
) -> Dict[str, float]:
    """Calculate absolute errors for each metric in 'actual_metrics' vs. 'predicted_metrics'.

    Parameters
    ----------
    actual_metrics : Dict[str, float]
        Dictionary of reference (true) metric values.
    predicted_metrics : Dict[str, float]
        Dictionary of predicted metric values.

    Returns
    -------
    Dict[str, float]
        For each metric key in 'actual_metrics' that also exists in 'predicted_metrics',
        returns abs(actual - predicted).

    Examples
    --------
    >>> actual = {"avg_degree": 2.0, "density": 0.05}
    >>> pred = {"avg_degree": 1.8, "density": 0.06, "extra": 10.0}
    >>> errs = calculate_error_metrics(actual, pred)
    >>> print(errs)
    {"avg_degree": 0.2, "density": 0.01}
    """
    return {
        key: abs(actual_metrics[key] - predicted_metrics[key])
        for key in actual_metrics
        if key in predicted_metrics
    }
