# src/graph/features.py

import networkx as nx
import numpy as np
import logging
from typing import List, Dict, Any, Union, Optional
from sknetwork.embedding import SVD  # type: ignore
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


def extract_centralities(graphs: List[np.ndarray]) -> Dict[str, List[List[float]]]:
    """Extract multiple centrality measures from a sequence of graphs.
    1. Degree: c_d(v) = degree(v) / (|V| - 1)
    2. Betweenness: c_b(v) = sum over all pairs s != v != t of (sigma_st(v) / sigma_st)
    3. Eigenvector: Ax = lambda x, where x is the centrality vector
    4. Closeness: c_c(v) = (|V| - 1) / sum of distances from v to all other nodes

    Args:
        graphs: List of adjacency matrices [n x n]

    Returns:
        Dictionary mapping centrality names to lists of node centralities per graph

    Raises:
        ValueError: If input graphs list is empty or contains invalid matrices
    """
    if not graphs:
        logger.error("Attempted to extract centralities from empty graph sequence")
        raise ValueError("Empty graph sequence")
    if not all(isinstance(g, np.ndarray) and g.ndim == 2 for g in graphs):
        logger.error("Invalid graph format detected")
        raise ValueError("All graphs must be 2D adjacency matrices")

    centrality_methods = {
        "degree": nx.degree_centrality,
        "betweenness": nx.betweenness_centrality,
        "eigenvector": nx.eigenvector_centrality_numpy,
        "closeness": nx.closeness_centrality,
    }

    logger.info(f"Computing centralities for {len(graphs)} graphs")
    try:
        result = {}
        for name, centrality_func in centrality_methods.items():
            logger.debug(f"Computing {name} centrality")
            result[name] = [
                [
                    float(val)
                    for val in centrality_func(nx.from_numpy_array(graph)).values()
                ]
                for graph in graphs
            ]
            logger.debug(f"Completed {name} centrality computation")

        logger.info("Centrality computation complete")
        return result
    except Exception as e:
        logger.error(f"Centrality computation failed: {str(e)}")
        raise RuntimeError(f"Centrality computation failed: {str(e)}")


def compute_embeddings(
    graphs: List[np.ndarray], method: str = "svd", n_components: int = 2
) -> List[np.ndarray]:
    """Compute low-dimensional graph embeddings preserving structural information.
    - SVD: X = U Sigma V^T, embedding = first k columns of U
    - LSVD: Applies SVD to Laplacian L = D - A
      where A is the adjacency matrix and D is the degree matrix

    Args:
        graphs: List of adjacency matrices
        method: Embedding type ('svd' or 'lsvd')
        n_components: Embedding dimension k

    Returns:
        List of embeddings as flattened vectors

    Raises:
        ValueError: For invalid method or input
    """
    if not graphs:
        logger.error("Attempted to compute embeddings for empty graph sequence")
        raise ValueError("Empty graph sequence")
    if method not in ["svd", "lsvd"]:
        logger.error(f"Invalid embedding method requested: {method}")
        raise ValueError(f"Unknown embedding method: {method}")

    logger.info(f"Computing {method.upper()} embeddings with {n_components} components")
    try:
        if method == "svd":
            embedder = SVD(n_components=n_components)
            logger.debug("Using SVD embedder")
            embeddings = [embedder.fit_transform(matrix) for matrix in graphs]
        else:
            logger.debug("Computing Laplacian SVD embeddings")
            embeddings = [
                np.linalg.svd(compute_laplacian(matrix))[1] 
                for matrix in graphs
            ]

        logger.info(f"Successfully computed embeddings for {len(graphs)} graphs")
        return embeddings
    except Exception as e:
        logger.error(f"Embedding computation failed: {str(e)}")
        raise RuntimeError(f"Embedding computation failed: {str(e)}")


def strangeness_point(
    data: Union[List[Any], np.ndarray],
    n_clusters: int = 1,
    random_state: Optional[int] = 42,
) -> np.ndarray:
    """Compute strangeness scores using distance to cluster centers.
    Strangeness alpha_i = minimum Euclidean distance between x_i and any cluster centroid mu_j

    Args:
        data: Input vectors to compute strangeness for
        n_clusters: Number of clusters k
        random_state: Random seed for reproducibility

    Returns:
        Array of strangeness scores [alpha_1, ..., alpha_n]

    Raises:
        ValueError: If data is empty or invalid
    """
    if not data:
        logger.error("Attempted to compute strangeness for empty data")
        raise ValueError("Empty data sequence")

    try:
        logger.debug(f"Computing strangeness with {n_clusters} clusters")
        data_array = np.array(data)
        # if data is 3D array (because embeddings are [2, n]), flatten it
        if data_array.ndim == 3:
            data_array = data_array.reshape(-1, data_array.shape[-1])
        kmeans = KMeans(n_clusters=n_clusters, n_init="auto", random_state=random_state)
        strangeness = kmeans.fit_transform(data_array)
        logger.debug(f"Strangeness shape: {strangeness.shape}")
        return strangeness
    except Exception as e:
        logger.error(f"Strangeness computation failed: {str(e)}")
        raise RuntimeError(f"Strangeness computation failed: {str(e)}")


def compute_laplacian(adjacency: np.ndarray) -> np.ndarray:
    """Compute unnormalized Laplacian matrix L = D - A.
    D: Degree matrix (diagonal)
    A: Adjacency matrix

    Args:
        adjacency: Graph adjacency matrix

    Returns:
        Unnormalized Laplacian matrix
    """
    logger.debug(f"Computing Laplacian for matrix of shape {adjacency.shape}")
    degrees = np.sum(adjacency, axis=1)
    degree_matrix = np.diag(degrees)
    laplacian = degree_matrix - adjacency
    logger.debug("Laplacian computation complete")
    return laplacian


def graph_to_adjacency(graph: nx.Graph) -> np.ndarray:
    """Convert NetworkX graph to adjacency matrix representation.
    A[i, j] = 1 if edge (i, j) exists, 0 otherwise

    Args:
        graph: NetworkX graph object

    Returns:
        Binary adjacency matrix [n x n]
    """
    logger.debug(
        f"Converting NetworkX graph with {graph.number_of_nodes()} nodes to adjacency matrix"
    )
    adjacency = nx.to_numpy_array(graph, dtype=np.float64)
    logger.debug(f"Generated adjacency matrix of shape {adjacency.shape}")
    return adjacency


def adjacency_to_graph(adjacency: np.ndarray) -> nx.Graph:
    """Convert adjacency matrix to NetworkX graph representation.
    Creates an undirected graph where edge (i, j) exists if A[i, j] > 0

    Args:
        adjacency: Binary adjacency matrix [n x n]

    Returns:
        NetworkX graph object

    Raises:
        ValueError: If matrix is not square or contains invalid values
    """
    if not isinstance(adjacency, np.ndarray) or adjacency.ndim != 2:
        logger.error("Invalid adjacency matrix format")
        raise ValueError("Input must be 2D numpy array")
    if adjacency.shape[0] != adjacency.shape[1]:
        logger.error(f"Non-square adjacency matrix: {adjacency.shape}")
        raise ValueError("Adjacency matrix must be square")

    logger.debug(
        f"Converting adjacency matrix of shape {adjacency.shape} to NetworkX graph"
    )
    graph = nx.from_numpy_array(adjacency)
    logger.debug(
        f"Created graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges"
    )
    return graph
