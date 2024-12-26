# src/graph/features.py

"""Feature extraction module for dynamic graphs.

This module provides functions to extract various graph features and embeddings,
optimized for web applications. Features include centrality measures, structural
embeddings, and anomaly detection metrics.

Key Features:
- Efficient computation of multiple centrality measures
- Low-dimensional graph embeddings (SVD, LSVD)
- Strangeness scores for anomaly detection
- Robust error handling and logging
- Memory-efficient processing for web applications
"""

import networkx as nx
import numpy as np
import logging
from typing import List, Dict, Any, Union, Optional
from sknetwork.embedding import SVD  # type: ignore
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix
from concurrent.futures import ThreadPoolExecutor
import warnings

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


def extract_centralities(
    graphs: List[np.ndarray],
    measures: Optional[List[str]] = None,
    batch_size: int = 10,
    n_jobs: int = -1,
) -> Dict[str, List[List[float]]]:
    """Extract multiple centrality measures from a sequence of graphs.

    Supported centrality measures:
    1. Degree: c_d(v) = degree(v) / (|V| - 1)
    2. Betweenness: c_b(v) = sum over all pairs s != v != t of (sigma_st(v) / sigma_st)
    3. Eigenvector: Ax = lambda x, where x is the centrality vector
    4. Closeness: c_c(v) = (|V| - 1) / sum of distances from v to all other nodes

    Args:
        graphs: List of adjacency matrices [n x n]
        measures: List of centrality measures to compute. If None, computes all.
        batch_size: Number of graphs to process in parallel
        n_jobs: Number of parallel jobs (-1 for all cores)

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

    available_measures = {
        "degree": nx.degree_centrality,
        "betweenness": nx.betweenness_centrality,
        "eigenvector": nx.eigenvector_centrality_numpy,
        "closeness": nx.closeness_centrality,
    }

    if measures is None:
        measures = list(available_measures.keys())
    else:
        invalid_measures = set(measures) - set(available_measures.keys())
        if invalid_measures:
            raise ValueError(f"Invalid centrality measures: {invalid_measures}")

    logger.info(f"Computing {len(measures)} centralities for {len(graphs)} graphs")

    def process_batch(
        batch_graphs: List[np.ndarray], measure: str
    ) -> List[List[float]]:
        """Process a batch of graphs for a given centrality measure."""
        centrality_func = available_measures[measure]
        try:
            return [
                [float(val) for val in centrality_func(nx.from_numpy_array(g)).values()]
                for g in batch_graphs
            ]
        except Exception as e:
            logger.error(f"Failed to compute {measure} centrality: {str(e)}")
            raise

    try:
        result = {}
        with ThreadPoolExecutor(max_workers=n_jobs if n_jobs > 0 else None) as executor:
            for measure in measures:
                logger.debug(f"Computing {measure} centrality")
                batches = [
                    graphs[i : i + batch_size]
                    for i in range(0, len(graphs), batch_size)
                ]

                futures = [
                    executor.submit(process_batch, batch, measure) for batch in batches
                ]

                # Collect results
                measure_results = []
                for future in futures:
                    measure_results.extend(future.result())

                result[measure] = measure_results
                logger.debug(f"Completed {measure} centrality computation")

        logger.info("Centrality computation complete")
        return result

    except Exception as e:
        logger.error(f"Centrality computation failed: {str(e)}")
        raise RuntimeError(f"Centrality computation failed: {str(e)}")


def compute_embeddings(
    graphs: List[np.ndarray],
    method: str = "svd",
    n_components: int = 2,
    use_sparse: bool = True,
    random_state: Optional[int] = None,
) -> List[np.ndarray]:
    """Compute low-dimensional graph embeddings preserving structural information.

    Methods:
    - SVD: Uses sknetwork's SVD embedding
    - LSVD: Computes unnormalized Laplacian SVD

    Args:
        graphs: List of adjacency matrices [n x n]
        method: Embedding type ('svd' or 'lsvd')
        n_components: Embedding dimension k
        use_sparse: Whether to use sparse matrix format
        random_state: Random seed for reproducibility

    Returns:
        List of embeddings, each is (n_nodes, n_components) array

    Raises:
        ValueError: If method is unknown or input is empty
    """
    if not graphs:
        logger.error("Attempted to compute embeddings for empty graph sequence")
        raise ValueError("Empty graph sequence")

    if method not in ["svd", "lsvd"]:
        logger.error(f"Invalid embedding method requested: {method}")
        raise ValueError(f"Unknown embedding method: {method}")

    logger.info(f"Computing {method.upper()} embeddings with {n_components} components")

    def to_sparse(matrix: np.ndarray) -> csr_matrix:
        """Convert dense matrix to sparse format if beneficial."""
        if (
            use_sparse
            and matrix.size > 1000
            and np.count_nonzero(matrix) / matrix.size < 0.1
        ):
            return csr_matrix(matrix)
        return matrix

    try:
        if method == "svd":
            embedder = SVD(n_components=n_components)
            logger.debug("Using SVD embedder")
            embeddings = []
            for matrix in graphs:
                sparse_matrix = to_sparse(matrix)
                emb = embedder.fit_transform(sparse_matrix)
                if emb.ndim == 1:
                    emb = emb[:, np.newaxis]
                embeddings.append(emb)

        else:  # LSVD
            embeddings = []
            for matrix in graphs:
                sparse_matrix = to_sparse(matrix)
                lap = compute_laplacian(sparse_matrix)
                U, _, _ = np.linalg.svd(lap.toarray() if use_sparse else lap)
                emb = U[:, :n_components]
                embeddings.append(emb)

        logger.info(f"Successfully computed embeddings for {len(graphs)} graphs")
        return embeddings

    except Exception as e:
        logger.error(f"Embedding computation failed: {str(e)}")
        raise RuntimeError(f"Embedding computation failed: {str(e)}")


def strangeness_point(
    data: Union[List[Any], np.ndarray],
    n_clusters: int = 1,
    random_state: Optional[int] = 42,
    batch_size: Optional[int] = None,
) -> np.ndarray:
    """Compute strangeness scores using distance to cluster centers.

    Strangeness alpha_i = minimum Euclidean distance between x_i and any cluster centroid mu_j

    Args:
        data: Input vectors to compute strangeness for
        n_clusters: Number of clusters k
        random_state: Random seed for reproducibility
        batch_size: Size of batches for large datasets

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

        # Handle 3D arrays (e.g., from embeddings)
        if data_array.ndim == 3:
            data_array = data_array.reshape(-1, data_array.shape[-1])

        # Use mini-batch KMeans for large datasets
        if batch_size and data_array.shape[0] > batch_size:
            from sklearn.cluster import MiniBatchKMeans

            kmeans = MiniBatchKMeans(
                n_clusters=n_clusters, batch_size=batch_size, random_state=random_state
            )
        else:
            kmeans = KMeans(
                n_clusters=n_clusters, n_init="auto", random_state=random_state
            )

        strangeness = kmeans.fit_transform(data_array)
        logger.debug(f"Strangeness shape: {strangeness.shape}")
        return strangeness

    except Exception as e:
        logger.error(f"Strangeness computation failed: {str(e)}")
        raise RuntimeError(f"Strangeness computation failed: {str(e)}")


def compute_laplacian(
    adjacency: Union[np.ndarray, csr_matrix], normalized: bool = False
) -> Union[np.ndarray, csr_matrix]:
    """Compute graph Laplacian matrix.

    Args:
        adjacency: Graph adjacency matrix (dense or sparse)
        normalized: Whether to compute normalized Laplacian

    Returns:
        Laplacian matrix (same format as input)
    """
    logger.debug(f"Computing {'normalized ' if normalized else ''}Laplacian")

    is_sparse = isinstance(adjacency, csr_matrix)
    degrees = np.array(adjacency.sum(axis=1)).flatten()

    if normalized:
        # Normalized Laplacian: L = I - D^(-1/2)AD^(-1/2)
        d_sqrt = np.sqrt(degrees)
        d_sqrt[d_sqrt == 0] = 1  # Handle isolated vertices

        if is_sparse:
            D_sqrt = csr_matrix((1 / d_sqrt, (range(len(d_sqrt)), range(len(d_sqrt)))))
            L = csr_matrix(np.eye(adjacency.shape[0])) - D_sqrt @ adjacency @ D_sqrt
        else:
            D_sqrt = np.diag(1 / d_sqrt)
            L = np.eye(adjacency.shape[0]) - D_sqrt @ adjacency @ D_sqrt
    else:
        # Unnormalized Laplacian: L = D - A
        if is_sparse:
            D = csr_matrix((degrees, (range(len(degrees)), range(len(degrees)))))
            L = D - adjacency
        else:
            D = np.diag(degrees)
            L = D - adjacency

    logger.debug("Laplacian computation complete")
    return L


def compute_graph_statistics(graphs: List[np.ndarray]) -> Dict[str, List[float]]:
    """Compute basic graph statistics for monitoring.

    Args:
        graphs: List of adjacency matrices

    Returns:
        Dictionary of statistics per graph:
        - density: Edge density
        - avg_degree: Average node degree
        - clustering: Average clustering coefficient
        - diameter: Graph diameter (or estimate for large graphs)
    """
    logger.info(f"Computing statistics for {len(graphs)} graphs")
    stats = {"density": [], "avg_degree": [], "clustering": [], "diameter": []}

    try:
        for adj in graphs:
            G = nx.from_numpy_array(adj)
            stats["density"].append(nx.density(G))
            stats["avg_degree"].append(np.mean(list(dict(G.degree()).values())))
            stats["clustering"].append(nx.average_clustering(G))

            # Handle disconnected graphs for diameter computation
            if not nx.is_connected(G):
                # Use the largest connected component
                largest_cc = max(nx.connected_components(G), key=len)
                subgraph = G.subgraph(largest_cc)
                diameter = nx.diameter(subgraph)
            else:
                diameter = nx.diameter(G)
            stats["diameter"].append(diameter)

        logger.info("Statistics computation complete")
        return stats

    except Exception as e:
        logger.error(f"Statistics computation failed: {str(e)}")
        raise RuntimeError(f"Statistics computation failed: {str(e)}")
