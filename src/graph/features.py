# src/graph/features.py

"""Feature extraction module for dynamic graphs.

This module provides functions to extract various graph features and embeddings,
optimized for web applications. Features include centrality measures, structural
embeddings, and link prediction metrics.

Key Features:
- Efficient computation of multiple centrality measures
- Low-dimensional graph embeddings (SVD, LSVD)
- Link prediction features (topology-based and similarity-based)
- Strangeness scores for anomaly detection
- Robust error handling and logging
- Memory-efficient processing for web applications
"""

import networkx as nx
import numpy as np
import logging
from typing import List, Dict, Any, Union, Optional, Tuple
from sknetwork.embedding import SVD  # type: ignore
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix
from concurrent.futures import ThreadPoolExecutor
import warnings
from scipy import sparse
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import itertools

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


def compute_link_prediction_features(
    graphs: List[np.ndarray],
    feature_types: List[str],
    community_labels: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Compute static link prediction features for a sequence of graphs.

    Args:
        graphs: List of adjacency matrices
        feature_types: List of feature types to compute
        community_labels: Optional array of community labels for each node

    Returns:
        Dictionary mapping feature names to feature matrices
    """
    features = {}

    # Convert first graph to NetworkX for computing features
    G = nx.from_numpy_array(graphs[0])
    n_nodes = G.number_of_nodes()

    # Get all node pairs
    node_pairs = list(itertools.combinations(range(n_nodes), 2))

    # Compute each requested feature
    if "common_neighbors" in feature_types:
        features["common_neighbors"] = np.array(
            [len(list(nx.common_neighbors(G, u, v))) for u, v in node_pairs]
        ).reshape(
            -1, 1
        )  # Reshape to 2D

    if "jaccard" in feature_types:
        features["jaccard"] = np.array(
            [nx.jaccard_coefficient(G, [(u, v)]).__next__()[2] for u, v in node_pairs]
        ).reshape(
            -1, 1
        )  # Reshape to 2D

    if "adamic_adar" in feature_types:
        features["adamic_adar"] = np.array(
            [nx.adamic_adar_index(G, [(u, v)]).__next__()[2] for u, v in node_pairs]
        ).reshape(
            -1, 1
        )  # Reshape to 2D

    if "preferential_attachment" in feature_types:
        features["preferential_attachment"] = np.array(
            [
                nx.preferential_attachment(G, [(u, v)]).__next__()[2]
                for u, v in node_pairs
            ]
        ).reshape(
            -1, 1
        )  # Reshape to 2D

    if "resource_allocation" in feature_types:
        features["resource_allocation"] = np.array(
            [
                nx.resource_allocation_index(G, [(u, v)]).__next__()[2]
                for u, v in node_pairs
            ]
        ).reshape(
            -1, 1
        )  # Reshape to 2D

    if "degree_similarity" in feature_types:
        degrees = dict(G.degree())
        features["degree_similarity"] = np.array(
            [
                abs(degrees[u] - degrees[v]) / max(degrees[u] + degrees[v], 1)
                for u, v in node_pairs
            ]
        ).reshape(
            -1, 1
        )  # Reshape to 2D

    if "community_similarity" in feature_types and community_labels is not None:
        features["community_similarity"] = np.array(
            [
                1.0 if community_labels[u] == community_labels[v] else 0.0
                for u, v in node_pairs
            ]
        ).reshape(
            -1, 1
        )  # Reshape to 2D

    return features


def compute_temporal_link_features(
    graphs: List[Union[nx.Graph, np.ndarray]],
    node_pairs: Optional[List[Tuple[int, int]]] = None,
    window_size: int = 3,
) -> Dict[str, np.ndarray]:
    """Compute temporal link prediction features from a sequence of graphs."""
    if not graphs:
        raise ValueError("Empty graph sequence")

    # Convert numpy arrays to NetworkX graphs
    if isinstance(graphs[0], np.ndarray):
        graphs = [nx.from_numpy_array(g) for g in graphs]

    if node_pairs is None:
        node_pairs = [
            (i, j) for i in graphs[0].nodes() for j in graphs[0].nodes() if i < j
        ]

    n_pairs = len(node_pairs)
    logger.info(f"Computing temporal features for {n_pairs} node pairs")

    try:
        result = {}

        # Link history
        link_history = np.zeros((len(graphs), n_pairs))
        for t, g in enumerate(graphs):
            link_history[t] = np.array(
                [1 if g.has_edge(u, v) else 0 for u, v in node_pairs]
            )

        # Compute features using sliding windows
        result["link_frequency"] = np.array(
            [
                np.mean(link_history[max(0, t - window_size) : t + 1], axis=0)
                for t in range(len(graphs))
            ]
        ).reshape(
            -1, 1
        )  # Reshape to 2D

        # Common neighbor history
        cn_history = np.zeros((len(graphs), n_pairs))
        for t, g in enumerate(graphs):
            cn_history[t] = np.array(
                [len(set(g.neighbors(u)) & set(g.neighbors(v))) for u, v in node_pairs]
            )

        result["cn_history"] = np.array(
            [
                np.mean(cn_history[max(0, t - window_size) : t + 1], axis=0)
                for t in range(len(graphs))
            ]
        ).reshape(
            -1, 1
        )  # Reshape to 2D

        # Temporal stability (variance in link presence)
        result["temporal_stability"] = np.array(
            [
                1 - np.var(link_history[max(0, t - window_size) : t + 1], axis=0)
                for t in range(len(graphs))
            ]
        ).reshape(
            -1, 1
        )  # Reshape to 2D

        logger.info("Temporal feature computation complete")
        return result

    except Exception as e:
        logger.error(f"Temporal feature computation failed: {str(e)}")
        raise RuntimeError(f"Temporal feature computation failed: {str(e)}")


def normalize_features(
    features: Dict[str, np.ndarray], method: str = "standard"
) -> Dict[str, np.ndarray]:
    """Normalize features using specified method.

    Args:
        features: Dictionary of feature matrices
        method: Normalization method ('standard' or 'minmax')

    Returns:
        Dictionary of normalized feature matrices
    """
    if not features:
        return features

    normalized = {}
    for feat_name, feat_matrix in features.items():
        # Skip if already normalized or if normalization not needed
        if feat_name in ["community_labels", "block_membership"]:
            normalized[feat_name] = feat_matrix
            continue

        # Handle sparse matrices
        is_sparse = sparse.issparse(feat_matrix)
        if is_sparse:
            feat_matrix = feat_matrix.toarray()

        # Ensure 2D shape
        if feat_matrix.ndim == 1:
            feat_matrix = feat_matrix.reshape(-1, 1)
        elif feat_matrix.ndim > 2:
            feat_matrix = feat_matrix.reshape(feat_matrix.shape[0], -1)

        # Select scaler
        if method == "standard":
            scaler = StandardScaler()
        elif method == "minmax":
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown normalization method: {method}")

        # Normalize
        try:
            normalized_matrix = scaler.fit_transform(feat_matrix)

            # Convert back to sparse if original was sparse
            if is_sparse:
                normalized_matrix = sparse.csr_matrix(normalized_matrix)

            normalized[feat_name] = normalized_matrix

        except Exception as e:
            logger.warning(f"Failed to normalize {feat_name}: {str(e)}")
            normalized[feat_name] = feat_matrix

    return normalized
