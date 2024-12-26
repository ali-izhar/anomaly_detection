# synthetic_data/extract_features.py

"""
Extract pairwise node features for link prediction using NetworkX.

1. Common Neighbors: Number of shared neighbors between nodes
2. Adamic Adar: Weighted common neighbors
3. Preferential Attachment: Product of node degrees
4. Resource Allocation: Similar to Adamic Adar but with different weighting
"""

import sys
import logging
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import networkx as nx
from scipy import sparse

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from create_graphs import GraphType, generate_graph_sequence
from src.graph.params import BAParams, ERParams, NWParams, SBMParams

logger = logging.getLogger(__name__)

# Available feature types
LINK_FEATURES = [
    "common_neighbors",
    "jaccard",
    "adamic_adar",
    "preferential_attachment",
    "resource_allocation",
    "degree_similarity",
]


def validate_graph_shapes(graphs: List[np.ndarray]) -> None:
    """Validate that all graphs in sequence have consistent shapes."""
    if not graphs:
        raise ValueError("Empty graph sequence")

    n_nodes = graphs[0].shape[0]
    if graphs[0].shape != (n_nodes, n_nodes):
        raise ValueError(
            f"First graph has invalid shape: {graphs[0].shape}, expected square matrix"
        )

    for i, g in enumerate(graphs):
        if not isinstance(g, np.ndarray):
            raise ValueError(f"Graph {i} is not a numpy array")
        if g.shape != (n_nodes, n_nodes):
            raise ValueError(
                f"Graph {i} has inconsistent shape: {g.shape}, expected {(n_nodes, n_nodes)}"
            )


def compute_node_features(G: nx.Graph) -> Dict[str, np.ndarray]:
    """Compute node-level features.

    Args:
        G: NetworkX graph

    Returns:
        Dictionary of node features
    """
    n = G.number_of_nodes()
    features = {}

    # Compute node degrees
    degrees = np.array([G.degree(i) for i in range(n)])
    features["degree"] = degrees / degrees.max()  # Normalize

    # Compute clustering coefficients
    clustering = np.array([nx.clustering(G, i) for i in range(n)])
    features["clustering"] = clustering

    # Compute PageRank
    pagerank = np.array(list(nx.pagerank(G).values()))
    features["pagerank"] = pagerank

    return features


def compute_link_features(graph: np.ndarray) -> Dict[str, sparse.csr_matrix]:
    """Compute pairwise node features specifically for link prediction in sparse BA graphs.
    Features optimized for preferential attachment behavior:
    1. Common Neighbors (normalized)
    2. Jaccard Coefficient
    3. Adamic-Adar (normalized)
    4. Preferential Attachment (normalized)
    5. Resource Allocation (normalized)
    6. Degree Similarity (new feature for BA graphs)
    """
    G = nx.from_numpy_array(graph)
    n_nodes = graph.shape[0]

    features = {}
    node_pairs = [(i, j) for i in range(n_nodes) for j in range(i + 1, n_nodes)]

    # Initialize matrices
    cn_matrix = sparse.lil_matrix((n_nodes, n_nodes))
    jc_matrix = sparse.lil_matrix((n_nodes, n_nodes))
    aa_matrix = sparse.lil_matrix((n_nodes, n_nodes))
    pa_matrix = sparse.lil_matrix((n_nodes, n_nodes))
    ra_matrix = sparse.lil_matrix((n_nodes, n_nodes))
    ds_matrix = sparse.lil_matrix((n_nodes, n_nodes))

    # Pre-compute node degrees and neighborhoods
    degrees = dict(G.degree())
    max_degree = max(degrees.values())
    avg_degree = sum(degrees.values()) / n_nodes
    min_degree_threshold = max(3, int(0.15 * avg_degree))

    # Pre-compute node neighborhoods for efficiency
    neighborhoods = {n: set(G.neighbors(n)) for n in range(n_nodes)}

    # Pre-compute max values for normalization
    max_aa = 0
    max_ra = 0

    # First pass to find max values for normalization
    for u, v in node_pairs:
        if not G.has_edge(u, v):
            # Skip pairs where both nodes have very low degree
            if degrees[u] < min_degree_threshold and degrees[v] < min_degree_threshold:
                continue

            common_neighbors = neighborhoods[u] & neighborhoods[v]
            if common_neighbors:
                aa_score = sum(1 / np.log(degrees[w]) for w in common_neighbors)
                ra_score = sum(1 / degrees[w] for w in common_neighbors)
                max_aa = max(max_aa, aa_score)
                max_ra = max(max_ra, ra_score)

    # Compute features only for non-connected pairs
    for u, v in node_pairs:
        if not G.has_edge(u, v):
            # Skip pairs where both nodes have very low degree
            if degrees[u] < min_degree_threshold and degrees[v] < min_degree_threshold:
                continue

            # Get neighborhoods
            u_neighbors = neighborhoods[u]
            v_neighbors = neighborhoods[v]
            common_neighbors = u_neighbors & v_neighbors
            union_neighbors = u_neighbors | v_neighbors
            cn_count = len(common_neighbors)

            # Compute degree similarity
            min_deg = min(degrees[u], degrees[v])
            max_deg = max(degrees[u], degrees[v])
            deg_sim = min_deg / max_deg if max_deg > 0 else 0

            # Only proceed if there are common neighbors or high degree similarity
            if cn_count > 0 or deg_sim > 0.3:
                # 1. Normalized Common Neighbors with degree-based threshold
                deg_threshold = 0.2 * max(degrees[u], degrees[v])
                if cn_count >= deg_threshold:
                    normalized_cn = cn_count / max_degree
                    cn_matrix[u, v] = normalized_cn
                    cn_matrix[v, u] = normalized_cn

                # 2. Jaccard Coefficient with degree-based threshold
                jaccard = cn_count / len(union_neighbors) if union_neighbors else 0
                if jaccard >= 0.15 * (deg_sim + 0.1):
                    jc_matrix[u, v] = jaccard
                    jc_matrix[v, u] = jaccard

                # 3. Normalized Adamic-Adar with degree-based threshold
                if common_neighbors:
                    aa_score = sum(1 / np.log(degrees[w]) for w in common_neighbors)
                    normalized_aa = aa_score / max_aa if max_aa > 0 else 0
                    if normalized_aa >= 0.15 * deg_sim:
                        aa_matrix[u, v] = normalized_aa
                        aa_matrix[v, u] = normalized_aa

                # 4. Normalized Resource Allocation
                if common_neighbors:
                    ra_score = sum(1 / degrees[w] for w in common_neighbors)
                    normalized_ra = ra_score / max_ra if max_ra > 0 else 0
                    if normalized_ra >= 0.15 * deg_sim:
                        ra_matrix[u, v] = normalized_ra
                        ra_matrix[v, u] = normalized_ra

                # 5. Preferential Attachment with degree-based threshold
                pa_score = (degrees[u] * degrees[v]) / (max_degree * max_degree)
                if pa_score > 0.1:
                    pa_matrix[u, v] = pa_score
                    pa_matrix[v, u] = pa_score

                # 6. Degree Similarity with higher threshold
                if deg_sim > 0.3:
                    ds_matrix[u, v] = deg_sim
                    ds_matrix[v, u] = deg_sim

    # Convert to CSR format for efficiency
    features["common_neighbors"] = cn_matrix.tocsr()
    features["jaccard"] = jc_matrix.tocsr()
    features["adamic_adar"] = aa_matrix.tocsr()
    features["preferential_attachment"] = pa_matrix.tocsr()
    features["resource_allocation"] = ra_matrix.tocsr()
    features["degree_similarity"] = ds_matrix.tocsr()

    return features


def extract_features(
    graphs: List[np.ndarray],
    feature_types: Optional[List[str]] = None,
) -> Dict[str, List[sparse.csr_matrix]]:
    """Extract features for link prediction from a sequence of graphs.

    Args:
        graphs: List of adjacency matrices [n_nodes x n_nodes]
        feature_types: List of features to extract. If None, extracts all link features

    Returns:
        Dictionary containing:
        {feature_name: List[sparse_matrix]} where each sparse matrix has shape [n_nodes, n_nodes]
        representing pairwise node features at each timestep
    """
    try:
        validate_graph_shapes(graphs)
        if feature_types is None:
            feature_types = LINK_FEATURES

        # Validate requested feature types
        invalid_features = set(feature_types) - set(LINK_FEATURES)
        if invalid_features:
            raise ValueError(
                f"Requested invalid feature types: {invalid_features}. "
                f"Available features: {LINK_FEATURES}"
            )

        seq_len = len(graphs)
        result = {feat: [] for feat in feature_types}

        # Extract features for each graph in sequence
        for t in range(seq_len):
            link_features = compute_link_features(graphs[t])
            for feat_name in feature_types:
                result[feat_name].append(link_features[feat_name])

        logger.info(
            f"Successfully extracted {len(feature_types)} link prediction features "
            f"for {seq_len} graphs"
        )
        return result

    except Exception as e:
        logger.error(f"Feature extraction failed: {str(e)}")
        raise


def main():
    """Example usage of feature extraction."""
    import argparse
    from create_graphs import (
        GraphType,
        generate_graph_sequence,
        BAParams,
        ERParams,
        NWParams,
        SBMParams,
    )

    parser = argparse.ArgumentParser(
        description="Extract link prediction features from graph sequences"
    )
    parser.add_argument(
        "--graph-type",
        type=str,
        choices=["BA", "ER", "NW", "SBM"],
        default="BA",
        help="Type of graph to generate",
    )
    args = parser.parse_args()

    try:
        # Example parameters for each graph type
        params_dict = {
            "BA": BAParams(
                n=500,  # Number of nodes
                seq_len=50,  # Sequence length
                min_segment=10,  # Minimum segment length
                min_changes=1,  # Minimum number of changes
                max_changes=2,  # Maximum number of changes
                initial_edges=3,  # Initial number of edges per new node
                min_edges=2,  # Minimum edges after change
                max_edges=5,  # Maximum edges after change
                pref_exp=1.0,  # Preferential attachment exponent
            ),
            "ER": ERParams(
                n=500,
                seq_len=50,
                min_segment=10,
                min_changes=1,
                max_changes=2,
                initial_prob=0.1,  # Initial edge probability
                min_prob=0.05,  # Minimum probability after change
                max_prob=0.15,  # Maximum probability after change
            ),
            "NW": NWParams(
                n=500,
                seq_len=50,
                min_segment=10,
                min_changes=1,
                max_changes=2,
                k_nearest=4,  # Number of nearest neighbors
                initial_prob=0.1,  # Initial rewiring probability
                min_prob=0.05,  # Minimum probability after change
                max_prob=0.15,  # Maximum probability after change
            ),
            "SBM": SBMParams(
                n=500,
                seq_len=50,
                min_segment=10,
                min_changes=1,
                max_changes=2,
                num_blocks=5,  # Number of communities
                min_block_size=80,  # Minimum community size
                max_block_size=120,  # Maximum community size
                initial_intra_prob=0.3,  # Initial within-community probability
                initial_inter_prob=0.05,  # Initial between-community probability
                min_intra_prob=0.25,  # Minimum intra-community probability
                max_intra_prob=0.35,  # Maximum intra-community probability
                min_inter_prob=0.03,  # Minimum inter-community probability
                max_inter_prob=0.07,  # Maximum inter-community probability
            ),
        }

        # Generate example graph sequence
        graph_type = GraphType[args.graph_type]
        params = params_dict[args.graph_type]
        sequence = generate_graph_sequence(graph_type, params)
        graphs = sequence["graphs"]

        features = extract_features(graphs, feature_types=LINK_FEATURES)
        print("\nExtracted Link Prediction Features:")
        print("-" * 50)
        for feat_name, feat_matrices in features.items():
            print(f"\n{feat_name}:")
            print(f"  Number of timesteps: {len(feat_matrices)}")
            print(f"  Feature matrix shape: {feat_matrices[0].shape}")
            print(f"  Non-zero entries: {feat_matrices[0].nnz}")

            # Calculate density excluding diagonal
            n = feat_matrices[0].shape[0]
            max_entries = n * (n - 1) / 2  # Only upper triangle
            density = feat_matrices[0].nnz / (
                2 * max_entries
            )  # Factor 2 because we store symmetric
            print(f"  Feature density: {density:.4f}")

            # Print some statistics
            values = feat_matrices[0].data
            if len(values) > 0:
                print(f"  Value range: [{values.min():.4f}, {values.max():.4f}]")
                print(f"  Mean value: {values.mean():.4f}")

    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
