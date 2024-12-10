"""
Feature Extractor for Graph Sequences

This module extracts features from sequences of graphs using the ChangePointDetector.
Features extracted:
1. Node-level features (degree, betweenness, closeness, eigenvector, svd, lsvd)
2. Global features by averaging node-level features
"""

import sys
import logging
from pathlib import Path
from typing import List, Dict, Union, Optional
import numpy as np

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.changepoint import ChangePointDetector

logger = logging.getLogger(__name__)


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


def extract_features(
    graphs: List[np.ndarray],
    feature_types: Optional[List[str]] = None,
    return_node_features: bool = False,
) -> Dict[str, Union[np.ndarray, List[List[float]]]]:
    """
    Extract features from a sequence of graphs.
    
    Args:
        graphs: List of adjacency matrices [n_nodes x n_nodes]
        feature_types: List of features to extract. If None, extracts all available features
        return_node_features: If True, returns both node-level and global features
                            If False, returns only global features (averaged over nodes)
    
    Returns:
        Dictionary containing:
        - If return_node_features=False:
            {feature_name: global_features} where global_features has shape [seq_len]
        - If return_node_features=True:
            {feature_name: {
                'node': node_features,     # shape [seq_len, n_nodes]
                'global': global_features  # shape [seq_len]
            }}
    
    Raises:
        ValueError: If graphs list is empty or contains invalid matrices
    """
    try:
        validate_graph_shapes(graphs)

        detector = ChangePointDetector()
        detector.initialize(graphs)
        raw_features = detector.extract_features()

        if feature_types is None:
            feature_types = [
                "degree",
                "betweenness",
                "eigenvector",
                "closeness",
                "svd",
                "lsvd",
            ]

        # Validate requested feature types
        available_features = set(raw_features.keys())
        invalid_features = set(feature_types) - available_features
        if invalid_features:
            raise ValueError(f"Requested invalid feature types: {invalid_features}")

        seq_len = len(graphs)
        n_nodes = graphs[0].shape[0]
        result = {}

        # Process each feature type
        for feat_name in feature_types:
            if feat_name not in raw_features:
                logger.warning(f"Skipping unavailable feature: {feat_name}")
                continue

            # Get feature values and ensure correct shape
            feat = np.array(raw_features[feat_name], dtype=np.float32)
            if feat.shape != (seq_len, n_nodes):
                raise ValueError(
                    f"{feat_name} feature has unexpected shape: {feat.shape}, "
                    f"expected ({seq_len}, {n_nodes})"
                )

            # Compute global features by averaging over nodes
            global_feat = feat.mean(axis=1)  # shape [seq_len]

            if return_node_features:
                result[feat_name] = {
                    'node': feat,  # shape [seq_len, n_nodes]
                    'global': global_feat  # shape [seq_len]
                }
            else:
                result[feat_name] = global_feat  # shape [seq_len]

        logger.info(
            f"Successfully extracted {len(feature_types)} feature types "
            f"for {seq_len} graphs with {n_nodes} nodes each"
        )
        return result

    except Exception as e:
        logger.error(f"Feature extraction failed: {str(e)}")
        raise


def main():
    """Example usage of feature extraction."""
    import argparse
    from create_graphs import GraphConfig, GraphType, generate_graph_sequence

    parser = argparse.ArgumentParser(description="Extract features from graph sequences")
    parser.add_argument(
        "--graph-type",
        type=str,
        choices=["BA", "ER", "NW"],
        default="BA",
        help="Type of graph to generate",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/graph_config.yaml",
        help="Path to graph configuration file",
    )
    parser.add_argument(
        "--node-features",
        action="store_true",
        help="Return both node-level and global features",
    )
    args = parser.parse_args()

    try:
        # Generate example graph sequence
        graph_type = GraphType[args.graph_type]
        config = GraphConfig.from_yaml(graph_type, config_path=args.config)
        sequence = generate_graph_sequence(config)
        graphs = sequence["graphs"]

        # Extract features
        features = extract_features(
            graphs,
            return_node_features=args.node_features
        )

        # Print results
        print("\nExtracted Features:")
        print("-" * 50)
        for feat_name, feat_data in features.items():
            if args.node_features:
                print(f"\n{feat_name}:")
                print(f"  Node features shape: {feat_data['node'].shape}")
                print(f"  Global features shape: {feat_data['global'].shape}")
                print(f"  Global feature values: {feat_data['global'][:5]}...")
            else:
                print(f"\n{feat_name}:")
                print(f"  Shape: {feat_data.shape}")
                print(f"  Values: {feat_data[:5]}...")

    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
