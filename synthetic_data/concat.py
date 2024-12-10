"""
Graph and Feature Concatenation

This module combines graph adjacency matrices with their extracted features
for all time instances in the sequences. Supports both node-level and global feature concatenation.
"""

import sys
import logging
from pathlib import Path
from typing import List, Dict, Union, Tuple, Optional
import numpy as np

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from extract_features import extract_features, validate_graph_shapes

logger = logging.getLogger(__name__)


def concat_features(
    graphs: List[np.ndarray],
    feature_types: Optional[List[str]] = None,
    use_node_features: bool = False,
    include_adjacency: bool = False,
    normalize: bool = True,
) -> Tuple[np.ndarray, Dict[str, Dict[str, float]]]:
    """
    Concatenate graph adjacency matrices with their features for each time instance.

    Args:
        graphs: List of adjacency matrices [n_nodes x n_nodes]
        feature_types: List of features to extract and concatenate. If None, uses all available
        use_node_features: If True, concatenates node-level features
                         If False, concatenates global features
        include_adjacency: If True, includes adjacency information in node-level features
        normalize: If True, normalizes features to [0,1] range

    Returns:
        Tuple containing:
        - Combined features array with shape:
          If use_node_features=True: [seq_len, n_nodes, n_features + n_nodes]
          If use_node_features=False: [seq_len, n_features + n_nodes * n_nodes]
        - Dictionary with normalization parameters for each feature

    Raises:
        ValueError: If graphs list is empty or contains invalid matrices
    """
    try:
        validate_graph_shapes(graphs)

        features = extract_features(
            graphs, feature_types=feature_types, return_node_features=use_node_features
        )

        seq_len = len(graphs)
        n_nodes = graphs[0].shape[0]
        norm_params = {}

        if use_node_features:
            n_features = len(features)
            if include_adjacency:
                # Combined: include both adjacency and features
                combined = np.zeros((seq_len, n_nodes, n_nodes + n_features))

                # Add adjacency information
                for t in range(seq_len):
                    combined[t, :, :n_nodes] = graphs[t]

                # Add node features
                for i, (feat_name, feat_data) in enumerate(features.items()):
                    node_features = feat_data["node"]
                    if normalize:
                        feat_min = node_features.min()
                        feat_max = node_features.max()
                        if feat_max > feat_min:
                            node_features = (node_features - feat_min) / (
                                feat_max - feat_min
                            )
                        norm_params[feat_name] = {"min": feat_min, "max": feat_max}
                    combined[:, :, n_nodes + i] = node_features
            else:
                # Node-level only: just the features
                combined = np.zeros((seq_len, n_nodes, n_features))
                for i, (feat_name, feat_data) in enumerate(features.items()):
                    node_features = feat_data["node"]
                    if normalize:
                        feat_min = node_features.min()
                        feat_max = node_features.max()
                        if feat_max > feat_min:
                            node_features = (node_features - feat_min) / (
                                feat_max - feat_min
                            )
                        norm_params[feat_name] = {"min": feat_min, "max": feat_max}
                    combined[:, :, i] = node_features
        else:
            # Global features with flattened adjacency
            n_features = len(features)
            combined = np.zeros((seq_len, n_nodes * n_nodes + n_features))

            # Add flattened adjacency
            flat_adj = np.array([g.flatten() for g in graphs])
            combined[:, : n_nodes * n_nodes] = flat_adj

            # Add global features
            for i, (feat_name, feat_data) in enumerate(features.items()):
                global_features = feat_data
                if normalize:
                    feat_min = global_features.min()
                    feat_max = global_features.max()
                    if feat_max > feat_min:
                        global_features = (global_features - feat_min) / (
                            feat_max - feat_min
                        )
                    norm_params[feat_name] = {"min": feat_min, "max": feat_max}
                combined[:, n_nodes * n_nodes + i] = global_features

        return combined, norm_params

    except Exception as e:
        logger.error(f"Feature concatenation failed: {str(e)}")
        raise


def main():
    """Example usage of feature concatenation."""
    import argparse
    from create_graphs import GraphConfig, GraphType, generate_graph_sequence

    parser = argparse.ArgumentParser(
        description="Concatenate graphs with their features"
    )
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
        help="Use node-level features instead of global",
    )
    parser.add_argument(
        "--include-adjacency",
        action="store_true",
        help="Include adjacency information in node-level features",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable feature normalization",
    )
    args = parser.parse_args()

    try:
        # Generate example graph sequence
        graph_type = GraphType[args.graph_type]
        config = GraphConfig.from_yaml(graph_type, config_path=args.config)
        sequence = generate_graph_sequence(config)
        graphs = sequence["graphs"]

        # Concatenate features
        combined, norm_params = concat_features(
            graphs,
            use_node_features=args.node_features,
            include_adjacency=args.include_adjacency,
            normalize=not args.no_normalize,
        )

        # Print results
        print("\nConcatenated Features:")
        print("-" * 50)
        print(f"Combined shape: {combined.shape}")
        if args.node_features:
            print(f"Format: [seq_len, n_nodes, n_features + n_nodes]")
        else:
            print(f"Format: [seq_len, n_features + n_nodes * n_nodes]")

        if not args.no_normalize:
            print("\nNormalization Parameters:")
            for feat_name, params in norm_params.items():
                print(f"  {feat_name}:")
                print(f"    Min: {params['min']:.4f}")
                print(f"    Max: {params['max']:.4f}")

    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
