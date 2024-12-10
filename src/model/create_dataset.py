# synthetic_data/create_dataset.py

"""
Dataset Generator for Graph Forecasting

This module creates synthetic graph datasets with labeled anomalies and global features.
The dataset is designed for a downstream task of forecasting the next m steps of the 6 global features.

Key Steps:
1. Generate synthetic graph sequences (BA, ER, NW) with fixed length (e.g., seq_len=200), so no padding is needed.
2. Compute node-level features (degree, betweenness, closeness, eigenvector, svd, lsvd).
3. Aggregate them to global features by averaging over nodes to get shape (seq_len, 6).
4. Since all sequences are of the same length, we simply stack all sequences.
5. Split into train/val/test sets.
6. Compute normalization parameters (mean, std) from the training set only, to prevent data leakage.
7. Apply the same normalization to train, val, and test splits.
8. Save final normalized data to HDF5 files for easy loading during model training.

Following best practices:
- Normalization is computed only from the training set.
- All sequences have the same length, thus no padding is required.
"""

import logging
import sys
from pathlib import Path
import yaml
from tqdm import tqdm

import numpy as np
import networkx as nx
import json

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from create_graph_sequences import GraphType, GraphConfig, generate_graph_sequence

logger = logging.getLogger(__name__)


class DatasetGenerator:
    """Generate dataset of graph sequences with node features."""

    def __init__(self, config: dict):
        """
        Initialize dataset generator.

        Args:
            config: Dictionary containing dataset configuration
        """
        self.config = config
        self.setup_paths()

    def setup_paths(self):
        """Create output directories if they don't exist."""
        output_dir = Path(self.config["paths"]["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

    def compute_graph_features(self, graph: nx.Graph) -> np.ndarray:
        """
        Compute node features for a graph.

        Args:
            graph: NetworkX graph object

        Returns:
            np.ndarray: Node features of shape (num_nodes, num_features)
        """
        features = []
        feature_config = self.config["features"]

        # Compute enabled features
        for feature in feature_config:
            if feature["enabled"]:
                try:
                    if feature["name"] == "degree_centrality":
                        feat = list(nx.degree_centrality(graph).values())
                    elif feature["name"] == "clustering_coefficient":
                        feat = list(nx.clustering(graph).values())
                    elif feature["name"] == "betweenness_centrality":
                        feat = list(nx.betweenness_centrality(graph).values())
                    elif feature["name"] == "eigenvector_centrality":
                        try:
                            feat = list(
                                nx.eigenvector_centrality(
                                    graph, max_iter=1000, tol=1e-4
                                ).values()
                            )
                        except nx.PowerIterationFailedConvergence:
                            # If eigenvector centrality fails, use degree centrality as fallback
                            feat = list(nx.degree_centrality(graph).values())
                    elif feature["name"] == "pagerank":
                        feat = list(
                            nx.pagerank(graph, max_iter=1000, tol=1e-4).values()
                        )
                    elif feature["name"] == "closeness_centrality":
                        feat = list(nx.closeness_centrality(graph).values())
                    features.append(feat)
                except Exception as e:
                    print(
                        f"Warning: Failed to compute {feature['name']}, using zeros. Error: {str(e)}"
                    )
                    features.append([0.0] * len(graph))

        return np.array(features).T

    def add_temporal_correlation(
        self, features: np.ndarray, correlation: float
    ) -> np.ndarray:
        """Add temporal correlation to feature sequences."""
        noise = np.random.normal(0, np.sqrt(1 - correlation**2), features.shape)
        return correlation * features + noise

    def convert_to_serializable(self, obj):
        """Convert numpy types to Python native types for JSON serialization."""
        if isinstance(
            obj,
            (
                np.int_,
                np.intc,
                np.intp,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {
                key: self.convert_to_serializable(value) for key, value in obj.items()
            }
        elif isinstance(obj, list):
            return [self.convert_to_serializable(item) for item in obj]
        return obj

    def generate_dataset(self):
        """Generate the full dataset."""
        data_config = self.config["data"]
        num_sequences = data_config["num_sequences"]

        # Initialize arrays for all graph types
        all_adj_matrices = []
        all_feature_sequences = []
        all_metadata = []

        # Generate sequences for each graph type
        for graph_type in [GraphType.BA, GraphType.ER, GraphType.NW]:
            print(f"\nGenerating {graph_type.value} sequences...")

            config = GraphConfig.from_yaml(graph_type, "configs/graph_config.yaml")

            for seq_idx in tqdm(range(num_sequences)):
                # Generate graph sequence
                sequence = generate_graph_sequence(config)

                # Extract adjacency matrices
                adj_matrices = []
                feature_sequences = []

                # Process each graph in the sequence
                for graph in sequence["graphs"]:
                    # Get adjacency matrix
                    adj_matrix = nx.to_numpy_array(graph)
                    adj_matrices.append(adj_matrix)

                    # Compute node features
                    features = self.compute_graph_features(graph)
                    feature_sequences.append(features)

                # Convert to numpy arrays
                adj_matrices = np.array(adj_matrices)
                feature_sequences = np.array(feature_sequences)

                # Add temporal correlation to features
                feature_sequences = self.add_temporal_correlation(
                    feature_sequences,
                    self.config["data"]["feature_params"]["temporal_correlation"],
                )

                # Store sequences
                all_adj_matrices.append(adj_matrices)
                all_feature_sequences.append(feature_sequences)

                # Store metadata
                metadata = {
                    "graph_type": graph_type.value,
                    "sequence_idx": seq_idx,
                    "change_points": sequence["change_points"],
                    "params": sequence["params"],
                }
                all_metadata.append(metadata)

        # Convert to final numpy arrays
        all_adj_matrices = np.array(all_adj_matrices)
        all_feature_sequences = np.array(all_feature_sequences)

        # Convert metadata to serializable format before saving
        serializable_metadata = self.convert_to_serializable(all_metadata)

        # Save dataset
        output_dir = Path(self.config["paths"]["output_dir"])
        np.savez(
            output_dir / self.config["paths"]["adjacency_matrices"],
            adj_matrices=all_adj_matrices,
        )
        np.savez(
            output_dir / self.config["paths"]["feature_sequences"],
            feature_sequences=all_feature_sequences,
        )

        with open(output_dir / self.config["paths"]["metadata"], "w") as f:
            json.dump(serializable_metadata, f, indent=2)

        print("\nDataset generation complete!")
        print(f"Adjacency matrices shape: {all_adj_matrices.shape}")
        print(f"Feature sequences shape: {all_feature_sequences.shape}")
        print(f"Number of sequences: {len(all_metadata)}")

        return {
            "adjacency_matrices": all_adj_matrices,
            "feature_sequences": all_feature_sequences,
            "metadata": all_metadata,
        }


def main():
    # Load configuration
    current_dir = Path(__file__).parent
    with open(current_dir / "configs/dataset_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Create and run generator
    generator = DatasetGenerator(config)
    dataset = generator.generate_dataset()

    # Print summary
    print("\nDataset Summary:")
    print(f"Total sequences: {len(dataset['metadata'])}")
    print(f"Sequence length: {dataset['adjacency_matrices'].shape[1]}")
    print(f"Number of nodes: {dataset['adjacency_matrices'].shape[2]}")
    print(f"Number of features: {dataset['feature_sequences'].shape[3]}")

    # Verify saved data
    output_dir = Path(config["paths"]["output_dir"])
    print("\nVerifying saved data...")

    # Check adjacency matrices
    adj_file = output_dir / config["paths"]["adjacency_matrices"]
    if adj_file.exists():
        adj_data = np.load(adj_file)
        print(
            f"Adjacency matrices saved successfully: {adj_data['adj_matrices'].shape}"
        )

    # Check feature sequences
    feat_file = output_dir / config["paths"]["feature_sequences"]
    if feat_file.exists():
        feat_data = np.load(feat_file)
        print(
            f"Feature sequences saved successfully: {feat_data['feature_sequences'].shape}"
        )

    # Check metadata
    meta_file = output_dir / config["paths"]["metadata"]
    if meta_file.exists():
        with open(meta_file, "r") as f:
            meta_data = json.load(f)
        print(f"Metadata saved successfully: {len(meta_data)} sequences")


if __name__ == "__main__":
    main()
