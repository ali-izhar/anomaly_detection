# src/model/dataset.py

"""
Dynamic Graph Dataset Loader

This module provides a PyTorch dataset class for loading and processing different variants
of the synthetic graph sequence datasets (node-level, global, or combined features).
"""

import h5py
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import logging
import yaml

import numpy as np
import torch

from torch_geometric_temporal.signal import (
    DynamicGraphTemporalSignal,
    temporal_signal_split,
)

logger = logging.getLogger(__name__)


class DynamicGraphDataset:
    """A dataset class for handling dynamic graph data with temporal features.
    Supports loading different dataset variants (node-level, global, combined)."""

    VARIANTS = ["node_level", "global", "combined"]

    def __init__(
        self,
        variant: str = "combined",
        data_dir: str = "datasets",
        graph_type: Optional[str] = None,
        config_path: Optional[str] = None,
    ):
        """Initialize the dataset."""
        if config_path is None:
            config_path = Path(__file__).parent / "configs/dataset_config.yaml"

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        if variant not in self.config["data"]["variants"]:
            raise ValueError(
                f"variant must be one of {self.config['data']['variants']}"
            )

        self.variant = variant
        self.data_dir = Path(data_dir) / variant
        self.graph_type = graph_type

        # Load variant-specific settings
        self.feature_config = self.config["features"][variant]
        self.temporal_config = self.config["temporal"]
        self.processing_config = self.config["processing"]

        self._load_data()
        self._validate_data()

    def _load_data(self):
        """Load the dataset from HDF5 file."""
        dataset_path = self.data_dir / "dataset.h5"
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found at {dataset_path}")

        self.data = h5py.File(dataset_path, "r")

        # Load data for specified graph type or all types
        self.graph_types = [self.graph_type] if self.graph_type else ["BA", "ER", "NW"]

        # Initialize data structures
        self.adjacency_matrices = []
        self.feature_sequences = []
        self.metadata = []

        # Load data for each graph type
        for gtype in self.graph_types:
            if gtype not in self.data:
                logger.warning(f"Graph type {gtype} not found in dataset")
                continue

            group = self.data[gtype]
            sequences = group["sequences"]

            # Load all sequences for this graph type
            for seq_idx in range(len(sequences)):
                seq = sequences[f"seq_{seq_idx}"]
                self.adjacency_matrices.append(seq["adjacency"][:])
                self.feature_sequences.append(seq["features"][:])

                # Create metadata
                meta = {
                    "graph_type": gtype,
                    "change_points": group["change_points"][f"seq_{seq_idx}"][
                        :
                    ].tolist(),
                    "params": eval(group["params"][f"seq_{seq_idx}"][()]),
                }
                self.metadata.append(meta)

        # Convert to numpy arrays
        self.adjacency_matrices = np.array(self.adjacency_matrices)
        self.feature_sequences = np.array(self.feature_sequences)

        # Set dimensions
        self.num_sequences = len(self.metadata)
        if self.num_sequences > 0:
            self.sequence_length = self.adjacency_matrices.shape[1]
            self.num_nodes = self.adjacency_matrices.shape[2]
            self.num_features = self.feature_sequences.shape[-1]

        # Process features and adjacency matrices
        for i in range(len(self.feature_sequences)):
            self.feature_sequences[i] = self._process_features(
                self.feature_sequences[i]
            )
            for j in range(len(self.adjacency_matrices[i])):
                self.adjacency_matrices[i][j] = self._process_adjacency(
                    self.adjacency_matrices[i][j]
                )

    def _validate_data(self):
        """Validate data dimensions and formats based on variant."""
        if self.num_sequences == 0:
            raise ValueError("No sequences loaded")

        # Check adjacency matrix dimensions
        assert (
            len(self.adjacency_matrices.shape) == 4
        ), "Adjacency matrices should be 4D"
        assert (
            self.adjacency_matrices.shape[2] == self.adjacency_matrices.shape[3]
        ), "Adjacency matrices should be square"

        # Check feature dimensions based on variant
        if self.variant == "node_level":
            assert (
                self.feature_sequences.shape[-1] == 6
            ), "Node-level features should have 6 features per node"
            assert len(self.feature_sequences.shape) == 4, "Node features should be 4D"
        elif self.variant == "global":
            expected_size = self.num_nodes * self.num_nodes + 6
            assert (
                self.feature_sequences.shape[-1] == expected_size
            ), f"Global features should have size {expected_size}"
            assert (
                len(self.feature_sequences.shape) == 3
            ), "Global features should be 3D"
        else:  # combined
            expected_size = self.num_nodes + 6
            assert (
                self.feature_sequences.shape[-1] == expected_size
            ), f"Combined features should have size {expected_size}"
            assert (
                len(self.feature_sequences.shape) == 4
            ), "Combined features should be 4D"

    def _get_edges_from_adjacency(
        self, adj_matrix: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert adjacency matrix to edge indices and weights."""
        edges = np.where(adj_matrix > 0)
        edge_index = torch.LongTensor(np.vstack((edges[0], edges[1])))
        edge_weight = torch.FloatTensor(adj_matrix[edges])
        return edge_index, edge_weight

    def _process_features(self, features: np.ndarray) -> np.ndarray:
        """Process features according to config settings."""
        if self.processing_config["normalize_features"]:
            # Normalize features to [0,1] range
            feature_min = features.min(axis=0, keepdims=True)
            feature_max = features.max(axis=0, keepdims=True)
            features = (features - feature_min) / (feature_max - feature_min + 1e-8)

        return features

    def _process_adjacency(self, adjacency: np.ndarray) -> np.ndarray:
        """Process adjacency matrix according to config settings."""
        if self.processing_config["normalize_adjacency"]:
            # Symmetric normalization: D^(-1/2) A D^(-1/2)
            degree = np.sum(adjacency, axis=-1)
            degree_sqrt_inv = 1.0 / (np.sqrt(degree) + 1e-8)
            adjacency = degree_sqrt_inv[:, None] * adjacency * degree_sqrt_inv[None, :]

        if self.processing_config["add_self_loops"]:
            # Add self-loops: A + I
            adjacency = adjacency + np.eye(adjacency.shape[0])

        return adjacency

    def get_temporal_signal(
        self, sequence_idx: int, window_size: Optional[int] = None
    ) -> DynamicGraphTemporalSignal:
        """Create a DynamicGraphTemporalSignal for a specific sequence."""
        # Use config window size if not specified
        window_size = window_size or self.temporal_config["window_size"]

        # Check minimum sequence length
        if self.sequence_length < self.temporal_config["min_sequence_length"]:
            raise ValueError(
                f"Sequence length {self.sequence_length} is less than minimum "
                f"required length {self.temporal_config['min_sequence_length']}"
            )

        edge_indices = []
        edge_weights = []
        features = []
        targets = []

        for t in range(self.sequence_length - window_size):
            # Get edges
            curr_adj = self.adjacency_matrices[sequence_idx, t]
            edge_index, edge_weight = self._get_edges_from_adjacency(curr_adj)

            # Convert to numpy for consistency
            edge_indices.append(edge_index.numpy())
            edge_weights.append(edge_weight.numpy())

            # Get features
            curr_features = self.feature_sequences[sequence_idx, t]
            if self.variant == "global":
                # For global variant, separate adjacency and features
                adj_size = self.num_nodes * self.num_nodes
                global_features = curr_features[
                    adj_size:
                ]  # Take only the global features
                # Replicate global features for each node
                node_features = np.tile(global_features, (self.num_nodes, 1))
                features.append(node_features)
            else:
                features.append(curr_features)

            # Target is next timestep's adjacency
            targets.append(self.adjacency_matrices[sequence_idx, t + 1])

        return DynamicGraphTemporalSignal(
            edge_indices=edge_indices,
            edge_weights=edge_weights,
            features=features,
            targets=targets,
        )

    def get_train_test_split(
        self, train_ratio: Optional[float] = None, sequence_idx: Optional[int] = None
    ) -> Tuple[DynamicGraphTemporalSignal, DynamicGraphTemporalSignal]:
        """Get train/test split for a sequence."""
        train_ratio = train_ratio or self.temporal_config["train_ratio"]

        if sequence_idx is None:
            sequence_idx = np.random.randint(0, self.num_sequences)

        temporal_signal = self.get_temporal_signal(sequence_idx)
        train_dataset, test_dataset = temporal_signal_split(
            temporal_signal, train_ratio
        )

        return train_dataset, test_dataset

    def get_graph_type_indices(self, graph_type: str) -> np.ndarray:
        """Get indices of sequences of a specific graph type."""
        return np.array(
            [
                i
                for i, meta in enumerate(self.metadata)
                if meta["graph_type"] == graph_type
            ]
        )

    def get_change_points(self, sequence_idx: int) -> List[int]:
        """Get change points for a specific sequence."""
        return self.metadata[sequence_idx]["change_points"]

    def get_sequence_metadata(self, sequence_idx: int) -> Dict:
        """Get full metadata for a specific sequence."""
        return self.metadata[sequence_idx]

    def get_train_val_test_split(self, sequence_idx: Optional[int] = None) -> Tuple[
        DynamicGraphTemporalSignal,
        DynamicGraphTemporalSignal,
        DynamicGraphTemporalSignal,
    ]:
        """Get train/validation/test split for a sequence."""
        train_ratio = self.temporal_config["train_ratio"]
        val_ratio = self.temporal_config["validation_ratio"]

        if sequence_idx is None:
            sequence_idx = np.random.randint(0, self.num_sequences)

        temporal_signal = self.get_temporal_signal(sequence_idx)

        # First split into train and test
        train_val_data, test_data = temporal_signal_split(temporal_signal, train_ratio)

        # Then split train into train and validation
        val_ratio_adjusted = val_ratio / train_ratio
        train_data, val_data = temporal_signal_split(
            train_val_data, 1 - val_ratio_adjusted
        )

        return train_data, val_data, test_data


if __name__ == "__main__":
    # Example usage
    for variant in ["node_level", "global", "combined"]:
        print(f"\nTesting {variant} dataset:")
        try:
            dataset = DynamicGraphDataset(variant=variant)

            print(f"Dataset Summary:")
            print(f"Variant: {dataset.variant}")
            print(f"Number of sequences: {dataset.num_sequences}")
            print(f"Sequence length: {dataset.sequence_length}")
            print(f"Number of nodes: {dataset.num_nodes}")
            print(f"Feature dimension: {dataset.num_features}")

            # Test loading specific graph type
            for graph_type in ["BA", "ER", "NW"]:
                indices = dataset.get_graph_type_indices(graph_type)
                print(f"\n{graph_type} sequences: {len(indices)}")

                if len(indices) > 0:
                    idx = indices[0]
                    change_points = dataset.get_change_points(idx)
                    print(f"First sequence change points: {change_points}")

                    # Test train/test split
                    train_data, test_data = dataset.get_train_test_split(
                        sequence_idx=idx
                    )
                    print(f"Train snapshots: {len(train_data.features)}")
                    print(f"Test snapshots: {len(test_data.features)}")

        except Exception as e:
            print(f"Error loading {variant} dataset: {str(e)}")
