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
from torch.utils.data import DataLoader

from torch_geometric_temporal.signal import (
    DynamicGraphTemporalSignalBatch,
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
        self.graph_config = self.config["graph"]

        # First load the data to get dimensions
        self._load_data()

        # Then validate graph settings after dimensions are known
        if self.graph_config["num_nodes"] != self.num_nodes:
            logger.warning(
                f"Dataset nodes ({self.num_nodes}) differs from config ({self.graph_config['num_nodes']})"
            )

        # Validate graph types
        if graph_type and graph_type not in self.graph_config["types"]:
            raise ValueError(f"graph_type must be one of {self.graph_config['types']}")

        self._validate_data()
        self._setup_logging()

    def _setup_logging(self):
        """Configure logging based on config."""
        log_config = self.config["logging"]
        log_level = getattr(logging, log_config["level"].upper())
        logger.setLevel(log_level)

        if not logger.handlers:
            # Add file handler
            log_dir = Path(log_config["save_dir"])
            log_dir.mkdir(exist_ok=True)
            fh = logging.FileHandler(log_dir / "dataset.log")
            fh.setLevel(log_level)
            logger.addHandler(fh)

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

        # Add validation for minimum sequence length from config
        if self.sequence_length < self.temporal_config["min_sequence_length"]:
            raise ValueError(
                f"Sequence length {self.sequence_length} is less than configured minimum "
                f"{self.temporal_config['min_sequence_length']}"
            )

        # Validate change points based on config
        for meta in self.metadata:
            changes = meta["change_points"]
            if len(changes) > self.graph_config["change_point"]["max_changes"]:
                logger.warning(
                    f"Sequence has {len(changes)} changes, exceeding configured maximum "
                    f"of {self.graph_config['change_point']['max_changes']}"
                )

            # Check minimum distance between change points
            min_dist = self.graph_config["change_point"]["min_distance"]
            for i in range(1, len(changes)):
                if changes[i] - changes[i - 1] < min_dist:
                    logger.warning(
                        f"Change points distance {changes[i] - changes[i-1]} is less than "
                        f"configured minimum {min_dist}"
                    )

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
        # Add undirected graph handling here
        if self.processing_config.get("undirected", True):
            # Make adjacency matrix symmetric for undirected graphs
            adjacency = (adjacency + adjacency.T) / 2

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
    ) -> DynamicGraphTemporalSignalBatch:
        """Create a DynamicGraphTemporalSignalBatch for a specific sequence."""
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
        batch_indices = []

        for t in range(self.sequence_length - window_size):
            # Get edges
            curr_adj = self.adjacency_matrices[sequence_idx, t]
            edge_index, edge_weight = self._get_edges_from_adjacency(curr_adj)

            # Convert to numpy for consistency
            edge_indices.append(edge_index.numpy())
            edge_weights.append(edge_weight.numpy())

            # Add batch indices for this timestep
            batch_idx = np.zeros(
                self.num_nodes, dtype=np.int64
            )  # All nodes belong to batch 0
            batch_indices.append(batch_idx)

            # Rest of the feature processing remains the same
            curr_features = self.feature_sequences[sequence_idx, t]
            if self.variant == "global":
                adj_size = self.num_nodes * self.num_nodes
                global_features = curr_features[adj_size:]
                node_features = np.tile(global_features, (self.num_nodes, 1))
                features.append(node_features)
            else:
                features.append(curr_features)

            targets.append(self.adjacency_matrices[sequence_idx, t + 1])

        return DynamicGraphTemporalSignalBatch(
            edge_indices=edge_indices,
            edge_weights=edge_weights,
            features=features,
            targets=targets,
            batch_indices=batch_indices,
            batches=batch_indices,
        )

    def get_batch_temporal_signal(
        self, sequence_indices: List[int], window_size: Optional[int] = None
    ) -> DynamicGraphTemporalSignalBatch:
        """Create a batched temporal signal from multiple sequences."""
        window_size = window_size or self.temporal_config["window_size"]

        edge_indices = []
        edge_weights = []
        features = []
        targets = []
        batch_indices = []

        for t in range(self.sequence_length - window_size):
            batch_edge_indices = []
            batch_edge_weights = []
            batch_features = []
            batch_targets = []
            batch_idx = []

            node_offset = 0
            for batch_id, seq_idx in enumerate(sequence_indices):
                # Get edges and offset them based on batch position
                curr_adj = self.adjacency_matrices[seq_idx, t]
                edge_index, edge_weight = self._get_edges_from_adjacency(curr_adj)
                edge_index = edge_index.numpy()
                edge_index[0] += node_offset  # Offset source nodes
                edge_index[1] += node_offset  # Offset target nodes

                batch_edge_indices.append(edge_index)
                batch_edge_weights.append(edge_weight.numpy())

                # Process features
                curr_features = self.feature_sequences[seq_idx, t]
                if self.variant == "global":
                    adj_size = self.num_nodes * self.num_nodes
                    global_features = curr_features[adj_size:]
                    node_features = np.tile(global_features, (self.num_nodes, 1))
                    batch_features.append(node_features)
                else:
                    batch_features.append(curr_features)

                # Add targets
                batch_targets.append(self.adjacency_matrices[seq_idx, t + 1])

                # Add batch indices for this sequence
                batch_idx.append(np.full(self.num_nodes, batch_id, dtype=np.int64))

                node_offset += self.num_nodes

            # Concatenate all batch data for this timestep
            edge_indices.append(np.concatenate(batch_edge_indices, axis=1))
            edge_weights.append(np.concatenate(batch_edge_weights))
            features.append(np.concatenate(batch_features))
            targets.append(np.concatenate(batch_targets))
            batch_indices.append(np.concatenate(batch_idx))

        return DynamicGraphTemporalSignalBatch(
            edge_indices=edge_indices,
            edge_weights=edge_weights,
            features=features,
            targets=targets,
            batch_indices=batch_indices,
            batches=batch_indices,
        )

    def get_train_test_split(
        self, train_ratio: Optional[float] = None, sequence_idx: Optional[int] = None
    ) -> Tuple[DynamicGraphTemporalSignalBatch, DynamicGraphTemporalSignalBatch]:
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
        DynamicGraphTemporalSignalBatch,
        DynamicGraphTemporalSignalBatch,
        DynamicGraphTemporalSignalBatch,
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

    def get_dataloader(
        self,
        sequence_indices: List[int],
        shuffle: Optional[bool] = None,
        batch_size: Optional[int] = None,
    ) -> DataLoader:
        """Create a DataLoader with configured settings."""
        shuffle = shuffle if shuffle is not None else self.config["training"]["shuffle"]
        batch_size = batch_size or self.config["training"]["batch_size"]

        return DataLoader(
            sequence_indices,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.config["training"]["num_workers"],
            pin_memory=self.config["training"]["pin_memory"],
            prefetch_factor=self.config["training"]["prefetch_factor"],
        )

    def get_validation_metrics(self) -> List[str]:
        """Get configured validation metrics."""
        return self.config["validation"]["metrics"]

    def get_test_metrics(self) -> List[str]:
        """Get configured test metrics."""
        return self.config["testing"]["metrics"]


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
