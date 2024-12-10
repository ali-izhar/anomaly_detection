"""
Dynamic Graph Dataset Loader

This module provides a PyTorch dataset class for loading and processing different variants
of the synthetic graph sequence datasets (node-level, global, or combined features).
"""

import numpy as np
import torch
from torch_geometric_temporal.signal import DynamicGraphTemporalSignal, temporal_signal_split
from pathlib import Path
import json
import h5py
from typing import Optional, Tuple, List, Dict
import logging

logger = logging.getLogger(__name__)


class DynamicGraphDataset:
    """
    A dataset class for handling dynamic graph data with temporal features.
    Supports loading different dataset variants (node-level, global, combined).
    """

    VARIANTS = ["node_level", "global", "combined"]

    def __init__(
        self,
        variant: str = "combined",
        data_dir: str = "datasets",
        graph_type: Optional[str] = None,
    ):
        """
        Initialize the dataset.

        Args:
            variant: str, one of ["node_level", "global", "combined"]
            data_dir: str, path to directory containing the dataset variants
            graph_type: Optional[str], if specified, load only this graph type (BA, ER, or NW)
        """
        if variant not in self.VARIANTS:
            raise ValueError(f"variant must be one of {self.VARIANTS}")

        self.variant = variant
        self.data_dir = Path(data_dir) / variant
        self.graph_type = graph_type
        self._load_data()
        self._validate_data()

    def _load_data(self):
        """Load the dataset from HDF5 file."""
        dataset_path = self.data_dir / "dataset.h5"
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found at {dataset_path}")

        self.data = h5py.File(dataset_path, 'r')
        
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
                    "change_points": group["change_points"][f"seq_{seq_idx}"][:].tolist(),
                    "params": eval(group["params"][f"seq_{seq_idx}"][()])
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

    def _validate_data(self):
        """Validate data dimensions and formats based on variant."""
        if self.num_sequences == 0:
            raise ValueError("No sequences loaded")

        # Check adjacency matrix dimensions
        assert len(self.adjacency_matrices.shape) == 4, "Adjacency matrices should be 4D"
        assert self.adjacency_matrices.shape[2] == self.adjacency_matrices.shape[3], "Adjacency matrices should be square"

        # Check feature dimensions based on variant
        if self.variant == "node_level":
            assert self.feature_sequences.shape[-1] == 6, "Node-level features should have 6 features per node"
            assert len(self.feature_sequences.shape) == 4, "Node features should be 4D"
        elif self.variant == "global":
            expected_size = self.num_nodes * self.num_nodes + 6
            assert self.feature_sequences.shape[-1] == expected_size, f"Global features should have size {expected_size}"
            assert len(self.feature_sequences.shape) == 3, "Global features should be 3D"
        else:  # combined
            expected_size = self.num_nodes + 6
            assert self.feature_sequences.shape[-1] == expected_size, f"Combined features should have size {expected_size}"
            assert len(self.feature_sequences.shape) == 4, "Combined features should be 4D"

    def _get_edges_from_adjacency(self, adj_matrix: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert adjacency matrix to edge indices and weights."""
        edges = np.where(adj_matrix > 0)
        edge_index = torch.LongTensor(np.vstack((edges[0], edges[1])))
        edge_weight = torch.FloatTensor(adj_matrix[edges])
        return edge_index, edge_weight

    def get_temporal_signal(self, sequence_idx: int, window_size: int = 10) -> DynamicGraphTemporalSignal:
        """Create a DynamicGraphTemporalSignal for a specific sequence."""
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
                global_features = curr_features[adj_size:]  # Take only the global features
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
        self, train_ratio: float = 0.8, sequence_idx: Optional[int] = None
    ) -> Tuple[DynamicGraphTemporalSignal, DynamicGraphTemporalSignal]:
        """Get train/test split for a sequence."""
        if sequence_idx is None:
            sequence_idx = np.random.randint(0, self.num_sequences)

        temporal_signal = self.get_temporal_signal(sequence_idx)
        train_dataset, test_dataset = temporal_signal_split(temporal_signal, train_ratio)

        return train_dataset, test_dataset

    def get_graph_type_indices(self, graph_type: str) -> np.ndarray:
        """Get indices of sequences of a specific graph type."""
        return np.array([
            i for i, meta in enumerate(self.metadata)
            if meta["graph_type"] == graph_type
        ])

    def get_change_points(self, sequence_idx: int) -> List[int]:
        """Get change points for a specific sequence."""
        return self.metadata[sequence_idx]["change_points"]

    def get_sequence_metadata(self, sequence_idx: int) -> Dict:
        """Get full metadata for a specific sequence."""
        return self.metadata[sequence_idx]


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
                    train_data, test_data = dataset.get_train_test_split(sequence_idx=idx)
                    print(f"Train snapshots: {len(train_data.features)}")
                    print(f"Test snapshots: {len(test_data.features)}")
        
        except Exception as e:
            print(f"Error loading {variant} dataset: {str(e)}")
