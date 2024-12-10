import numpy as np
import torch
from torch_geometric_temporal.signal import DynamicGraphTemporalSignal, temporal_signal_split
from pathlib import Path
import json


class DynamicGraphDataset:
    """
    A dataset class for handling dynamic graph data with temporal features.

    This class processes sequences of adjacency matrices and node features
    to create temporal graph signals suitable for link prediction tasks.
    """

    def __init__(self, data_dir: str = "data/processed"):
        """
        Initialize the dataset.

        Args:
            data_dir: str, path to directory containing the processed data files
                Should contain:
                - adjacency_matrices.npz
                - feature_sequences.npz
                - metadata.json
        """
        self.data_dir = Path(data_dir)
        self._load_data()
        self._validate_data()

    def _load_data(self):
        """Load the preprocessed data from files."""
        # Load adjacency matrices
        adj_path = self.data_dir / "adjacency_matrices.npz"
        adj_data = np.load(adj_path)
        self.adjacency_matrices = adj_data["adj_matrices"]

        # Load feature sequences
        feat_path = self.data_dir / "feature_sequences.npz"
        feat_data = np.load(feat_path)
        self.feature_sequences = feat_data["feature_sequences"]

        # Load metadata
        meta_path = self.data_dir / "metadata.json"
        with open(meta_path, "r") as f:
            self.metadata = json.load(f)

        # Set dimensions
        self.num_sequences = self.adjacency_matrices.shape[0]
        self.sequence_length = self.adjacency_matrices.shape[1]
        self.num_nodes = self.adjacency_matrices.shape[2]
        self.num_features = self.feature_sequences.shape[3]

    def _validate_data(self):
        """Validate data dimensions and formats."""
        assert (
            len(self.adjacency_matrices.shape) == 4
        ), "Adjacency matrices should be 4D"
        assert len(self.feature_sequences.shape) == 4, "Feature sequences should be 4D"
        assert (
            self.adjacency_matrices.shape[:-1] == self.feature_sequences.shape[:-1]
        ), "Adjacency matrices and features must have matching dimensions"
        assert len(self.metadata) == self.num_sequences, "Metadata length mismatch"

    def _get_edges_from_adjacency(self, adj_matrix):
        """
        Convert adjacency matrix to edge indices and weights.

        Args:
            adj_matrix: numpy array of shape (num_nodes, num_nodes)

        Returns:
            tuple: (edge_index, edge_weight) tensors for PyTorch Geometric
        """
        edges = np.where(adj_matrix > 0)
        edge_index = torch.LongTensor(np.vstack((edges[0], edges[1])))
        edge_weight = torch.FloatTensor(adj_matrix[edges])
        return edge_index, edge_weight

    def get_temporal_signal(self, sequence_idx, window_size=10):
        """
        Create a DynamicGraphTemporalSignal for a specific sequence.

        Args:
            sequence_idx: int, index of the sequence to use
            window_size: int, number of timesteps to include in prediction window

        Returns:
            DynamicGraphTemporalSignal object containing the temporal graph data
        """
        edge_indices = []
        edge_weights = []
        features = []
        targets = []

        for t in range(self.sequence_length - window_size):
            # Get edges for current timestep
            curr_adj = self.adjacency_matrices[sequence_idx, t]
            if isinstance(curr_adj, torch.Tensor):
                curr_adj = curr_adj.numpy()
            edge_index, edge_weight = self._get_edges_from_adjacency(curr_adj)
            
            # Convert edge data to numpy
            edge_indices.append(edge_index.numpy() if isinstance(edge_index, torch.Tensor) else edge_index)
            edge_weights.append(edge_weight.numpy() if isinstance(edge_weight, torch.Tensor) else edge_weight)

            # Features for current timestep
            curr_features = self.feature_sequences[sequence_idx, t]
            if isinstance(curr_features, torch.Tensor):
                curr_features = curr_features.numpy()
            features.append(curr_features)

            # Target is the next timestep's adjacency matrix
            target_adj = self.adjacency_matrices[sequence_idx, t + 1]
            if isinstance(target_adj, torch.Tensor):
                target_adj = target_adj.numpy()
            targets.append(target_adj)

        return DynamicGraphTemporalSignal(
            edge_indices=edge_indices,
            edge_weights=edge_weights,
            features=features,
            targets=targets,
        )

    def get_train_test_split(self, train_ratio=0.8, sequence_idx=None):
        """
        Get train/test split for a sequence.

        Args:
            train_ratio: float between 0 and 1, ratio of data to use for training
            sequence_idx: optional int to specify which sequence to use

        Returns:
            tuple: (train_dataset, test_dataset) of DynamicGraphTemporalSignal
        """
        if sequence_idx is None:
            sequence_idx = np.random.randint(0, self.num_sequences)

        temporal_signal = self.get_temporal_signal(sequence_idx)
        train_dataset, test_dataset = temporal_signal_split(temporal_signal, train_ratio)

        return train_dataset, test_dataset

    def get_graph_type_indices(self, graph_type: str) -> np.ndarray:
        """
        Get indices of sequences of a specific graph type.

        Args:
            graph_type: str, one of ['barabasi_albert', 'erdos_renyi', 'newman_watts']

        Returns:
            np.ndarray: indices of sequences of the specified type
        """
        return np.array(
            [
                i
                for i, meta in enumerate(self.metadata)
                if meta["graph_type"] == graph_type
            ]
        )

    def get_change_points(self, sequence_idx: int) -> list:
        """
        Get change points for a specific sequence.

        Args:
            sequence_idx: int, index of the sequence

        Returns:
            list: timesteps where graph parameters change
        """
        return self.metadata[sequence_idx]["change_points"]


# Example usage
if __name__ == "__main__":
    # Create dataset
    dataset = DynamicGraphDataset()

    print("\nDataset Summary:")
    print(f"Number of sequences: {dataset.num_sequences}")
    print(f"Sequence length: {dataset.sequence_length}")
    print(f"Number of nodes: {dataset.num_nodes}")
    print(f"Number of features: {dataset.num_features}")

    # Get sequences by graph type
    for graph_type in ["barabasi_albert", "erdos_renyi", "newman_watts"]:
        indices = dataset.get_graph_type_indices(graph_type)
        print(f"\n{graph_type} sequences: {len(indices)}")

        # Get a random sequence of this type
        if len(indices) > 0:
            idx = np.random.choice(indices)
            change_points = dataset.get_change_points(idx)
            print(f"Sample sequence {idx} change points: {change_points}")

            # Get train/test split
            train_data, test_data = dataset.get_train_test_split(sequence_idx=idx)
            
            # Access the number of snapshots through the features list
            print(f"Train snapshots: {len(train_data.features)}")
            print(f"Test snapshots: {len(test_data.features)}")
