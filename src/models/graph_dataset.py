# src/models/graph_dataset.py

"""
Graph Sequence Dataset

A PyTorch dataset for loading and preprocessing graph sequences with features and change points.
Handles batching, padding, and data augmentation for LSTM-GNN training.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class GraphDataConfig:
    """Configuration for data loading and preprocessing."""

    # Data loading
    window_size: int = 20  # Increased from 10 to handle longer sequences
    stride: int = 5  # Increased for more efficient sampling
    forecast_horizon: int = 10  # Increased to match sequence scales
    batch_size: int = 32  # Keep as is

    # Constants from dataset
    num_nodes: int = 100  # Fixed number of nodes
    max_seq_length: int = 200  # Maximum sequence length
    min_seq_length: int = 161  # Minimum sequence length from inspection
    num_change_points: int = 2  # Fixed number of change points

    # Feature dimensions
    svd_dim: int = 2  # SVD embedding dimension
    lsvd_dim: int = 16  # LSVD embedding dimension

    # Feature selection
    use_centrality: bool = True
    use_spectral: bool = True
    centrality_features: List[str] = None
    spectral_features: List[str] = None

    # Data augmentation
    enable_augmentation: bool = False
    noise_level: float = 0.01

    def __post_init__(self):
        if self.centrality_features is None:
            self.centrality_features = [
                "degree",
                "betweenness",
                "closeness",
                "eigenvector",
            ]
        if self.spectral_features is None:
            self.spectral_features = ["svd", "lsvd"]

        # Validate configuration
        self.validate_config()

    def validate_config(self):
        """Validate configuration parameters."""
        assert self.window_size > 0, "Window size must be positive"
        assert self.forecast_horizon > 0, "Forecast horizon must be positive"
        assert (
            self.window_size + self.forecast_horizon <= self.min_seq_length
        ), "Window size + forecast horizon exceeds minimum sequence length"
        assert self.stride > 0, "Stride must be positive"
        assert self.batch_size > 0, "Batch size must be positive"


class GraphSequenceDataset(Dataset):
    """Dataset for graph sequences with features and change points."""

    def __init__(
        self,
        data_dir: Union[str, Path],
        split: str,
        config: Optional[GraphDataConfig] = None,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.config = config or GraphDataConfig()
        self.data = self._load_data()
        self.validate_data()
        self.indices = self._create_sequence_indices()

    def validate_data(self):
        """Validate loaded data dimensions."""
        # Check sequence lengths
        assert len(self.data["sequence_lengths"]) > 0, "No sequences found"
        assert (
            min(self.data["sequence_lengths"]) >= self.config.min_seq_length
        ), f"Sequence length below minimum: {min(self.data['sequence_lengths'])}"
        assert (
            max(self.data["sequence_lengths"]) <= self.config.max_seq_length
        ), f"Sequence length exceeds maximum: {max(self.data['sequence_lengths'])}"

        # Validate feature dimensions
        num_sequences = len(self.data["sequence_lengths"])
        for feat_name, feat_data in self.data["features"].items():
            if feat_name in self.config.centrality_features:
                assert feat_data.shape == (
                    num_sequences,
                    self.config.max_seq_length,
                    self.config.num_nodes,
                ), f"Invalid shape for {feat_name}: {feat_data.shape}"
            elif feat_name == "svd":
                assert feat_data.shape == (
                    num_sequences,
                    self.config.max_seq_length,
                    self.config.num_nodes,
                    self.config.svd_dim,
                ), f"Invalid shape for SVD: {feat_data.shape}"
            elif feat_name == "lsvd":
                assert feat_data.shape == (
                    num_sequences,
                    self.config.max_seq_length,
                    self.config.num_nodes,
                    self.config.lsvd_dim,
                ), f"Invalid shape for LSVD: {feat_data.shape}"

        # Validate change points
        if self.data["change_points"]:
            for cp in self.data["change_points"]:
                assert (
                    len(cp) == self.config.num_change_points
                ), f"Invalid number of change points: {len(cp)}"

    def _load_data(self) -> Dict:
        """Load data from HDF5 file."""
        path = self.data_dir / self.split / "data.h5"
        logger.info(f"Loading data from {path}")
        if not path.exists():
            raise FileNotFoundError(f"No data found at {path}")

        with h5py.File(path, "r") as f:
            # Load sequence information
            sequence_lengths = f["lengths"][:]

            # Load features
            features = {}
            feat_group = f["features"]

            if self.config.use_centrality:
                for feat in self.config.centrality_features:
                    if feat in feat_group:
                        features[feat] = torch.FloatTensor(feat_group[feat][:])

            if self.config.use_spectral:
                for feat in self.config.spectral_features:
                    if feat in feat_group:
                        features[feat] = torch.FloatTensor(feat_group[feat][:])

            # Load adjacency matrices
            adj_group = f["adjacency"]
            adjacency = {
                i: torch.FloatTensor(adj_group[f"sequence_{i}"][:])
                for i in range(len(sequence_lengths))
            }

            # Load change points
            change_points = []
            if "change_points" in f:
                cp_group = f["change_points"]
                for i in range(len(sequence_lengths)):
                    change_points.append(torch.LongTensor(cp_group[f"sequence_{i}"][:]))

        return {
            "features": features,
            "adjacency": adjacency,
            "sequence_lengths": sequence_lengths,
            "change_points": change_points,
        }

    def _create_sequence_indices(self) -> List[Tuple[int, int, int]]:
        """Create sliding window indices for sequences."""
        indices = []

        for seq_idx, length in enumerate(self.data["sequence_lengths"]):
            # Ensure we have enough room for both input window and forecast horizon
            max_start = length - (self.config.window_size + self.config.forecast_horizon)
            
            # Create windows with given size and stride
            for start in range(0, max_start + 1, self.config.stride):
                end = start + self.config.window_size
                indices.append((seq_idx, start, end))

        return indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sequence window with features."""
        seq_idx, start, end = self.indices[idx]

        # Get input window
        x = self._get_features(seq_idx, start, end)  # window_size length

        # Get target window
        target_start = end
        target_end = end + self.config.forecast_horizon
        y = self._get_features(seq_idx, target_start, target_end)  # forecast_horizon length

        # Get adjacency matrices for the input window
        # Take first adjacency matrix as representative for the window
        # since graph structure changes are captured in features
        adj = self.data["adjacency"][seq_idx][start]  # [N x N]

        # Get change points in the sequence window
        cp = self._get_change_points(seq_idx, start, target_end)

        sample = {
            "x": x,  # Input features (window_size length)
            "y": y,  # Target features (forecast_horizon length)
            "adj": adj,  # Adjacency matrix [N x N]
            "change_points": cp,  # Change points
            "seq_idx": seq_idx,  # Sequence index
            "window": (start, end),  # Window indices
        }

        if self.config.enable_augmentation and self.split == "train":
            sample = self._augment_sample(sample)

        return sample

    def _get_features(
        self, seq_idx: int, start: int, end: int
    ) -> Dict[str, torch.Tensor]:
        """Get features for a specific sequence window."""
        features = {}
        window_length = end - start  # Calculate actual window length

        if self.config.use_centrality:
            for feat in self.config.centrality_features:
                if feat in self.data["features"]:
                    feat_data = self.data["features"][feat][seq_idx, start:end]
                    assert feat_data.shape == (
                        window_length,
                        self.config.num_nodes,
                    ), f"Invalid feature shape for {feat}: {feat_data.shape}, expected ({window_length}, {self.config.num_nodes})"
                    features[feat] = feat_data

        if self.config.use_spectral:
            for feat in self.config.spectral_features:
                if feat in self.data["features"]:
                    feat_data = self.data["features"][feat][seq_idx, start:end]
                    emb_dim = self.config.svd_dim if feat == "svd" else self.config.lsvd_dim
                    assert feat_data.shape == (
                        window_length,
                        self.config.num_nodes,
                        emb_dim,
                    ), f"Invalid embedding shape for {feat}: {feat_data.shape}, expected ({window_length}, {self.config.num_nodes}, {emb_dim})"
                    features[feat] = feat_data

        return features

    def _get_change_points(self, seq_idx: int, start: int, end: int) -> torch.Tensor:
        """Get change points within a sequence window."""
        if not self.data["change_points"]:
            return torch.tensor([], dtype=torch.long)

        cp = self.data["change_points"][seq_idx]
        # Keep only change points within the window
        mask = (cp >= start) & (cp < end)
        return cp[mask] - start  # Adjust indices relative to window

    def _augment_sample(self, sample: Dict) -> Dict:
        """Apply data augmentation to a sample."""
        if not self.config.enable_augmentation:
            return sample

        # Add Gaussian noise to features
        for feat_type in sample["x"]:
            noise = torch.randn_like(sample["x"][feat_type]) * self.config.noise_level
            sample["x"][feat_type] = sample["x"][feat_type] + noise

        return sample


def create_dataloader(
    dataset: GraphSequenceDataset,
    batch_size: Optional[int] = None,
    shuffle: bool = None,
    num_workers: int = 0,
) -> DataLoader:
    """Create a DataLoader with appropriate settings for the split."""

    if batch_size is None:
        batch_size = dataset.config.batch_size

    if shuffle is None:
        shuffle = dataset.split == "train"

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_sequences,
    )


def collate_sequences(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function for batching sequence data."""
    # Combine features
    x = {feat: torch.stack([b["x"][feat] for b in batch]) for feat in batch[0]["x"]}
    y = {feat: torch.stack([b["y"][feat] for b in batch]) for feat in batch[0]["y"]}

    # Stack adjacency matrices
    adj = torch.stack([b["adj"] for b in batch])

    # Collect metadata
    seq_indices = torch.tensor([b["seq_idx"] for b in batch])
    windows = torch.tensor([b["window"] for b in batch])

    # Pad and stack change points
    max_changes = max(len(b["change_points"]) for b in batch)
    if max_changes > 0:
        change_points = torch.stack(
            [
                F.pad(
                    b["change_points"],
                    (0, max_changes - len(b["change_points"])),
                    value=-1,
                )
                for b in batch
            ]
        )
    else:
        change_points = torch.empty((len(batch), 0), dtype=torch.long)

    return {
        "x": x,
        "y": y,
        "adj": adj,
        "change_points": change_points,
        "seq_indices": seq_indices,
        "windows": windows,
    }


# USAGE
# # Create dataset
# config = GraphDataConfig(
#     window_size=10,
#     forecast_horizon=5,
#     batch_size=32,
#     use_centrality=True,
#     use_spectral=True
# )

# # Create datasets for each split
# train_dataset = GraphSequenceDataset("dataset", "train", config)
# val_dataset = GraphSequenceDataset("dataset", "val", config)
# test_dataset = GraphSequenceDataset("dataset", "test", config)

# # Create dataloaders
# train_loader = create_dataloader(train_dataset, shuffle=True)
# val_loader = create_dataloader(val_dataset, shuffle=False)
# test_loader = create_dataloader(test_dataset, shuffle=False)

# # Training loop
# for batch in train_loader:
#     x = batch["x"]           # Input features
#     y = batch["y"]           # Target features
#     adj = batch["adj"]       # Adjacency matrices
#     cp = batch["change_points"]  # Change points
#     # ... training step
