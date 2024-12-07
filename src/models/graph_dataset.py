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
    window_size: int = 10  # Sequence window for training
    stride: int = 1  # Stride for sliding window
    forecast_horizon: int = 5  # Number of steps to predict
    batch_size: int = 32  # Training batch size

    # Feature selection
    use_centrality: bool = True  # Use centrality features
    use_spectral: bool = True  # Use spectral embeddings
    centrality_features: List[str] = None  # Specific centrality features to use
    spectral_features: List[str] = None  # Specific spectral features to use

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
        self.indices = self._create_sequence_indices()

    def _load_data(self) -> Dict:
        """Load data from HDF5 file."""
        path = self.data_dir / self.split / "data.h5"
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

    def _create_sequence_indices(self) -> List[Tuple[int, int]]:
        """Create sliding window indices for sequences."""
        indices = []

        for seq_idx, length in enumerate(self.data["sequence_lengths"]):
            # Create windows with given size and stride
            for start in range(
                0,
                length - self.config.window_size - self.config.forecast_horizon + 1,
                self.config.stride,
            ):
                end = start + self.config.window_size
                indices.append((seq_idx, start, end))

        return indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sequence window with features."""
        seq_idx, start, end = self.indices[idx]

        # Get input window
        x = self._get_features(seq_idx, start, end)

        # Get target window
        target_start = end
        target_end = end + self.config.forecast_horizon
        y = self._get_features(seq_idx, target_start, target_end)

        # Get adjacency matrices
        adj = self.data["adjacency"][seq_idx][start:end]

        # Get change points in the sequence window
        cp = self._get_change_points(seq_idx, start, target_end)

        sample = {
            "x": x,  # Input features
            "y": y,  # Target features
            "adj": adj,  # Adjacency matrices
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

        if self.config.use_centrality:
            for feat in self.config.centrality_features:
                if feat in self.data["features"]:
                    features[feat] = self.data["features"][feat][seq_idx, start:end]

        if self.config.use_spectral:
            for feat in self.config.spectral_features:
                if feat in self.data["features"]:
                    features[feat] = self.data["features"][feat][seq_idx, start:end]

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
