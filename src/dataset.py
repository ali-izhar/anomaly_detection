# src/dataset.py

"""
PyTorch Dataset for Graph Sequences with Link Prediction Features

This module provides dataset classes for loading and processing graph sequences:
- Handles both static and temporal features
- Supports different sequence lengths and window sizes
- Provides options for feature normalization and selection
"""

from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset
import logging

logger = logging.getLogger(__name__)


def custom_collate(batch: List[Dict]) -> Dict:
    """Custom collate function to handle variable-sized tensors.

    Args:
        batch: List of dictionaries from __getitem__

    Returns:
        Batched dictionary with properly stacked tensors
    """
    # Initialize output dictionary
    output = {
        "adjacency": [],
        "features": {},
        "metadata": {
            "sequence_idx": [],
            "window_start": [],
            "window_end": [],
            "change_points": [],  # Will be a list of tensors
            "community_labels": [],
        },
    }

    # Initialize feature lists
    if batch:
        for feat_name in batch[0]["features"].keys():
            output["features"][feat_name] = []

    # Collect tensors from each batch item
    for item in batch:
        # Stack adjacency matrices
        output["adjacency"].append(item["adjacency"])

        # Stack features
        for feat_name, feat_tensor in item["features"].items():
            output["features"][feat_name].append(feat_tensor)

        # Collect metadata
        output["metadata"]["sequence_idx"].append(item["metadata"]["sequence_idx"])
        output["metadata"]["window_start"].append(item["metadata"]["window_start"])
        output["metadata"]["window_end"].append(item["metadata"]["window_end"])
        output["metadata"]["change_points"].append(item["metadata"]["change_points"])

        if "community_labels" in item["metadata"]:
            output["metadata"]["community_labels"].append(
                item["metadata"]["community_labels"]
            )

    # Convert lists to tensors where possible
    output["adjacency"] = torch.stack(output["adjacency"])

    for feat_name in output["features"]:
        output["features"][feat_name] = torch.stack(output["features"][feat_name])

    # Handle metadata
    output["metadata"]["sequence_idx"] = torch.tensor(
        output["metadata"]["sequence_idx"]
    )
    output["metadata"]["window_start"] = torch.tensor(
        output["metadata"]["window_start"]
    )
    output["metadata"]["window_end"] = torch.tensor(output["metadata"]["window_end"])
    # Keep change_points as list of tensors since they may have different lengths

    if output["metadata"]["community_labels"]:
        output["metadata"]["community_labels"] = torch.stack(
            output["metadata"]["community_labels"]
        )
    else:
        del output["metadata"]["community_labels"]

    return output


class GraphSequenceDataset(Dataset):
    """Dataset for graph sequences with link prediction features."""

    def __init__(
        self,
        data_path: str,
        split: str = "train",
        window_size: int = 5,
        stride: int = 1,
        feature_set: Optional[List[str]] = None,
        normalize: bool = True,
        device: Union[str, torch.device] = "cpu",
    ):
        """Initialize the dataset.

        Args:
            data_path: Path to HDF5 dataset file
            split: Dataset split to use ('train', 'val', or 'test')
            window_size: Number of graphs to include in each sequence window
            stride: Step size between sequence windows
            feature_set: List of features to use (if None, use all available)
            normalize: Whether to normalize features
            device: Device to load tensors to
        """
        self.data_path = data_path
        self.split = split
        self.window_size = window_size
        self.stride = stride
        self.feature_set = feature_set
        self.normalize = normalize
        self.device = device

        # Load dataset metadata and prepare indices
        self._load_metadata()
        self._prepare_indices()

    def _load_metadata(self):
        """Load dataset metadata and compute statistics."""
        with h5py.File(self.data_path, "r") as f:
            # Load global metadata
            self.num_sequences = f["metadata"].attrs["num_sequences"]
            self.sequence_length = f["metadata"].attrs["sequence_length"]
            self.num_nodes = f["metadata"].attrs["num_nodes"]

            # Get sequence information
            self.sequence_lengths = []
            self.feature_names = []
            self.feature_dims = {}

            # Get feature information from first sequence
            seq = f["sequences/sequence_0"]
            feat_group = f["features/sequence_0"]

            # Get all available features if none specified
            if self.feature_set is None:
                self.feature_set = list(feat_group.keys())
            else:
                # Filter out features that don't exist
                self.feature_set = [f for f in self.feature_set if f in feat_group]
                if not self.feature_set:
                    raise ValueError(
                        "None of the requested features were found in the dataset"
                    )

            # Store feature information
            for feat_name in self.feature_set:
                self.feature_dims[feat_name] = feat_group[feat_name].shape[1:]

            # Store sequence lengths
            for i in range(self.num_sequences):
                seq_name = f"sequence_{i}"
                seq_group = f["sequences"][seq_name]
                # Count number of graphs in sequence
                n_graphs = sum(1 for k in seq_group.keys() if k.startswith("graph_"))
                self.sequence_lengths.append(n_graphs)

            # Compute feature statistics for normalization
            if self.normalize:
                self.feature_stats = self._compute_feature_statistics(f)

    def _compute_feature_statistics(self, h5file) -> Dict:
        """Compute mean and std of features for normalization."""
        stats = {}

        # Sample sequences for statistics computation
        num_samples = min(10, self.num_sequences)  # Use at most 10 sequences
        sample_indices = np.random.choice(
            self.num_sequences, num_samples, replace=False
        )

        for feat_name in self.feature_set:
            values = []
            for idx in sample_indices:
                seq_name = f"sequence_{idx}"
                feat_data = h5file[f"features/{seq_name}/{feat_name}"][:]
                values.append(feat_data)

            values = np.concatenate(values, axis=0)
            # Compute stats along all dimensions except the last
            stats[feat_name] = {
                "mean": np.mean(
                    values, axis=tuple(range(values.ndim - 1)), keepdims=True
                ),
                "std": np.std(values, axis=tuple(range(values.ndim - 1)), keepdims=True)
                + 1e-8,
            }

        return stats

    def _prepare_indices(self):
        """Prepare indices for sequence windows."""
        self.window_indices = []

        for seq_idx in range(self.num_sequences):
            seq_len = self.sequence_lengths[seq_idx]

            # Calculate valid windows for this sequence
            for start in range(0, seq_len - self.window_size + 1, self.stride):
                end = start + self.window_size
                self.window_indices.append((seq_idx, start, end))

    def __len__(self) -> int:
        """Return the number of sequence windows."""
        return len(self.window_indices)

    def __getitem__(self, idx: int) -> Dict:
        """Get a sequence window with features.

        Args:
            idx: Index of the sequence window

        Returns:
            Dictionary containing:
                - adjacency: Graph adjacency matrices for the window
                - features: Dictionary of features
                - metadata: Sequence metadata
        """
        seq_idx, start, end = self.window_indices[idx]

        with h5py.File(self.data_path, "r") as f:
            seq_name = f"sequence_{seq_idx}"

            # Load adjacency matrices for the window
            adj_matrices = []
            for t in range(start, end):
                adj = torch.tensor(
                    f[f"sequences/{seq_name}/graph_{t}"][:], dtype=torch.float32
                )
                adj_matrices.append(adj)
            adjacency = torch.stack(adj_matrices)

            # Load features
            features = {}
            for feat_name in self.feature_set:
                feat_data = torch.tensor(
                    f[f"features/{seq_name}/{feat_name}"][:], dtype=torch.float32
                )

                # Handle temporal features differently
                if any(feat_name.startswith(p) for p in ["temporal_", "link_", "cn_"]):
                    # For temporal features, select the window
                    if feat_data.ndim > 1:
                        feat_data = feat_data[start:end]
                else:
                    # For static features, replicate across the window
                    feat_data = feat_data.unsqueeze(0).expand(
                        self.window_size, *feat_data.shape
                    )

                # Normalize if requested
                if self.normalize and feat_name in self.feature_stats:
                    stats = self.feature_stats[feat_name]
                    feat_data = (feat_data - stats["mean"]) / stats["std"]

                features[feat_name] = feat_data

            # Load metadata
            metadata = {
                "sequence_idx": seq_idx,
                "window_start": start,
                "window_end": end,
                "change_points": torch.tensor(
                    [], dtype=torch.long
                ),  # Default empty tensor
            }

            # Add community labels if available
            if "community_labels" in f[f"metadata/{seq_name}"]:
                metadata["community_labels"] = torch.tensor(
                    f[f"metadata/{seq_name}/community_labels"][:], dtype=torch.long
                )

            # Add change points if available
            try:
                if "change_points" in f[f"metadata/{seq_name}"]:
                    change_points = torch.tensor(
                        f[f"metadata/{seq_name}/change_points"][:], dtype=torch.long
                    )
                    # Filter change points to only those in the current window
                    mask = (change_points >= start) & (change_points < end)
                    metadata["change_points"] = change_points[mask] - start
            except (KeyError, AttributeError):
                pass  # Keep default empty tensor

            # Add link prediction data if available
            try:
                if "link_prediction" in f[f"metadata/{seq_name}"]:
                    lp_data = {}
                    lp_group = f[f"metadata/{seq_name}/link_prediction"]
                    for split in ["train", "val", "test"]:
                        if split in lp_group:
                            split_data = lp_group[split]
                            lp_data[split] = {
                                "positive": torch.tensor(
                                    split_data["positive"][:], dtype=torch.long
                                ),
                                "negative": torch.tensor(
                                    split_data["negative"][:], dtype=torch.long
                                ),
                            }
                    metadata["link_prediction"] = lp_data
            except (KeyError, AttributeError):
                pass

            return {
                "adjacency": adjacency.to(self.device),
                "features": {k: v.to(self.device) for k, v in features.items()},
                "metadata": metadata,
            }

    def get_feature_dims(self) -> Dict[str, Tuple]:
        """Get the dimensions of each feature.

        Returns:
            Dictionary mapping feature names to their dimensions
        """
        return self.feature_dims

    def get_collate_fn(self):
        """Get the collate function for DataLoader.

        Returns:
            Collate function that properly handles the dataset's format
        """

        def collate_fn(batch):
            # Stack adjacency matrices
            adjacency = torch.stack([item["adjacency"] for item in batch])

            # Collect features
            features = {}
            for feat_name in self.feature_set:
                features[feat_name] = torch.stack(
                    [item["features"][feat_name] for item in batch]
                )

            # Collect metadata
            metadata = {
                "sequence_idx": [item["metadata"]["sequence_idx"] for item in batch],
                "window_start": [item["metadata"]["window_start"] for item in batch],
                "window_end": [item["metadata"]["window_end"] for item in batch],
                "change_points": torch.stack(
                    [
                        (
                            item["metadata"]["change_points"]
                            if "change_points" in item["metadata"]
                            else torch.tensor([], dtype=torch.long)
                        )
                        for item in batch
                    ]
                ),
            }

            # Handle optional metadata
            if "community_labels" in batch[0]["metadata"]:
                metadata["community_labels"] = torch.stack(
                    [item["metadata"]["community_labels"] for item in batch]
                )

            if "link_prediction" in batch[0]["metadata"]:
                lp_data = {split: {} for split in ["train", "val", "test"]}
                for split in lp_data:
                    if split not in batch[0]["metadata"]["link_prediction"]:
                        continue

                    max_pos = max(
                        len(item["metadata"]["link_prediction"][split]["positive"])
                        for item in batch
                    )
                    max_neg = max(
                        len(item["metadata"]["link_prediction"][split]["negative"])
                        for item in batch
                    )

                    # Pad positive edges
                    padded_pos = []
                    for item in batch:
                        edges = item["metadata"]["link_prediction"][split]["positive"]
                        padding = torch.full(
                            (max_pos - len(edges), 2), -1, dtype=torch.long
                        )
                        padded = torch.cat([edges, padding])
                        padded_pos.append(padded)

                    # Pad negative edges
                    padded_neg = []
                    for item in batch:
                        edges = item["metadata"]["link_prediction"][split]["negative"]
                        padding = torch.full(
                            (max_neg - len(edges), 2), -1, dtype=torch.long
                        )
                        padded = torch.cat([edges, padding])
                        padded_neg.append(padded)

                    lp_data[split]["positive"] = torch.stack(padded_pos)
                    lp_data[split]["negative"] = torch.stack(padded_neg)

                metadata["link_prediction"] = lp_data

            return {"adjacency": adjacency, "features": features, "metadata": metadata}

        return collate_fn
