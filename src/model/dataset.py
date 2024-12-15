# src/model/dataset.py

"""Dynamic Graph Dataset Loader for temporal graph sequences with different feature variants."""

import h5py
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Union
import yaml
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


class DynamicGraphDataset(Dataset):
    """Dataset class for dynamic graph sequences."""

    def __init__(
        self,
        variant: str = "node_level",
        data_dir: str = "datasets",
        graph_type: Optional[str] = None,
        config_path: Optional[str] = None,
    ):
        """Initialize dataset.

        Args:
            variant: Feature type ("node_level", "global", "combined")
            data_dir: Dataset directory path
            graph_type: Specific graph type to load (BA, ER, or NW)
            config_path: Path to config file
        """
        super().__init__()

        if config_path is None:
            config_path = Path(__file__).parent / "dataset_config.yaml"

        with open(config_path) as f:
            logger.debug(f"Loading DynamicGraphDataset config from {config_path}")
            self.config = yaml.safe_load(f)

        # Validate inputs
        if variant not in self.config["data"]["variants"]:
            raise ValueError(
                f"variant must be one of {self.config['data']['variants']}"
            )
        if graph_type and graph_type not in self.config["graph"]["types"]:
            raise ValueError(
                f"graph_type must be one of {self.config['graph']['types']}"
            )

        # Store parameters
        self.variant = variant
        self.data_dir = Path(data_dir) / variant
        self.graph_type = graph_type

        # Load data
        self._load_data()
        self._validate_data()
        self._create_temporal_samples()

    def _load_data(self):
        """Load dataset from HDF5 file."""
        dataset_path = self.data_dir / "dataset.h5"
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found at {dataset_path}")

        # Load data using context manager to ensure file is closed
        with h5py.File(dataset_path, "r") as data:
            graph_types = (
                [self.graph_type] if self.graph_type else self.config["graph"]["types"]
            )

            # Initialize data structures
            self.adjacency_matrices = []
            self.feature_sequences = []
            self.metadata = []

            # Load sequences for each graph type
            for gtype in graph_types:
                if gtype not in data:
                    logger.warning(f"Graph type {gtype} not found in dataset")
                    continue

                group = data[gtype]
                sequences = group["sequences"]

                for seq_idx in range(len(sequences)):
                    seq = sequences[f"seq_{seq_idx}"]
                    # Convert to numpy arrays immediately
                    self.adjacency_matrices.append(np.array(seq["adjacency"]))
                    self.feature_sequences.append(np.array(seq["features"]))

                    self.metadata.append(
                        {
                            "graph_type": gtype,
                            "change_points": np.array(
                                group["change_points"][f"seq_{seq_idx}"]
                            ).tolist(),
                            "params": eval(group["params"][f"seq_{seq_idx}"][()]),
                        }
                    )

        # Convert to arrays and set dimensions
        self.adjacency_matrices = np.array(self.adjacency_matrices)
        self.feature_sequences = np.array(self.feature_sequences)

        self.num_sequences = len(self.metadata)
        if self.num_sequences > 0:
            self.sequence_length = self.adjacency_matrices.shape[1]
            self.num_nodes = self.adjacency_matrices.shape[2]
            self.num_features = self.feature_sequences.shape[-1]

        # Normalize features if enabled
        if self.config["processing"]["normalize_features"]:
            for i in range(len(self.feature_sequences)):
                self.feature_sequences[i] = self._normalize_features(
                    self.feature_sequences[i]
                )

        logger.debug(
            f"Loaded {self.num_sequences} sequences for {self.variant} variant and {graph_types} graph types"
        )

    def _validate_data(self):
        """Validate data dimensions."""
        if self.num_sequences == 0:
            raise ValueError("No sequences loaded")

        # Validate dimensions based on variant
        expected_dim = self.config["features"][self.variant]["dimension"]
        if self.feature_sequences.shape[-1] != expected_dim:
            raise ValueError(
                f"Expected feature dimension {expected_dim}, got {self.feature_sequences.shape[-1]}"
            )

    def _create_temporal_samples(self):
        """Create temporal windows for sampling."""
        window = self.config["processing"]["temporal_window"]
        stride = self.config["processing"]["stride"]

        self.temporal_samples = []
        for seq_idx in range(self.num_sequences):
            for start_idx in range(0, self.sequence_length - window, stride):
                self.temporal_samples.append(
                    {
                        "sequence_idx": seq_idx,
                        "start_idx": start_idx,
                        "end_idx": start_idx + window,
                    }
                )

    def __len__(self) -> int:
        return len(self.temporal_samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a temporal sample."""
        sample = self.temporal_samples[idx]
        seq_idx = sample["sequence_idx"]
        start_idx = sample["start_idx"]
        end_idx = sample["end_idx"]

        # Get window of data
        features = self.feature_sequences[seq_idx, start_idx:end_idx]
        adj_matrices = self.adjacency_matrices[seq_idx, start_idx:end_idx]
        target = self.adjacency_matrices[seq_idx, end_idx]

        # Get edge indices and weights
        edge_indices = []
        edge_weights = []
        for adj in adj_matrices:
            edges = np.where(adj > 0)
            edge_indices.append(torch.LongTensor(np.vstack((edges[0], edges[1]))))
            edge_weights.append(torch.FloatTensor(adj[edges]))

        return {
            "features": torch.FloatTensor(features),
            "edge_indices": edge_indices,
            "edge_weights": edge_weights,
            "target": torch.FloatTensor(target),
            "metadata": self.metadata[seq_idx],
        }

    def get_dataloader(
        self, batch_size: Optional[int] = None, shuffle: bool = True
    ) -> DataLoader:
        """Create a DataLoader."""
        if batch_size is None:
            batch_size = self.config["training"]["batch_size"]

        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.config["training"]["num_workers"],
            collate_fn=self._collate_fn,
            persistent_workers=True,
        )

    def _collate_fn(
        self, batch: List[Dict]
    ) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        """Collate batch of samples."""
        return {
            "features": torch.stack([item["features"] for item in batch]),
            "edge_indices": [item["edge_indices"] for item in batch],
            "edge_weights": [item["edge_weights"] for item in batch],
            "targets": torch.stack([item["target"] for item in batch]),
            "metadata": [item["metadata"] for item in batch],
        }

    def get_train_val_test_split(
        self, seed: Optional[int] = None
    ) -> Tuple[List[int], List[int], List[int]]:
        """Get dataset split indices."""
        if seed is not None:
            np.random.seed(seed)

        indices = np.random.permutation(len(self))
        train_size = int(self.config["training"]["train_ratio"] * len(self))
        val_size = int(self.config["training"]["val_ratio"] * len(self))

        return (
            indices[:train_size],
            indices[train_size : train_size + val_size],
            indices[train_size + val_size :],
        )

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features to [0,1] range."""
        feature_min = features.min(axis=0, keepdims=True)
        feature_max = features.max(axis=0, keepdims=True)
        return (features - feature_min) / (feature_max - feature_min + 1e-8)


def test_dataset(variant: str = "node_level", graph_type: Optional[str] = None):
    """Test dataset functionality."""
    logger.info(
        f"Testing {variant} dataset" + (f" for {graph_type}" if graph_type else "")
    )

    dataset = DynamicGraphDataset(variant=variant, graph_type=graph_type)

    # Print dataset info
    print("\nDataset Summary:")
    print(f"Total samples: {len(dataset)}")
    print(f"Number of sequences: {dataset.num_sequences}")
    print(f"Sequence length: {dataset.sequence_length}")
    print(f"Number of nodes: {dataset.num_nodes}")
    print(f"Feature dimension: {dataset.num_features}")

    # Get a sample
    print("\nSample from dataset:")
    sample = dataset[0]
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(f"{key} shape: {value.shape}")
        elif isinstance(value, list):
            print(f"{key} length: {len(value)}")
            if value:  # If list is not empty
                if isinstance(value[0], torch.Tensor):
                    print(f"  First element shape: {value[0].shape}")
        else:
            print(f"{key} type: {type(value)}")

    # Test dataloader
    print("\nTesting DataLoader:")
    dataloader = dataset.get_dataloader(batch_size=4)
    batch = next(iter(dataloader))
    print("\nBatch contents:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"{key} shape: {value.shape}")
        elif isinstance(value, list):
            print(f"{key} length: {len(value)}")
            if value and isinstance(value[0], list):
                print(f"  First sequence length: {len(value[0])}")
                if value[0]:  # If inner list is not empty
                    print(f"  First element shape: {value[0][0].shape}")

    # Test train/val/test split
    print("\nTesting train/val/test split:")
    train_idx, val_idx, test_idx = dataset.get_train_val_test_split(seed=42)
    print(f"Train samples: {len(train_idx)}")
    print(f"Val samples: {len(val_idx)}")
    print(f"Test samples: {len(test_idx)}")

    # Test graph type distribution
    print("\nGraph type distribution:")
    type_counts = {}
    for meta in dataset.metadata:
        gtype = meta["graph_type"]
        type_counts[gtype] = type_counts.get(gtype, 0) + 1
    for gtype, count in type_counts.items():
        print(f"{gtype}: {count} sequences")

    # Test feature statistics
    print("\nFeature statistics:")
    features = dataset.feature_sequences
    print(f"Min: {features.min():.4f}")
    print(f"Max: {features.max():.4f}")
    print(f"Mean: {features.mean():.4f}")
    print(f"Std: {features.std():.4f}")

    # Test edge statistics
    print("\nEdge statistics:")
    adj = dataset.adjacency_matrices
    edge_density = adj.sum() / (
        adj.shape[0] * adj.shape[1] * adj.shape[2] * adj.shape[3]
    )
    print(f"Average edge density: {edge_density:.4f}")
    print(f"Total edges: {int(adj.sum())}")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Test each variant
    # variants = ["node_level", "global", "combined"]
    variants = ["node_level"]
    for variant in variants:
        try:
            print("\n" + "=" * 50)
            test_dataset(variant)
        except Exception as e:
            print(f"Error testing {variant} dataset: {str(e)}")

    # Test specific graph type
    print("\n" + "=" * 50)
    print("Testing BA graph type specifically:")
    try:
        test_dataset("node_level", graph_type="BA")
    except Exception as e:
        print(f"Error testing BA graph type: {str(e)}")
