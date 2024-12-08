# src/models2/datasets.py

import os
import sys
import h5py
import torch
import time
import psutil
import logging
import numpy as np
from pathlib import Path
import torch.utils.data as data
from typing import Dict, Tuple
from torch import Tensor

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.utils.normalization import Normalizer


def print_hdf5_structure(file_path: str) -> None:
    """Print the structure and metadata of an HDF5 file.

    Args:
        file_path: Path to the HDF5 file
    """

    def print_attrs(name: str, obj: h5py.Dataset) -> None:
        logging.info(f"Found object: {name} | Type: {type(obj).__name__}")
        if hasattr(obj, "shape"):
            logging.info(f"  Shape: {obj.shape}")
        if hasattr(obj, "dtype"):
            logging.info(f"  Dtype: {obj.dtype}")

    try:
        with h5py.File(file_path, "r") as f:
            logging.info(f"\nStructure of {file_path}:")
            f.visititems(print_attrs)
    except Exception as e:
        logging.error(f"Error reading {file_path}: {str(e)}")


class GraphTimeSeriesDataset(data.Dataset):
    """Dataset class for graph time series data.

    Handles loading and preprocessing of temporal graph data from HDF5 files.
    """

    def __init__(self, data_dir: str, split: str, config: Dict) -> None:
        """
        Args:
            data_dir: Path to the processed data directory
            split: One of 'train', 'val', 'test'
            config: Configuration dictionary containing data parameters
        """
        super().__init__()
        self.split = split
        self.sequence_length = config["data"]["sequence_length"]
        self.m_horizon = config["data"]["m_horizon"]
        self.num_features = config["data"]["num_features"]

        # Load data from HDF5 file
        h5_path = os.path.join(data_dir, split, "data.h5")
        self._load_data(h5_path)

        # Handle normalization
        self._setup_normalizer(data_dir, split)

        # Log dataset info
        logging.info(f"\nLoaded data shapes for {split}:")
        logging.info(f"Features shape: {self.features.shape}")
        logging.info(f"Adjacency matrices shape: {self.adj_matrices.shape}")
        logging.info(f"Lengths shape: {self.lengths.shape}")

    def _load_data(self, h5_path: str) -> None:
        """Load data from HDF5 file."""
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"Data file not found: {h5_path}")

        try:
            with h5py.File(h5_path, "r") as f:
                self.features = torch.from_numpy(f["features/all"][:])
                self.lengths = torch.from_numpy(f["lengths"][:])

                # Handle adjacency matrices
                if isinstance(f["adjacency"], h5py.Group):
                    adj_data = [
                        f[f"adjacency/sequence_{i}"][:]
                        for i in range(len(self.features))
                    ]
                    self.adj_matrices = torch.from_numpy(np.array(adj_data))
                else:
                    self.adj_matrices = torch.from_numpy(f["adjacency"][:])

                # Store metadata
                self.graph_types = [gt.decode("utf-8") for gt in f["graph_types"][:]]
                self.change_points = {
                    i: f[f"change_points/sequence_{i}"][:]
                    for i in range(len(self.features))
                    if f"change_points/sequence_{i}" in f
                }
        except Exception as e:
            raise RuntimeError(f"Error loading data from {h5_path}: {str(e)}")

    def _setup_normalizer(self, data_dir: str, split: str) -> None:
        """Setup data normalization."""
        if split == "train":
            self.normalizer = Normalizer()
            # Reshape features to 2D for normalization
            all_features = self.features.reshape(-1, self.features.shape[-1])
            self.normalizer.fit(all_features.numpy())
            self.normalizer.save(os.path.join(data_dir, "normalizer.pkl"))
        else:
            self.normalizer = Normalizer.load(os.path.join(data_dir, "normalizer.pkl"))

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, Tensor], Tensor]:
        if idx >= len(self):
            raise IndexError(
                f"Index {idx} out of bounds for dataset of size {len(self)}"
            )

        # Get single sample
        length = self.lengths[idx].item()
        adj_matrices = self.adj_matrices[idx, :length]  # (T, N, N)
        features = self.features[idx, :length]  # (T, num_features)

        # Normalize features
        features = torch.from_numpy(self.normalizer.transform(features.numpy())).float()

        # Prepare input sequence and target
        input_seq = {
            "adj_matrices": adj_matrices[: -self.m_horizon],  # (T - m, N, N)
            "features": features[: -self.m_horizon],  # (T - m, num_features)
        }
        target_seq = features[-self.m_horizon :]  # (m, num_features)

        return input_seq, target_seq


def get_dataloaders(
    data_dir: str, config: Dict, num_workers: int = 4, pin_memory: bool = True
) -> Tuple[data.DataLoader, data.DataLoader, data.DataLoader]:
    """Create DataLoader instances for train, validation and test sets.

    Args:
        data_dir: Path to data directory
        config: Configuration dictionary
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    datasets = {
        split: GraphTimeSeriesDataset(data_dir, split, config)
        for split in ["train", "val", "test"]
    }

    loaders = {
        "train": data.DataLoader(
            datasets["train"],
            batch_size=config["data"]["batch_size"],
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "val": data.DataLoader(
            datasets["val"],
            batch_size=config["data"]["batch_size"],
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "test": data.DataLoader(
            datasets["test"],
            batch_size=config["data"]["batch_size"],
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
    }

    return loaders["train"], loaders["val"], loaders["test"]


def inspect_dataset(
    dataset: GraphTimeSeriesDataset, split_name: str
) -> Dict[str, float]:
    """Inspect and validate a dataset's properties.

    Args:
        dataset: Dataset instance to inspect
        split_name: Name of the data split (train/val/test)

    Returns:
        Dictionary containing dataset statistics
    """
    logging.info(f"\n=== Inspecting {split_name} dataset ===")

    # Basic info
    logging.info(f"Dataset size: {len(dataset)} samples")

    # Get a sample
    input_seq, target_seq = dataset[0]

    # Inspect shapes
    logging.info("\nShapes:")
    logging.info(f"Adjacency matrices: {input_seq['adj_matrices'].shape}")
    logging.info(f"Features: {input_seq['features'].shape}")
    logging.info(f"Target sequence: {target_seq.shape}")

    # Calculate statistics
    features = input_seq["features"]
    stats = {
        "mean": features.mean().item(),
        "std": features.std().item(),
        "min": features.min().item(),
        "max": features.max().item(),
        "has_nan": torch.isnan(features).any().item(),
        "has_inf": torch.isinf(features).any().item(),
    }

    # Log statistics
    logging.info("\nFeature statistics:")
    logging.info(f"Mean: {stats['mean']:.4f}")
    logging.info(f"Std: {stats['std']:.4f}")
    logging.info(f"Min: {stats['min']:.4f}")
    logging.info(f"Max: {stats['max']:.4f}")

    # Data validation
    logging.info("\nData validation:")
    logging.info(f"Contains NaN: {stats['has_nan']}")
    logging.info(f"Contains Inf: {stats['has_inf']}")

    return stats


def get_memory_usage() -> float:
    """Get current memory usage of the process.

    Returns:
        Memory usage in megabytes
    """
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def main() -> None:
    """Main function to test and validate the dataset implementation."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Configuration
    config = {
        "data": {
            "sequence_length": 188,
            "m_horizon": 12,
            "num_features": 6,
            "batch_size": 32,
        }
    }

    # Data directory
    data_dir = "dataset"

    # Verify directory structure
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(data_dir, split)
        h5_path = os.path.join(split_dir, "data.h5")

        if not os.path.exists(split_dir):
            logging.error(f"Directory not found: {split_dir}")
            continue

        if not os.path.exists(h5_path):
            logging.error(f"HDF5 file not found: {h5_path}")
            continue

        logging.info(f"Found data file: {h5_path}")
        print_hdf5_structure(h5_path)

    try:
        # Track memory and time
        start_memory = get_memory_usage()
        start_time = time.time()

        # Create dataloaders
        logging.info("Loading dataloaders...")
        train_loader, val_loader, test_loader = get_dataloaders(
            data_dir, config, num_workers=4, pin_memory=torch.cuda.is_available()
        )

        # Log memory usage
        memory_used = get_memory_usage() - start_memory
        time_taken = time.time() - start_time
        logging.info(f"Data loading completed in {time_taken:.2f} seconds")
        logging.info(f"Memory usage: {memory_used:.2f} MB")

        # Inspect datasets
        dataset_stats = {}
        for name, loader in [
            ("Training", train_loader),
            ("Validation", val_loader),
            ("Test", test_loader),
        ]:
            dataset_stats[name] = inspect_dataset(loader.dataset, name)

        # Test batch iteration
        logging.info("\nTesting batch iteration...")
        batch = next(iter(train_loader))
        input_seq, target_seq = batch

        logging.info("\nFirst batch shapes:")
        logging.info(f"Input adjacency matrices: {input_seq['adj_matrices'].shape}")
        logging.info(f"Input features: {input_seq['features'].shape}")
        logging.info(f"Target sequence: {target_seq.shape}")

        # Final memory usage
        final_memory = get_memory_usage() - start_memory
        logging.info(f"\nFinal memory usage: {final_memory:.2f} MB")

    except Exception as e:
        logging.error(f"Error occurred: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
