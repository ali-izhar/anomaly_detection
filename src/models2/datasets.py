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

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from torch.utils.data import Dataset, DataLoader
from src.utils.normalization import Normalizer


def print_hdf5_structure(file_path):
    """Helper function to print the structure of an HDF5 file."""
    def print_attrs(name, obj):
        print(f"Found object: {name} | Type: {type(obj).__name__}")
        if hasattr(obj, 'shape'):
            print(f"  Shape: {obj.shape}")
        if hasattr(obj, 'dtype'):
            print(f"  Dtype: {obj.dtype}")
    
    try:
        with h5py.File(file_path, 'r') as f:
            print(f"\nStructure of {file_path}:")
            f.visititems(print_attrs)
    except Exception as e:
        print(f"Error reading {file_path}: {str(e)}")


class GraphTimeSeriesDataset(Dataset):
    def __init__(self, data_dir, split, config):
        """
        Args:
            data_dir (str): Path to the processed data.
            split (str): One of 'train', 'val', 'test'.
            config (dict): Configuration dictionary.
        """
        self.split = split
        self.sequence_length = config["data"]["sequence_length"]
        self.m_horizon = config["data"]["m_horizon"]
        self.num_features = config["data"]["num_features"]

        # Load data from HDF5 file
        h5_path = os.path.join(data_dir, split, "data.h5")
        
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"Data file not found: {h5_path}")
        
        try:
            with h5py.File(h5_path, "r") as f:
                # Load features
                self.features = f["features/all"][:]      # Shape: (num_samples, T, num_features)
                self.lengths = f["lengths"][:]            # Load sequence lengths
                
                # Load and reshape adjacency matrix if needed
                if isinstance(f["adjacency"], h5py.Group):
                    adj_data = []
                    for i in range(len(self.features)):
                        seq_adj = f[f"adjacency/sequence_{i}"][:]
                        adj_data.append(seq_adj)
                    self.adj_matrices = np.array(adj_data)
                else:
                    self.adj_matrices = f["adjacency"][:]
                
                # Store graph types if needed
                self.graph_types = [gt.decode('utf-8') for gt in f["graph_types"][:]]
                
                # Store change points dictionary
                self.change_points = {}
                for i in range(len(self.features)):
                    cp_key = f"change_points/sequence_{i}"
                    if cp_key in f:
                        self.change_points[i] = f[cp_key][:]

        except Exception as e:
            raise Exception(f"Error loading data from {h5_path}: {str(e)}")

        # Print shapes for debugging
        print(f"\nLoaded data shapes for {split}:")
        print(f"Features shape: {self.features.shape}")
        print(f"Adjacency matrices shape: {self.adj_matrices.shape}")
        print(f"Lengths shape: {self.lengths.shape}")

        # Initialize normalizer using training data
        if split == "train":
            self.normalizer = Normalizer()
            # Reshape features to 2D for normalization
            all_features = self.features.reshape(-1, self.features.shape[-1])
            self.normalizer.fit(all_features)
            # Save normalizer in the parent data directory
            self.normalizer.save(os.path.join(data_dir, "normalizer.pkl"))
        else:
            # Load pre-fitted normalizer from parent data directory
            self.normalizer = Normalizer.load(os.path.join(data_dir, "normalizer.pkl"))

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds for dataset of size {len(self)}")
            
        # Get single sample
        adj_matrices = self.adj_matrices[idx]  # Shape should be (T, N, N)
        features = self.features[idx]          # Shape: (T, num_features)
        length = self.lengths[idx]             # Actual sequence length
        
        # Trim sequences to actual length if needed
        adj_matrices = adj_matrices[:length]
        features = features[:length]

        # Normalize features
        features = self.normalizer.transform(features)

        # Convert to tensors
        adj_matrices = torch.tensor(adj_matrices, dtype=torch.float32)
        features = torch.tensor(features, dtype=torch.float32)

        # Input sequence and target
        input_seq = {
            "adj_matrices": adj_matrices[:-self.m_horizon],  # (T - m, N, N)
            "features": features[:-self.m_horizon],          # (T - m, num_features)
        }
        target_seq = features[-self.m_horizon:]             # (m, num_features) - just the horizon steps

        return input_seq, target_seq


def get_dataloaders(data_dir, config):
    train_dataset = GraphTimeSeriesDataset(data_dir, "train", config)
    val_dataset = GraphTimeSeriesDataset(data_dir, "val", config)
    test_dataset = GraphTimeSeriesDataset(data_dir, "test", config)

    train_loader = DataLoader(
        train_dataset, batch_size=config["data"]["batch_size"], shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["data"]["batch_size"], shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config["data"]["batch_size"], shuffle=False
    )

    return train_loader, val_loader, test_loader


def inspect_dataset(dataset, split_name):
    """Helper function to inspect a dataset's properties."""
    print(f"\n=== Inspecting {split_name} dataset ===")

    # Basic info
    print(f"Dataset size: {len(dataset)} samples")

    # Get a sample
    input_seq, target_seq = dataset[0]

    # Inspect shapes
    print("\nShapes:")
    print(f"Adjacency matrices: {input_seq['adj_matrices'].shape}")
    print(f"Features: {input_seq['features'].shape}")
    print(f"Target sequence: {target_seq.shape}")

    # Check data statistics
    print("\nFeature statistics:")
    features = input_seq["features"]
    print(f"Mean: {features.mean().item():.4f}")
    print(f"Std: {features.std().item():.4f}")
    print(f"Min: {features.min().item():.4f}")
    print(f"Max: {features.max().item():.4f}")

    # Check for NaN/Inf
    print("\nData validation:")
    print(f"Contains NaN: {torch.isnan(features).any().item()}")
    print(f"Contains Inf: {torch.isinf(features).any().item()}")


def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Updated configuration to match actual data dimensions
    config = {
        "data": {
            "sequence_length": 188,  # Match the actual sequence length we see
            "m_horizon": 12,
            "num_features": 6,
            "batch_size": 32,
        }
    }

    # Path to your data directory
    data_dir = "dataset"
    
    # Verify directory structure
    logger.info(f"Checking data directory structure...")
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(data_dir, split)
        if not os.path.exists(split_dir):
            logger.error(f"Directory not found: {split_dir}")
            continue
        
        h5_path = os.path.join(split_dir, "data.h5")
        if not os.path.exists(h5_path):
            logger.error(f"HDF5 file not found: {h5_path}")
            continue
        
        logger.info(f"Found data file: {h5_path}")

    try:
        # Record starting memory
        start_memory = get_memory_usage()
        start_time = time.time()

        logger.info("Loading dataloaders...")
        train_loader, val_loader, test_loader = get_dataloaders(data_dir, config)

        # Memory usage after loading
        memory_after_loading = get_memory_usage()
        loading_time = time.time() - start_time

        logger.info(f"Data loading completed in {loading_time:.2f} seconds")
        logger.info(f"Memory usage: {memory_after_loading - start_memory:.2f} MB")

        # Inspect each dataset
        inspect_dataset(train_loader.dataset, "Training")
        inspect_dataset(val_loader.dataset, "Validation")
        inspect_dataset(test_loader.dataset, "Test")

        # Test batch iteration
        logger.info("\nTesting batch iteration...")
        for batch_idx, (input_seq, target_seq) in enumerate(train_loader):
            if batch_idx == 0:
                print("\nFirst batch shapes:")
                print(f"Input adjacency matrices: {input_seq['adj_matrices'].shape}")
                print(f"Input features: {input_seq['features'].shape}")
                print(f"Target sequence: {target_seq.shape}")
                break

        # Memory usage after everything
        final_memory = get_memory_usage()
        logger.info(f"\nFinal memory usage: {final_memory - start_memory:.2f} MB")

    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        raise e


if __name__ == "__main__":
    main()
