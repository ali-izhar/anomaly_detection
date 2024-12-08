# data/datasets.py

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from utils.normalization import Normalizer


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

        # Load data
        self.data = np.load(
            os.path.join(data_dir, f"{split}_data.npy"), allow_pickle=True
        )
        # data should be a list of dictionaries with 'adj_matrices' and 'features'

        # Initialize normalizer using training data
        if split == "train":
            self.normalizer = Normalizer()
            all_features = np.concatenate(
                [sample["features"] for sample in self.data], axis=0
            )
            self.normalizer.fit(all_features)
        else:
            # Load pre-fitted normalizer
            self.normalizer = Normalizer.load(os.path.join(data_dir, "normalizer.pkl"))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        adj_matrices = sample["adj_matrices"]  # Shape: (T, N, N)
        features = sample["features"]  # Shape: (T, 6)

        # Normalize features
        features = self.normalizer.transform(features)  # Shape: (T, 6)

        # Convert to tensors
        adj_matrices = torch.tensor(adj_matrices, dtype=torch.float32)  # (T, N, N)
        features = torch.tensor(features, dtype=torch.float32)  # (T, 6)

        # Input sequence and target
        # For simplicity, use full sequence as input and predict next m steps
        input_seq = {
            "adj_matrices": adj_matrices[: -self.m_horizon],  # (T - m, N, N)
            "features": features[: -self.m_horizon],  # (T - m, 6)
        }
        target_seq = features[self.m_horizon :]  # (m, 6)

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
