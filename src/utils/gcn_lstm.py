# src/utils/gcn_lstm.py

from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Union

import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle
from scipy.spatial.distance import cdist


class DataLoader:
    """Handles loading and basic preprocessing of data."""

    @staticmethod
    def load_data(filename: Union[str, Path]) -> np.ndarray:
        """Load soccer matrix data from a .npz file."""
        filename = Path(filename)
        if not filename.exists():
            raise FileNotFoundError(f"The file '{filename}' does not exist.")

        try:
            data = np.load(filename)
        except Exception as e:
            raise IOError(f"Error reading file '{filename}': {e}")

        if "data" not in data:
            raise ValueError(f"File '{filename}' missing 'data' key.")

        return data["data"]

    @staticmethod
    def preprocess_data(
        data: np.ndarray,
        pitch_control_data: Optional[np.ndarray] = None,
        feature: str = "influence",
    ) -> np.ndarray:
        """Preprocess the data based on selected feature."""
        if feature == "pitch_control":
            if pitch_control_data is None:
                raise ValueError(
                    "pitch_control_data required for 'pitch_control' feature."
                )

            X = data[1:]  # Skip first time frame
            X[:, :, 0] = pitch_control_data
            nan_rows = np.isnan(pitch_control_data).any(axis=1)
            return X[~nan_rows]

        elif feature == "influence":
            return data

        raise ValueError(f"Unsupported feature: {feature}")


class GraphConstructor:
    """Handles graph construction and processing."""

    @staticmethod
    def create_edge_indices_and_distances(
        X: np.ndarray, distance_threshold: float = 15
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Create edge indices and distances for graph construction."""
        if not isinstance(X, np.ndarray) or X.ndim != 3 or X.shape[2] < 3:
            raise ValueError("Invalid input array shape.")

        if not isinstance(distance_threshold, (int, float)) or distance_threshold <= 0:
            raise ValueError("Invalid distance threshold.")

        edge_indices = []
        edge_distances = []

        for t in range(X.shape[0]):
            coordinates = X[t, :, 1:3]
            distances = cdist(coordinates, coordinates, "euclidean")
            adjacency = (distances < distance_threshold) & (distances != 0)

            edges = []
            dists = []
            for i in range(adjacency.shape[0]):
                for j in range(adjacency.shape[1]):
                    if adjacency[i, j]:
                        edges.append((i, j))
                        dists.append(distances[i, j])

            edge_indices.append(torch.tensor(edges, dtype=torch.long).t().contiguous())
            edge_distances.append(torch.tensor(dists, dtype=torch.float))

        return edge_indices, edge_distances

    @staticmethod
    def pad_edge_data(
        edge_indices: List[torch.Tensor],
        edge_distances: List[torch.Tensor],
        max_edges: int,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Pad edge data for consistent sizes."""
        if len(edge_indices) != len(edge_distances):
            raise ValueError("Mismatched edge indices and distances lengths.")

        if not isinstance(max_edges, int) or max_edges <= 0:
            raise ValueError("Invalid max_edges value.")

        padded_indices = []
        padded_distances = []

        for idx, dist in zip(edge_indices, edge_distances):
            # Pad indices
            pad_size_idx = max_edges - idx.size(1)
            if pad_size_idx > 0:
                pad_idx = torch.full(
                    (2, pad_size_idx), -1, dtype=idx.dtype, device=idx.device
                )
                padded_idx = torch.cat([idx, pad_idx], dim=1)
            else:
                padded_idx = idx

            # Pad distances
            pad_size_dist = max_edges - dist.size(0)
            if pad_size_dist > 0:
                pad_dist = torch.full(
                    (pad_size_dist,), float("inf"), dtype=dist.dtype, device=dist.device
                )
                padded_dist = torch.cat([dist, pad_dist])
            else:
                padded_dist = dist

            padded_indices.append(padded_idx)
            padded_distances.append(padded_dist)

        return padded_indices, padded_distances


class SequenceProcessor:
    """Handles sequence creation and processing."""

    @staticmethod
    def create_sequences(
        input_data: np.ndarray,
        output_data: np.ndarray,
        n_steps_in: int,
        n_steps_out: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create input-output sequences."""
        X, y = [], []
        for i in range(len(input_data) - n_steps_in - n_steps_out + 1):
            X.append(input_data[i : (i + n_steps_in)])
            y.append(output_data[(i + n_steps_in) : (i + n_steps_in + n_steps_out)])
        return np.array(X), np.array(y)

    @staticmethod
    def normalize_feature(feature: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """Normalize features using min-max scaling."""
        min_val = np.min(feature)
        max_val = np.max(feature)
        return (feature - min_val) / (max_val - min_val), min_val, max_val

    @staticmethod
    def split_data(
        X_sequences: np.ndarray,
        y_sequences: np.ndarray,
        edge_indices_sequences: List,
        edge_distances_sequences: List,
        train_split: float = 0.8,
    ) -> Tuple:
        """Split data into training and testing sets."""
        n_train = int(train_split * X_sequences.shape[0])
        return (
            X_sequences[:n_train],
            y_sequences[:n_train],
            edge_indices_sequences[:n_train],
            edge_distances_sequences[:n_train],
            X_sequences[n_train:],
            y_sequences[n_train:],
            edge_indices_sequences[n_train:],
            edge_distances_sequences[n_train:],
        )

    @staticmethod
    def convert_to_tensors(*arrays: np.ndarray) -> Tuple[torch.Tensor, ...]:
        """Convert numpy arrays to PyTorch tensors."""
        return tuple(torch.tensor(arr, dtype=torch.float) for arr in arrays)


class ResultAnalyzer:
    """Handles result analysis and visualization."""

    @staticmethod
    def calculate_residue(
        true_influences: np.ndarray,
        predicted_influences: np.ndarray,
        epsilon: float = 1e-10,
    ) -> np.ndarray:
        """Calculate residue between true and predicted values."""
        if true_influences.shape != predicted_influences.shape:
            raise ValueError("Shape mismatch between true and predicted influences.")
        return np.abs(true_influences - predicted_influences) / (
            true_influences + epsilon
        )

    @staticmethod
    def calculate_metrics(residue: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate various residue metrics."""
        return {
            "median_residue_per_player": np.median(residue, axis=(0, 1)),
            "mean_residue_per_player": np.mean(residue, axis=(0, 1)),
            "average_residue_all_players": np.mean(residue),
        }

    @classmethod
    def load_results(cls, folder_path: Union[str, Path]) -> Dict[str, Any]:
        """Load and process all result files."""
        folder_path = Path(folder_path)
        results = {}

        for file_path in folder_path.glob("*.pkl"):
            with open(file_path, "rb") as f:
                model_results = pickle.load(f)

            # Calculate missing metrics if needed
            if "residue_all_samples" not in model_results:
                residue = cls.calculate_residue(
                    model_results["y_true"], model_results["y_pred"]
                )
                model_results["residue_all_samples"] = residue
                model_results.update(cls.calculate_metrics(residue))

                # Save updated results
                with open(file_path, "wb") as f:
                    pickle.dump(model_results, f)

            results[file_path.stem] = model_results

        return results

    @staticmethod
    def plot_residue(
        results: Dict[str, Any],
        title: str = "Median Residue per Player",
        filter_game: Optional[int] = None,
        filter_models: Optional[List[str]] = None,
    ) -> None:
        """Plot residue comparison across models."""
        plt.figure(figsize=(14, 6))
        players = np.arange(1, 23)
        max_val = 0

        for model_name, data in results.items():
            if (
                filter_game is not None
                and data["game"] != filter_game
                or filter_models is not None
                and data["model"] not in filter_models
            ):
                continue

            residue = data["median_residue_per_player"]
            plt.plot(players, residue, "o-", label=model_name)
            max_val = max(max_val, np.max(residue))

        plt.xlabel("Player", fontsize=16)
        plt.ylabel("Median Residue", fontsize=16)
        plt.xticks(players, fontsize=12)
        plt.title(title, fontsize=18)
        plt.legend(fontsize=12)
        plt.show()
