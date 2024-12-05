# synthetic_data/visualize_dataset.py

"""
Dataset Visualization Script

This script reads the training data from the dataset directory and creates visualizations showing:
1. Timeline view of sequences with anomaly labels
2. Feature distributions and correlations
3. Training data statistics and balance
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List
from pathlib import Path
import h5py
import yaml
import argparse

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.changepoint import ChangePointDetector
from visualize_martingales import MartingaleVisualizer


class DatasetVisualizer:
    """Visualizer for graph anomaly detection dataset."""

    def __init__(self, data_dir: str = "dataset", config_path: str = None):
        self.data_dir = Path(data_dir)

        # Load martingale config
        if config_path is None:
            config_path = Path(__file__).parent / "martingale_config.yaml"

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Get visualization parameters from config
        self.feature_names = self.config["visualization"]["feature_names"]
        self.colors = sns.color_palette(
            self.config["visualization"]["colors"], n_colors=len(self.feature_names)
        )
        plt.style.use("seaborn-v0_8-darkgrid")

    def load_training_data(self) -> Tuple[np.ndarray, np.ndarray, List[List[int]]]:
        """Load training sequences, labels and change points from H5 files."""
        train_dir = self.data_dir / "train"

        with h5py.File(train_dir / "data.h5", "r") as hf:
            sequences = hf["sequences"][:]
            labels = hf["labels"][:]
            padded_change_points = hf["change_points"][:]
            lengths = hf["change_point_lengths"][:]

            # Unpad change points
            change_points = [
                cp[:length].tolist()
                for cp, length in zip(padded_change_points, lengths)
            ]

        return sequences, labels, change_points

    def plot_sequence_timeline(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        change_points: List[List[int]],
        sequence_idx: int,
        save_path: str = None,
    ) -> None:
        """Plot timeline view of a single sequence with features, martingales, and labels."""
        plt.figure(figsize=(15, 12))
        gs = plt.GridSpec(3, 1, height_ratios=[2, 2, 1], hspace=0.3)

        # Get change points for this sequence
        seq_change_points = change_points[sequence_idx]

        # Top subplot: Features
        ax1 = plt.subplot(gs[0])
        for i, (name, color) in enumerate(zip(self.feature_names, self.colors)):
            feature_values = features[sequence_idx, :, i]
            normalized_values = (feature_values - np.mean(feature_values)) / np.std(
                feature_values
            )
            ax1.plot(normalized_values, label=name, color=color, alpha=0.7)

        # Add change point vertical lines
        for cp in seq_change_points:
            ax1.axvline(x=cp, color="r", linestyle="--", alpha=0.5)

        ax1.set_title(f"Sequence {sequence_idx}: Feature Values Over Time")
        ax1.set_xlabel("Time Step")
        ax1.set_ylabel("Feature Value (Normalized)")
        ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        # Middle subplot: Martingales
        ax2 = plt.subplot(gs[1])
        martingale_config = self.config["martingale"]
        
        # Extract features for this sequence
        seq_features = features[sequence_idx]  # Shape: (sequence_length, num_features)
        
        # First compute all martingales
        martingales = {"reset": {}}
        detector = ChangePointDetector()
        
        # Normalize entire feature sequence first
        normalized_features = np.zeros_like(seq_features)
        for i in range(seq_features.shape[1]):  # For each feature
            feature_values = seq_features[:, i]
            normalized_features[:, i] = (feature_values - np.mean(feature_values)) / (np.std(feature_values) + 1e-8)

        # Compute martingales for each feature
        for i, name in enumerate(self.feature_names):
            # Get normalized feature values
            feature_values = normalized_features[:, i].reshape(-1, 1)
            
            # Compute martingales
            result = detector.martingale_test(
                data=feature_values,
                threshold=martingale_config["threshold"],
                epsilon=martingale_config["epsilon"],
                reset=True,
            )
            martingales["reset"][name] = result

        # Plot martingales directly
        for (name, results), color in zip(martingales["reset"].items(), self.colors):
            martingale_values = np.array([
                x.item() if isinstance(x, np.ndarray) else x 
                for x in results["martingales"]
            ])
            ax2.semilogy(
                np.maximum(martingale_values, 1e-10),  # Ensure positive values for log scale
                label=name,
                color=color,
                linewidth=1.5,
                alpha=0.6,
            )

        # Compute and plot combined martingales
        martingale_arrays = []
        for name in self.feature_names:
            values = np.array([
                x.item() if isinstance(x, np.ndarray) else x 
                for x in martingales["reset"][name]["martingales"]
            ])
            martingale_arrays.append(values)

        M_sum = np.sum(martingale_arrays, axis=0)
        M_avg = M_sum / len(self.feature_names)

        ax2.semilogy(
            M_avg,
            color="#FF4B4B",
            label="Average",
            linewidth=2.5,
            alpha=0.9,
        )
        ax2.semilogy(
            M_sum,
            color="#2F2F2F",
            label="Sum",
            linewidth=2.5,
            linestyle="-.",
            alpha=0.8,
        )

        # Add threshold line
        ax2.axhline(
            y=martingale_config["threshold"],
            color="r",
            linestyle="--",
            label="Threshold",
        )

        # Add change points
        for cp in seq_change_points:
            ax2.axvline(x=cp, color="r", linestyle="--", alpha=0.5)

        # Customize plot
        ax2.grid(True, linestyle="--", alpha=0.3)
        ax2.set_title("Martingale Values Over Time")
        ax2.set_xlabel("Time Step")
        ax2.set_ylabel("Martingale Value (log scale)")
        ax2.legend(
            fontsize=10,
            ncol=3,
            loc="upper left",
            bbox_to_anchor=(0, 1.02),
            frameon=True,
            facecolor="none",
            edgecolor="none",
        )

        # Bottom subplot: Labels
        ax3 = plt.subplot(gs[2])
        sequence_labels = labels[sequence_idx]
        ax3.fill_between(
            range(len(sequence_labels)),
            0,
            sequence_labels,
            color="red",
            alpha=0.3,
            label="Anomaly",
        )

        # Add change points
        for cp in seq_change_points:
            ax3.axvline(x=cp, color="r", linestyle="--", alpha=0.5)

        ax3.set_ylim(-0.1, 1.1)
        ax3.set_title("Anomaly Labels")
        ax3.set_xlabel("Time Step")
        ax3.set_ylabel("Label")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.show()

    def plot_feature_distributions(
        self, features: np.ndarray, save_path: str = None
    ) -> None:
        """Plot distribution of each feature across all sequences."""
        plt.figure(figsize=(15, 10))

        for i, (name, color) in enumerate(zip(self.feature_names, self.colors)):
            plt.subplot(2, 3, i + 1)
            feature_values = features[:, :, i].flatten()
            sns.histplot(feature_values, color=color, kde=True)
            plt.title(f"{name} Distribution")
            plt.xlabel("Value")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def plot_feature_correlations(
        self, features: np.ndarray, save_path: str = None
    ) -> None:
        """Plot correlation matrix between features."""
        flat_features = features.reshape(-1, features.shape[-1])

        plt.figure(figsize=(10, 8))
        correlation_matrix = np.corrcoef(flat_features.T)
        sns.heatmap(
            correlation_matrix,
            xticklabels=self.feature_names,
            yticklabels=self.feature_names,
            annot=True,
            cmap="coolwarm",
            center=0,
            vmin=-1,
            vmax=1,
        )
        plt.title("Feature Correlations")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def plot_label_statistics(self, labels: np.ndarray, save_path: str = None) -> None:
        """Plot statistics about anomaly labels."""
        plt.figure(figsize=(15, 5))

        # Plot 1: Overall label distribution
        plt.subplot(1, 2, 1)
        total_samples = labels.size
        anomaly_samples = np.sum(labels)
        normal_samples = total_samples - anomaly_samples

        plt.pie(
            [normal_samples, anomaly_samples],
            labels=["Normal", "Anomaly"],
            autopct="%1.1f%%",
            colors=["lightblue", "lightcoral"],
        )
        plt.title("Label Distribution")

        # Plot 2: Anomaly distribution across sequences
        plt.subplot(1, 2, 2)
        anomaly_per_seq = np.mean(labels, axis=1) * 100
        plt.hist(anomaly_per_seq, bins=20, color="lightcoral")
        plt.title("Anomaly Percentage per Sequence")
        plt.xlabel("Percentage of Anomalies")
        plt.ylabel("Number of Sequences")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def visualize_training_data(self, output_dir: str = "visualizations") -> None:
        """Create comprehensive visualizations of the training data."""
        print("Loading training data...")
        sequences, labels, change_points = self.load_training_data()

        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get shapes
        n_sequences, seq_length, n_features = sequences.shape
        print(f"\nTraining Data Shape:")
        print(f"Number of sequences: {n_sequences}")
        print(f"Sequence length: {seq_length}")
        print(f"Number of features: {n_features}")

        # 1. Plot example sequences with change points
        print("\nPlotting example sequences...")
        for i in range(min(5, n_sequences)):
            self.plot_sequence_timeline(
                sequences,
                labels,
                change_points,
                i,
                save_path=output_dir / f"sequence_{i}.png",
            )

        # 2. Plot feature distributions
        print("\nPlotting feature distributions...")
        self.plot_feature_distributions(
            sequences, save_path=output_dir / "feature_distributions.png"
        )

        # 3. Plot feature correlations
        print("\nPlotting feature correlations...")
        self.plot_feature_correlations(
            sequences, save_path=output_dir / "feature_correlations.png"
        )

        # 4. Plot label statistics
        print("\nPlotting label statistics...")
        self.plot_label_statistics(
            labels, save_path=output_dir / "label_statistics.png"
        )

        print(f"\nVisualizations saved to: {output_dir}/")


def main():
    """Main entry point for visualization."""
    parser = argparse.ArgumentParser(description="Visualize graph sequence dataset")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        help="Path to martingale config YAML file",
        default="martingale_config.yaml",
    )
    parser.add_argument(
        "--data-dir",
        "-d",
        type=str,
        default="dataset",
        help="Path to dataset directory",
    )

    args = parser.parse_args()

    print("Visualizing training data...")
    visualizer = DatasetVisualizer(data_dir=args.data_dir, config_path=args.config)
    visualizer.visualize_training_data()


if __name__ == "__main__":
    main()
