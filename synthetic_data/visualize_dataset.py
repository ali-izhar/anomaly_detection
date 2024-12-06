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
from typing import Tuple, List, Dict, Any
from pathlib import Path
import h5py
import yaml
import argparse

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))


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

    def load_training_data(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, List[List[int]], List[Dict], np.ndarray]:
        """Load training sequences, labels, change points, martingales and sequence lengths from H5 files."""
        train_dir = self.data_dir / "train"

        with h5py.File(train_dir / "data.h5", "r") as hf:
            # Load sequences, labels and lengths from sequences group
            sequences = hf["sequences/features"][:]
            labels = hf["sequences/labels"][:]
            sequence_lengths = hf["sequences/lengths"][:]

            # Load change points
            padded_change_points = hf["change_points/points"][:]
            lengths = hf["change_points/lengths"][:]
            change_points = [
                cp[:length].tolist()
                for cp, length in zip(padded_change_points, lengths)
            ]

            # Load martingales
            martingales = []
            mart_group = hf["martingales"]

            # Get feature names from first sequence's martingales
            first_seq = mart_group["sequence_0"]
            feature_names = list(first_seq["reset"].keys())
            self.feature_names = feature_names  # Update feature names from data

            # Update colors to match number of features
            self.colors = sns.color_palette(
                self.config["visualization"]["colors"], n_colors=len(self.feature_names)
            )

            # Iterate through sequences
            for seq_idx in range(len(sequences)):
                seq_group = mart_group[f"sequence_{seq_idx}"]

                # Load reset and cumulative martingales
                seq_mart = {"reset": {}, "cumulative": {}}

                for mart_type in ["reset", "cumulative"]:
                    type_group = seq_group[mart_type]
                    for feat_name in feature_names:
                        seq_mart[mart_type][feat_name] = {
                            "martingales": type_group[feat_name][:]
                        }

                martingales.append(seq_mart)

        return sequences, labels, change_points, martingales, sequence_lengths

    def plot_sequence_timeline(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        change_points: List[List[int]],
        martingales: List[Dict],
        sequence_idx: int,
        sequence_lengths: np.ndarray,
        save_path: str = None,
    ) -> None:
        """Plot timeline view of a single sequence with martingales and labels."""
        # Get actual sequence length
        seq_length = int(sequence_lengths[sequence_idx])

        plt.figure(figsize=(15, 12))
        gs = plt.GridSpec(3, 1, height_ratios=[2, 2, 1], hspace=0.3)

        # Get change points and martingales for this sequence
        seq_change_points = change_points[sequence_idx]
        seq_martingales = martingales[sequence_idx]

        # Top subplot: Reset Martingales
        ax1 = plt.subplot(gs[0])
        self._plot_martingale_sequence(
            ax1,
            seq_martingales["reset"],
            "Reset Martingale Measures",
            seq_change_points,
            cumulative=False,
            sequence_length=seq_length,
        )

        # Middle subplot: Cumulative Martingales
        ax2 = plt.subplot(gs[1])
        self._plot_martingale_sequence(
            ax2,
            seq_martingales["cumulative"],
            "Cumulative Martingale Measures",
            seq_change_points,
            cumulative=True,
            sequence_length=seq_length,
        )

        # Bottom subplot: Labels
        ax3 = plt.subplot(gs[2])
        sequence_labels = labels[sequence_idx, :seq_length]
        ax3.fill_between(
            range(seq_length),
            0,
            sequence_labels,
            color="red",
            alpha=0.3,
            label="Anomaly",
        )

        # Add change points
        for cp in seq_change_points:
            if cp < seq_length:
                ax3.axvline(x=cp, color="r", linestyle="--", alpha=0.5)

        ax3.set_ylim(-0.1, 1.1)
        ax3.set_xlim(0, seq_length)
        ax3.set_title("Anomaly Labels")
        ax3.set_xlabel("Time Step")
        ax3.set_ylabel("Label")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.show()

    def _plot_martingale_sequence(
        self,
        ax: plt.Axes,
        martingales: Dict[str, Dict[str, Any]],
        title: str,
        change_points: List[int],
        cumulative: bool = False,
        sequence_length: int = None,
    ) -> None:
        """Plot martingale sequences with change points."""
        # Add change point indicators
        for cp in change_points:
            if cp < sequence_length:
                ax.axvspan(cp - 5, cp + 5, color="red", alpha=0.1)

        # Plot individual feature martingales
        for name, color in zip(self.feature_names, self.colors):
            martingale_values = np.array(martingales[name]["martingales"])
            if sequence_length:
                martingale_values = martingale_values[:sequence_length]

            if cumulative:
                ax.semilogy(
                    martingale_values,
                    color=color,
                    label=name.capitalize(),
                    linewidth=1.5,
                    alpha=0.6,
                )
            else:
                ax.plot(
                    martingale_values,
                    color=color,
                    label=name.capitalize(),
                    linewidth=1.5,
                    alpha=0.6,
                )

        # Compute and plot combined martingales
        martingale_arrays = []
        for name in self.feature_names:
            values = np.array(martingales[name]["martingales"])
            if sequence_length:
                values = values[:sequence_length]
            martingale_arrays.append(values)

        martingale_arrays = np.array(martingale_arrays)
        M_sum = np.sum(martingale_arrays, axis=0)
        M_avg = M_sum / len(self.feature_names)

        if cumulative:
            ax.semilogy(
                M_avg, color="#FF4B4B", label="Average", linewidth=2.5, alpha=0.9
            )
            ax.semilogy(
                M_sum,
                color="#2F2F2F",
                label="Sum",
                linewidth=2.5,
                linestyle="-.",
                alpha=0.8,
            )
        else:
            ax.plot(M_avg, color="#FF4B4B", label="Average", linewidth=2.5, alpha=0.9)
            ax.plot(
                M_sum,
                color="#2F2F2F",
                label="Sum",
                linewidth=2.5,
                linestyle="-.",
                alpha=0.8,
            )

        # Add threshold line
        ax.axhline(
            y=self.config["martingale"]["threshold"],
            color="r",
            linestyle="--",
            label="Threshold",
        )

        # Set x-axis limit to sequence length
        if sequence_length:
            ax.set_xlim(0, sequence_length)

        # Customize plot
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.set_title(title, fontsize=12, pad=15)
        ax.set_xlabel("Time Steps", fontsize=10)
        ax.set_ylabel(
            "Martingale Values" + (" (log scale)" if cumulative else ""),
            fontsize=10,
        )

        # Add legend
        legend = ax.legend(
            fontsize=10,
            ncol=3,
            loc="upper right" if not cumulative else "upper left",
            bbox_to_anchor=(1, 1.02) if not cumulative else (0, 1.02),
            frameon=True,
            facecolor="none",
            edgecolor="none",
        )
        legend.get_frame().set_facecolor("none")
        legend.get_frame().set_alpha(0)

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
        sequences, labels, change_points, martingales, sequence_lengths = (
            self.load_training_data()
        )

        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get shapes and statistics
        n_sequences = len(sequences)
        max_seq_length = sequences.shape[1]
        n_features = sequences.shape[2]
        avg_seq_length = np.mean(sequence_lengths)
        std_seq_length = np.std(sequence_lengths)

        print(f"\nTraining Data Shape:")
        print(f"Number of sequences: {n_sequences}")
        print(f"Maximum sequence length: {max_seq_length}")
        print(f"Average sequence length: {avg_seq_length:.1f} ± {std_seq_length:.1f}")
        print(f"Number of features: {n_features}")

        # Plot example sequences with change points and martingales
        print("\nPlotting example sequences...")
        for i in range(min(5, n_sequences)):
            self.plot_sequence_timeline(
                sequences,
                labels,
                change_points,
                martingales,
                i,
                sequence_lengths,
                save_path=output_dir / f"sequence_{i}.png",
            )

        # For feature distributions and correlations, use only valid timesteps
        valid_features = []
        valid_labels = []
        for i, length in enumerate(sequence_lengths):
            valid_features.append(sequences[i, : int(length)])
            valid_labels.append(labels[i, : int(length)])

        valid_features = np.concatenate(valid_features)
        valid_labels = np.concatenate(valid_labels)

        # Plot feature distributions using only valid timesteps
        print("\nPlotting feature distributions...")
        self.plot_feature_distributions(
            valid_features.reshape(1, -1, sequences.shape[-1]),
            save_path=output_dir / "feature_distributions.png",
        )

        # Plot feature correlations using only valid timesteps
        print("\nPlotting feature correlations...")
        self.plot_feature_correlations(
            valid_features.reshape(1, -1, sequences.shape[-1]),
            save_path=output_dir / "feature_correlations.png",
        )

        # Plot label statistics using only valid timesteps
        print("\nPlotting label statistics...")
        self.plot_label_statistics(
            valid_labels.reshape(1, -1), save_path=output_dir / "label_statistics.png"
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
        default="configs/martingale_config.yaml",
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
