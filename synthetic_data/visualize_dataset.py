# synthetic_data/visualize_dataset.py

"""
Dataset Visualization Script

This script reads the train/val/test data from the dataset and creates visualizations showing:
1. Sequence-level feature time-series for multiple feature types, with change points (up to 5 sequences per split).
2. Feature distributions and correlations across the training set.
3. Basic statistics for each split (e.g., sequence lengths, node counts, etc.).

Can also plot multiple sequences side-by-side on a single figure (dashboard) with dotted lines at actual change points.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional
from pathlib import Path
import h5py
import yaml
import argparse

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))


class DatasetVisualizer:
    """Visualizer for the created graph sequences dataset with change points."""

    def __init__(self, data_dir: str = "dataset", config_path: str = None):
        self.data_dir = Path(data_dir)

        # Load a config file for visualization settings if available
        if config_path is None:
            self.config = {
                "visualization": {
                    "colors": "husl",
                }
            }
        else:
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)

        plt.style.use("seaborn-v0_8-darkgrid")

        # Define feature names in the order they appear in features["all"]:
        # (seq_len, 6) -> 0: degree, 1: betweenness, 2: eigenvector, 3: closeness, 4: svd, 5: lsvd
        self.feature_names = [
            "Degree",
            "Betweenness",
            "Eigenvector",
            "Closeness",
            "SVD",
            "LSVD",
        ]

        # Set colors
        self.colors = sns.color_palette(
            self.config["visualization"].get("colors", "husl"),
            n_colors=len(self.feature_names),
        )

    def load_data_split(self, split: str) -> Dict[str, Any]:
        """Load a specific data split (train/val/test) from HDF5, including change points."""
        split_dir = self.data_dir / split
        data_path = split_dir / "data.h5"
        if not data_path.exists():
            raise FileNotFoundError(f"No data found for split '{split}' at {data_path}")

        with h5py.File(data_path, "r") as hf:
            # Load sequence lengths
            sequence_lengths = hf["lengths"][:]
            # Load graph types if available
            graph_types = (
                hf["graph_types"][:].astype(str) if "graph_types" in hf else None
            )

            # Load adjacency
            graphs_group = hf["adjacency"]
            num_sequences = len(sequence_lengths)
            first_graph = graphs_group["sequence_0"][:]
            max_seq_len, n_nodes, _ = first_graph.shape
            all_graphs = np.zeros(
                (num_sequences, max_seq_len, n_nodes, n_nodes), dtype=np.float32
            )
            all_graphs[0] = first_graph
            for i in range(1, num_sequences):
                all_graphs[i] = graphs_group[f"sequence_{i}"][:]

            # Load features (now a single dataset "all" with shape (num_sequences, max_seq_len, 6))
            feat_group = hf["features"]
            if "all" not in feat_group:
                raise ValueError("Features dataset 'all' not found")
            features = {"all": feat_group["all"][:]}

            # Load change points if available
            change_points = []
            if "change_points" in hf:
                cp_group = hf["change_points"]
                for i in range(num_sequences):
                    seq_cp = cp_group[f"sequence_{i}"][:]
                    change_points.append(seq_cp)

        return {
            "features": features,
            "graphs": all_graphs,
            "sequence_lengths": sequence_lengths,
            "graph_types": graph_types,
            "change_points": change_points,
        }

    def plot_sequence_features(
        self,
        features: Dict[str, np.ndarray],
        sequence_lengths: np.ndarray,
        sequence_idx: int,
        change_points: Optional[List[int]] = None,
        save_path: str = None,
    ) -> None:
        """
        Plot the time-series of features for a single sequence.
        If change_points are provided, add vertical dashed lines at those points.
        """
        seq_length = int(sequence_lengths[sequence_idx])
        feat_data = features["all"][sequence_idx, :seq_length, :]  # shape: (seq_len, 6)

        plt.figure(figsize=(12, 8))
        for i, (feat_name, color) in enumerate(zip(self.feature_names, self.colors)):
            plt.plot(
                feat_data[:, i],
                label=feat_name,
                color=color,
                linewidth=1.5,
                alpha=0.8,
            )

        # Plot change points if present
        if change_points is not None and len(change_points) > 0:
            cp_array = np.array(change_points)
            valid_cps = cp_array[cp_array < seq_length]
            for cp in valid_cps:
                plt.axvline(x=cp, color="black", linestyle="--", alpha=0.7)

        plt.title(f"Sequence {sequence_idx} Feature Time-Series")
        plt.xlabel("Time Step")
        plt.ylabel("Normalized Feature Value")
        plt.legend()
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    def plot_dashboard_of_sequences(
        self,
        features: Dict[str, np.ndarray],
        sequence_lengths: np.ndarray,
        sequence_indices: List[int],
        change_points: Optional[List[List[int]]] = None,
        rows: int = 2,
        cols: int = 3,
        save_path: str = None,
    ) -> None:
        """
        Plot multiple sequences in a single figure (dashboard).
        Each subplot shows one sequence with all features and dotted lines at change points.
        """
        fig, axes = plt.subplots(rows, cols, figsize=(15, 8))
        axes = axes.flatten()

        for i, seq_idx in enumerate(sequence_indices):
            ax = axes[i]
            seq_length = int(sequence_lengths[seq_idx])
            feat_data = features["all"][seq_idx, :seq_length, :]  # (seq_len, 6)

            cp_list = (
                change_points[i]
                if (change_points is not None and i < len(change_points))
                else None
            )

            # Plot features
            for feat_name, color, f_idx in zip(
                self.feature_names, self.colors, range(len(self.feature_names))
            ):
                ax.plot(
                    feat_data[:, f_idx],
                    label=feat_name,
                    color=color,
                    linewidth=1.5,
                    alpha=0.8,
                )

            # Add dotted lines for change points if available
            if cp_list is not None:
                for c in cp_list:
                    if 0 <= c < seq_length:
                        ax.axvline(x=c, color="black", linestyle="--", alpha=0.7)

            ax.set_title(f"Sequence {seq_idx}", fontsize=10)
            ax.set_xlabel("Time Step", fontsize=9)
            ax.set_ylabel("Feature Value", fontsize=9)
            ax.tick_params(axis="both", which="major", labelsize=8)
            ax.grid(True, alpha=0.3)

        # Turn off any unused subplots
        for j in range(i + 1, rows * cols):
            axes[j].axis("off")

        # Add a single legend for all features below the plots
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=len(self.feature_names))
        fig.tight_layout(rect=[0, 0.05, 1, 1])

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    def plot_feature_distributions(
        self,
        features: Dict[str, np.ndarray],
        sequence_lengths: np.ndarray,
        save_path: str = None,
    ) -> None:
        """
        Plot distribution of each feature from the training set.
        features["all"] is (num_sequences, max_seq_len, 6).
        We'll flatten across all sequences and valid timesteps.
        """
        feat_data = features["all"]
        num_sequences, max_seq_len, n_features = feat_data.shape

        # Flatten each feature across all sequences and valid lengths
        valid_values = []
        for f_idx in range(n_features):
            vals = []
            for i, length in enumerate(sequence_lengths):
                vals.append(feat_data[i, :length, f_idx])
            vals = np.concatenate(vals)
            valid_values.append(vals)

        plt.figure(figsize=(15, 10))
        for i, (feat_name, color) in enumerate(zip(self.feature_names, self.colors)):
            plt.subplot(2, 3, i + 1)
            sns.histplot(valid_values[i], color=color, kde=True)
            plt.title(f"{feat_name} Distribution")
            plt.xlabel("Value")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    def plot_feature_correlations(
        self,
        features: Dict[str, np.ndarray],
        sequence_lengths: np.ndarray,
        save_path: str = None,
    ) -> None:
        """
        Plot correlation matrix between features for the training set.
        We have (N, T, 6). We'll flatten all sequences and timesteps into one large array of shape (total_samples, 6).
        """
        feat_data = features["all"]
        rows = []
        for i, length in enumerate(sequence_lengths):
            rows.append(feat_data[i, :length, :])  # (length, 6)
        all_data = np.vstack(rows)  # (total_timesteps_across_all_sequences, 6)

        correlation_matrix = np.corrcoef(all_data.T)

        plt.figure(figsize=(10, 8))
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
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    def visualize_data_split(
        self,
        split_name: str,
        output_dir: Path,
        visualize_distributions: bool = False,
        visualize_correlations: bool = False,
    ) -> None:
        """Visualize data for a given split. Only first 5 sequences are shown."""
        print(f"\nLoading {split_name} data...")
        data = self.load_data_split(split_name)
        features = data["features"]
        graphs = data["graphs"]
        sequence_lengths = data["sequence_lengths"]
        graph_types = data["graph_types"]
        change_points = data["change_points"]  # Actual changepoints

        n_sequences = len(sequence_lengths)
        max_seq_length = graphs.shape[1]
        n_nodes = graphs.shape[2]

        print(f"\n{split_name.capitalize()} Data Stats:")
        print(f"Number of sequences: {n_sequences}")
        print(f"Max sequence length: {max_seq_length}")
        print(f"Number of nodes: {n_nodes}")
        if graph_types is not None:
            unique_types, counts = np.unique(graph_types, return_counts=True)
            print("Graph types distribution:")
            for t, c in zip(unique_types, counts):
                print(f"  {t}: {c} sequences")

        # Visualize up to 5 sequences from this split
        max_to_plot = min(5, n_sequences)
        print(
            f"\nPlotting up to {max_to_plot} example sequences from {split_name} set..."
        )
        for i in range(max_to_plot):
            cp = change_points[i] if change_points else None
            self.plot_sequence_features(
                features,
                sequence_lengths,
                i,
                change_points=cp,
                save_path=output_dir / f"{split_name}_sequence_{i}_features.png",
            )

        # Dashboard of multiple sequences if we have at least 5 sequences
        if n_sequences >= 5:
            print(
                f"\nPlotting a dashboard of multiple sequences from {split_name} set..."
            )
            seq_indices = list(range(5))
            cp_subset = (
                [change_points[i] for i in seq_indices] if change_points else None
            )
            self.plot_dashboard_of_sequences(
                features,
                sequence_lengths,
                sequence_indices=seq_indices,
                change_points=cp_subset,
                rows=2,
                cols=3,
                save_path=output_dir / f"{split_name}_dashboard_of_sequences.png",
            )

        # For train split, we can also visualize distributions and correlations
        if split_name == "train":
            if visualize_distributions:
                print("\nPlotting feature distributions for training set...")
                self.plot_feature_distributions(
                    features,
                    sequence_lengths,
                    save_path=output_dir / "train_feature_distributions.png",
                )

            if visualize_correlations:
                print("\nPlotting feature correlations for training set...")
                self.plot_feature_correlations(
                    features,
                    sequence_lengths,
                    save_path=output_dir / "train_feature_correlations.png",
                )

    def visualize_all_splits(self, output_dir: str = "visualizations") -> None:
        """Visualize data for train/val/test splits."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Visualize training data (with distributions and correlations)
        self.visualize_data_split(
            "train",
            output_dir,
            visualize_distributions=True,
            visualize_correlations=True,
        )

        # Visualize validation data (just the first 5 sequences and dashboard)
        if (self.data_dir / "val").exists():
            self.visualize_data_split("val", output_dir)

        # Visualize test data (just the first 5 sequences and dashboard)
        if (self.data_dir / "test").exists():
            self.visualize_data_split("test", output_dir)

        print(f"\nVisualizations saved to: {output_dir}/")


def main():
    """Main entry point for visualization."""
    parser = argparse.ArgumentParser(description="Visualize graph sequence dataset")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        help="Path to optional config YAML file",
        default=None,
    )
    parser.add_argument(
        "--data-dir",
        "-d",
        type=str,
        default="dataset",
        help="Path to dataset directory",
    )

    args = parser.parse_args()

    print("Visualizing dataset (train/val/test)...")
    visualizer = DatasetVisualizer(data_dir=args.data_dir, config_path=args.config)
    visualizer.visualize_all_splits()


if __name__ == "__main__":
    main()
