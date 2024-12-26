# synthetic_data/inspect_dataset.py

"""
Dataset Inspector for Sparse Link Prediction

Provides tools to inspect and visualize BA graph sequences with link prediction features:
- Graph structure evolution
- Link prediction feature distributions
- Change point analysis
- Sparsity patterns
"""

import sys
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from scipy import sparse

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from extract_features import LINK_FEATURES

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DatasetInspector:
    """Inspector for analyzing BA graph sequences with link prediction features"""

    def __init__(self, dataset_path: str):
        """Initialize inspector with dataset path"""
        self.dataset_path = dataset_path
        self.format = dataset_path.split(".")[-1]
        logger.info(f"Loading dataset from {dataset_path}")
        self._load_dataset()

    def _load_dataset(self):
        """Load dataset from file"""
        if self.format == "h5":
            self.data = h5py.File(self.dataset_path, "r")
            logger.info("Successfully loaded H5 dataset")
        else:
            raise ValueError(
                f"Unsupported format: {self.format}. Only H5 format is supported."
            )

    def get_basic_stats(self) -> Dict:
        """Get basic dataset statistics"""
        stats = {}

        for split in ["train", "val", "test"]:
            if split not in self.data:
                logger.warning(f"Split {split} not found in dataset")
                continue

            split_group = self.data[split]
            num_sequences = len(split_group)

            if num_sequences == 0:
                logger.warning(f"No sequences found in {split} split")
                continue

            # Get stats from first sequence
            seq0 = split_group["sequence_0"]
            seq_length = seq0["adjacency"].shape[0]
            n_nodes = seq0["adjacency"].shape[1]
            feature_names = list(seq0["features"].keys())

            # Compute sparsity
            sparsity = []
            for i in range(num_sequences):
                seq = split_group[f"sequence_{i}"]
                adj_matrices = seq["adjacency"][:]
                sparsity.append(
                    np.mean(
                        [1 - (adj.sum() / (n_nodes * n_nodes)) for adj in adj_matrices]
                    )
                )

            stats[split] = {
                "num_sequences": num_sequences,
                "sequence_length": seq_length,
                "num_nodes": n_nodes,
                "feature_types": feature_names,
                "avg_sparsity": np.mean(sparsity),
                "std_sparsity": np.std(sparsity),
            }

        return stats

    def analyze_change_points(self) -> Dict:
        """Analyze change point distributions"""
        cp_stats = {}

        for split in ["train", "val", "test"]:
            if split not in self.data:
                continue

            split_group = self.data[split]
            change_points = []

            for i in range(len(split_group)):
                seq = split_group[f"sequence_{i}"]
                if "change_points" in seq:
                    cp = seq["change_points"][:]
                    change_points.append(cp)

            if not change_points:
                logger.warning(f"No change points found in {split} split")
                continue

            num_changes = [len(cp) for cp in change_points]
            cp_locations = (
                np.concatenate(change_points) if change_points else np.array([])
            )

            cp_stats[split] = {
                "avg_num_changes": np.mean(num_changes),
                "std_num_changes": np.std(num_changes),
                "avg_cp_locations": (
                    np.mean(cp_locations) if len(cp_locations) > 0 else 0
                ),
                "std_cp_locations": (
                    np.std(cp_locations) if len(cp_locations) > 0 else 0
                ),
                "min_gap": (
                    np.min(np.diff(np.sort(cp_locations)))
                    if len(cp_locations) > 1
                    else 0
                ),
                "max_gap": (
                    np.max(np.diff(np.sort(cp_locations)))
                    if len(cp_locations) > 1
                    else 0
                ),
            }

        return cp_stats

    def analyze_features(self, sample_size: int = 5) -> Dict:
        """Analyze link prediction feature distributions"""
        feature_stats = {}
        stats = self.get_basic_stats()  # Get n_nodes from basic stats

        for split in ["train", "val", "test"]:
            if split not in self.data:
                continue

            split_group = self.data[split]
            features_dict = {}

            # Sample sequences
            num_sequences = len(split_group)
            seq_indices = np.random.choice(
                num_sequences, size=min(sample_size, num_sequences), replace=False
            )

            n_nodes = stats[split]["num_nodes"] if split in stats else None
            seq_length = stats[split]["sequence_length"] if split in stats else None
            if n_nodes is None or seq_length is None:
                logger.warning(
                    f"Could not determine n_nodes or sequence_length for {split} split"
                )
                continue

            # Calculate total possible edges (excluding self-loops)
            total_possible_edges = (n_nodes * (n_nodes - 1)) // 2

            for seq_idx in seq_indices:
                seq = split_group[f"sequence_{seq_idx}"]
                feat_group = seq["features"]

                for feat_name in LINK_FEATURES:
                    if feat_name not in feat_group:
                        continue

                    if feat_name not in features_dict:
                        features_dict[feat_name] = {
                            "values": [],
                            "total_nonzero": 0,
                            "total_elements": 0,
                            "total_matrices": 0,
                        }

                    try:
                        # Convert sparse matrices to dense for statistics
                        feat_matrices = feat_group[feat_name][:]
                        for matrix in feat_matrices:
                            # Handle different feature types appropriately
                            if feat_name == "degree_similarity":
                                # For dense features, count actual non-zeros
                                nonzero_count = np.count_nonzero(matrix)
                                total_elements = matrix.size
                            else:
                                # For sparse features, only consider upper triangle
                                nonzero_count = np.count_nonzero(
                                    matrix[np.triu_indices_from(matrix, k=1)]
                                )
                                total_elements = total_possible_edges

                            # Get non-zero values only
                            values = matrix[matrix != 0]
                            if len(values) > 0:
                                features_dict[feat_name]["values"].extend(values)
                            features_dict[feat_name]["total_nonzero"] += nonzero_count
                            features_dict[feat_name]["total_elements"] += total_elements
                            features_dict[feat_name]["total_matrices"] += 1

                    except Exception as e:
                        logger.warning(
                            f"Error processing feature {feat_name}: {str(e)}"
                        )
                        continue

            # Compute statistics
            feature_stats[split] = {}
            for feat_name, feat_data in features_dict.items():
                values = feat_data["values"]
                if values:
                    # Calculate sparsity based on actual non-zero elements
                    sparsity = 1.0 - (
                        feat_data["total_nonzero"] / feat_data["total_elements"]
                    )

                    # Validate sparsity
                    if not (0 <= sparsity <= 1):
                        logger.warning(
                            f"Invalid sparsity for {feat_name} in {split} split: {sparsity}. "
                            f"Nonzeros: {feat_data['total_nonzero']}, "
                            f"Total: {feat_data['total_elements']}"
                        )
                        sparsity = np.clip(sparsity, 0, 1)

                    avg_nonzero = (
                        feat_data["total_nonzero"] / feat_data["total_matrices"]
                    )

                    feature_stats[split][feat_name] = {
                        "mean": np.mean(values),
                        "std": np.std(values),
                        "min": np.min(values),
                        "max": np.max(values),
                        "sparsity": sparsity,
                        "density": 1.0 - sparsity,
                        "avg_nonzero_per_graph": avg_nonzero,
                        "total_nonzero": feat_data["total_nonzero"],
                        "total_elements": feat_data["total_elements"],
                    }
                else:
                    feature_stats[split][feat_name] = {
                        "mean": 0,
                        "std": 0,
                        "min": 0,
                        "max": 0,
                        "sparsity": 1.0,
                        "density": 0.0,
                        "avg_nonzero_per_graph": 0,
                        "total_nonzero": 0,
                        "total_elements": 0,
                    }

        return feature_stats

    def plot_feature_distributions(self, save_path: Optional[str] = None):
        """Plot link prediction feature distributions"""
        feature_stats = self.analyze_features()

        if not feature_stats:
            logger.warning("No feature statistics available for plotting")
            return

        # Get available features across all splits
        all_features = set()
        for split_stats in feature_stats.values():
            all_features.update(split_stats.keys())

        fig, axes = plt.subplots(
            len(all_features), 1, figsize=(12, 4 * len(all_features))
        )
        if len(all_features) == 1:
            axes = [axes]

        for i, feat_name in enumerate(sorted(all_features)):
            ax = axes[i]

            # Plot distribution for each split
            for split, stats in feature_stats.items():
                if feat_name not in stats:
                    continue

                feat_stats = stats[feat_name]
                if feat_stats["std"] > 0:  # Only plot if there's variance
                    x = np.linspace(
                        feat_stats["mean"] - 3 * feat_stats["std"],
                        feat_stats["mean"] + 3 * feat_stats["std"],
                        100,
                    )
                    y = np.exp(
                        -0.5 * ((x - feat_stats["mean"]) / feat_stats["std"]) ** 2
                    )
                    ax.plot(
                        x, y, label=f"{split} (sparsity: {feat_stats['sparsity']:.2f})"
                    )

            ax.set_title(f"{feat_name} Distribution")
            ax.set_xlabel("Feature Value")
            ax.set_ylabel("Density")
            ax.legend()

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.close()

    def plot_sparsity_evolution(
        self, split: str = "train", seq_idx: int = 0, save_path: Optional[str] = None
    ):
        """Plot evolution of graph sparsity over time"""
        if split not in self.data:
            logger.warning(f"Split {split} not found in dataset")
            return

        split_group = self.data[split]
        if seq_idx >= len(split_group):
            logger.warning(f"Sequence {seq_idx} not found in {split} split")
            return

        seq = split_group[f"sequence_{seq_idx}"]
        adj_matrices = seq["adjacency"][:]
        n_nodes = adj_matrices.shape[1]

        # Compute sparsity at each timestep
        sparsity = [1 - (adj.sum() / (n_nodes * n_nodes)) for adj in adj_matrices]
        change_points = seq["change_points"][:] if "change_points" in seq else []

        plt.figure(figsize=(12, 6))
        plt.plot(sparsity, label="Sparsity")

        # Plot change points
        for cp in change_points:
            plt.axvline(x=cp, color="r", linestyle="--", alpha=0.5)

        plt.title(f"Graph Sparsity Evolution ({split} sequence {seq_idx})")
        plt.xlabel("Timestep")
        plt.ylabel("Sparsity")
        plt.legend()

        if save_path:
            plt.savefig(save_path)
        plt.close()

    def plot_feature_evolution(
        self,
        feature_name: str,
        split: str = "train",
        seq_idx: int = 0,
        save_path: Optional[str] = None,
    ):
        """Plot evolution of a specific feature over time"""
        if split not in self.data:
            logger.warning(f"Split {split} not found in dataset")
            return

        split_group = self.data[split]
        if seq_idx >= len(split_group):
            logger.warning(f"Sequence {seq_idx} not found in {split} split")
            return

        seq = split_group[f"sequence_{seq_idx}"]
        if "features" not in seq or feature_name not in seq["features"]:
            logger.warning(f"Feature {feature_name} not found in sequence")
            return

        feature_matrices = seq["features"][feature_name][:]
        change_points = seq["change_points"][:] if "change_points" in seq else []

        # Compute feature statistics over time
        mean_values = []
        std_values = []
        sparsity_values = []

        for matrix in feature_matrices:
            values = matrix[matrix != 0]
            mean_values.append(np.mean(values) if len(values) > 0 else 0)
            std_values.append(np.std(values) if len(values) > 0 else 0)
            sparsity_values.append(
                1 - len(values) / (matrix.shape[0] * matrix.shape[1])
            )

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # Plot mean and std
        ax1.plot(mean_values, label="Mean")
        ax1.fill_between(
            range(len(mean_values)),
            np.array(mean_values) - np.array(std_values),
            np.array(mean_values) + np.array(std_values),
            alpha=0.3,
        )

        # Plot sparsity
        ax2.plot(sparsity_values, label="Sparsity", color="g")

        # Plot change points
        for cp in change_points:
            ax1.axvline(x=cp, color="r", linestyle="--", alpha=0.5)
            ax2.axvline(x=cp, color="r", linestyle="--", alpha=0.5)

        ax1.set_title(f"{feature_name} Evolution ({split} sequence {seq_idx})")
        ax1.set_ylabel("Feature Value")
        ax1.legend()

        ax2.set_xlabel("Timestep")
        ax2.set_ylabel("Sparsity")
        ax2.legend()

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.close()

    def print_feature_summary(self):
        """Print detailed feature summary"""
        stats = self.get_basic_stats()
        feature_stats = self.analyze_features()

        print("\nDataset Summary")
        print("-" * 50)

        for split, split_stats in stats.items():
            print(f"\n{split.upper()} Split:")
            print(f"  Sequences: {split_stats['num_sequences']}")
            print(f"  Sequence Length: {split_stats['sequence_length']}")
            print(f"  Nodes: {split_stats['num_nodes']}")
            print(
                f"  Average Graph Sparsity: {split_stats['avg_sparsity']:.4f} ± {split_stats['std_sparsity']:.4f}"
            )

            if split in feature_stats:
                print("\n  Feature Statistics:")
                for feat_name in sorted(feature_stats[split].keys()):
                    feat_stats = feature_stats[split][feat_name]
                    print(f"\n    {feat_name}:")
                    print(f"      Mean Value: {feat_stats['mean']:.4f}")
                    print(f"      Std Dev:    {feat_stats['std']:.4f}")
                    print(
                        f"      Range:      [{feat_stats['min']:.4f}, {feat_stats['max']:.4f}]"
                    )
                    print(f"      Sparsity:   {feat_stats['sparsity']:.4f}")
                    print(f"      Density:    {feat_stats['density']:.4f}")
                    print(
                        f"      Avg Nonzero/Graph: {feat_stats['avg_nonzero_per_graph']:.1f}"
                    )

                # Print feature correlations if more than one feature
                if len(feature_stats[split]) > 1:
                    print("\n  Feature Value Correlations:")
                    features = sorted(feature_stats[split].keys())
                    corr_matrix = np.zeros((len(features), len(features)))

                    # Get a sample sequence for correlation analysis
                    split_group = self.data[split]
                    if len(split_group) > 0:
                        seq = split_group[f"sequence_0"]
                        feat_group = seq["features"]

                        # Compute correlations using the first timestep
                        for i, feat1 in enumerate(features):
                            for j, feat2 in enumerate(features):
                                if i <= j:  # Only compute upper triangle
                                    try:
                                        # Get feature matrices
                                        matrix1 = feat_group[feat1][0]
                                        matrix2 = feat_group[feat2][0]

                                        # Get non-zero values
                                        values1 = matrix1[matrix1 != 0]
                                        values2 = matrix2[matrix2 != 0]

                                        if len(values1) > 0 and len(values2) > 0:
                                            # Use minimum length for correlation
                                            min_len = min(len(values1), len(values2))
                                            corr = np.corrcoef(
                                                values1[:min_len], values2[:min_len]
                                            )[0, 1]
                                            corr_matrix[i, j] = corr
                                            corr_matrix[j, i] = corr
                                    except Exception as e:
                                        logger.warning(
                                            f"Error computing correlation between {feat1} and {feat2}: {str(e)}"
                                        )
                                        corr_matrix[i, j] = np.nan
                                        corr_matrix[j, i] = np.nan

                        # Print correlation matrix
                        print("\n    Correlation Matrix:")
                        print("    " + " ".join(f"{f[:8]:>8}" for f in features))
                        for i, feat in enumerate(features):
                            print(f"    {feat[:8]:<8}", end=" ")
                            for j in range(len(features)):
                                val = corr_matrix[i, j]
                                print(
                                    f"{val:8.2f}" if not np.isnan(val) else "    N/A ",
                                    end=" ",
                                )
                            print()


def main():
    """Example usage of dataset inspection"""
    import argparse

    parser = argparse.ArgumentParser(description="Inspect BA graph sequence dataset")
    parser.add_argument(
        "--dataset", type=str, help="Path to dataset file", required=True
    )
    parser.add_argument(
        "--output", type=str, default="analysis", help="Output directory for analysis"
    )
    args = parser.parse_args()

    try:
        # Create inspector
        inspector = DatasetInspector(args.dataset)

        # Create output directories
        stats_dir = os.path.join(args.output, "statistics")
        plots_dir = os.path.join(args.output, "plots")
        os.makedirs(stats_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)

        # Save basic statistics
        with open(os.path.join(stats_dir, "dataset_stats.txt"), "w") as f:
            stats = inspector.get_basic_stats()
            f.write("Dataset Statistics:\n")
            f.write("-" * 50 + "\n")
            for split, split_stats in stats.items():
                f.write(f"\n{split.upper()}:\n")
                for key, value in split_stats.items():
                    f.write(f"  {key}: {value}\n")

        # Save change point analysis
        with open(os.path.join(stats_dir, "change_point_stats.txt"), "w") as f:
            cp_stats = inspector.analyze_change_points()
            f.write("Change Point Analysis:\n")
            f.write("-" * 50 + "\n")
            for split, split_stats in cp_stats.items():
                f.write(f"\n{split.upper()}:\n")
                for key, value in split_stats.items():
                    f.write(f"  {key}: {value:.4f}\n")

        # Generate plots
        inspector.plot_feature_distributions(
            save_path=os.path.join(plots_dir, "feature_distributions.png")
        )

        # Plot evolution for each feature
        for feature in LINK_FEATURES:
            inspector.plot_feature_evolution(
                feature_name=feature,
                save_path=os.path.join(plots_dir, f"{feature}_evolution.png"),
            )

        # Plot sparsity evolution
        inspector.plot_sparsity_evolution(
            save_path=os.path.join(plots_dir, "sparsity_evolution.png")
        )

        # Print feature summary
        with open(os.path.join(stats_dir, "feature_summary.txt"), "w") as f:
            # Redirect stdout to file
            old_stdout = sys.stdout
            sys.stdout = f
            inspector.print_feature_summary()
            sys.stdout = old_stdout

        print(f"\nAnalysis saved to {args.output}/")

    except Exception as e:
        logger.error(f"Error during inspection: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
