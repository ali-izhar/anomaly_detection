"""Dataset Inspector for Link Prediction Datasets

This module provides tools to analyze and visualize datasets generated for link prediction:
- Visualize graph sequences and their changes
- Analyze feature distributions and correlations
- Verify change points and community structure
- Profile dataset statistics and balance
"""

import h5py
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_regression

logger = logging.getLogger(__name__)


class DatasetInspector:
    """Inspector for analyzing link prediction datasets"""

    def __init__(self, dataset_path: str):
        """Initialize with path to HDF5 dataset"""
        self.dataset_path = dataset_path
        self.splits = ["train", "val", "test"]

    def load_dataset(self) -> None:
        """Load dataset and basic statistics"""
        with h5py.File(self.dataset_path, "r") as f:
            self.config = self._load_config(f)
            self.stats = self._compute_basic_stats(f)

    def _load_config(self, f: h5py.File) -> Dict:
        """Load configuration from dataset"""
        config = {}
        config_group = f["config"]

        for key in config_group.attrs.keys():
            config[key] = config_group.attrs[key]

        for group_name in config_group.keys():
            config[group_name] = {}
            group = config_group[group_name]

            for key in group.attrs.keys():
                config[group_name][key] = group.attrs[key]

            for subgroup_name in group.keys():
                config[group_name][subgroup_name] = {}
                subgroup = group[subgroup_name]
                for key in subgroup.attrs.keys():
                    config[group_name][subgroup_name][key] = subgroup.attrs[key]

        return config

    def _compute_basic_stats(self, f: h5py.File) -> Dict:
        """Compute basic dataset statistics"""
        stats = {
            "num_sequences": {},
            "avg_nodes": {},
            "avg_edges": {},
            "avg_density": {},
            "avg_changes": {},
            "feature_dims": {},
        }

        for split in self.splits:
            if split not in f:
                continue

            split_group = f[split]
            sequences = list(split_group.keys())
            stats["num_sequences"][split] = len(sequences)

            nodes = []
            edges = []
            densities = []
            changes = []

            for seq_name in sequences:
                seq = split_group[seq_name]
                adj_matrices = seq["adjacency"][:]

                # Compute graph statistics
                nodes.append(adj_matrices.shape[1])
                edges.append(np.sum(adj_matrices, axis=(1, 2)) / 2)
                densities.append(edges[-1] / (nodes[-1] * (nodes[-1] - 1) / 2))

                # Count changes
                changes.append(len(seq["change_points"]))

                # Get feature dimensions if not already recorded
                if split not in stats["feature_dims"]:
                    stats["feature_dims"][split] = {
                        name: data.shape for name, data in seq["features"].items()
                    }

            stats["avg_nodes"][split] = np.mean(nodes)
            stats["avg_edges"][split] = np.mean(edges)
            stats["avg_density"][split] = np.mean(densities)
            stats["avg_changes"][split] = np.mean(changes)

        return stats

    def plot_graph_statistics(self, split: str = "train") -> None:
        """Plot graph statistics over time for a random sequence"""
        with h5py.File(self.dataset_path, "r") as f:
            split_group = f[split]
            seq_name = np.random.choice(list(split_group.keys()))
            seq = split_group[seq_name]

            adj_matrices = seq["adjacency"][:]
            change_points = seq["change_points"][:]

            # Compute statistics over time
            edges = np.sum(adj_matrices, axis=(1, 2)) / 2
            density = edges / (adj_matrices.shape[1] * (adj_matrices.shape[1] - 1) / 2)

            # Plot
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

            # Edge count
            ax1.plot(edges, label="Edge Count")
            ax1.set_title(f"Edge Count Over Time - {seq_name}")
            for cp in change_points:
                ax1.axvline(x=cp, color="r", linestyle="--", alpha=0.5)
            ax1.legend()

            # Density
            ax2.plot(density, label="Density")
            ax2.set_title("Graph Density Over Time")
            for cp in change_points:
                ax2.axvline(x=cp, color="r", linestyle="--", alpha=0.5)
            ax2.legend()

            plt.tight_layout()
            plt.show()

    def plot_feature_distributions(
        self, split: str = "train", sample_size: int = 1000
    ) -> None:
        """Plot distributions of features"""
        with h5py.File(self.dataset_path, "r") as f:
            split_group = f[split]
            seq_name = np.random.choice(list(split_group.keys()))
            seq = split_group[seq_name]

            features = seq["features"]
            feature_names = list(features.keys())

            # Sample random indices
            n_samples = min(sample_size, features[feature_names[0]].shape[0])
            indices = np.random.choice(features[feature_names[0]].shape[0], n_samples)

            # Create subplot grid
            n_features = len(feature_names)
            n_cols = 3
            n_rows = (n_features + n_cols - 1) // n_cols

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
            axes = axes.flatten()

            for i, feat_name in enumerate(feature_names):
                feat_data = features[feat_name][:][indices].flatten()

                # Check if feature has variance
                if np.std(feat_data) < 1e-10:
                    # For zero-variance features, just show the value
                    axes[i].text(
                        0.5,
                        0.5,
                        f"Constant value: {feat_data[0]:.3f}",
                        horizontalalignment="center",
                        verticalalignment="center",
                        transform=axes[i].transAxes,
                    )
                    axes[i].set_xticks([])
                    axes[i].set_yticks([])
                else:
                    try:
                        # Try to plot with KDE
                        sns.histplot(feat_data, kde=True, ax=axes[i])
                    except (np.linalg.LinAlgError, ValueError):
                        # Fallback to simple histogram if KDE fails
                        sns.histplot(feat_data, kde=False, ax=axes[i])

                axes[i].set_title(f"{feat_name} Distribution")

            # Remove empty subplots
            for i in range(n_features, len(axes)):
                fig.delaxes(axes[i])

            plt.tight_layout()
            plt.show()

    def plot_change_point_analysis(self, split: str = "train") -> None:
        """Analyze and visualize change points"""
        with h5py.File(self.dataset_path, "r") as f:
            split_group = f[split]

            # Collect change point statistics
            segment_lengths = []
            num_changes = []

            for seq_name in split_group.keys():
                seq = split_group[seq_name]
                change_points = list(seq["change_points"][:])
                num_changes.append(len(change_points))

                # Calculate segment lengths
                prev_point = 0
                for cp in change_points + [seq["adjacency"].shape[0]]:
                    segment_lengths.append(cp - prev_point)
                    prev_point = cp

            # Plot distributions
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            # Number of changes
            sns.histplot(num_changes, ax=ax1)
            ax1.set_title("Distribution of Number of Changes")
            ax1.set_xlabel("Number of Changes")

            # Segment lengths
            sns.histplot(segment_lengths, ax=ax2)
            ax2.set_title("Distribution of Segment Lengths")
            ax2.set_xlabel("Segment Length")

            plt.tight_layout()
            plt.show()

    def plot_community_structure(self, split: str = "train") -> None:
        """Visualize community structure of a random graph"""
        with h5py.File(self.dataset_path, "r") as f:
            split_group = f[split]
            seq_name = np.random.choice(list(split_group.keys()))
            seq = split_group[seq_name]

            # Get first adjacency matrix and community labels
            adj_matrix = seq["adjacency"][0]
            community_labels = seq["metadata/community_labels"][:]

            # Create graph
            G = nx.from_numpy_array(adj_matrix)

            # Plot
            plt.figure(figsize=(10, 10))
            pos = nx.spring_layout(G)
            nx.draw_networkx_nodes(
                G, pos, node_color=community_labels, cmap=plt.cm.tab20
            )
            nx.draw_networkx_edges(G, pos, alpha=0.2)
            plt.title(f"Community Structure - {seq_name}")
            plt.show()

    def print_dataset_summary(self) -> None:
        """Print summary statistics of the dataset"""
        print("\n=== Dataset Summary ===")
        print(f"\nDataset path: {self.dataset_path}")

        print("\nSequence counts:")
        for split, count in self.stats["num_sequences"].items():
            print(f"  {split}: {count}")

        print("\nAverage statistics per split:")
        for split in self.splits:
            if split in self.stats["num_sequences"]:
                print(f"\n{split.upper()}:")
                print(f"  Nodes: {self.stats['avg_nodes'][split]:.1f}")
                print(f"  Edges: {self.stats['avg_edges'][split]:.1f}")
                print(f"  Density: {self.stats['avg_density'][split]:.3f}")
                print(f"  Changes: {self.stats['avg_changes'][split]:.1f}")

        print("\nFeature dimensions:")
        for split, dims in self.stats["feature_dims"].items():
            print(f"\n{split.upper()}:")
            for feat, shape in dims.items():
                print(f"  {feat}: {shape}")

    def plot_feature_correlations(self, split: str = "train") -> None:
        """Plot correlation matrix of features"""
        with h5py.File(self.dataset_path, "r") as f:
            split_group = f[split]
            seq_name = np.random.choice(list(split_group.keys()))
            seq = split_group[seq_name]

            # Get feature data
            features = {}
            for feat_name, feat_data in seq["features"].items():
                features[feat_name] = feat_data[:]

            # Create correlation matrix
            feature_names = list(features.keys())
            corr_matrix = np.zeros((len(feature_names), len(feature_names)))

            for i, feat1 in enumerate(feature_names):
                for j, feat2 in enumerate(feature_names):
                    # Compute correlation only for features with same shape
                    if features[feat1].shape == features[feat2].shape:
                        corr_matrix[i, j] = np.corrcoef(
                            features[feat1].flatten(), features[feat2].flatten()
                        )[0, 1]
                    else:
                        corr_matrix[i, j] = np.nan

            # Plot correlation matrix
            plt.figure(figsize=(12, 10))
            sns.heatmap(
                corr_matrix,
                xticklabels=feature_names,
                yticklabels=feature_names,
                annot=True,
                cmap="coolwarm",
                vmin=-1,
                vmax=1,
                center=0,
            )
            plt.title("Feature Correlations")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

    def analyze_feature_importance(
        self, split: str = "train", n_samples: int = 10000
    ) -> None:
        """Analyze feature importance using mutual information"""
        with h5py.File(self.dataset_path, "r") as f:
            split_group = f[split]
            seq_name = np.random.choice(list(split_group.keys()))
            seq = split_group[seq_name]

            # Get feature data
            features = {}
            for feat_name, feat_data in seq["features"].items():
                features[feat_name] = feat_data[:]

            # Sample data points
            feature_names = list(features.keys())
            base_shape = features[feature_names[0]].shape[0]
            n_samples = min(n_samples, base_shape)
            indices = np.random.choice(base_shape, n_samples)

            # Filter features with matching shape and create feature matrix
            valid_features = []
            feature_data = []

            for feat in feature_names:
                if features[feat].shape[0] == base_shape and feat != "link_frequency":
                    valid_features.append(feat)
                    feature_data.append(features[feat].flatten()[indices])

            if not valid_features:
                logger.warning("No valid features found for importance analysis")
                return

            X = np.column_stack(feature_data)

            # Use link frequency as target
            if "link_frequency" not in features:
                logger.warning("Link frequency feature not found")
                return

            y = features["link_frequency"].flatten()[indices]

            # Compute mutual information
            mi_scores = mutual_info_regression(X, y)

            # Create DataFrame with matched lengths
            importance_df = pd.DataFrame(
                {"Feature": valid_features, "Importance": mi_scores}
            )

            # Sort by importance
            importance_df = importance_df.sort_values("Importance", ascending=True)

            # Plot feature importance
            plt.figure(figsize=(12, 6))
            sns.barplot(data=importance_df, x="Importance", y="Feature")
            plt.title("Feature Importance (Mutual Information)")
            plt.tight_layout()
            plt.show()

    def analyze_community_structure(self, split: str = "train") -> None:
        """Analyze community structure statistics"""
        with h5py.File(self.dataset_path, "r") as f:
            split_group = f[split]

            community_sizes = []
            intra_densities = []
            inter_densities = []

            for seq_name in list(split_group.keys())[:5]:  # Analyze first 5 sequences
                seq = split_group[seq_name]
                adj_matrix = seq["adjacency"][0]  # First timestep

                try:
                    community_labels = seq["metadata/community_labels"][:]
                    unique_communities = np.unique(community_labels)

                    # Analyze each community
                    for comm in unique_communities:
                        # Get community nodes
                        comm_nodes = np.where(community_labels == comm)[0]
                        community_sizes.append(len(comm_nodes))

                        # Compute intra-community density
                        intra_edges = adj_matrix[comm_nodes][:, comm_nodes]
                        intra_density = np.sum(intra_edges) / (
                            len(comm_nodes) * (len(comm_nodes) - 1)
                        )
                        intra_densities.append(intra_density)

                        # Compute inter-community density
                        other_nodes = np.where(community_labels != comm)[0]
                        inter_edges = adj_matrix[comm_nodes][:, other_nodes]
                        inter_density = np.sum(inter_edges) / (
                            len(comm_nodes) * len(other_nodes)
                        )
                        inter_densities.append(inter_density)

                except KeyError:
                    logger.warning(f"Community labels not found for {seq_name}")
                    continue

            # Plot community statistics
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

            # Community sizes
            sns.histplot(community_sizes, ax=ax1)
            ax1.set_title("Distribution of Community Sizes")
            ax1.set_xlabel("Community Size")

            # Density comparison
            data = pd.DataFrame(
                {
                    "Density": intra_densities + inter_densities,
                    "Type": ["Intra-community"] * len(intra_densities)
                    + ["Inter-community"] * len(inter_densities),
                }
            )
            sns.boxplot(data=data, x="Type", y="Density", ax=ax2)
            ax2.set_title("Community Density Comparison")

            plt.tight_layout()
            plt.show()

            # Print statistics
            print("\nCommunity Structure Statistics:")
            print(f"Average community size: {np.mean(community_sizes):.1f}")
            print(f"Average intra-community density: {np.mean(intra_densities):.3f}")
            print(f"Average inter-community density: {np.mean(inter_densities):.3f}")
            print(
                f"Modularity ratio: {np.mean(intra_densities)/np.mean(inter_densities):.2f}"
            )


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Inspect link prediction dataset")
    parser.add_argument("dataset_path", type=str, help="Path to HDF5 dataset file")
    args = parser.parse_args()

    inspector = DatasetInspector(args.dataset_path)
    inspector.load_dataset()

    # Print dataset summary
    inspector.print_dataset_summary()

    print("\nGenerating visualizations...")

    # Basic visualizations
    print("\n1. Graph Statistics")
    inspector.plot_graph_statistics()

    print("\n2. Feature Distributions")
    inspector.plot_feature_distributions()

    print("\n3. Change Point Analysis")
    inspector.plot_change_point_analysis()

    print("\n4. Community Structure")
    inspector.plot_community_structure()

    print("\n5. Feature Analysis")
    inspector.plot_feature_correlations()
    inspector.analyze_feature_importance()

    print("\n6. Community Analysis")
    inspector.analyze_community_structure()


if __name__ == "__main__":
    main()
