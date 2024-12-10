"""
Dataset Inspector

This module provides tools to inspect and visualize different types of graph sequence datasets:
- Node-level features with adjacency
- Global features with adjacency
- Combined features
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
from tqdm import tqdm
import networkx as nx
from ast import literal_eval

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

logger = logging.getLogger(__name__)


class DatasetInspector:
    """Inspector for analyzing graph sequence datasets"""

    def __init__(self, dataset_path: str):
        """Initialize inspector with dataset path"""
        self.dataset_path = dataset_path
        self.format = dataset_path.split(".")[-1]
        self._load_dataset()
        self._determine_dataset_type()

    def _determine_dataset_type(self):
        """Determine the type of dataset based on feature shape"""
        stats = self.get_basic_stats()
        feature_shape = stats[list(stats.keys())[0]]["feature_shape"]

        if len(feature_shape) == 3:  # (seq_len, n_nodes, n_features)
            if feature_shape[2] == 6:
                self.dataset_type = "node_level"
            elif feature_shape[2] == 36:  # 30 (adjacency) + 6 (features)
                self.dataset_type = "combined"
            else:
                raise ValueError(f"Unexpected feature dimension: {feature_shape[2]}")
        elif len(feature_shape) == 2:  # (seq_len, n_features + adj_size)
            self.dataset_type = "global"
        else:
            raise ValueError(f"Unexpected feature shape: {feature_shape}")

    def _load_dataset(self):
        """Load dataset from file"""
        if self.format == "h5":
            self.data = h5py.File(self.dataset_path, "r")
        elif self.format == "npz":
            self.data = {
                graph_type: np.load(
                    str(Path(self.dataset_path).parent / f"{graph_type}_dataset.npz"),
                    allow_pickle=True,
                )
                for graph_type in ["BA", "ER", "NW"]
            }
        else:
            raise ValueError(f"Unsupported format: {self.format}")

    def get_basic_stats(self) -> Dict:
        """Get basic dataset statistics"""
        stats = {}

        for graph_type in ["BA", "ER", "NW"]:
            if self.format == "h5":
                group = self.data[graph_type]
                num_sequences = len(group["sequences"])
                seq_length = group["sequences/seq_0/adjacency"].shape[0]
                n_nodes = group["sequences/seq_0/adjacency"].shape[1]
                feature_shape = group["sequences/seq_0/features"].shape
            else:
                data = self.data[graph_type]
                num_sequences = len(data["adjacency"])
                seq_length = data["adjacency"][0].shape[0]
                n_nodes = data["adjacency"][0].shape[1]
                feature_shape = data["features"][0].shape

            stats[graph_type] = {
                "num_sequences": num_sequences,
                "sequence_length": seq_length,
                "num_nodes": n_nodes,
                "feature_shape": feature_shape,
            }

        return stats

    def analyze_change_points(self) -> Dict:
        """Analyze change point distributions"""
        cp_stats = {}

        for graph_type in ["BA", "ER", "NW"]:
            if self.format == "h5":
                cp_group = self.data[graph_type]["change_points"]
                change_points = [cp_group[f"seq_{i}"][:] for i in range(len(cp_group))]
            else:
                change_points = self.data[graph_type]["change_points"]

            num_changes = [len(cp) for cp in change_points]
            cp_locations = np.concatenate(change_points)

            cp_stats[graph_type] = {
                "avg_num_changes": np.mean(num_changes),
                "std_num_changes": np.std(num_changes),
                "avg_cp_locations": np.mean(cp_locations),
                "std_cp_locations": np.std(cp_locations),
                "min_gap": np.min(np.diff(np.sort(cp_locations))),
                "max_gap": np.max(np.diff(np.sort(cp_locations))),
            }

        return cp_stats

    def analyze_features(self, sample_size: int = 1000) -> Dict:
        """Analyze feature distributions based on dataset type"""
        feature_stats = {}

        for graph_type in ["BA", "ER", "NW"]:
            if self.format == "h5":
                features = [
                    self.data[graph_type][f"sequences/seq_{i}/features"][:]
                    for i in range(
                        min(sample_size, len(self.data[graph_type]["sequences"]))
                    )
                ]
            else:
                features = self.data[graph_type]["features"][:sample_size]

            # Process features based on dataset type
            if self.dataset_type in ["node_level", "combined"]:
                # For node-level features, compute stats across all nodes and timesteps
                features = np.concatenate(
                    [f.reshape(-1, f.shape[-1]) for f in features]
                )
                n_features = 6 if self.dataset_type == "node_level" else 36
                feature_names = self._get_feature_names()
            else:  # global
                features = np.concatenate(features)
                n_features = 6
                feature_names = self._get_feature_names(include_adjacency=True)

            feature_stats[graph_type] = {
                "mean": np.mean(features, axis=0),
                "std": np.std(features, axis=0),
                "min": np.min(features, axis=0),
                "max": np.max(features, axis=0),
                "names": feature_names,
            }

        return feature_stats

    def _get_feature_names(self, include_adjacency: bool = False) -> List[str]:
        """Get feature names based on dataset type"""
        base_features = [
            "Degree",
            "Betweenness",
            "Eigenvector",
            "Closeness",
            "SVD",
            "LSVD",
        ]

        if self.dataset_type == "combined":
            return [f"Adj_{i}" for i in range(30)] + base_features
        elif self.dataset_type == "global" and include_adjacency:
            return [f"Adj_{i}" for i in range(900)] + base_features
        else:
            return base_features

    def plot_feature_distributions(self, save_path: Optional[str] = None):
        """Plot feature value distributions with proper labeling"""
        feature_stats = self.analyze_features()

        fig, axes = plt.subplots(3, 1, figsize=(15, 20))

        for i, (graph_type, stats) in enumerate(feature_stats.items()):
            data = stats["mean"]

            if self.dataset_type == "global":
                # For global features, split adjacency and features
                adj_data = data[:-6]  # Adjacency matrix elements
                feat_data = data[-6:]  # Global features

                # Plot adjacency distribution
                ax_adj = axes[i].twinx()
                sns.kdeplot(
                    data=adj_data, ax=ax_adj, color="lightgray", label="Adjacency"
                )
                ax_adj.set_ylabel("Adjacency Density")

                # Plot feature boxplots
                sns.boxplot(data=feat_data, ax=axes[i])
                axes[i].set_xticklabels(stats["names"][-6:], rotation=45)
            else:
                # For node-level and combined features
                sns.boxplot(data=data, ax=axes[i])
                if len(stats["names"]) > 10:  # For combined features
                    axes[i].set_xticklabels(
                        [n if j % 5 == 0 else "" for j, n in enumerate(stats["names"])],
                        rotation=45,
                    )
                else:
                    axes[i].set_xticklabels(stats["names"], rotation=45)

            axes[i].set_title(f"{graph_type} Feature Distributions")
            axes[i].set_xlabel("Features")
            axes[i].set_ylabel("Feature Value")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def plot_change_point_distribution(self, save_path: Optional[str] = None):
        """Plot change point distributions"""
        cp_stats = self.analyze_change_points()

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

        # Plot average number of changes
        graph_types = list(cp_stats.keys())
        avg_changes = [stats["avg_num_changes"] for stats in cp_stats.values()]
        std_changes = [stats["std_num_changes"] for stats in cp_stats.values()]

        ax1.bar(graph_types, avg_changes, yerr=std_changes)
        ax1.set_title("Average Number of Change Points")
        ax1.set_ylabel("Number of Changes")

        # Plot change point locations with proper data handling
        for graph_type, stats in cp_stats.items():
            # Create proper distribution data
            loc_mean = stats["avg_cp_locations"]
            loc_std = stats["std_cp_locations"]
            if loc_std > 0:  # Only plot if there's variance
                # Generate points for a smoother distribution
                x = np.linspace(loc_mean - 3 * loc_std, loc_mean + 3 * loc_std, 100)
                y = np.exp(-0.5 * ((x - loc_mean) / loc_std) ** 2) / (
                    loc_std * np.sqrt(2 * np.pi)
                )
                ax2.plot(x, y, label=graph_type)
            else:
                # For zero variance, just plot a vertical line
                ax2.axvline(x=loc_mean, label=graph_type, linestyle="--")

        ax2.set_title("Change Point Location Distribution")
        ax2.set_xlabel("Sequence Position")
        ax2.set_ylabel("Density")
        ax2.legend()

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def visualize_graph_evolution(
        self,
        graph_type: str,
        seq_idx: int,
        timesteps: List[int],
        save_path: Optional[str] = None,
    ):
        """Visualize graph structure at specific timesteps"""
        if self.format == "h5":
            adjacency = self.data[graph_type][f"sequences/seq_{seq_idx}/adjacency"]
        else:
            adjacency = self.data[graph_type]["adjacency"][seq_idx]

        fig, axes = plt.subplots(1, len(timesteps), figsize=(5 * len(timesteps), 5))
        if len(timesteps) == 1:
            axes = [axes]

        for i, t in enumerate(timesteps):
            G = nx.from_numpy_array(adjacency[t])
            pos = nx.spring_layout(G, seed=42)
            nx.draw(
                G,
                pos,
                ax=axes[i],
                node_size=100,
                node_color="lightblue",
                with_labels=False,
            )
            axes[i].set_title(f"Time {t}")
            # Add axis labels and remove ticks for better visualization
            axes[i].set_xlabel("X")
            axes[i].set_ylabel("Y")
            axes[i].set_xticks([])
            axes[i].set_yticks([])

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def print_feature_summary(self):
        """Print detailed feature summary based on dataset type"""
        stats = self.get_basic_stats()
        feature_stats = self.analyze_features()

        print(f"\nDataset Type: {self.dataset_type}")
        print("-" * 50)

        for graph_type in stats.keys():
            print(f"\n{graph_type} Features:")
            print(f"  Shape: {stats[graph_type]['feature_shape']}")

            if self.dataset_type == "global":
                print("  Global Features:")
                for i, name in enumerate(feature_stats[graph_type]["names"][-6:]):
                    idx = -6 + i
                    print(f"    {name}:")
                    print(f"      Mean: {feature_stats[graph_type]['mean'][idx]:.4f}")
                    print(f"      Std:  {feature_stats[graph_type]['std'][idx]:.4f}")
                print("  Adjacency Statistics:")
                adj_mean = np.mean(feature_stats[graph_type]["mean"][:-6])
                adj_std = np.mean(feature_stats[graph_type]["std"][:-6])
                print(f"    Mean: {adj_mean:.4f}")
                print(f"    Std:  {adj_std:.4f}")
            else:
                for i, name in enumerate(feature_stats[graph_type]["names"]):
                    print(f"    {name}:")
                    print(f"      Mean: {feature_stats[graph_type]['mean'][i]:.4f}")
                    print(f"      Std:  {feature_stats[graph_type]['std'][i]:.4f}")


def main():
    """Example usage of dataset inspection"""
    import argparse

    parser = argparse.ArgumentParser(description="Inspect graph sequence dataset")
    parser.add_argument(
        "--dataset", type=str, help="Path to dataset file", required=True
    )
    parser.add_argument(
        "--output", type=str, default="figures", help="Output directory for plots"
    )
    args = parser.parse_args()

    try:
        # Create inspector
        inspector = DatasetInspector(args.dataset)

        # Print basic stats
        print("\nDataset Statistics:")
        print("-" * 50)
        stats = inspector.get_basic_stats()
        for graph_type, graph_stats in stats.items():
            print(f"\n{graph_type}:")
            for key, value in graph_stats.items():
                print(f"  {key}: {value}")

        # Print change point analysis
        print("\nChange Point Analysis:")
        print("-" * 50)
        cp_stats = inspector.analyze_change_points()
        for graph_type, cp_stat in cp_stats.items():
            print(f"\n{graph_type}:")
            for key, value in cp_stat.items():
                print(f"  {key}: {value:.2f}")

        # Create plots
        os.makedirs(args.output, exist_ok=True)
        inspector.plot_feature_distributions(
            save_path=os.path.join(args.output, "feature_distributions.png")
        )
        inspector.plot_change_point_distribution(
            save_path=os.path.join(args.output, "change_point_distribution.png")
        )

        # Visualize example graphs
        for graph_type in ["BA", "ER", "NW"]:
            inspector.visualize_graph_evolution(
                graph_type=graph_type,
                seq_idx=0,
                timesteps=[0, 50, 100, 150],
                save_path=os.path.join(args.output, f"{graph_type}_evolution.png"),
            )

    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
