"""
Dataset Inspector for Link Prediction Data

This script provides functionality to inspect and analyze the generated link prediction dataset:
- Visualize graph sequences and their features
- Analyze community structure and link prediction data
- Plot feature distributions and correlations
- Verify data splits and sampling strategies
"""

import os
import json
import logging
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
import h5py
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DatasetInspector:
    """Class for inspecting and analyzing the link prediction dataset"""

    def __init__(self, dataset_path: str):
        """Initialize with path to dataset file"""
        self.dataset_path = dataset_path
        self._load_dataset()

    def _load_dataset(self):
        """Load dataset from HDF5 file"""
        try:
            self.data = h5py.File(self.dataset_path, "r")
            logger.info(f"Successfully loaded dataset from {self.dataset_path}")

            # Load global metadata
            self.num_sequences = self.data["metadata"].attrs["num_sequences"]
            self.sequence_length = self.data["metadata"].attrs["sequence_length"]
            self.num_nodes = self.data["metadata"].attrs["num_nodes"]
            self.config = json.loads(self.data["metadata"].attrs["config"])

            logger.info(f"Dataset contains {self.num_sequences} sequences")
            logger.info(f"Each sequence has {self.sequence_length} timesteps")
            logger.info(f"Each graph has {self.num_nodes} nodes")

        except Exception as e:
            logger.error(f"Failed to load dataset: {str(e)}")
            raise

    def plot_graph_sequence(
        self,
        sequence_idx: int,
        timesteps: Optional[List[int]] = None,
        with_communities: bool = True,
        layout: str = "spring",
    ) -> None:
        """Plot graphs from a sequence at specified timesteps.

        Args:
            sequence_idx: Index of the sequence to plot
            timesteps: List of timesteps to plot. If None, plots evenly spaced timesteps
            with_communities: Whether to color nodes by community
            layout: Graph layout algorithm ('spring' or 'circular')
        """
        try:
            seq_name = f"sequence_{sequence_idx}"
            if timesteps is None:
                # Select 4 evenly spaced timesteps
                timesteps = np.linspace(0, self.sequence_length - 1, 4, dtype=int)

            fig, axes = plt.subplots(1, len(timesteps), figsize=(5 * len(timesteps), 5))
            if len(timesteps) == 1:
                axes = [axes]

            # Get community labels if available
            community_labels = None
            if (
                with_communities
                and "community_labels" in self.data[f"metadata/{seq_name}"]
            ):
                community_labels = self.data[f"metadata/{seq_name}/community_labels"][:]

            for ax, t in zip(axes, timesteps):
                # Load adjacency matrix
                adj_matrix = self.data[f"sequences/{seq_name}/graph_{t}"][:]
                G = nx.from_numpy_array(adj_matrix)

                # Set layout
                if layout == "spring":
                    pos = nx.spring_layout(G)
                else:
                    pos = nx.circular_layout(G)

                # Draw graph
                if community_labels is not None:
                    nx.draw_networkx_nodes(
                        G,
                        pos,
                        node_color=community_labels,
                        cmap=plt.cm.tab20,
                        node_size=100,
                        ax=ax,
                    )
                else:
                    nx.draw_networkx_nodes(
                        G, pos, node_color="lightblue", node_size=100, ax=ax
                    )
                nx.draw_networkx_edges(G, pos, alpha=0.2, ax=ax)
                ax.set_title(f"t = {t}")

            plt.tight_layout()
            plt.show()

        except Exception as e:
            logger.error(f"Failed to plot graph sequence: {str(e)}")
            raise

    def plot_feature_distributions(
        self, sequence_idx: int, features: Optional[List[str]] = None
    ) -> None:
        """Plot distributions of link prediction features.

        Args:
            sequence_idx: Index of the sequence to analyze
            features: List of feature names to plot. If None, plots all features
        """
        try:
            seq_name = f"sequence_{sequence_idx}"

            # Get available features
            if features is None:
                features = list(self.data[f"features/{seq_name}"].keys())

            # Calculate grid dimensions
            n_features = len(features)
            n_cols = min(3, n_features)
            n_rows = (n_features + n_cols - 1) // n_cols

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
            if n_rows == 1 and n_cols == 1:
                axes = np.array([[axes]])
            elif n_rows == 1 or n_cols == 1:
                axes = axes.reshape(-1, 1) if n_cols == 1 else axes.reshape(1, -1)

            for idx, feature in enumerate(features):
                row, col = idx // n_cols, idx % n_cols
                ax = axes[row, col]

                # Load feature data
                feature_data = self.data[f"features/{seq_name}/{feature}"][:]

                # Plot distribution
                if feature_data.ndim > 1 and feature_data.shape[1] > 1:
                    # For multi-dimensional features, plot mean across dimensions
                    feature_data = np.mean(feature_data, axis=1)

                sns.histplot(feature_data, ax=ax)
                ax.set_title(feature)
                ax.set_xlabel("Value")
                ax.set_ylabel("Count")

            # Remove empty subplots
            for idx in range(n_features, n_rows * n_cols):
                row, col = idx // n_cols, idx % n_cols
                fig.delaxes(axes[row, col])

            plt.tight_layout()
            plt.show()

        except Exception as e:
            logger.error(f"Failed to plot feature distributions: {str(e)}")
            raise

    def analyze_link_prediction_data(self, sequence_idx: int) -> None:
        """Analyze link prediction data for a sequence.

        Args:
            sequence_idx: Index of the sequence to analyze
        """
        try:
            seq_name = f"sequence_{sequence_idx}"

            # Get link prediction data
            lp_data = self.data[f"link_prediction/{seq_name}"]

            # Print statistics for each split
            for split in ["train", "val", "test"]:
                split_data = lp_data[split]
                n_pos = len(split_data["positive"])
                n_neg = len(split_data["negative"])

                print(f"\n{split.upper()} Split:")
                print(f"Positive edges: {n_pos}")
                print(f"Negative edges: {n_neg}")
                print(f"Positive/Negative ratio: {n_pos/n_neg:.2f}")

            # Analyze community-based edge distribution if available
            if "community_labels" in self.data[f"metadata/{seq_name}"]:
                community_labels = self.data[f"metadata/{seq_name}/community_labels"][:]
                n_communities = len(np.unique(community_labels))

                print("\nCommunity-based Edge Distribution:")
                for split in ["train", "val", "test"]:
                    split_data = lp_data[split]
                    pos_edges = split_data["positive"][:]

                    # Count intra-community and inter-community edges
                    intra_comm = sum(
                        community_labels[u] == community_labels[v] for u, v in pos_edges
                    )
                    inter_comm = len(pos_edges) - intra_comm

                    print(f"\n{split.upper()} Split:")
                    print(
                        f"Intra-community edges: {intra_comm} ({intra_comm/len(pos_edges):.2%})"
                    )
                    print(
                        f"Inter-community edges: {inter_comm} ({inter_comm/len(pos_edges):.2%})"
                    )

        except Exception as e:
            logger.error(f"Failed to analyze link prediction data: {str(e)}")
            raise

    def plot_community_structure(self, sequence_idx: int, timestep: int = 0) -> None:
        """Visualize community structure of a graph.

        Args:
            sequence_idx: Index of the sequence to analyze
            timestep: Timestep to visualize
        """
        try:
            seq_name = f"sequence_{sequence_idx}"

            # Load graph and community labels
            adj_matrix = self.data[f"sequences/{seq_name}/graph_{timestep}"][:]
            community_labels = self.data[f"metadata/{seq_name}/community_labels"][:]

            G = nx.from_numpy_array(adj_matrix)
            pos = nx.spring_layout(G)

            plt.figure(figsize=(12, 8))

            # Draw nodes colored by community
            nx.draw_networkx_nodes(
                G, pos, node_color=community_labels, cmap=plt.cm.tab20, node_size=100
            )
            nx.draw_networkx_edges(G, pos, alpha=0.2)

            # Add title with community statistics
            n_communities = len(np.unique(community_labels))
            plt.title(
                f"Community Structure (t={timestep})\n"
                f"Number of communities: {n_communities}"
            )

            plt.show()

            # Print community statistics
            print("\nCommunity Statistics:")
            for comm in range(n_communities):
                size = np.sum(community_labels == comm)
                print(
                    f"Community {comm}: {size} nodes ({size/len(community_labels):.2%})"
                )

        except Exception as e:
            logger.error(f"Failed to plot community structure: {str(e)}")
            raise

    def plot_temporal_features(
        self, sequence_idx: int, features: Optional[List[str]] = None
    ) -> None:
        """Plot temporal features over time.

        Args:
            sequence_idx: Index of the sequence to analyze
            features: List of temporal features to plot. If None, plots all temporal features
        """
        try:
            seq_name = f"sequence_{sequence_idx}"

            # Get temporal features
            if features is None:
                features = [
                    f
                    for f in self.data[f"features/{seq_name}"].keys()
                    if any(f.startswith(p) for p in ["temporal_", "link_", "cn_"])
                ]

            if not features:
                logger.warning("No temporal features found to plot")
                return

            # Plot features in batches to avoid memory issues
            batch_size = 3  # Reduced batch size
            for i in range(0, len(features), batch_size):
                batch_features = features[i : i + batch_size]
                n_features = len(batch_features)

                fig, axes = plt.subplots(n_features, 1, figsize=(10, 3 * n_features))
                if n_features == 1:
                    axes = [axes]

                for ax, feature in zip(axes, batch_features):
                    try:
                        feature_data = self.data[f"features/{seq_name}/{feature}"][:]

                        if feature_data.ndim > 1:
                            # For high-dimensional data, downsample more aggressively
                            if feature_data.shape[1] > 100:
                                sample_size = min(100, feature_data.shape[1])
                                sample_indices = np.linspace(
                                    0, feature_data.shape[1] - 1, sample_size, dtype=int
                                )
                                feature_data = feature_data[:, sample_indices]

                            # Compute statistics with reduced precision
                            mean = np.mean(feature_data, axis=1)
                            std = np.std(feature_data, axis=1)
                            timesteps = np.arange(len(mean))

                            # Plot with reduced number of points if sequence is long
                            if len(timesteps) > 100:
                                stride = len(timesteps) // 100
                                timesteps = timesteps[::stride]
                                mean = mean[::stride]
                                std = std[::stride]

                            # Plot mean line only first
                            line = ax.plot(timesteps, mean, label="Mean", linewidth=1)[
                                0
                            ]
                            color = line.get_color()

                            # Then add fill_between with same color
                            ax.fill_between(
                                timesteps,
                                mean - std,
                                mean + std,
                                color=color,
                                alpha=0.1,
                                label="±1 std",
                            )
                        else:
                            # For 1D data, plot with reduced points if necessary
                            if len(feature_data) > 100:
                                stride = len(feature_data) // 100
                                feature_data = feature_data[::stride]
                                timesteps = np.arange(0, len(feature_data))
                            else:
                                timesteps = np.arange(len(feature_data))
                            ax.plot(timesteps, feature_data, linewidth=1)

                        ax.set_title(feature, fontsize=10)
                        ax.set_xlabel("Timestep", fontsize=8)
                        ax.set_ylabel("Value", fontsize=8)
                        if feature_data.ndim > 1:
                            ax.legend(loc="upper right", fontsize=8)

                        # Set reasonable y-axis limits
                        if feature_data.size > 0:
                            percentiles = np.percentile(feature_data, [5, 95])
                            margin = (percentiles[1] - percentiles[0]) * 0.1
                            ax.set_ylim(
                                percentiles[0] - margin, percentiles[1] + margin
                            )

                        # Reduce number of ticks
                        ax.xaxis.set_major_locator(plt.MaxNLocator(6))
                        ax.yaxis.set_major_locator(plt.MaxNLocator(6))

                    except Exception as e:
                        logger.error(f"Failed to plot feature {feature}: {str(e)}")
                        ax.text(
                            0.5,
                            0.5,
                            f"Failed to plot {feature}",
                            ha="center",
                            va="center",
                            transform=ax.transAxes,
                        )

                plt.tight_layout()
                plt.show()
                plt.close("all")  # Close all figures to free memory

        except Exception as e:
            logger.error(f"Failed to plot temporal features: {str(e)}")
            plt.close("all")  # Ensure all figures are closed in case of error
            raise

    def plot_feature_correlations(
        self, sequence_idx: int, features: Optional[List[str]] = None
    ) -> None:
        """Plot correlation matrix of features.

        Args:
            sequence_idx: Index of the sequence to analyze
            features: List of features to include. If None, uses all features
        """
        try:
            seq_name = f"sequence_{sequence_idx}"

            # Get features
            if features is None:
                features = list(self.data[f"features/{seq_name}"].keys())

            # Collect and preprocess feature data
            feature_data = []
            valid_features = []
            base_length = None

            for feature in features:
                try:
                    data = self.data[f"features/{seq_name}/{feature}"][:]

                    # Handle multi-dimensional features
                    if data.ndim > 1:
                        # For 2D data, take mean across second dimension
                        data = np.mean(data, axis=1)

                    # Ensure data is 1D
                    data = np.ravel(data)

                    # Set base length if not set
                    if base_length is None:
                        base_length = len(data)

                    # Only include features with matching lengths
                    if len(data) == base_length:
                        feature_data.append(data)
                        valid_features.append(feature)
                    else:
                        logger.warning(
                            f"Skipping feature '{feature}' due to shape mismatch. "
                            f"Expected length {base_length}, got {len(data)}"
                        )

                except Exception as e:
                    logger.warning(f"Failed to process feature '{feature}': {str(e)}")

            if not valid_features:
                logger.warning("No valid features found for correlation analysis")
                return

            # Convert to numpy array and compute correlation matrix
            feature_data = np.array(feature_data)
            corr_matrix = np.corrcoef(feature_data)

            # Plot correlation matrix
            plt.figure(figsize=(12, 10))
            mask = np.triu(np.ones_like(corr_matrix), k=1)  # Mask upper triangle
            sns.heatmap(
                corr_matrix,
                xticklabels=valid_features,
                yticklabels=valid_features,
                mask=mask,
                annot=True,
                fmt=".2f",
                cmap="coolwarm",
                center=0,
                vmin=-1,
                vmax=1,
                square=True,
            )

            plt.title("Feature Correlations")
            plt.xticks(rotation=45, ha="right")
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.show()
            plt.close("all")  # Close figure to free memory

        except Exception as e:
            logger.error(f"Failed to plot feature correlations: {str(e)}")
            plt.close("all")  # Ensure figure is closed in case of error
            raise


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Inspect link prediction dataset")
    parser.add_argument(
        "--dataset", type=str, required=True, help="Path to dataset file"
    )
    parser.add_argument(
        "--sequence", type=int, default=0, help="Sequence index to analyze"
    )
    args = parser.parse_args()

    try:
        inspector = DatasetInspector(args.dataset)

        # Basic dataset info
        print("\n=== Dataset Information ===")
        print(f"Number of sequences: {inspector.num_sequences}")
        print(f"Sequence length: {inspector.sequence_length}")
        print(f"Number of nodes: {inspector.num_nodes}")

        # Analyze sequence
        print(f"\n=== Analyzing Sequence {args.sequence} ===")

        # Plot graph sequence
        print("\nPlotting graph sequence...")
        inspector.plot_graph_sequence(args.sequence)

        # Plot community structure
        print("\nPlotting community structure...")
        inspector.plot_community_structure(args.sequence)

        # Analyze link prediction data
        print("\nAnalyzing link prediction data...")
        inspector.analyze_link_prediction_data(args.sequence)

        # Plot feature distributions
        print("\nPlotting feature distributions...")
        inspector.plot_feature_distributions(args.sequence)

        # Plot temporal features
        print("\nPlotting temporal features...")
        inspector.plot_temporal_features(args.sequence)

        # Plot feature correlations
        print("\nPlotting feature correlations...")
        inspector.plot_feature_correlations(args.sequence)

    except Exception as e:
        logger.error(f"Dataset inspection failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
