# src/graph/visualizer.py

from typing import Dict, List, Optional, Tuple, Union

import logging
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .utils import graph_to_adjacency

logger = logging.getLogger(__name__)


class NetworkVisualizer:
    """Visualizer for network graphs, adjacency matrices, and features."""

    DEFAULT_STYLE = {
        "node_size": 300,
        "node_color": "#1f77b4",
        "edge_color": "#7f7f7f",
        "font_size": 10,
        "width": 1,
        "alpha": 0.7,
        "cmap": "viridis",
        "with_labels": True,
        "arrows": False,
    }

    LAYOUTS = {
        "spring": nx.spring_layout,
        "circular": nx.circular_layout,
        "random": nx.random_layout,
        "shell": nx.shell_layout,
        "spectral": nx.spectral_layout,
        "kamada_kawai": nx.kamada_kawai_layout,
    }

    def __init__(self, style: Dict = None):
        """Initialize visualizer with custom style."""
        self.style = self.DEFAULT_STYLE.copy()
        if style:
            self.style.update(style)

    def plot_adjacency(
        self,
        adj_matrix: np.ndarray,
        ax: Optional[Axes] = None,
        title: str = "Adjacency Matrix",
        show_values: bool = False,
    ) -> Tuple[Figure, Axes]:
        """Plot adjacency matrix as heatmap.

        Args:
            adj_matrix: Adjacency matrix to plot
            ax: Matplotlib axes to plot on (default: None)
            title: Plot title (default: "Adjacency Matrix")
            show_values: Whether to show matrix values (default: False)
        Returns:
            (figure, axes) tuple
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        else:
            fig = ax.figure

        # Handle empty matrix
        if adj_matrix.size == 0:
            ax.text(0.5, 0.5, "Empty Graph", ha="center", va="center")
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
            return fig, ax

        # Plot heatmap
        sns.heatmap(
            adj_matrix,
            ax=ax,
            cmap=self.style["cmap"],
            square=True,
            annot=show_values,
            fmt=".2f" if show_values else "",
            cbar_kws={"shrink": 0.8},
        )

        ax.set_title(title)
        ax.set_xlabel("Node")
        ax.set_ylabel("Node")

        return fig, ax

    def plot_network(
        self,
        graph: Union[nx.Graph, np.ndarray],
        ax: Optional[Axes] = None,
        title: str = "Network Graph",
        layout: str = "spring",
        node_color: Optional[List] = None,
        edge_color: Optional[List] = None,
        node_size: Optional[List] = None,
        edge_width: Optional[List] = None,
    ) -> Tuple[Figure, Axes]:
        """Plot network graph with nodes and edges.

        Args:
            graph: NetworkX graph or adjacency matrix
            ax: Matplotlib axes to plot on (default: None)
            title: Plot title (default: "Network Graph")
            layout: Layout algorithm to use (default: "spring")
            node_color: Node colors (default: None)
            edge_color: Edge colors (default: None)
            node_size: Node sizes (default: None)
            edge_width: Edge widths (default: None)
        Returns:
            (figure, axes) tuple
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        else:
            fig = ax.figure

        # Convert adjacency matrix to graph if needed
        if isinstance(graph, np.ndarray):
            graph = nx.from_numpy_array(graph)
        else:
            # Convert node labels to integers for visualization
            graph = nx.convert_node_labels_to_integers(graph)

        # Handle empty graph
        if graph.number_of_nodes() == 0:
            ax.text(0.5, 0.5, "Empty Graph", ha="center", va="center")
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
            return fig, ax

        # Get layout
        if layout not in self.LAYOUTS:
            logger.warning(f"Unknown layout: {layout}, falling back to spring layout")
            layout = "spring"
        pos = self.LAYOUTS[layout](graph)

        # Draw network
        nx.draw_networkx(
            graph,
            pos=pos,
            ax=ax,
            node_color=node_color or self.style["node_color"],
            edge_color=edge_color or self.style["edge_color"],
            node_size=node_size or self.style["node_size"],
            width=edge_width or self.style["width"],
            alpha=self.style["alpha"],
            with_labels=self.style["with_labels"],
            arrows=self.style["arrows"],
            font_size=self.style["font_size"],
        )

        ax.set_title(title)
        ax.axis("off")

        return fig, ax

    def plot_graph_sequence(
        self,
        graphs: List[Union[nx.Graph, np.ndarray]],
        n_cols: int = 4,
        plot_type: str = "network",
        layout: str = "spring",
        titles: Optional[List[str]] = None,
    ) -> Tuple[Figure, List[Axes]]:
        """Plot sequence of graphs in a grid.

        Args:
            graphs: List of graphs to plot
            n_cols: Number of columns in grid (default: 4)
            plot_type: Type of plot ("network" or "adjacency")
            layout: Layout for network plots (default: "spring")
            titles: List of subplot titles (default: None)
        Returns:
            (figure, list of axes) tuple
        """
        n_graphs = len(graphs)
        n_rows = (n_graphs + n_cols - 1) // n_cols

        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), squeeze=False
        )
        axes = axes.flatten()

        for i, graph in enumerate(graphs):
            title = titles[i] if titles and i < len(titles) else f"Graph {i+1}"

            if plot_type == "network":
                self.plot_network(graph, ax=axes[i], title=title, layout=layout)
            else:
                if isinstance(graph, nx.Graph):
                    graph = graph_to_adjacency(graph)
                self.plot_adjacency(graph, ax=axes[i], title=title)

        # Remove empty subplots
        for i in range(n_graphs, len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout()
        return fig, axes.tolist()

    def plot_graph_evolution(
        self,
        graphs: List[Union[nx.Graph, np.ndarray]],
        change_points: Optional[List[int]] = None,
        metrics: Optional[Dict[str, List[float]]] = None,
        figsize: Tuple[int, int] = (12, 8),
    ) -> Tuple[Figure, List[Axes]]:
        """Plot graph evolution with optional change points and metrics.

        Args:
            graphs: List of graphs in sequence
            change_points: List of change point indices (default: None)
            metrics: Dict of metric names to values (default: None)
            figsize: Figure size (width, height) (default: (12, 8))
        Returns:
            (figure, list of axes) tuple
        """
        n_metrics = len(metrics) if metrics else 0
        n_rows = 1 + (n_metrics > 0)

        fig = plt.figure(figsize=figsize)
        gs = plt.GridSpec(n_rows, 1, height_ratios=[2] + [1] * (n_rows - 1))
        axes = []

        # Plot graph metrics
        if metrics:
            ax = fig.add_subplot(gs[0])
            for name, values in metrics.items():
                ax.plot(values, label=name, alpha=0.7)

            if change_points:
                for cp in change_points:
                    ax.axvline(cp, color="r", linestyle="--", alpha=0.5)

            ax.set_xlabel("Time")
            ax.set_ylabel("Metric Value")
            ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
            ax.grid(True, alpha=0.3)
            axes.append(ax)

        # Plot graph density evolution
        ax = fig.add_subplot(gs[-1])
        densities = [
            nx.density(g) if isinstance(g, nx.Graph) else g.sum() / (g.shape[0] ** 2)
            for g in graphs
        ]
        ax.plot(densities, label="Density", color="blue", alpha=0.7)

        if change_points:
            for cp in change_points:
                ax.axvline(cp, color="r", linestyle="--", alpha=0.5)

        ax.set_xlabel("Time")
        ax.set_ylabel("Graph Density")
        ax.grid(True, alpha=0.3)
        axes.append(ax)

        plt.tight_layout()
        return fig, axes

    def plot_feature_evolution(
        self,
        features: List[Dict],
        change_points: Optional[List[int]] = None,
        ax: Optional[Axes] = None,
        title: Optional[str] = None,
        feature_name: Optional[str] = None,
    ) -> Tuple[Figure, Axes]:
        """Plot evolution of a network feature with standard deviation bands.

        Args:
            features: List of feature dictionaries for each time step
            change_points: List of change point indices (default: None)
            ax: Matplotlib axes to plot on (default: None)
            title: Plot title (default: None)
            feature_name: Name of feature to plot (default: None)
        Returns:
            (figure, axes) tuple
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
        else:
            fig = ax.figure

        if not features or not feature_name:
            ax.text(0.5, 0.5, "No Features", ha="center", va="center")
            ax.set_title(title or "Feature Evolution")
            return fig, ax

        # Extract feature values
        time = np.arange(len(features))
        if isinstance(features[0][feature_name], list):
            # For list features (like degrees), compute mean and std
            mean_values = [
                np.mean(f[feature_name]) if len(f[feature_name]) > 0 else 0
                for f in features
            ]
            std_values = [
                np.std(f[feature_name]) if len(f[feature_name]) > 0 else 0
                for f in features
            ]

            # Plot mean line
            ax.plot(time, mean_values, label=f"{feature_name} (mean)", alpha=0.7)

            # Add std bands
            ax.fill_between(
                time,
                np.array(mean_values) - np.array(std_values),
                np.array(mean_values) + np.array(std_values),
                alpha=0.2,
                label=f"±1 std",
            )
        else:
            # For scalar features (like density)
            values = [f[feature_name] for f in features]
            ax.plot(values, label=feature_name, alpha=0.7)

        # Mark change points
        if change_points:
            for cp in change_points:
                ax.axvline(cp, color="r", linestyle="--", alpha=0.5)

        ax.set_title(title or f"{feature_name.replace('_', ' ').title()} Evolution")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)
        ax.legend()

        return fig, ax

    def plot_all_features(
        self,
        features: List[Dict],
        change_points: Optional[List[int]] = None,
        n_cols: int = 2,
        figsize: Optional[Tuple[int, int]] = None,
    ) -> Tuple[Figure, List[Axes]]:
        """Plot evolution of all network features.

        Args:
            features: List of feature dictionaries for each time step
            change_points: List of change point indices (default: None)
            n_cols: Number of columns in the grid (default: 2)
            figsize: Figure size (width, height) (default: None)
        Returns:
            (figure, list of axes) tuple
        """
        if not features:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No Features", ha="center", va="center")
            ax.set_xticks([])
            ax.set_yticks([])
            return fig, [ax]

        # Get list of features
        feature_names = list(features[0].keys())
        n_features = len(feature_names)
        n_rows = (n_features + n_cols - 1) // n_cols

        if figsize is None:
            figsize = (10 * n_cols, 5 * n_rows)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_features == 1:
            axes = [axes]
        axes = np.array(axes).flatten()

        # Plot each feature
        for i, feature_name in enumerate(feature_names):
            self.plot_feature_evolution(
                features,
                change_points=change_points,
                ax=axes[i],
                feature_name=feature_name,
            )

        # Remove empty subplots
        for i in range(n_features, len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout()
        return fig, axes.tolist()
