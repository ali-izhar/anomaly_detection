# src/graph/visualizer.py

"""Visualizes network graphs and features."""

from typing import Dict, List, Optional, Tuple, Union

import logging
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .plotting_config import (
    FIGURE_DIMENSIONS as FD,
    TYPOGRAPHY as TYPO,
    LINE_STYLE as LS,
    COLORS,
    get_network_style,
)
from ..graph.utils import graph_to_adjacency, adjacency_to_graph

logger = logging.getLogger(__name__)

# Constants for NetworkVisualizer
GRAPH_CONSTANTS = {
    # Layout types
    "LAYOUTS": {
        "SPRING": "spring",
        "CIRCULAR": "circular",
        "RANDOM": "random",
        "SHELL": "shell",
        "SPECTRAL": "spectral",
        "KAMADA_KAWAI": "kamada_kawai",
    },
    # Plot types
    "PLOT_TYPES": {
        "NETWORK": "network",
        "ADJACENCY": "adjacency",
    },
    # Default style values
    "DEFAULT_STYLES": {
        "NODE_COLOR": "lightblue",
        "EDGE_COLOR": "gray",
        "NODE_SIZE": 300,
        "EDGE_WIDTH": 1,
        "ALPHA": 0.7,
        "FONT_SIZE": 8,
    },
    # Text constants
    "TEXT": {
        "EMPTY_GRAPH": "Empty Graph",
        "GRAPH_PREFIX": "Graph ",
    },
    # Default titles
    "TITLES": {
        "ADJACENCY": "Adjacency Matrix",
        "NETWORK": "Network Graph",
        "FEATURE_EVOLUTION": "Feature Evolution",
    },
    # Plot labels
    "LABELS": {
        "NODE": "Node",
        "TIME": "Time",
        "VALUE": "Value",
        "METRIC_VALUE": "Metric Value",
        "GRAPH_DENSITY": "Graph Density",
        "DENSITY": "Density",
    },
    # Style keys
    "STYLE_KEYS": {
        "CMAP": "cmap",
        "NODE_COLOR": "node_color",
        "EDGE_COLOR": "edge_color",
        "NODE_SIZE": "node_size",
        "WIDTH": "width",
        "ALPHA": "alpha",
        "FONT_SIZE": "font_size",
    },
    # Layout parameter keys
    "LAYOUT_PARAMS": {
        "POS": "pos",
    },
}


class NetworkVisualizer:
    """Visualizer for network graphs, adjacency matrices, and features."""

    LAYOUTS = {
        GRAPH_CONSTANTS["LAYOUTS"]["SPRING"]: nx.spring_layout,
        GRAPH_CONSTANTS["LAYOUTS"]["CIRCULAR"]: nx.circular_layout,
        GRAPH_CONSTANTS["LAYOUTS"]["RANDOM"]: nx.random_layout,
        GRAPH_CONSTANTS["LAYOUTS"]["SHELL"]: nx.shell_layout,
        GRAPH_CONSTANTS["LAYOUTS"]["SPECTRAL"]: nx.spectral_layout,
        GRAPH_CONSTANTS["LAYOUTS"]["KAMADA_KAWAI"]: nx.kamada_kawai_layout,
    }

    def __init__(self, style: Dict = None):
        """Initialize visualizer with custom style."""
        plt.style.use("seaborn-v0_8-paper")  # Use paper style as base
        self.style = get_network_style(style)

    def plot_adjacency(
        self,
        adj_matrix: np.ndarray,
        ax: Optional[Axes] = None,
        title: str = GRAPH_CONSTANTS["TITLES"]["ADJACENCY"],
        show_values: bool = False,
    ) -> Tuple[Figure, Axes]:
        """Plot adjacency matrix as heatmap."""
        if ax is None:
            fig, ax = plt.subplots(
                figsize=(FD["SINGLE_COLUMN_WIDTH"] / 2, FD["STANDARD_HEIGHT"] / 2)
            )
        else:
            fig = ax.figure

        # Handle empty matrix
        if adj_matrix.size == 0:
            ax.text(
                0.5,
                0.5,
                GRAPH_CONSTANTS["TEXT"]["EMPTY_GRAPH"],
                ha="center",
                va="center",
                fontsize=TYPO["TICK_SIZE"],
            )
            ax.set_title(title, fontsize=TYPO["TITLE_SIZE"], pad=4)
            ax.set_xticks([])
            ax.set_yticks([])
            return fig, ax

        # Plot heatmap without colorbar
        sns.heatmap(
            adj_matrix,
            ax=ax,
            cmap=self.style[GRAPH_CONSTANTS["STYLE_KEYS"]["CMAP"]],
            square=True,
            annot=show_values,
            fmt=".0f" if show_values else "",
            cbar=False,  # Remove colorbar
            annot_kws={"size": TYPO["ANNOTATION_SIZE"]} if show_values else {},
        )

        ax.set_title(title, fontsize=TYPO["TITLE_SIZE"], pad=4)
        ax.set_xlabel(
            GRAPH_CONSTANTS["LABELS"]["NODE"], fontsize=TYPO["LABEL_SIZE"], labelpad=2
        )
        ax.set_ylabel(
            GRAPH_CONSTANTS["LABELS"]["NODE"], fontsize=TYPO["LABEL_SIZE"], labelpad=2
        )
        ax.tick_params(labelsize=TYPO["TICK_SIZE"], pad=1)

        return fig, ax

    def plot_network(
        self,
        graph: Union[nx.Graph, np.ndarray],
        ax: Optional[Axes] = None,
        title: str = GRAPH_CONSTANTS["TITLES"]["NETWORK"],
        layout: str = GRAPH_CONSTANTS["LAYOUTS"]["SPRING"],
        layout_params: Optional[Dict] = None,
        node_color: Optional[List] = None,
        edge_color: Optional[List] = None,
        node_size: Optional[List] = None,
        edge_width: Optional[List] = None,
    ) -> Tuple[Figure, Axes]:
        """Plot network graph with nodes and edges.

        Args:
            graph: NetworkX graph or adjacency matrix
            ax: Matplotlib axes to plot on
            title: Plot title
            layout: Graph layout algorithm ('spring', 'circular', 'spectral', 'random', 'shell')
            layout_params: Parameters specific to the chosen layout algorithm, or pre-computed positions
            node_color: Colors for nodes
            edge_color: Colors for edges
            node_size: Sizes for nodes
            edge_width: Widths for edges
        Returns:
            (figure, axes) tuple
        """
        if ax is None:
            fig, ax = plt.subplots(
                figsize=(FD["SINGLE_COLUMN_WIDTH"] / 2, FD["STANDARD_HEIGHT"] / 2)
            )
        else:
            fig = ax.figure

        # Convert adjacency matrix to graph if needed
        if isinstance(graph, np.ndarray):
            graph = adjacency_to_graph(graph)
        else:
            graph = nx.convert_node_labels_to_integers(graph)

        # Handle empty graph
        if graph.number_of_nodes() == 0:
            ax.text(
                0.5,
                0.5,
                GRAPH_CONSTANTS["TEXT"]["EMPTY_GRAPH"],
                ha="center",
                va="center",
                fontsize=TYPO["TICK_SIZE"],
            )
            ax.set_title(title, fontsize=TYPO["TITLE_SIZE"], pad=4)
            ax.set_xticks([])
            ax.set_yticks([])
            return fig, ax

        # Get layout with appropriate parameters
        layout_params = layout_params or {}
        if GRAPH_CONSTANTS["LAYOUT_PARAMS"]["POS"] in layout_params:
            # Use pre-computed positions
            pos = layout_params[GRAPH_CONSTANTS["LAYOUT_PARAMS"]["POS"]]
        else:
            # Compute new layout
            if layout not in self.LAYOUTS:
                logger.warning(
                    f"Unknown layout: {layout}, falling back to {GRAPH_CONSTANTS['LAYOUTS']['SPRING']} layout"
                )
                layout = GRAPH_CONSTANTS["LAYOUTS"]["SPRING"]
            pos = self.LAYOUTS[layout](graph, **layout_params)

        # Draw the network
        nx.draw_networkx(
            graph,
            pos=pos,
            ax=ax,
            node_color=node_color
            or self.style.get(
                GRAPH_CONSTANTS["STYLE_KEYS"]["NODE_COLOR"],
                GRAPH_CONSTANTS["DEFAULT_STYLES"]["NODE_COLOR"],
            ),
            edge_color=edge_color
            or self.style.get(
                GRAPH_CONSTANTS["STYLE_KEYS"]["EDGE_COLOR"],
                GRAPH_CONSTANTS["DEFAULT_STYLES"]["EDGE_COLOR"],
            ),
            node_size=node_size
            or self.style.get(
                GRAPH_CONSTANTS["STYLE_KEYS"]["NODE_SIZE"],
                GRAPH_CONSTANTS["DEFAULT_STYLES"]["NODE_SIZE"],
            ),
            width=edge_width
            or self.style.get(
                GRAPH_CONSTANTS["STYLE_KEYS"]["WIDTH"],
                GRAPH_CONSTANTS["DEFAULT_STYLES"]["EDGE_WIDTH"],
            ),
            alpha=self.style.get(
                GRAPH_CONSTANTS["STYLE_KEYS"]["ALPHA"],
                GRAPH_CONSTANTS["DEFAULT_STYLES"]["ALPHA"],
            ),
            with_labels=True,
            font_size=self.style.get(
                GRAPH_CONSTANTS["STYLE_KEYS"]["FONT_SIZE"],
                GRAPH_CONSTANTS["DEFAULT_STYLES"]["FONT_SIZE"],
            ),
        )

        ax.set_title(title, fontsize=TYPO["TITLE_SIZE"], pad=4)
        ax.set_xticks([])
        ax.set_yticks([])
        return fig, ax

    def plot_graph_sequence(
        self,
        graphs: List[Union[nx.Graph, np.ndarray]],
        n_cols: int = 4,
        plot_type: str = GRAPH_CONSTANTS["PLOT_TYPES"]["NETWORK"],
        layout: str = GRAPH_CONSTANTS["LAYOUTS"]["SPRING"],
        titles: Optional[List[str]] = None,
    ) -> Tuple[Figure, List[Axes]]:
        """Plot sequence of graphs in a grid."""
        n_graphs = len(graphs)
        n_rows = (n_graphs + n_cols - 1) // n_cols

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(FD["SINGLE_COLUMN_WIDTH"], FD["STANDARD_HEIGHT"] * n_rows / 2),
            squeeze=False,
        )
        axes = axes.flatten()

        for i, graph in enumerate(graphs):
            title = (
                titles[i]
                if titles and i < len(titles)
                else f"{GRAPH_CONSTANTS['TEXT']['GRAPH_PREFIX']}{i+1}"
            )

            if plot_type == GRAPH_CONSTANTS["PLOT_TYPES"]["NETWORK"]:
                self.plot_network(graph, ax=axes[i], title=title, layout=layout)
            else:
                if isinstance(graph, nx.Graph):
                    graph = graph_to_adjacency(graph)
                self.plot_adjacency(graph, ax=axes[i], title=title)

        # Remove empty subplots
        for i in range(n_graphs, len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout(pad=0.5, h_pad=0.5, w_pad=0.5)
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

            ax.set_xlabel(GRAPH_CONSTANTS["LABELS"]["TIME"])
            ax.set_ylabel(GRAPH_CONSTANTS["LABELS"]["METRIC_VALUE"])
            ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
            ax.grid(True, alpha=0.3)
            axes.append(ax)

        # Plot graph density evolution
        ax = fig.add_subplot(gs[-1])
        densities = [
            nx.density(g) if isinstance(g, nx.Graph) else g.sum() / (g.shape[0] ** 2)
            for g in graphs
        ]
        ax.plot(
            densities,
            label=GRAPH_CONSTANTS["LABELS"]["DENSITY"],
            color="blue",
            alpha=0.7,
        )

        if change_points:
            for cp in change_points:
                ax.axvline(cp, color="r", linestyle="--", alpha=0.5)

        ax.set_xlabel(GRAPH_CONSTANTS["LABELS"]["TIME"])
        ax.set_ylabel(GRAPH_CONSTANTS["LABELS"]["GRAPH_DENSITY"])
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
        """Plot evolution of a network feature."""
        if ax is None:
            fig, ax = plt.subplots(
                figsize=(FD["SINGLE_COLUMN_WIDTH"] / 2, FD["STANDARD_HEIGHT"] / 2)
            )
        else:
            fig = ax.figure

        if not features or not feature_name:
            ax.text(
                0.5,
                0.5,
                GRAPH_CONSTANTS["TEXT"]["EMPTY_GRAPH"],
                ha="center",
                va="center",
                fontsize=TYPO["TICK_SIZE"],
            )
            ax.set_title(
                title or GRAPH_CONSTANTS["TITLES"]["FEATURE_EVOLUTION"],
                fontsize=TYPO["TITLE_SIZE"],
                pad=4,
            )
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
            ax.plot(
                time,
                mean_values,
                label=feature_name,  # Only show feature name in legend
                color=COLORS["actual"],
                alpha=LS["LINE_ALPHA"],
                linewidth=LS["LINE_WIDTH"],
            )

            # Add std bands without legend entry
            ax.fill_between(
                time,
                np.array(mean_values) - np.array(std_values),
                np.array(mean_values) + np.array(std_values),
                color=COLORS["actual"],
                alpha=0.1,
            )
        else:
            # For scalar features (like density)
            values = [f[feature_name] for f in features]
            ax.plot(
                values,
                label=feature_name,
                color=COLORS["actual"],
                alpha=LS["LINE_ALPHA"],
                linewidth=LS["LINE_WIDTH"],
            )

        # Mark change points
        if change_points:
            for cp in change_points:
                ax.axvline(
                    cp,
                    color=COLORS["change_point"],
                    linestyle="--",
                    alpha=0.5,
                    linewidth=LS["LINE_WIDTH"] * 0.8,
                )

        ax.set_title(
            title or f"{feature_name.replace('_', ' ').title()}",
            fontsize=TYPO["TITLE_SIZE"],
            pad=4,
        )
        ax.set_xlabel(
            GRAPH_CONSTANTS["LABELS"]["TIME"], fontsize=TYPO["LABEL_SIZE"], labelpad=2
        )
        ax.set_ylabel(
            GRAPH_CONSTANTS["LABELS"]["VALUE"], fontsize=TYPO["LABEL_SIZE"], labelpad=2
        )
        ax.tick_params(labelsize=TYPO["TICK_SIZE"], pad=1)
        ax.grid(True, alpha=LS["GRID_ALPHA"], linewidth=LS["GRID_WIDTH"])

        # Configure legend
        if ax.get_legend():
            ax.legend(
                fontsize=TYPO["LEGEND_SIZE"],
                framealpha=0.8,
                borderaxespad=0.2,
                handlelength=1.0,
                columnspacing=0.8,
            )

        return fig, ax

    def plot_all_features(
        self,
        features: List[Dict],
        change_points: Optional[List[int]] = None,
        n_cols: int = 2,
        figsize: Optional[Tuple[int, int]] = None,
    ) -> Tuple[Figure, List[Axes]]:
        """Plot evolution of all network features."""
        if not features:
            fig, ax = plt.subplots(
                figsize=(FD["SINGLE_COLUMN_WIDTH"] / 2, FD["STANDARD_HEIGHT"] / 2)
            )
            ax.text(
                0.5,
                0.5,
                GRAPH_CONSTANTS["TEXT"]["EMPTY_GRAPH"],
                ha="center",
                va="center",
                fontsize=TYPO["TICK_SIZE"],
            )
            ax.set_xticks([])
            ax.set_yticks([])
            return fig, [ax]

        # Get list of features
        feature_names = list(features[0].keys())
        n_features = len(feature_names)
        n_rows = (n_features + n_cols - 1) // n_cols

        if figsize is None:
            figsize = (FD["SINGLE_COLUMN_WIDTH"], FD["STANDARD_HEIGHT"] * n_rows / 1.5)

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

            # Hide x-axis label for all plots except those in the last row
            if i < n_features and i < len(axes):
                is_bottom_row = i >= len(axes) - n_cols
                if not is_bottom_row:
                    axes[i].set_xlabel("")

        # Remove empty subplots
        for i in range(n_features, len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout(pad=0.5, h_pad=0.8, w_pad=0.8)
        return fig, axes.tolist()

    def plot_network_with_adjacency(
        self,
        adj_matrix: np.ndarray,
        title: str = None,
        layout: str = GRAPH_CONSTANTS["LAYOUTS"]["SPRING"],
        node_color: Union[str, List[str]] = None,
        fig: plt.Figure = None,
    ) -> None:
        """Plot network visualization alongside its adjacency matrix.

        Args:
            adj_matrix: Adjacency matrix of the network
            title: Title for the overall figure
            layout: Network layout algorithm ('spring', 'circular', 'random', 'shell')
            node_color: Color(s) for network nodes
            fig: Optional matplotlib figure to plot on. If None, creates new figure.
        """
        if fig is None:
            fig = plt.figure(figsize=(12, 5))

        # Create two subplots
        gs = fig.add_gridspec(1, 2)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])

        # Plot network
        self.plot_network(
            adj_matrix,
            ax=ax1,
            title="Network View",
            layout=layout,
            node_color=node_color,
        )

        # Plot adjacency matrix
        self.plot_adjacency(adj_matrix, ax=ax2, title="Adjacency Matrix")

        if title:
            fig.suptitle(title, y=1.05)

        # Adjust layout
        plt.tight_layout()
