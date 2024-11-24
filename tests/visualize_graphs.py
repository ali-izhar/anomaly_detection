"""
Graph Visualization Module

This module provides a comprehensive visualization system for different types of graphs
(BA, ER, NW) and their analysis results. It creates detailed dashboards showing graph
evolution, structural changes, and detection results.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from collections import defaultdict
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path

from src.graph.features import (
    extract_centralities,
    compute_embeddings,
    adjacency_to_graph,
    compute_laplacian,
)


@dataclass
class GraphMetrics:
    """Container for graph metrics and analysis results."""

    basic_metrics: Dict[str, List[float]]
    centrality_metrics: Dict[str, List[float]]
    structural_metrics: Dict[str, List[float]]
    spectral_metrics: Dict[str, List[float]]
    embedding_metrics: Dict[str, List[float]]
    path_metrics: Dict[str, List[float]]

    @classmethod
    def from_graph_sequence(cls, graphs: List[np.ndarray]) -> "GraphMetrics":
        """Compute all metrics for a sequence of graphs."""
        basic_metrics = defaultdict(list)
        centrality_metrics = defaultdict(list)
        structural_metrics = defaultdict(list)
        spectral_metrics = defaultdict(list)
        embedding_metrics = defaultdict(list)
        path_metrics = defaultdict(list)

        for adj_matrix in graphs:
            G = adjacency_to_graph(adj_matrix)
            degrees = [d for n, d in G.degree()]

            # Basic metrics
            basic_metrics["avg_degree"].append(np.mean(degrees))
            basic_metrics["density"].append(nx.density(G))
            basic_metrics["clustering_coeff"].append(nx.average_clustering(G))

            # Centrality metrics
            centralities = extract_centralities([adj_matrix])
            for name, values in centralities.items():
                centrality_metrics[f"{name}_mean"].append(np.mean(values))

            # Structural metrics
            structural_metrics["Clustering"].append(nx.average_clustering(G))
            structural_metrics["Density"].append(nx.density(G))
            if nx.is_connected(G):
                path_metrics["Avg Path Length"].append(
                    nx.average_shortest_path_length(G)
                )
                path_metrics["Diameter"].append(nx.diameter(G))
                path_metrics["Radius"].append(nx.radius(G))
            else:
                giant = G.subgraph(max(nx.connected_components(G), key=len))
                path_metrics["Avg Path Length"].append(
                    nx.average_shortest_path_length(giant)
                )
                path_metrics["Diameter"].append(nx.diameter(giant))
                path_metrics["Radius"].append(nx.radius(giant))

            # Spectral metrics
            laplacian = compute_laplacian(adj_matrix)
            eigenvalues = np.real(np.linalg.eigvals(laplacian))
            eigenvalues.sort()
            spectral_metrics["Spectral Gap"].append(eigenvalues[1])
            spectral_metrics["Algebraic Connectivity"].append(eigenvalues[1])
            spectral_metrics["Spectral Radius"].append(eigenvalues[-1])
            spectral_metrics["Eigenvalue Spread"].append(
                eigenvalues[-1] - eigenvalues[1]
            )

            # Embedding metrics
            svd_emb = compute_embeddings([adj_matrix], method="svd")
            lsvd_emb = compute_embeddings([adj_matrix], method="lsvd")
            embedding_metrics["SVD"].append(np.linalg.norm(svd_emb[0]))
            embedding_metrics["LSVD"].append(np.linalg.norm(lsvd_emb[0]))

        return cls(
            basic_metrics=dict(basic_metrics),
            centrality_metrics=dict(centrality_metrics),
            structural_metrics=dict(structural_metrics),
            spectral_metrics=dict(spectral_metrics),
            embedding_metrics=dict(embedding_metrics),
            path_metrics=dict(path_metrics),
        )


@dataclass
class GraphVisualizer:
    """Main visualization class for graph sequences and their analysis."""

    graphs: List[np.ndarray]
    change_points: List[int]
    martingale_scores: Optional[np.ndarray] = None
    detection_results: Optional[Dict] = None
    graph_type: str = "BA"
    metrics: GraphMetrics = field(init=False)

    def __post_init__(self):
        """Initialize derived attributes."""
        self.metrics = GraphMetrics.from_graph_sequence(self.graphs)

    def _create_graph_evolution_plot(self, fig: plt.Figure, gs: plt.GridSpec) -> None:
        """Create plot showing graph structure evolution at change points."""
        time_points = [0] + self.change_points
        G_first = adjacency_to_graph(self.graphs[0])
        pos = nx.spring_layout(G_first, k=1, iterations=50)

        for i, t in enumerate(time_points):
            ax = fig.add_subplot(gs[0, i])
            G = adjacency_to_graph(self.graphs[t])

            degrees = dict(G.degree())
            node_sizes = [
                1000 * (v + 1) / max(degrees.values()) for v in degrees.values()
            ]
            node_colors = list(nx.degree_centrality(G).values())

            nx.draw(
                G,
                pos,
                node_color=node_colors,
                node_size=node_sizes,
                with_labels=True,
                font_size=6,
                edge_color="gray",
                alpha=0.7,
                ax=ax,
                cmap=plt.cm.viridis,
            )

            stats = (
                f"N={G.number_of_nodes()}, E={G.number_of_edges()}\n"
                f"Avg deg={np.mean(list(degrees.values())):.1f}"
            )
            ax.set_title(f"Graph Structure t={t}\n{stats}", fontsize=8)

    def _create_centrality_plot(self, fig: plt.Figure, gs: plt.GridSpec) -> None:
        """Create plot showing centrality measures evolution."""
        ax = fig.add_subplot(gs[1, :])
        metrics = ["degree", "betweenness", "eigenvector", "closeness"]

        for metric in metrics:
            values = self.metrics.centrality_metrics[f"{metric}_mean"]
            mean, std = np.mean(values), np.std(values)
            line = ax.plot(values, "-", label=f"{metric} (μ={mean:.2f}, σ={std:.2f})")

        for cp in self.change_points:
            ax.axvline(x=cp, color="r", linestyle="--", alpha=0.5)

        ax.set_title("Centrality Measures Evolution")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Value")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)

    def _create_embedding_plot(self, fig: plt.Figure, gs: plt.GridSpec) -> None:
        """Create plot showing embedding evolution."""
        ax = fig.add_subplot(gs[2, :])

        for name in ["SVD", "LSVD"]:
            values = self.metrics.embedding_metrics[name]
            mean, std = np.mean(values), np.std(values)
            ax.plot(values, "-", label=f"{name} norm (μ={mean:.2f}, σ={std:.2f})")

        for cp in self.change_points:
            ax.axvline(x=cp, color="r", linestyle="--", alpha=0.5)

        ax.set_title("Graph Embedding Evolution")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Norm")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)

    def _create_structural_plot(self, fig: plt.Figure, gs: plt.GridSpec) -> None:
        """Create plot showing structural properties evolution."""
        ax = fig.add_subplot(gs[3, :])

        for metric, values in self.metrics.structural_metrics.items():
            mean, std = np.mean(values), np.std(values)
            ax.plot(values, "-", label=f"{metric} (μ={mean:.2f}, σ={std:.2f})")

        for cp in self.change_points:
            ax.axvline(x=cp, color="r", linestyle="--", alpha=0.5)

        ax.set_title("Structural Properties Evolution")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Value")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)

    def _create_spectral_plot(self, fig: plt.Figure, gs: plt.GridSpec) -> None:
        """Create plot showing spectral properties evolution."""
        ax = fig.add_subplot(gs[4, :])

        for metric, values in self.metrics.spectral_metrics.items():
            mean, std = np.mean(values), np.std(values)
            ax.plot(values, "-", label=f"{metric} (μ={mean:.2f}, σ={std:.2f})")

        for cp in self.change_points:
            ax.axvline(x=cp, color="r", linestyle="--", alpha=0.5)

        ax.set_title("Spectral Properties Evolution")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Value")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)

    def _create_feature_groups_plot(self, fig: plt.Figure, gs: plt.GridSpec) -> None:
        """Create plots for feature groups (basic metrics, path-based metrics)."""
        feature_groups = {
            "Basic Metrics": (5, 0, self.metrics.basic_metrics),
            "Centrality Statistics": (5, 2, self.metrics.centrality_metrics),
            "Path-based Metrics": (6, 0, self.metrics.path_metrics),
            "Spectral Properties": (
                6,
                2,
                {
                    "spectral_gap": self.metrics.spectral_metrics["Spectral Gap"],
                    "svd_norm": self.metrics.embedding_metrics["SVD"],
                },
            ),
        }

        for group_name, (row, col, metrics) in feature_groups.items():
            ax = fig.add_subplot(gs[row, col : col + 2])

            for name, values in metrics.items():
                mean, std = np.mean(values), np.std(values)
                ax.plot(values, "-", label=f"{name} (μ={mean:.2f}, σ={std:.2f})")

            for cp in self.change_points:
                ax.axvline(x=cp, color="r", linestyle="--", alpha=0.5)

            ax.set_title(group_name)
            ax.set_xlabel("Time step")
            ax.set_ylabel("Value")
            ax.legend(
                fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2
            )
            ax.grid(True, alpha=0.3)

    def create_dashboard(self, output_dir: str = "outputs") -> None:
        """Generate comprehensive dashboard with all visualizations."""
        output_dir = Path(output_dir) / f"{self.graph_type.lower()}_outputs"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Setup figure
        fig = plt.figure(figsize=(20, 35))
        fig.suptitle(f"{self.graph_type} Graph Analysis Dashboard", fontsize=16, y=0.95)

        # Create grid for subplots
        gs = fig.add_gridspec(
            7, 4, height_ratios=[1.5, 1, 1, 1, 1, 1, 1], hspace=0.4, wspace=0.3
        )

        # Create all plots
        self._create_graph_evolution_plot(fig, gs)
        self._create_centrality_plot(fig, gs)
        self._create_embedding_plot(fig, gs)
        self._create_structural_plot(fig, gs)
        self._create_spectral_plot(fig, gs)
        self._create_feature_groups_plot(fig, gs)

        # Save dashboard
        plt.savefig(output_dir / "dashboard.png", dpi=300, bbox_inches="tight")
        plt.close()

        # Save metrics
        self._save_metrics(output_dir)

    def _save_metrics(self, output_dir: Path) -> None:
        """Save all computed metrics to a file."""
        with open(output_dir / "metrics.txt", "w") as f:
            for t in [0] + self.change_points:
                f.write(f"\n=== Time step {t} ===\n")
                for metric_group in [
                    self.metrics.basic_metrics,
                    self.metrics.centrality_metrics,
                    self.metrics.structural_metrics,
                    self.metrics.spectral_metrics,
                    self.metrics.embedding_metrics,
                    self.metrics.path_metrics,
                ]:
                    for name, values in metric_group.items():
                        f.write(f"{name}: {values[t]:.3f}\n")
