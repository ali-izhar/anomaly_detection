import sys
import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.graph.features import (
    extract_centralities,
    compute_embeddings,
    adjacency_to_graph,
)
from create_nw_graphs import generate_nw_graphs


def visualize_nw_graph(
    adj_matrix: np.ndarray, title: str, pos: Optional[Dict] = None
) -> Dict:
    """Visualize a single Newman-Watts small-world graph.

    Args:
        adj_matrix: Adjacency matrix
        title: Plot title
        pos: Optional node positions for consistent layout

    Returns:
        Node positions dictionary
    """
    G = adjacency_to_graph(adj_matrix)

    plt.figure(figsize=(10, 10))
    if pos is None:
        # Circular layout for better visualization of small-world structure
        pos = nx.circular_layout(G)

    # Draw regular edges and shortcuts differently
    regular_edges = [
        (i, (i + 1) % G.number_of_nodes()) for i in range(G.number_of_nodes())
    ]
    shortcut_edges = [e for e in G.edges() if e not in regular_edges]

    nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=800)

    # Draw regular edges
    nx.draw_networkx_edges(G, pos, edgelist=regular_edges, edge_color="gray", alpha=0.7)

    # Draw shortcuts
    nx.draw_networkx_edges(
        G, pos, edgelist=shortcut_edges, edge_color="red", style="dashed", alpha=0.5
    )

    nx.draw_networkx_labels(G, pos, font_size=8, font_weight="bold")

    plt.title(title)
    return pos


def analyze_nw_features(adj_matrix: np.ndarray) -> Dict:
    """Analyze Newman-Watts graph features.

    Args:
        adj_matrix: Adjacency matrix

    Returns:
        Dictionary of graph metrics
    """
    G = adjacency_to_graph(adj_matrix)

    # Compute small-world metrics
    random_G = nx.erdos_renyi_graph(G.number_of_nodes(), nx.density(G))
    lattice_G = nx.watts_strogatz_graph(G.number_of_nodes(), 4, 0)

    # Get largest connected components
    G_components = list(nx.connected_components(G))
    random_components = list(nx.connected_components(random_G))

    G_largest = G.subgraph(max(G_components, key=len))
    random_largest = random_G.subgraph(max(random_components, key=len))

    metrics = {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "avg_degree": np.mean([d for n, d in G.degree()]),
        "clustering_coeff": nx.average_clustering(G),
        "density": nx.density(G),
        "num_components": len(G_components),
        "largest_component_size": len(max(G_components, key=len)),
    }

    # Compute path length only for largest component
    metrics["avg_shortest_path"] = nx.average_shortest_path_length(G_largest)

    # Small-world metrics (using largest components)
    C = nx.average_clustering(G)
    L = nx.average_shortest_path_length(G_largest)
    C_rand = nx.average_clustering(random_G)
    L_rand = nx.average_shortest_path_length(random_largest)

    metrics["clustering_ratio"] = C / C_rand  # Should be >> 1 for small-world
    metrics["path_length_ratio"] = L / L_rand  # Should be ~ 1 for small-world
    metrics["small_world_coeff"] = (C / C_rand) / (L / L_rand)

    return metrics


def plot_path_length_distribution(G: nx.Graph, title: str) -> None:
    """Plot distribution of shortest path lengths.

    Args:
        G: NetworkX graph
        title: Plot title
    """
    # Compute all shortest paths
    path_lengths = []
    for u in G.nodes():
        lengths = nx.single_source_shortest_path_length(G, u)
        path_lengths.extend(lengths.values())

    plt.figure(figsize=(8, 6))
    plt.hist(path_lengths, bins="auto", alpha=0.7, color="blue")
    plt.axvline(
        np.mean(path_lengths),
        color="r",
        linestyle="--",
        label=f"Mean = {np.mean(path_lengths):.2f}",
    )

    plt.xlabel("Path Length")
    plt.ylabel("Frequency")
    plt.title(f"Shortest Path Distribution: {title}")
    plt.legend()
    plt.grid(True, alpha=0.3)


def create_dashboard(
    graphs: List[np.ndarray], time_points: List[int], pos: Dict
) -> None:
    """Create a comprehensive dashboard of Newman-Watts small-world graph analytics.

    Args:
        graphs: List of adjacency matrices
        time_points: Time points to analyze
        pos: Node positions for consistent layout
    """
    fig = plt.figure(figsize=(20, 15))
    fig.suptitle(
        "Newman-Watts Small-World Graph Analysis Dashboard", fontsize=16, y=0.95
    )

    # Create grid for subplots
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

    # 1. Graph Structures (Row 1)
    for i, t in enumerate(time_points):
        ax = fig.add_subplot(gs[0, i])
        G = adjacency_to_graph(graphs[t])

        # Draw regular edges and shortcuts differently
        regular_edges = [
            (i, (i + 1) % G.number_of_nodes()) for i in range(G.number_of_nodes())
        ]
        shortcut_edges = [e for e in G.edges() if e not in regular_edges]

        nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=500, ax=ax)

        # Draw regular edges
        nx.draw_networkx_edges(
            G, pos, edgelist=regular_edges, edge_color="gray", alpha=0.7, ax=ax
        )

        # Draw shortcuts
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=shortcut_edges,
            edge_color="red",
            style="dashed",
            alpha=0.5,
            ax=ax,
        )

        nx.draw_networkx_labels(G, pos, font_size=6, ax=ax)
        ax.set_title(f"t={t}")

    # 2. Path Length Distributions (Row 2)
    for i, t in enumerate(time_points):
        ax = fig.add_subplot(gs[1, i])
        G = adjacency_to_graph(graphs[t])

        # Compute path lengths
        path_lengths = []
        for u in G.nodes():
            lengths = nx.single_source_shortest_path_length(G, u)
            path_lengths.extend(lengths.values())

        ax.hist(path_lengths, bins="auto", alpha=0.7, color="blue")
        ax.axvline(
            np.mean(path_lengths),
            color="r",
            linestyle="--",
            label=f"Mean = {np.mean(path_lengths):.2f}",
        )

        ax.set_xlabel("Path Length")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Path Length Distribution t={t}")
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)

    # 3. Small-World Metrics Evolution (Row 3, First 2 columns)
    ax_sw = fig.add_subplot(gs[2, :2])
    metrics_over_time = {
        "clustering_ratio": [],
        "path_length_ratio": [],
        "small_world_coeff": [],
    }

    for t in time_points:
        metrics = analyze_nw_features(graphs[t])
        for metric in metrics_over_time.keys():
            metrics_over_time[metric].append(metrics[metric])

    for metric, values in metrics_over_time.items():
        ax_sw.plot(time_points, values, "-o", label=metric)
    ax_sw.set_xlabel("Time step")
    ax_sw.set_ylabel("Ratio")
    ax_sw.set_title("Small-World Properties Evolution")
    ax_sw.legend()
    ax_sw.grid(True, alpha=0.3)

    # 4. Centrality Evolution (Row 3, Last 2 columns)
    ax_centrality = fig.add_subplot(gs[2, 2:])
    centralities = extract_centralities([graphs[t] for t in time_points])

    for metric, values in centralities.items():
        means = [np.mean(v) for v in values]
        ax_centrality.plot(time_points, means, "-o", label=metric)
    ax_centrality.set_xlabel("Time step")
    ax_centrality.set_ylabel("Average centrality")
    ax_centrality.set_title("Centrality Evolution")
    ax_centrality.legend()
    ax_centrality.grid(True, alpha=0.3)

    plt.savefig("nw_outputs/nw_dashboard.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    # Generate NW graphs
    result = generate_nw_graphs()
    graphs = result["graphs"]
    change_point = result["change_point"]

    # Time points to analyze
    time_points = [
        0,  # Start
        change_point - 1,  # Just before change
        change_point,  # At change
        len(graphs) - 1,  # End
    ]

    # Create output directory
    os.makedirs("nw_outputs", exist_ok=True)

    # First pass to get consistent layout
    G_first = adjacency_to_graph(graphs[0])
    pos = nx.circular_layout(G_first)  # Circular layout for small-world graphs

    # Generate individual plots
    for t in time_points:
        print(f"\n=== Time step {t} ===")

        # Analyze features
        metrics = analyze_nw_features(graphs[t])
        for metric, value in metrics.items():
            print(f"{metric}: {value:.3f}")

        # Visualize graph
        pos = visualize_nw_graph(graphs[t], f"NW Graph at t={t}", pos)
        plt.savefig(f"nw_outputs/nw_graph_t{t}.png")
        plt.close()

        # Plot path length distribution
        G = adjacency_to_graph(graphs[t])
        plot_path_length_distribution(G, f"t={t}")
        plt.savefig(f"nw_outputs/nw_pathlength_dist_t{t}.png")
        plt.close()

    # Create dashboard
    create_dashboard(graphs, time_points, pos)


if __name__ == "__main__":
    main()
