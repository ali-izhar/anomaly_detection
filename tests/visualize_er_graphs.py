import sys
import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Dict, Optional

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.graph.features import (
    extract_centralities,
    compute_embeddings,
    adjacency_to_graph,
)
from create_er_graphs import generate_er_graphs


def visualize_er_graph(
    adj_matrix: np.ndarray, title: str, pos: Optional[Dict] = None
) -> Dict:
    """Visualize a single ER graph.

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
        pos = nx.spring_layout(G, k=1.5, iterations=50)

    # Draw nodes with uniform size but varying edge thickness
    edge_weights = [G[u][v].get("weight", 1.0) for u, v in G.edges()]

    nx.draw(
        G,
        pos,
        node_color="lightgreen",
        node_size=800,
        with_labels=True,
        font_size=8,
        font_weight="bold",
        edge_color="gray",
        width=edge_weights,
        alpha=0.7,
    )

    plt.title(title)
    return pos


def analyze_er_features(adj_matrix: np.ndarray) -> Dict:
    """Analyze ER graph features.

    Args:
        adj_matrix: Adjacency matrix

    Returns:
        Dictionary of graph metrics
    """
    G = adjacency_to_graph(adj_matrix)

    # Compute degree distribution
    degrees = [d for n, d in G.degree()]

    metrics = {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "avg_degree": np.mean(degrees),
        "density": nx.density(G),
        "clustering_coeff": nx.average_clustering(G),
        "degree_histogram": np.histogram(degrees, bins="auto")[0].tolist(),
        "assortativity": nx.degree_assortativity_coefficient(G),
    }

    # Component analysis
    components = list(nx.connected_components(G))
    metrics["num_components"] = len(components)
    metrics["largest_component_size"] = len(max(components, key=len))

    # Path length for largest component
    giant = G.subgraph(max(components, key=len))
    metrics["avg_shortest_path"] = nx.average_shortest_path_length(giant)

    return metrics


def plot_degree_distribution(G: nx.Graph, title: str) -> None:
    """Plot degree distribution with Poisson fit.

    Args:
        G: NetworkX graph
        title: Plot title
    """
    degrees = [d for n, d in G.degree()]
    degree_count = nx.degree_histogram(G)

    plt.figure(figsize=(8, 6))
    plt.plot(range(len(degree_count)), degree_count, "g-", alpha=0.6, label="Observed")

    # Add Poisson fit
    lambda_poisson = np.mean(degrees)
    k = np.arange(0, len(degree_count))
    poisson = (
        lambda_poisson**k
        * np.exp(-lambda_poisson)
        / np.math.factorial(len(degree_count) - 1)
    )
    poisson = poisson * sum(degree_count) / sum(poisson)
    plt.plot(k, poisson, "r--", alpha=0.8, label="Poisson fit")

    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    plt.title(f"Degree Distribution: {title}")
    plt.legend()
    plt.grid(True, alpha=0.3)


def create_dashboard(
    graphs: List[np.ndarray], time_points: List[int], pos: Dict
) -> None:
    """Create a comprehensive dashboard of ER graph analytics.

    Args:
        graphs: List of adjacency matrices
        time_points: Time points to analyze
        pos: Node positions for consistent layout
    """
    fig = plt.figure(figsize=(20, 15))
    fig.suptitle("Erdős-Rényi Graph Analysis Dashboard", fontsize=16, y=0.95)

    # Create grid for subplots
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

    # 1. Graph Structures (Row 1)
    for i, t in enumerate(time_points):
        ax = fig.add_subplot(gs[0, i])
        G = adjacency_to_graph(graphs[t])

        # Draw graph with edge weights
        edge_weights = [G[u][v].get("weight", 1.0) for u, v in G.edges()]
        nx.draw(
            G,
            pos,
            node_color="lightgreen",
            node_size=800,
            with_labels=True,
            font_size=6,
            edge_color="gray",
            width=edge_weights,
            alpha=0.7,
            ax=ax,
        )
        ax.set_title(f"t={t}")

    # 2. Degree Distributions with Poisson Fit (Row 2)
    for i, t in enumerate(time_points):
        ax = fig.add_subplot(gs[1, i])
        G = adjacency_to_graph(graphs[t])
        degrees = [d for n, d in G.degree()]
        degree_count = nx.degree_histogram(G)

        # Observed distribution
        ax.plot(
            range(len(degree_count)), degree_count, "g-", alpha=0.6, label="Observed"
        )

        # Poisson fit
        lambda_poisson = np.mean(degrees)
        k = np.arange(0, len(degree_count))
        poisson = (
            lambda_poisson**k
            * np.exp(-lambda_poisson)
            / np.math.factorial(len(degree_count) - 1)
        )
        poisson = poisson * sum(degree_count) / sum(poisson)
        ax.plot(k, poisson, "r--", alpha=0.8, label="Poisson fit")

        ax.set_xlabel("Degree")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Degree Distribution t={t}")
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)

    # 3. Component Analysis (Row 3, First 2 columns)
    ax_components = fig.add_subplot(gs[2, :2])
    metrics_over_time = {
        "num_components": [],
        "largest_component_size": [],
        "density": [],
    }

    for t in time_points:
        metrics = analyze_er_features(graphs[t])
        for metric in metrics_over_time.keys():
            metrics_over_time[metric].append(metrics[metric])

    for metric, values in metrics_over_time.items():
        ax_components.plot(time_points, values, "-o", label=metric)
    ax_components.set_xlabel("Time step")
    ax_components.set_ylabel("Value")
    ax_components.set_title("Component Analysis")
    ax_components.legend()
    ax_components.grid(True, alpha=0.3)

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

    plt.savefig("er_outputs/er_dashboard.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    # Generate ER graphs
    result = generate_er_graphs()
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
    os.makedirs("er_outputs", exist_ok=True)

    # First pass to get consistent layout
    G_first = adjacency_to_graph(graphs[0])
    pos = nx.spring_layout(G_first, k=1.5, iterations=50)

    # Generate individual plots
    for t in time_points:
        print(f"\n=== Time step {t} ===")

        # Analyze features
        metrics = analyze_er_features(graphs[t])
        for metric, value in metrics.items():
            if metric != "degree_histogram":
                print(f"{metric}: {value:.3f}")

        # Visualize graph
        visualize_er_graph(graphs[t], f"ER Graph at t={t}", pos)
        plt.savefig(f"er_outputs/er_graph_t{t}.png")
        plt.close()

        # Plot degree distribution
        G = adjacency_to_graph(graphs[t])
        plot_degree_distribution(G, f"t={t}")
        plt.savefig(f"er_outputs/er_degree_dist_t{t}.png")
        plt.close()

    # Create dashboard
    create_dashboard(graphs, time_points, pos)


if __name__ == "__main__":
    main()
