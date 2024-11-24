import sys
import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
from collections import defaultdict

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.graph.features import (
    extract_centralities,
    compute_embeddings,
    adjacency_to_graph,
    compute_laplacian,
)
from create_nw_graphs import generate_nw_graphs


def analyze_nw_features(adj_matrix: np.ndarray) -> Dict:
    """Analyze Newman-Watts graph features."""
    G = adjacency_to_graph(adj_matrix)

    # Compute degree distribution
    degrees = [d for n, d in G.degree()]

    metrics = {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "avg_degree": np.mean(degrees),
        "max_degree": max(degrees),
        "min_degree": min(degrees),
        "density": nx.density(G),
        "clustering_coeff": nx.average_clustering(G),
        "degree_histogram": np.histogram(degrees, bins="auto")[0].tolist(),
    }

    centralities = extract_centralities([adj_matrix])
    for name, values_list in centralities.items():
        values = values_list[0]
        metrics[f"{name}_centrality_mean"] = np.mean(values)
        metrics[f"{name}_centrality_std"] = np.std(values)
        metrics[f"{name}_centrality_max"] = np.max(values)

    embeddings = compute_embeddings([adj_matrix], method="svd")
    metrics["svd_norm"] = np.linalg.norm(embeddings[0])

    laplacian = compute_laplacian(adj_matrix)
    eigenvalues = np.real(np.linalg.eigvals(laplacian))
    metrics["spectral_gap"] = np.real(np.sort(eigenvalues)[1])

    # Get largest connected component of G
    if nx.is_connected(G):
        giant = G
        metrics["avg_shortest_path"] = nx.average_shortest_path_length(G)
        metrics["diameter"] = nx.diameter(G)
        metrics["radius"] = nx.radius(G)
    else:
        giant = G.subgraph(max(nx.connected_components(G), key=len))
        metrics["avg_shortest_path"] = nx.average_shortest_path_length(giant)
        metrics["diameter"] = nx.diameter(giant)
        metrics["radius"] = nx.radius(giant)
        metrics["largest_component_size"] = (
            giant.number_of_nodes() / G.number_of_nodes()
        )

    # Small-world specific metrics
    # Generate random graph with same degree sequence for more accurate comparison
    random_G = nx.configuration_model(degrees, seed=42)
    random_G = nx.Graph(random_G)  # Remove parallel edges
    random_G.remove_edges_from(nx.selfloop_edges(random_G))  # Remove self-loops

    # Get largest connected component of random graph
    random_giant = random_G.subgraph(max(nx.connected_components(random_G), key=len))

    C = nx.average_clustering(G)
    L = metrics["avg_shortest_path"]
    C_rand = nx.average_clustering(random_G)
    L_rand = nx.average_shortest_path_length(random_giant)

    # Handle division by zero
    EPSILON = 1e-10  # Small constant to avoid division by zero
    metrics["clustering_ratio"] = C / (
        C_rand + EPSILON
    )  # Should be >> 1 for small-world
    metrics["path_length_ratio"] = L / L_rand  # Should be ~ 1 for small-world
    metrics["small_world_coeff"] = (C / (C_rand + EPSILON)) / (L / L_rand)

    return metrics


def create_dashboard(
    graphs: List[np.ndarray],
    time_points: List[int],
    pos: Dict,
    change_points: List[int],
) -> None:
    """Create comprehensive dashboard of NW graph analytics."""
    fig = plt.figure(figsize=(20, 35))
    fig.suptitle(
        "Newman-Watts Small-World Graph Analysis Dashboard", fontsize=16, y=0.95
    )

    # Create grid for subplots - 7 rows
    gs = fig.add_gridspec(
        7, 4, height_ratios=[1.5, 1, 1, 1, 1, 1, 1], hspace=0.4, wspace=0.3
    )

    # Row 1: Graph Structure Evolution (Larger)
    for i, t in enumerate(time_points):
        ax = fig.add_subplot(gs[0, i])
        G = adjacency_to_graph(graphs[t])

        # Draw regular edges and shortcuts differently
        regular_edges = [
            (i, (i + 1) % G.number_of_nodes()) for i in range(G.number_of_nodes())
        ]
        shortcut_edges = [e for e in G.edges() if e not in regular_edges]

        # Node sizes based on degree
        degrees = dict(G.degree())
        node_sizes = [1000 * (v + 1) / max(degrees.values()) for v in degrees.values()]
        node_colors = list(nx.degree_centrality(G).values())

        nx.draw_networkx_nodes(
            G,
            pos,
            node_color=node_colors,
            node_size=node_sizes,
            cmap=plt.cm.viridis,
            ax=ax,
        )
        nx.draw_networkx_edges(
            G, pos, edgelist=regular_edges, edge_color="gray", alpha=0.7, ax=ax
        )
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

        stats = (
            f"N={G.number_of_nodes()}, E={G.number_of_edges()}\n"
            f"k={len(regular_edges)/G.number_of_nodes():.1f}, "
            f"p={len(shortcut_edges)/len(regular_edges):.2f}"
        )
        ax.set_title(f"Graph Structure t={t}\n{stats}", fontsize=8)

    # Row 2: Centrality Measures Evolution
    ax_centrality = fig.add_subplot(gs[1, :])
    all_time_points = list(range(len(graphs)))
    centralities = extract_centralities([graphs[t] for t in all_time_points])

    for metric, values in centralities.items():
        means = [np.mean(v) for v in values]
        stds = [np.std(v) for v in values]
        maxs = [np.max(v) for v in values]

        line = ax_centrality.plot(
            all_time_points,
            means,
            "-",
            label=f"{metric} (μ={means[-1]:.2f}, σ={stds[-1]:.2f}, max={maxs[-1]:.2f})",
        )
        ax_centrality.fill_between(
            all_time_points,
            [m - s for m, s in zip(means, stds)],
            [m + s for m, s in zip(means, stds)],
            alpha=0.2,
            color=line[0].get_color(),
        )

    for cp in change_points:
        ax_centrality.axvline(x=cp, color="r", linestyle="--", alpha=0.5)
    ax_centrality.set_xlabel("Time step")
    ax_centrality.set_ylabel("Centrality Value")
    ax_centrality.set_title("Centrality Measures Evolution (with ±1σ bands)")
    ax_centrality.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax_centrality.grid(True, alpha=0.3)

    # Row 3: Small-World Metrics Evolution
    ax_sw = fig.add_subplot(gs[2, :])
    sw_metrics = defaultdict(list)

    for t in all_time_points:
        metrics = analyze_nw_features(graphs[t])
        sw_metrics["Clustering Ratio"].append(metrics["clustering_ratio"])
        sw_metrics["Path Length Ratio"].append(metrics["path_length_ratio"])
        sw_metrics["Small-World Coefficient"].append(metrics["small_world_coeff"])

    for metric, values in sw_metrics.items():
        mean, std = np.mean(values), np.std(values)
        ax_sw.plot(
            all_time_points,
            values,
            "-",
            label=f"{metric} (μ={mean:.2f}, σ={std:.2f})",
        )

    for cp in change_points:
        ax_sw.axvline(x=cp, color="r", linestyle="--", alpha=0.5)
    ax_sw.set_xlabel("Time step")
    ax_sw.set_ylabel("Ratio")
    ax_sw.set_title("Small-World Properties Evolution")
    ax_sw.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax_sw.grid(True, alpha=0.3)

    # Row 4: Structural Properties
    ax_struct = fig.add_subplot(gs[3, :])
    structural_metrics = defaultdict(list)

    for t in all_time_points:
        G = adjacency_to_graph(graphs[t])
        structural_metrics["Clustering"].append(nx.average_clustering(G))
        structural_metrics["Density"].append(nx.density(G))
        if nx.is_connected(G):
            structural_metrics["Avg Path Length"].append(
                nx.average_shortest_path_length(G)
            )
            structural_metrics["Diameter"].append(nx.diameter(G))
        else:
            giant = G.subgraph(max(nx.connected_components(G), key=len))
            structural_metrics["Avg Path Length"].append(
                nx.average_shortest_path_length(giant)
            )
            structural_metrics["Diameter"].append(nx.diameter(giant))

    for metric, values in structural_metrics.items():
        mean, std = np.mean(values), np.std(values)
        ax_struct.plot(
            all_time_points,
            values,
            "-",
            label=f"{metric} (μ={mean:.2f}, σ={std:.2f}, Δ={values[-1]-values[0]:.2f})",
        )

    for cp in change_points:
        ax_struct.axvline(x=cp, color="r", linestyle="--", alpha=0.5)
    ax_struct.set_xlabel("Time step")
    ax_struct.set_ylabel("Value")
    ax_struct.set_title("Structural Properties Evolution")
    ax_struct.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax_struct.grid(True, alpha=0.3)

    # Row 5: Spectral Properties
    ax_spectral = fig.add_subplot(gs[4, :])
    spectral_metrics = defaultdict(list)

    for t in all_time_points:
        laplacian = compute_laplacian(graphs[t])
        eigenvalues = np.real(np.linalg.eigvals(laplacian))
        eigenvalues.sort()

        spectral_metrics["Spectral Gap"].append(eigenvalues[1])
        spectral_metrics["Algebraic Connectivity"].append(eigenvalues[1])
        spectral_metrics["Spectral Radius"].append(eigenvalues[-1])
        spectral_metrics["Eigenvalue Spread"].append(eigenvalues[-1] - eigenvalues[1])

    for metric, values in spectral_metrics.items():
        mean, std = np.mean(values), np.std(values)
        ax_spectral.plot(
            all_time_points,
            values,
            "-",
            label=f"{metric} (μ={mean:.2f}, σ={std:.2f}, range=[{min(values):.2f}, {max(values):.2f}])",
        )

    for cp in change_points:
        ax_spectral.axvline(x=cp, color="r", linestyle="--", alpha=0.5)
    ax_spectral.set_xlabel("Time step")
    ax_spectral.set_ylabel("Value")
    ax_spectral.set_title("Spectral Properties Evolution")
    ax_spectral.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax_spectral.grid(True, alpha=0.3)

    # Feature Evolution Groups (Rows 6-7)
    feature_groups = {
        "Basic Metrics": ["avg_degree", "density", "clustering_coeff"],
        "Small-World Metrics": [
            "clustering_ratio",
            "path_length_ratio",
            "small_world_coeff",
        ],
        "Path-based Metrics": ["avg_shortest_path", "diameter", "radius"],
        "Spectral Properties": ["spectral_gap", "svd_norm"],
    }

    # Track features over time
    feature_values = defaultdict(list)
    for t in all_time_points:
        metrics = analyze_nw_features(graphs[t])
        for feature in sum(feature_groups.values(), []):
            if feature in metrics:
                feature_values[feature].append(metrics[feature])

    # Plot feature groups in 2x2 grid
    for i, (group_name, features) in enumerate(feature_groups.items()):
        row = 5 + (i // 2)
        col = i % 2 * 2
        ax = fig.add_subplot(gs[row, col : col + 2])

        for feature in features:
            if feature in feature_values:
                values = feature_values[feature]
                ax.plot(
                    all_time_points,
                    values,
                    "-",
                    label=feature.replace("_", " "),
                    alpha=0.7,
                    linewidth=1,
                )

        for cp in change_points:
            ax.axvline(x=cp, color="r", linestyle="--", alpha=0.5)

        ax.set_xlabel("Time step")
        ax.set_ylabel("Value")
        ax.set_title(f"{group_name}", fontsize=10)
        ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
        ax.grid(True, alpha=0.3)

        # Add statistics annotations
        for j, feature in enumerate(features):
            if feature in feature_values:
                values = np.array(feature_values[feature])
                mean, std = np.mean(values), np.std(values)
                ax.text(
                    0.02,
                    0.98 - j * 0.15,
                    f"{feature}:\nμ={mean:.2f}, σ={std:.2f}",
                    transform=ax.transAxes,
                    fontsize=8,
                    verticalalignment="top",
                    bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
                )

    # Add explanatory footer
    footer_text = """
    Change Points:
    • Initial: k₁=4 neighbors, p₁=0.1 rewiring
    • First change (t=50): k₂=6, p₂=0.3
    • Second change (t=100): k₃=3, p₃=0.05
    • Third change (t=150): k₄=5, p₄=0.2

    Visual Elements:
    • Node colors: Darker = higher degree centrality
    • Node sizes: Larger = more connections
    • Gray edges: Regular lattice connections
    • Red dashed edges: Rewired connections
    • Red dashed lines: Change points
    • Shaded bands: ±1σ variation
    """

    fig.text(
        0.1,
        0.02,
        footer_text,
        fontsize=8,
        family="monospace",
        bbox=dict(
            facecolor="white", alpha=0.8, edgecolor="lightgray", boxstyle="round,pad=1"
        ),
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig("nw_outputs/nw_dashboard.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    # Generate NW graphs
    result = generate_nw_graphs()
    graphs = result["graphs"]
    change_points = result["change_points"]

    # Select key time points to visualize
    time_points = [
        0,  # Start
        change_points[0],  # First change
        change_points[1],  # Second change
        change_points[2],  # Third change
    ]

    # Create output directory
    os.makedirs("nw_outputs", exist_ok=True)

    # First pass to get consistent layout
    G_first = adjacency_to_graph(graphs[0])
    pos = nx.spring_layout(
        G_first, k=1.5, iterations=50
    )  # Changed from circular_layout

    with open("nw_outputs/nw_metrics.txt", "w") as f:
        for t in time_points:
            f.write(f"\n=== Time step {t} ===\n")
            metrics = analyze_nw_features(graphs[t])
            for metric, value in metrics.items():
                if isinstance(value, list):
                    f.write(f"{metric}: {value}\n")
                else:
                    f.write(f"{metric}: {value:.3f}\n")

    # Create dashboard
    create_dashboard(graphs, time_points, pos, change_points)


if __name__ == "__main__":
    main()
