#!/usr/bin/env python3
"""Generate and visualize sample synthetic networks:
1. Stochastic Block Model (SBM)
2. Barabási-Albert (BA)
3. Erdős-Rényi (ER)
4. Newman-Watts-Strogatz (NWS)
"""

import os
import sys
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.gridspec import GridSpec

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph.generator import GraphGenerator
from configs.loader import get_config


def create_directory(dir_path):
    """Create directory if it doesn't exist."""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Created directory: {dir_path}")


def adjacency_to_networkx(adj_matrix):
    """Convert adjacency matrix to NetworkX graph."""
    return nx.from_numpy_array(adj_matrix)


def plot_network_with_adjacency(
    G,
    title,
    params,
    pos=None,
    node_color=None,
    community_labels=None,
    filename=None,
    cmap="viridis",
):
    """
    Create a two-part visualization:
    1. Network structure (top)
    2. Adjacency matrix (bottom)

    Args:
        G: NetworkX graph
        title: Plot title
        params: Dictionary of model parameters to display
        pos: Node positions (if None, calculated using spring layout)
        node_color: Node colors (if None, default color is used)
        community_labels: Dict mapping nodes to community labels
        filename: Output filename (if None, display only)
        cmap: Colormap to use for node colors and adjacency matrix
    """
    # Create figure with GridSpec for layout control
    fig = plt.figure(figsize=(8, 10))
    gs = GridSpec(2, 1, height_ratios=[3, 2], hspace=0.3)

    # Network visualization (top)
    ax_network = fig.add_subplot(gs[0])

    # Create position layout if not provided
    if pos is None:
        pos = nx.spring_layout(G, seed=42)

    # Set default node color if not provided
    if node_color is None:
        if community_labels:
            node_color = [community_labels[node] for node in G.nodes()]
        else:
            node_color = "skyblue"

    # Draw the network
    nx.draw_networkx(
        G,
        pos=pos,
        ax=ax_network,
        with_labels=False,
        node_size=100,
        node_color=node_color,
        cmap=plt.cm.get_cmap(cmap),
        edge_color="gray",
        alpha=0.8,
        width=0.5,
    )

    # Add colorbar if using community colors or color mapping
    if isinstance(node_color, list) and len(set(node_color)) > 1:
        if community_labels:
            unique_values = sorted(set(community_labels.values()))
            label = "Community"
        else:
            unique_values = np.linspace(min(node_color), max(node_color), 10)
            label = "Centrality"

        sm = ScalarMappable(
            cmap=plt.cm.get_cmap(cmap),
            norm=Normalize(vmin=min(unique_values), vmax=max(unique_values)),
        )
        sm.set_array([])
        cbar = plt.colorbar(
            sm,
            ax=ax_network,
            label=label,
            orientation="horizontal",
            pad=0.01,
            aspect=40,
        )
        cbar.ax.tick_params(labelsize=8)

    # Add title and turn off axis
    ax_network.set_title(title, fontsize=14, pad=10, fontweight="bold")
    ax_network.axis("off")

    # Add model parameters as text
    param_text = ", ".join(
        [
            f"{k}={v}"
            for k, v in params.items()
            if k not in ["seq_len", "min_segment", "min_changes", "max_changes", "seed"]
        ]
    )
    ax_network.text(
        0.5,
        -0.05,
        f"Parameters: {param_text}",
        transform=ax_network.transAxes,
        ha="center",
        fontsize=9,
    )

    # Adjacency matrix visualization (bottom)
    ax_matrix = fig.add_subplot(gs[1])

    # Get adjacency matrix
    adj_matrix = nx.to_numpy_array(G)

    # Use seaborn for better heatmap
    sns.heatmap(
        adj_matrix,
        cmap=cmap,
        ax=ax_matrix,
        cbar=True,
        square=True,
        linewidths=0,
        linecolor="none",
        xticklabels=False,
        yticklabels=False,
    )

    ax_matrix.set_title("Adjacency Matrix", fontsize=12)
    ax_matrix.set_xlabel("Node Index", fontsize=10)
    ax_matrix.set_ylabel("Node Index", fontsize=10)

    # Tight layout and save or display
    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved visualization to {filename}")
    else:
        plt.show()


def create_combined_visualization(networks, output_dir):
    """Create a 2x4 grid visualization.
    Uses 2 rows and 4 columns:
    - Row 1: All network structures
    - Row 2: All adjacency matrices

    Args:
        networks: Dictionary containing the four networks and their metadata
        output_dir: Directory to save the visualization
    """
    # Define a consistent set of colors
    color_schemes = {
        "sbm": {"network": "viridis", "matrix": "viridis"},
        "ba": {"network": "plasma", "matrix": "plasma"},
        "er": {"network": "Blues", "matrix": "Blues"},
        "nws": {"network": "magma", "matrix": "magma"},
    }

    # Use paper-style formatting based on plot_martingale.py
    plt.style.use("seaborn-v0_8-paper")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "Times New Roman"],
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "lines.linewidth": 1.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    # Create the figure - landscape orientation for 2x4
    fig = plt.figure(figsize=(10, 5.5))

    # Create a 2x4 grid: networks on top, adjacency matrices on bottom
    gs = GridSpec(
        2,
        4,
        figure=fig,
        height_ratios=[1, 1],
        width_ratios=[1, 1, 1, 1],
        hspace=0.35,
        wspace=0.15,
    )

    # Column 1: SBM
    ax_sbm_net = fig.add_subplot(gs[0, 0])
    ax_sbm_adj = fig.add_subplot(gs[1, 0])

    # Column 2: BA
    ax_ba_net = fig.add_subplot(gs[0, 1])
    ax_ba_adj = fig.add_subplot(gs[1, 1])

    # Column 3: ER
    ax_er_net = fig.add_subplot(gs[0, 2])
    ax_er_adj = fig.add_subplot(gs[1, 2])

    # Column 4: NWS
    ax_nws_net = fig.add_subplot(gs[0, 3])
    ax_nws_adj = fig.add_subplot(gs[1, 3])

    # Draw SBM Network and Matrix
    sbm_graph = networks["sbm"]["graph"]
    sbm_pos = networks["sbm"]["pos"]
    sbm_colors = networks["sbm"]["colors"]
    sbm_params = networks["sbm"]["params"]

    # Draw networks - restore spines for network plots
    ax_sbm_net.spines["top"].set_visible(True)
    ax_sbm_net.spines["right"].set_visible(True)

    nx.draw_networkx(
        sbm_graph,
        pos=sbm_pos,
        ax=ax_sbm_net,
        with_labels=False,
        node_size=60,
        node_color=sbm_colors,
        cmap=plt.colormaps.get_cmap(color_schemes["sbm"]["network"]),
        edge_color="gray",
        alpha=0.8,
        width=0.5,
    )
    ax_sbm_net.set_title("SBM", fontsize=12, fontweight="bold")
    ax_sbm_net.axis("off")

    # Add parameter text with formatting
    param_text = "n=50, κ=2, p$_{\\mathrm{in}}$=0.95, p$_{\\mathrm{out}}$=0.01"
    ax_sbm_net.text(
        0.5,
        -0.1,
        param_text,
        transform=ax_sbm_net.transAxes,
        ha="center",
        fontsize=9,
        style="italic",
    )

    # SBM Adjacency matrix
    sbm_matrix = nx.to_numpy_array(sbm_graph)
    sns.heatmap(
        sbm_matrix,
        cmap=color_schemes["sbm"]["matrix"],
        ax=ax_sbm_adj,
        cbar=False,
        square=True,
        linewidths=0,
        xticklabels=False,
        yticklabels=False,
    )
    ax_sbm_adj.set_title("", fontsize=12)
    ax_sbm_adj.set_xlabel("", fontsize=10, labelpad=5)
    ax_sbm_adj.set_ylabel("", fontsize=10, labelpad=5)

    # Draw BA Network and Matrix
    ba_graph = networks["ba"]["graph"]
    ba_colors = networks["ba"]["colors"]
    ba_params = networks["ba"]["params"]

    # Restore spines for network plots
    ax_ba_net.spines["top"].set_visible(True)
    ax_ba_net.spines["right"].set_visible(True)

    nx.draw_networkx(
        ba_graph,
        ax=ax_ba_net,
        with_labels=False,
        node_size=60,
        node_color=ba_colors,
        cmap=plt.colormaps.get_cmap(color_schemes["ba"]["network"]),
        edge_color="gray",
        alpha=0.8,
        width=0.5,
    )
    ax_ba_net.set_title("BA", fontsize=12, fontweight="bold")
    ax_ba_net.axis("off")

    # Add parameter text
    param_text = "n=50, m=1"
    ax_ba_net.text(
        0.5,
        -0.1,
        param_text,
        transform=ax_ba_net.transAxes,
        ha="center",
        fontsize=9,
        style="italic",
    )

    # BA Adjacency matrix
    ba_matrix = nx.to_numpy_array(ba_graph)
    sns.heatmap(
        ba_matrix,
        cmap=color_schemes["ba"]["matrix"],
        ax=ax_ba_adj,
        cbar=False,
        square=True,
        linewidths=0,
        xticklabels=False,
        yticklabels=False,
    )
    ax_ba_adj.set_title("", fontsize=12)
    ax_ba_adj.set_xlabel("", fontsize=10, labelpad=5)
    ax_ba_adj.set_ylabel("", fontsize=10)

    # Draw ER Network and Matrix
    er_graph = networks["er"]["graph"]
    er_params = networks["er"]["params"]

    # Restore spines for network plots
    ax_er_net.spines["top"].set_visible(True)
    ax_er_net.spines["right"].set_visible(True)

    nx.draw_networkx(
        er_graph,
        ax=ax_er_net,
        with_labels=False,
        node_size=60,
        node_color="skyblue",
        edge_color="gray",
        alpha=0.8,
        width=0.5,
    )
    ax_er_net.set_title("ER", fontsize=12, fontweight="bold")
    ax_er_net.axis("off")

    # Add parameter text
    param_text = "n=50, p=0.050"
    ax_er_net.text(
        0.5,
        -0.1,
        param_text,
        transform=ax_er_net.transAxes,
        ha="center",
        fontsize=9,
        style="italic",
    )

    # ER Adjacency matrix
    er_matrix = nx.to_numpy_array(er_graph)
    sns.heatmap(
        er_matrix,
        cmap=color_schemes["er"]["matrix"],
        ax=ax_er_adj,
        cbar=False,
        square=True,
        linewidths=0,
        xticklabels=False,
        yticklabels=False,
    )
    ax_er_adj.set_title("", fontsize=12)
    ax_er_adj.set_xlabel("", fontsize=10, labelpad=5)
    ax_er_adj.set_ylabel("", fontsize=10)

    # Draw NWS Network and Matrix
    nws_graph = networks["nws"]["graph"]
    nws_pos = networks["nws"]["pos"]
    nws_colors = networks["nws"]["colors"]
    nws_params = networks["nws"]["params"]

    # Restore spines for network plots
    ax_nws_net.spines["top"].set_visible(True)
    ax_nws_net.spines["right"].set_visible(True)

    nx.draw_networkx(
        nws_graph,
        pos=nws_pos,
        ax=ax_nws_net,
        with_labels=False,
        node_size=60,
        node_color=nws_colors,
        cmap=plt.colormaps.get_cmap(color_schemes["nws"]["network"]),
        edge_color="gray",
        alpha=0.8,
        width=0.5,
    )
    ax_nws_net.set_title("NWS", fontsize=12, fontweight="bold")
    ax_nws_net.axis("off")

    # Add parameter text
    param_text = "n=50, k=6, p=0.10"
    ax_nws_net.text(
        0.5,
        -0.1,
        param_text,
        transform=ax_nws_net.transAxes,
        ha="center",
        fontsize=9,
        style="italic",
    )

    # NWS Adjacency matrix
    nws_matrix = nx.to_numpy_array(nws_graph)
    sns.heatmap(
        nws_matrix,
        cmap=color_schemes["nws"]["matrix"],
        ax=ax_nws_adj,
        cbar=False,
        square=True,
        linewidths=0,
        xticklabels=False,
        yticklabels=False,
    )
    ax_nws_adj.set_title("", fontsize=12)
    ax_nws_adj.set_xlabel("", fontsize=10, labelpad=5)
    ax_nws_adj.set_ylabel("", fontsize=10)

    # Add a single colorbar for all adjacency matrices
    cax = fig.add_axes([0.925, 0.15, 0.015, 0.7])  # [left, bottom, width, height]
    sm = ScalarMappable(cmap="viridis", norm=Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cax)
    cbar.set_label("Edge Presence", fontsize=10, labelpad=10)

    # Add custom y-axis label for adjacency matrices that spans all plots
    fig.text(0.02, 0.25, "", va="center", rotation="vertical", fontsize=10)

    # Adjust layout
    plt.tight_layout(rect=[0.02, 0, 0.91, 0.98])

    # Save the figure
    plt.savefig(
        os.path.join(output_dir, "synthetic_networks_grid.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print(
        f"Saved combined visualization to {os.path.join(output_dir, 'synthetic_networks_grid.png')}"
    )


def generate_synthetic_networks(output_dir="figures"):
    """Generate and visualize all synthetic networks."""
    print("Generating synthetic networks...")

    # Create output directory
    create_directory(output_dir)

    # Set random seed for reproducibility
    seed = 42

    # Dictionary to store all networks and their metadata
    networks = {}

    # 1. Stochastic Block Model (SBM)
    print("Generating SBM network...")
    sbm_config = get_config(
        "stochastic_block_model",
        n=50,  # Override for visualization clarity
        seq_len=1,  # Just one graph
        min_segment=1,
        min_changes=0,
        max_changes=0,
        seed=seed,
    )

    sbm_gen = GraphGenerator(sbm_config["model"])
    sbm_result = sbm_gen.generate_sequence(sbm_config["params"].__dict__)
    G_sbm = adjacency_to_networkx(sbm_result["graphs"][0])

    # Create community labels for SBM
    n_nodes = sbm_config["params"].n
    n_blocks = sbm_config["params"].num_blocks
    community_labels = {}
    nodes_per_block = n_nodes // n_blocks
    for i in range(n_nodes):
        block = min(i // nodes_per_block, n_blocks - 1)
        community_labels[i] = block

    # Use spectral layout to better show community structure
    pos_sbm = nx.spectral_layout(G_sbm)

    # Display parameters
    sbm_display_params = {
        "nodes": n_nodes,
        "communities": n_blocks,
        "p_intra": sbm_config["params"].intra_prob,
        "p_inter": sbm_config["params"].inter_prob,
    }

    # Store SBM data
    networks["sbm"] = {
        "graph": G_sbm,
        "pos": pos_sbm,
        "colors": [community_labels[node] for node in G_sbm.nodes()],
        "params": sbm_display_params,
    }

    # 2. Barabási-Albert (BA)
    print("Generating BA network...")
    ba_config = get_config(
        "barabasi_albert",
        n=50,  # Override for visualization clarity
        seq_len=1,  # Just one graph
        min_segment=1,
        min_changes=0,
        max_changes=0,
        seed=seed,
    )

    ba_gen = GraphGenerator(ba_config["model"])
    ba_result = ba_gen.generate_sequence(ba_config["params"].__dict__)
    G_ba = adjacency_to_networkx(ba_result["graphs"][0])

    # Color nodes by degree centrality
    degree_centrality = nx.degree_centrality(G_ba)
    node_colors_ba = [degree_centrality[node] for node in G_ba.nodes()]

    # Display parameters
    ba_display_params = {"nodes": ba_config["params"].n, "m": ba_config["params"].m}

    # Store BA data
    networks["ba"] = {
        "graph": G_ba,
        "pos": None,  # Use spring layout
        "colors": node_colors_ba,
        "params": ba_display_params,
    }

    # 3. Erdős-Rényi (ER)
    print("Generating ER network...")
    er_config = get_config(
        "erdos_renyi",
        n=50,  # Override for visualization clarity
        seq_len=1,  # Just one graph
        min_segment=1,
        min_changes=0,
        max_changes=0,
        seed=seed,
    )

    er_gen = GraphGenerator(er_config["model"])
    er_result = er_gen.generate_sequence(er_config["params"].__dict__)
    G_er = adjacency_to_networkx(er_result["graphs"][0])

    # Display parameters
    er_display_params = {"nodes": er_config["params"].n, "p": er_config["params"].prob}

    # Store ER data
    networks["er"] = {
        "graph": G_er,
        "pos": None,  # Use spring layout
        "colors": "skyblue",
        "params": er_display_params,
    }

    # 4. Newman-Watts-Strogatz (NWS)
    print("Generating NWS network...")
    nws_config = get_config(
        "watts_strogatz",
        n=50,  # Override for visualization clarity
        seq_len=1,  # Just one graph
        min_segment=1,
        min_changes=0,
        max_changes=0,
        seed=seed,
    )

    nws_gen = GraphGenerator(nws_config["model"])
    nws_result = nws_gen.generate_sequence(nws_config["params"].__dict__)
    G_nws = adjacency_to_networkx(nws_result["graphs"][0])

    # Use circular layout to better show small-world properties
    pos_nws = nx.circular_layout(G_nws)

    # Color nodes by closeness centrality
    closeness_centrality = nx.closeness_centrality(G_nws)
    node_colors_nws = [closeness_centrality[node] for node in G_nws.nodes()]

    # Display parameters
    nws_display_params = {
        "nodes": nws_config["params"].n,
        "k": nws_config["params"].k_nearest,
        "p": nws_config["params"].rewire_prob,
    }

    # Store NWS data
    networks["nws"] = {
        "graph": G_nws,
        "pos": pos_nws,
        "colors": node_colors_nws,
        "params": nws_display_params,
    }

    # Create the combined 2x4 grid visualization
    create_combined_visualization(networks, output_dir)

    print("Successfully generated all network visualizations.")


if __name__ == "__main__":
    # Directory where figures will be saved
    output_dir = "paper/Figures"
    create_directory(output_dir)
    generate_synthetic_networks(output_dir)
