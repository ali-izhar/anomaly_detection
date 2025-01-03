# archive/draw_graph.py

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.decomposition import TruncatedSVD


def create_example_graphs():
    """Create different example graphs to demonstrate various centrality measures"""
    graphs = {}

    # 1. Star graph for degree centrality
    G_degree = nx.star_graph(5)  # Center node has high degree
    graphs["degree"] = (G_degree, "Star Network\nCenter node has highest degree")

    # 2. Bridge graph for betweenness
    G_between = nx.Graph()
    # Create two clusters connected by a bridge
    G_between.add_edges_from([(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (2, 3)])
    graphs["betweenness"] = (
        G_between,
        "Bridge Network\nNodes 2,3 have high betweenness",
    )

    # 3. Influence graph for eigenvector
    G_eigen = nx.Graph()
    # Create a network where some nodes connect to important neighbors
    G_eigen.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0), (1, 4), (2, 5), (0, 6)])
    graphs["eigenvector"] = (
        G_eigen,
        "Influence Network\nNodes 0,1,2 have high eigenvector centrality",
    )

    # 4. Information flow for closeness
    G_close = nx.Graph()
    # Create a network where central node reaches others quickly
    G_close.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (0, 7)])
    graphs["closeness"] = (
        G_close,
        "Information Flow Network\nNode 0 has high closeness",
    )

    return graphs


def plot_centrality_measures():
    """Create visualizations explaining different centrality measures"""
    graphs = create_example_graphs()

    # Create more compact figure
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(
        2, 3, height_ratios=[1, 1], width_ratios=[1, 1, 1], hspace=0.3, wspace=0.2
    )

    # Visualization parameters
    node_size = 800
    font_size = 8
    title_size = 10
    detail_size = 6  # Even smaller font for detailed explanations

    plot_params = {
        "node_size": node_size,
        "cmap": plt.cm.viridis,
        "font_size": font_size,
        "with_labels": True,
        "alpha": 0.2,
    }

    # 1. Degree Centrality
    ax1 = fig.add_subplot(gs[0, 0])
    G = graphs["degree"][0]
    pos = nx.spring_layout(G, seed=42, k=1)
    degree_cent = nx.degree_centrality(G)

    nx.draw_networkx_edges(G, pos, alpha=plot_params["alpha"], ax=ax1)
    nodes = nx.draw_networkx_nodes(
        G,
        pos,
        node_color=list(degree_cent.values()),
        node_size=plot_params["node_size"],
        cmap=plot_params["cmap"],
        ax=ax1,
    )
    nx.draw_networkx_labels(G, pos, font_size=plot_params["font_size"], ax=ax1)

    ax1.set_title("Degree Centrality\n(# of connections)", fontsize=title_size, pad=10)
    ax1.text(
        0.02,
        -0.1,
        "Star Network\nCenter node has highest degree\nMeasures direct connections",
        fontsize=detail_size,
        transform=ax1.transAxes,
    )

    # 2. Betweenness Centrality
    ax2 = fig.add_subplot(gs[0, 1])
    G = graphs["betweenness"][0]
    pos = nx.spring_layout(G, seed=42, k=1)
    between_cent = nx.betweenness_centrality(G)

    nx.draw_networkx_edges(G, pos, alpha=plot_params["alpha"], ax=ax2)
    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=list(between_cent.values()),
        node_size=plot_params["node_size"],
        cmap=plot_params["cmap"],
        ax=ax2,
    )
    nx.draw_networkx_labels(G, pos, font_size=plot_params["font_size"], ax=ax2)

    ax2.set_title("Betweenness\n(bridge importance)", fontsize=title_size, pad=10)
    ax2.text(
        0.02,
        -0.1,
        "Bridge Network\nNodes 2,3 control information flow\nMeasures path control",
        fontsize=detail_size,
        transform=ax2.transAxes,
    )

    # 3. Eigenvector Centrality
    ax3 = fig.add_subplot(gs[0, 2])
    G = graphs["eigenvector"][0]
    pos = nx.spring_layout(G, seed=42, k=1)
    eigen_cent = nx.eigenvector_centrality(G)

    nx.draw_networkx_edges(G, pos, alpha=plot_params["alpha"], ax=ax3)
    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=list(eigen_cent.values()),
        node_size=plot_params["node_size"],
        cmap=plot_params["cmap"],
        ax=ax3,
    )
    nx.draw_networkx_labels(G, pos, font_size=plot_params["font_size"], ax=ax3)

    ax3.set_title("Eigenvector\n(neighbor importance)", fontsize=title_size, pad=10)
    ax3.text(
        0.02,
        -0.1,
        "Influence Network\nNodes 0,1,2 connected to important nodes\nMeasures connection quality",
        fontsize=detail_size,
        transform=ax3.transAxes,
    )

    # 4. Closeness Centrality
    ax4 = fig.add_subplot(gs[1, 0])
    G = graphs["closeness"][0]
    pos = nx.spring_layout(G, seed=42, k=1)
    close_cent = nx.closeness_centrality(G)

    nx.draw_networkx_edges(G, pos, alpha=plot_params["alpha"], ax=ax4)
    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=list(close_cent.values()),
        node_size=plot_params["node_size"],
        cmap=plot_params["cmap"],
        ax=ax4,
    )
    nx.draw_networkx_labels(G, pos, font_size=plot_params["font_size"], ax=ax4)

    ax4.set_title("Closeness\n(avg. distance to others)", fontsize=title_size, pad=10)
    ax4.text(
        0.02,
        -0.1,
        "Information Flow Network\nNode 0 reaches others quickly\nMeasures average distance",
        fontsize=detail_size,
        transform=ax4.transAxes,
    )

    # 5. SVD Embedding
    ax5 = fig.add_subplot(gs[1, 1])
    G = graphs["eigenvector"][0]  # Use influence network
    A = nx.adjacency_matrix(G).todense()
    svd = TruncatedSVD(n_components=2)
    pos_matrix = svd.fit_transform(A)
    pos_scaled = pos_matrix / np.abs(pos_matrix).max()
    svd_pos = {i: (pos_scaled[i, 0], pos_scaled[i, 1]) for i in G.nodes()}

    nx.draw(
        G,
        svd_pos,
        node_color=list(eigen_cent.values()),
        node_size=plot_params["node_size"],
        cmap=plot_params["cmap"],
        with_labels=True,
        font_size=plot_params["font_size"],
        ax=ax5,
    )

    ax5.set_title("SVD Embedding\n(global structure)", fontsize=title_size, pad=10)
    ax5.text(
        0.02,
        -0.1,
        "Node positions based on overall connectivity\nPreserves global network structure\nUses singular value decomposition",
        fontsize=detail_size,
        transform=ax5.transAxes,
    )

    # 6. Laplacian SVD
    ax6 = fig.add_subplot(gs[1, 2])
    L = nx.laplacian_matrix(G).todense()
    eigvals, eigvecs = np.linalg.eigh(L)
    idx = np.argsort(eigvals)[1:3]  # Skip first eigenvalue (always 0)
    pos_matrix = eigvecs[:, idx]
    pos_scaled = pos_matrix / np.abs(pos_matrix).max()
    lsvd_pos = {i: (pos_scaled[i, 0], pos_scaled[i, 1]) for i in G.nodes()}

    nx.draw(
        G,
        lsvd_pos,
        node_color=list(eigen_cent.values()),
        node_size=plot_params["node_size"],
        cmap=plot_params["cmap"],
        with_labels=True,
        font_size=plot_params["font_size"],
        ax=ax6,
    )

    ax6.set_title("Laplacian SVD\n(local structure)", fontsize=title_size, pad=10)
    ax6.text(
        0.02,
        -0.1,
        "Preserves community structure\nFocuses on local relationships\nUses graph Laplacian",
        fontsize=detail_size,
        transform=ax6.transAxes,
    )

    # Colorbar
    cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.02])
    cbar = plt.colorbar(nodes, cax=cbar_ax, orientation="horizontal")
    cbar.ax.tick_params(labelsize=font_size)
    cbar.set_label("Importance Score", fontsize=font_size)

    fig.suptitle(
        "Network Centrality Measures & Embeddings", fontsize=title_size + 2, y=0.95
    )

    # Remove axis for all subplots
    for ax in fig.get_axes():
        ax.set_axis_off()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_centrality_measures()
