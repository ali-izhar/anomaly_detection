import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from pathlib import Path
from community import community_louvain


def load_edge_list(file_path):
    """Load edge list file into a pandas DataFrame."""
    df = pd.read_csv(
        file_path, sep=" ", header=None, names=["source", "target", "weight"]
    )
    return df


def create_graph_from_df(df):
    """Create a NetworkX graph from DataFrame of edges."""
    G = nx.Graph()
    for _, row in df.iterrows():
        G.add_edge(row["source"], row["target"])  # weight=1 for binary edges
    return G


def visualize_graph(G, title, pos=None, communities=None):
    """Visualize the network with optional community-based coloring."""
    plt.figure(figsize=(10, 6))
    if pos is None:
        pos = nx.spring_layout(G, k=1, iterations=50)

    if communities is not None:
        colors = [communities[node] for node in G.nodes()]
        nx.draw(
            G,
            pos,
            node_color=colors,
            node_size=100,
            with_labels=False,
            edge_color="gray",
            alpha=0.7,
            cmap=plt.cm.tab20,
        )
    else:
        nx.draw(
            G,
            pos,
            node_color="lightblue",
            node_size=100,
            with_labels=False,
            edge_color="gray",
            alpha=0.7,
        )

    plt.title(title)
    plt.tight_layout()
    return pos


def analyze_graph_communities(G):
    """Run community detection (Louvain) and return the partition + stats."""
    communities = community_louvain.best_partition(G)
    num_communities = len(set(communities.values()))
    modularity = community_louvain.modularity(communities, G)
    sizes = pd.Series(communities.values()).value_counts().sort_index()
    return communities, num_communities, modularity, sizes


def analyze_single_folder(data_dir):
    """
    Analyze all edge_list_{t}.txt files in a single directory data_dir.
    Returns a combined DataFrame of all edges plus some basic stats.
    """
    edge_list_files = sorted(
        [f for f in os.listdir(data_dir) if f.startswith("edge_list")]
    )
    all_dfs = []

    print(f"\nAnalyzing folder: {data_dir}")
    print("Loading edge lists...")
    for file_name in edge_list_files:
        df = load_edge_list(Path(data_dir) / file_name)
        df["file"] = file_name
        all_dfs.append(df)

    combined_df = pd.concat(all_dfs, ignore_index=True)

    # Overall stats
    print("\n=== Overall Dataset Statistics ===")
    print(f"Total number of edge lists: {len(edge_list_files)}")
    print(f"Total number of edges overall: {len(combined_df)}")

    all_nodes = set(combined_df["source"].unique()) | set(
        combined_df["target"].unique()
    )
    print(f"Total unique nodes: {len(all_nodes)}")

    # Plot distribution of node degrees across entire folder
    all_nodes_series = pd.concat([combined_df["source"], combined_df["target"]])
    degree_counts = all_nodes_series.value_counts()
    plt.figure(figsize=(8, 5))
    sns.histplot(degree_counts, bins=30)
    plt.title(f"Node Degree Distribution (Log Scale) - {data_dir}")
    plt.xlabel("Degree")
    plt.ylabel("Count")
    plt.yscale("log")
    plt.tight_layout()
    plt.show()

    # Temporal analysis
    print("\n=== Per-file (temporal) Analysis ===")
    pos = None
    prev_communities = None

    for file_name in edge_list_files:
        snapshot_idx = int(file_name.split("_")[-1].split(".")[0])
        print(f"\nAnalyzing {file_name} (t={snapshot_idx})...")
        df_snap = combined_df[combined_df["file"] == file_name]
        G = create_graph_from_df(df_snap)

        # Basic stats
        print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
        if G.number_of_nodes() > 0:
            avg_degree = sum(dict(G.degree()).values()) / G.number_of_nodes()
        else:
            avg_degree = 0
        print(f"Average degree: {avg_degree:.2f}")
        print(f"Density: {nx.density(G):.4f}")

        # Community analysis
        comm, num_comm, mod_val, sizes = analyze_graph_communities(G)
        print(f"Detected communities: {num_comm}, Modularity: {mod_val:.4f}")
        print("Community sizes:", dict(sizes))

        if prev_communities is not None:
            common_nodes = set(comm.keys()) & set(prev_communities.keys())
            if common_nodes:
                changes = sum(
                    1 for node in common_nodes if comm[node] != prev_communities[node]
                )
                print(
                    f"Community membership changes from previous snapshot: {changes} nodes, "
                    f"{changes / len(common_nodes) * 100:.1f}%"
                )

        prev_communities = comm

        # Optional: visualize the graph with communities
        pos = visualize_graph(G, f"{data_dir} - {file_name}", pos, comm)
        plt.show()

    return combined_df


def main():
    """
    Inspect the splitted subfolders:
      data/SBM/train, data/SBM/val, data/SBM/test
    And verify the splits, anomaly points, densities, etc.
    """
    root_dir = Path("data/SBM")
    subfolders = ["train", "val", "test"]

    for subf in subfolders:
        folder_path = root_dir / subf
        if folder_path.exists():
            _ = analyze_single_folder(folder_path)
        else:
            print(f"Folder not found: {folder_path}")


if __name__ == "__main__":
    main()
