import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from pathlib import Path
from community import community_louvain

def load_edge_list(file_path):
    """Load edge list file into a pandas DataFrame."""
    df = pd.read_csv(file_path, sep=' ', header=None, names=['source', 'target', 'weight'])
    return df

def create_graph_from_df(df):
    """Create a NetworkX graph from DataFrame."""
    G = nx.Graph()
    for _, row in df.iterrows():
        G.add_edge(row['source'], row['target'])  # No need for weight parameter since all edges are binary
    return G

def visualize_graph(G, title, pos=None, communities=None):
    """Visualize network with community structure."""
    plt.figure(figsize=(12, 8))
    
    if pos is None:
        pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Color nodes by community if available
    if communities is not None:
        colors = [communities[node] for node in G.nodes()]
        nx.draw(G, pos, node_color=colors, node_size=100, 
               with_labels=False, edge_color='gray', alpha=0.7,
               cmap=plt.cm.tab20)
    else:
        nx.draw(G, pos, node_color='lightblue', node_size=100,
               with_labels=False, edge_color='gray', alpha=0.7)
    
    plt.title(title)
    plt.tight_layout()
    return pos

def analyze_graph_communities(G):
    """Analyze community structure of a graph."""
    # Detect communities using Louvain method
    communities = community_louvain.best_partition(G)
    num_communities = len(set(communities.values()))
    
    # Calculate modularity
    modularity = community_louvain.modularity(communities, G)
    
    # Calculate community sizes
    community_sizes = pd.Series(communities.values()).value_counts().sort_index()
    
    return communities, num_communities, modularity, community_sizes

def analyze_dataset(data_dir):
    """Analyze all edge list files in the dataset collectively."""
    edge_list_files = sorted([f for f in os.listdir(data_dir) if f.startswith('edge_list')])
    all_dfs = []
    
    print("Loading all edge lists...")
    for file_name in edge_list_files:
        df = load_edge_list(data_dir / file_name)
        df['file'] = file_name  # Add source file information
        all_dfs.append(df)
    
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Overall dataset statistics
    print("\n=== Overall Dataset Statistics ===")
    print(f"Total number of edge lists: {len(edge_list_files)}")
    print(f"Total number of edges: {len(combined_df)}")
    all_nodes = set(combined_df['source'].unique()) | set(combined_df['target'].unique())
    print(f"Total number of unique nodes: {len(all_nodes)}")
    
    # Per-file statistics
    print("\n=== Per-file Statistics ===")
    file_stats = combined_df.groupby('file').agg({
        'source': ['count', lambda x: len(set(x) | set(combined_df['target']))]
    })
    file_stats.columns = ['Edge Count', 'Node Count']
    print(file_stats)
    
    # Node degree analysis
    all_nodes_series = pd.concat([combined_df['source'], combined_df['target']])
    degree_counts = all_nodes_series.value_counts()
    
    plt.figure(figsize=(10, 6))
    sns.histplot(data=degree_counts, bins=30)
    plt.title('Overall Node Degree Distribution')
    plt.xlabel('Degree')
    plt.ylabel('Count')
    plt.yscale('log')
    plt.tight_layout()
    plt.show()
    
    # Analyze temporal evolution
    print("\n=== Temporal Evolution Analysis ===")
    pos = None  # Store layout to keep node positions consistent
    prev_communities = None
    
    for file_name in edge_list_files:
        print(f"\nAnalyzing {file_name}...")
        df = combined_df[combined_df['file'] == file_name]
        G = create_graph_from_df(df)
        
        # Basic graph statistics
        print(f"Number of nodes: {G.number_of_nodes()}")
        print(f"Number of edges: {G.number_of_edges()}")
        print(f"Average degree: {sum(dict(G.degree()).values())/G.number_of_nodes():.2f}")
        print(f"Density: {nx.density(G):.4f}")
        
        # Community analysis
        communities, num_communities, modularity, community_sizes = analyze_graph_communities(G)
        print(f"Number of detected communities: {num_communities}")
        print(f"Modularity: {modularity:.4f}")
        print("Community sizes:", dict(community_sizes))
        
        # Compare with previous communities if available
        if prev_communities is not None:
            common_nodes = set(communities.keys()) & set(prev_communities.keys())
            if common_nodes:
                changes = sum(1 for node in common_nodes if communities[node] != prev_communities[node])
                print(f"Community changes from previous snapshot: {changes} nodes ({changes/len(common_nodes)*100:.1f}%)")
        
        prev_communities = communities
        
        # Visualize graph with communities
        pos = visualize_graph(G, f"Graph Structure - {file_name}", pos, communities)
        plt.show()

def main():
    data_dir = Path('data/SBM')  # Update path to match the SBM dataset location
    analyze_dataset(data_dir)

if __name__ == "__main__":
    main()
