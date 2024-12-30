"""
Generate large-scale Stochastic Block Model dataset with temporal dynamics
"""

import numpy as np
import networkx as nx
import os
from pathlib import Path

def generate_sbm_snapshot(n_nodes, n_communities, p_in_range=(0.7, 0.9), p_out_range=(0.1, 0.2)):
    """
    Generate a binary SBM snapshot with clear community structure.
    
    Args:
        n_nodes: Number of nodes (38 for UCSB)
        n_communities: Number of communities
        p_in_range: Range of probabilities for intra-community edges
        p_out_range: Range of probabilities for inter-community edges
    """
    # Calculate sizes of communities
    sizes = [n_nodes // n_communities] * n_communities
    sizes[-1] += n_nodes % n_communities  # Add remainder to last community
    
    # Generate probabilities
    p_in = np.random.uniform(*p_in_range)
    p_out = np.random.uniform(*p_out_range)
    
    # Create probability matrix
    p_matrix = np.full((n_communities, n_communities), p_out)
    np.fill_diagonal(p_matrix, p_in)
    
    # Generate SBM graph
    graph = nx.stochastic_block_model(sizes, p_matrix)
    
    # Convert to binary adjacency matrix
    adj_matrix = np.zeros((n_nodes, n_nodes))
    
    # Add binary edges
    for i, j in graph.edges():
        adj_matrix[i, j] = 1
        adj_matrix[j, i] = 1  # Symmetric
    
    return adj_matrix

def evolve_graph(prev_adj, change_prob=0.15):
    """
    Evolve the binary graph for temporal consistency.
    
    Args:
        prev_adj: Previous adjacency matrix
        change_prob: Probability of edge change
    """
    n_nodes = prev_adj.shape[0]
    new_adj = prev_adj.copy()
    
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):  # Upper triangle only due to symmetry
            if np.random.random() < change_prob:
                # Flip edge state (0->1 or 1->0)
                new_adj[i, j] = 1 - prev_adj[i, j]
                new_adj[j, i] = new_adj[i, j]  # Symmetric
    
    return new_adj

def generate_temporal_dataset(n_timesteps, n_nodes, n_communities, output_dir):
    """
    Generate temporal binary SBM dataset with community structure.
    
    Args:
        n_timesteps: Number of time steps
        n_nodes: Number of nodes
        n_communities: Number of communities
        output_dir: Directory to save the edge lists
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate initial snapshot
    adj_matrix = generate_sbm_snapshot(
        n_nodes=n_nodes,
        n_communities=n_communities,
        p_in_range=(0.7, 0.9),    # High intra-community connectivity
        p_out_range=(0.1, 0.2)    # Low inter-community connectivity
    )
    
    # Save initial snapshot
    save_edge_list(adj_matrix, os.path.join(output_dir, f"edge_list_0.txt"))
    
    # Generate and save subsequent snapshots
    for t in range(1, n_timesteps):
        # Evolve graph with temporal consistency
        adj_matrix = evolve_graph(
            adj_matrix,
            change_prob=0.15        # 15% chance of edge flipping
        )
        
        # Save snapshot
        save_edge_list(adj_matrix, os.path.join(output_dir, f"edge_list_{t}.txt"))
        
        if t % 100 == 0:
            print(f"Generated timestep {t}/{n_timesteps}")

def save_edge_list(adj_matrix, filename):
    """Save adjacency matrix as edge list with binary edges."""
    n_nodes = adj_matrix.shape[0]
    with open(filename, 'w') as f:
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):  # Upper triangle only due to symmetry
                if adj_matrix[i, j] > 0:
                    f.write(f"{i} {j} 1\n")

if __name__ == "__main__":
    # Parameters matching UCSB structure but with binary edges
    N_TIMESTEPS = 1000  # Same as paper
    N_NODES = 38      # Same as UCSB
    N_COMMUNITIES = 4  # Creates reasonable community sizes
    
    # Create data directory if it doesn't exist
    output_dir = Path("data/SBM")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating binary temporal SBM dataset with community structure...")
    print(f"Nodes: {N_NODES}")
    print(f"Communities: {N_COMMUNITIES}")
    print(f"Timesteps: {N_TIMESTEPS}")
    print(f"Output directory: {output_dir}")
    
    generate_temporal_dataset(
        n_timesteps=N_TIMESTEPS,
        n_nodes=N_NODES,
        n_communities=N_COMMUNITIES,
        output_dir=str(output_dir)
    )
    
    print("Dataset generation complete!") 