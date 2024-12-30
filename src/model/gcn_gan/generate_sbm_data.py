"""
Generate large-scale Stochastic Block Model dataset with temporal dynamics
"""

import os
import numpy as np
import networkx as nx
from pathlib import Path
from tqdm import tqdm

def generate_sbm_graph(n_nodes, n_communities, p_in, p_out, min_size=8, max_size=12):
    """Generate a single SBM graph."""
    # Calculate community sizes
    sizes = np.random.randint(min_size, max_size + 1, n_communities)
    # Adjust last community size to match total nodes
    sizes[-1] = n_nodes - sum(sizes[:-1])
    
    # Create probability matrix
    p_matrix = np.full((n_communities, n_communities), p_out)
    np.fill_diagonal(p_matrix, p_in)
    
    # Generate graph
    G = nx.stochastic_block_model(sizes, p_matrix)
    return G

def save_edge_list(G, filepath):
    """Save graph as edge list with binary edges."""
    with open(filepath, 'w') as f:
        for u, v in G.edges():
            f.write(f"{u} {v} 1\n")

def generate_temporal_dataset(output_dir, n_timesteps=5000, n_nodes=38, n_communities=4, p_in_range=(0.6, 0.8), p_out_range=(0.05, 0.2), change_prob=0.15, max_change=0.03):
    """Generate temporal SBM dataset with smooth transitions."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize probabilities at the middle of their ranges
    p_in = np.mean(p_in_range)
    p_out = np.mean(p_out_range)
    
    print(f"Generating {n_timesteps} timesteps of SBM data...")
    print(f"Initial p_in: {p_in:.3f}, p_out: {p_out:.3f}")
    
    for t in tqdm(range(n_timesteps)):
        # Randomly update probabilities with smooth transitions
        if np.random.random() < change_prob:
            # Calculate target probabilities
            target_p_in = np.random.uniform(*p_in_range)
            target_p_out = np.random.uniform(*p_out_range)
            
            # Move current probabilities toward targets
            p_in += np.clip(target_p_in - p_in, -max_change, max_change)
            p_out += np.clip(target_p_out - p_out, -max_change, max_change)
            
            # Ensure probabilities stay within their ranges
            p_in = np.clip(p_in, p_in_range[0], p_in_range[1])
            p_out = np.clip(p_out, p_out_range[0], p_out_range[1])
        
        # Generate and save graph
        G = generate_sbm_graph(n_nodes, n_communities, p_in, p_out)
        save_edge_list(G, output_dir / f"edge_list_{t}.txt")
    
    # Save dataset parameters
    with open(output_dir / "dataset_info.txt", "w") as f:
        f.write(f"Number of timesteps: {n_timesteps}\n")
        f.write(f"Number of nodes: {n_nodes}\n")
        f.write(f"Number of communities: {n_communities}\n")
        f.write(f"p_in range: {p_in_range}\n")
        f.write(f"p_out range: {p_out_range}\n")
        f.write(f"Change probability: {change_prob}\n")
        f.write(f"Maximum change per step: {max_change}\n")
        f.write(f"Final p_in: {p_in:.3f}\n")
        f.write(f"Final p_out: {p_out:.3f}\n")

def main():
    # Parameters aligned with paper's methodology
    output_dir = "./data/SBM"
    n_timesteps = 1000  # Reduced for faster training/testing
    n_nodes = 38       # Keep consistent with model
    n_communities = 4  # As per paper's community structure
    
    # Update probability ranges for more realistic community structure
    p_in_range = (0.6, 0.8)    # Higher intra-community probability
    p_out_range = (0.05, 0.2)  # Lower inter-community probability
    change_prob = 0.15         # Slightly more dynamic
    max_change = 0.03         # Smoother transitions
    
    print("Starting SBM dataset generation...")
    print(f"Output directory: {output_dir}")
    print(f"Number of timesteps: {n_timesteps}")
    print(f"Number of nodes: {n_nodes}")
    print(f"Number of communities: {n_communities}")
    print(f"Intra-community probability range: {p_in_range}")
    print(f"Inter-community probability range: {p_out_range}")
    print(f"Change probability: {change_prob}")
    print(f"Maximum change per step: {max_change}")
    
    generate_temporal_dataset(
        output_dir=output_dir,
        n_timesteps=n_timesteps,
        n_nodes=n_nodes,
        n_communities=n_communities,
        p_in_range=p_in_range,
        p_out_range=p_out_range,
        change_prob=change_prob,
        max_change=max_change
    )
    
    print("\nDataset generation complete!")

if __name__ == "__main__":
    main() 