# Synthetic Graph Sequence Generator

Generates sequences of evolving graphs with change points for training and evaluating graph change detection algorithms.

## Graph Types
- Barabási-Albert (BA)
- Erdős-Rényi (ER) 
- Newman-Watts (NW)

## Configuration
Configuration files in `configs/`:
- `graph_config.yaml`: Graph generation parameters
- `martingale_config.yaml`: Martingale computation parameters  
- `dataset_config.yaml`: Dataset creation parameters

## Graph Generation Process

1. For each sequence:
   - Randomly sample sequence length from `[min_seq_length, max_seq_length]`
   - Randomly sample number of nodes from `[min_n, max_n]`
   - Generate random number of change points from `[min_changes, max_changes]`
   - Ensure minimum segment length between changes

2. Parameter Changes:
   - Initial parameters set from config
   - At each change point, parameters randomly sampled from configured ranges:
     - BA: Number of edges ($m$)
     - ER: Connection probability ($p$)
     - NW: Both $k$ and $p$ parameters
   - Changes can range from subtle to dramatic

3. Visualization:
   - Dashboard shows graph structure evolution
   - Plots centrality, embedding, structural and spectral metrics
   - Limited to 4 representative graphs for clarity
