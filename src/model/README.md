<i>This file contains the documentation for (1) Dataset and (2) Model architectures.</i>

# 1. Dynamic Graph Dataset

The dataset contains sequences of evolving graphs with the following characteristics:

- **Graph Types**: BA (Barabási-Albert), ER (Erdős-Rényi), NW (Newman-Watts)
- **Sequence Length**: 200 timesteps per sequence
- **Number of Nodes**: 30 nodes per graph
- **Number of Sequences**: 300 sequences per graph type (900 total)

## Feature Variants

The dataset supports three feature variants:

1. **Node Level** (`node_level`):
   - 6 features per node
   - Shape: `[sequence_length, num_nodes, 6]`
   - Features include: degree, betweenness, eigenvector, closeness, SVD, LSVD
   - Best for: Learning node-level patterns and local structure changes

2. **Global** (`global`):
   - 906 features (900 flattened adjacency + 6 global features)
   - Shape: `[sequence_length, 906]`
   - Best for: Learning global graph patterns and structural changes

3. **Combined** (`combined`):
   - 36 features (30 node-specific + 6 global features)
   - Shape: `[sequence_length, num_nodes, 36]`
   - Best for: Learning both local and global patterns simultaneously

## Temporal Processing

The dataset uses a sliding window approach for temporal processing:

- **Window Size**: 10 timesteps
- **Stride**: 1 timestep
- **Sample Creation**: For each sequence of length 200:
  - Creates (200 - window_size) = 190 samples
  - Each sample contains 10 consecutive timesteps
  - Predicts the graph structure at timestep t+1

Example of temporal window:

**Input window** $[t_0, t_1, t_2, ..., t_9] \rightarrow \text{Predict } t_{10}$

**Slide by stride = 1:**

$$[t_1, t_2, t_3, ..., t_{10}] \rightarrow \text{Predict } t_{11}$$
$$[t_2, t_3, t_4, ..., t_{11}] \rightarrow \text{Predict } t_{12}$$
$$[t_3, t_4, t_5, ..., t_{12}] \rightarrow \text{Predict } t_{13}$$
$$...$$
$$[t_{189}, t_{190}, t_{191}, ..., t_{199}] \rightarrow \text{Predict } t_{200}$$

## Data Format

Each sample contains:
- **Features**: `[temporal_window, num_nodes, num_features]`
  - Temporal sequence of node features
  - Normalized to range [0, 1]
- **Edge Indices**: List of `[2, num_edges]` for each timestep
  - COO format sparse representation
  - Each column represents an edge (source, target)
- **Edge Weights**: List of `[num_edges]` for each timestep
  - Weight values for each edge in edge_indices
- **Target**: Next timestep adjacency matrix `[num_nodes, num_nodes]`
  - Binary matrix representing graph at t+1
- **Metadata**: Graph type, change points, and parameters
  - Useful for analysis and evaluation

## Configuration Parameters

The dataset behavior is controlled through `dataset_config.yaml`:

```yaml
data:
    variants: ["node_level", "global", "combined"]  # Available feature types
    default_variant: "node_level"                   # Default if none specified

graph:
    num_nodes: 30                # Fixed number of nodes per graph
    types: ["BA", "ER", "NW"]    # Available graph models

features:
    node_level:
        dimension: 6             # Number of features per node
    global:
        dimension: 906           # Flattened adjacency (30x30=900) + 6 features
    combined:
        dimension: 36            # Node-specific (30) + global features (6)

processing:
    normalize_features: false    # Features are normalized during generation
    temporal_window: 10          # Number of timesteps to consider for prediction
    stride: 1                    # Step size for sliding window

training:
    batch_size: 32             # Number of samples per batch
    num_workers: 4             # Parallel data loading processes
    train_ratio: 0.7           # Proportion of data for training
    val_ratio: 0.15            # Proportion of data for validation
```

## Usage Examples

### Basic Usage
```python
from dataset import DynamicGraphDataset

# Initialize dataset
dataset = DynamicGraphDataset(
    variant="node_level",  # Feature type to use
    graph_type="BA"        # Optional: specific graph type
)

# Create dataloader
dataloader = dataset.get_dataloader(
    batch_size=32,        # Samples per batch
    shuffle=True          # Randomize sample order
)

# Get train/val/test splits
train_idx, val_idx, test_idx = dataset.get_train_val_test_split(seed=42)
```

### Batch Contents
Each batch from the dataloader contains:
```python
batch = {
    'features': torch.FloatTensor,      # Shape: [batch_size, temporal_window, num_nodes, features]
    'edge_indices': List[torch.Tensor], # Length: batch_size, each [2, num_edges]
    'edge_weights': List[torch.Tensor], # Length: batch_size, each [num_edges]
    'targets': torch.FloatTensor,       # Shape: [batch_size, num_nodes, num_nodes]
    'metadata': List[Dict]              # Length: batch_size
}
```

## Statistics

Typical dataset statistics:

### All Graph Types
- **Total Samples**: 171,000 (900 sequences × 190 samples)
- **Edge Density**: ~15.55%
- **Feature Range**: [0, 1] (normalized)
- **Memory Usage**: ~2-4GB when loaded

### BA Graphs Only
- **Total Samples**: 57,000 (300 sequences × 190 samples)
- **Edge Density**: ~23.22%
- **Characteristics**: Higher average feature values, scale-free structure

## Notes
- Features are pre-normalized during dataset generation
- Edge indices use sparse COO format for memory efficiency
- Temporal windows create overlapping samples for better temporal pattern learning
- Batch processing is optimized for GPU training


# 2. Graph Neural Network (GNN) Models