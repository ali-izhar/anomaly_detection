# Synthetic Graph Sequence Dataset Generator

This module generates synthetic graph sequences with change points and extracts various graph features. It supports multiple graph types and feature extraction methods.

## Overview

The generator creates sequences of graphs with structural changes at random points. Each sequence consists of graphs that evolve over time, with sudden changes in their generating parameters at certain timestamps. The changes are significant (at least 30% difference in parameters) to ensure detectable structural shifts.

### Supported Graph Types
1. **BA (Barabási-Albert)**
   - Scale-free networks
   - Parameters: number of edges $(m)$
   - Changes: Edge count varies between `min_edges` and `max_edges`

2. **ER (Erdős-Rényi)**
   - Random networks
   - Parameters: connection probability $(p)$
   - Changes: Probability varies between `min_p` and `max_p`

3. **NW (Newman-Watts)**
   - Small-world networks
   - Parameters: nearest neighbors $(k)$ and rewiring probability $(p)$
   - Changes: Either $k$ or $p$ changes significantly

### Feature Types
1. **Node-level Features**:
   - Degree centrality: Local connectivity measure
   - Betweenness centrality: Path-based importance
   - Eigenvector centrality: Influence measure
   - Closeness centrality: Distance-based centrality
   - SVD embeddings: Spectral feature
   - LSVD embeddings: Laplacian-based feature

2. **Global Features**:
   - Average of node-level features across all nodes
   - Flattened adjacency matrices for structural information

## Dataset Variants

The generator creates three types of datasets:

1. **Node-level Dataset**
   - Content: Pure node-level features
   - Shape: `(seq_len, n_nodes, 6)`
   - Features: 6 features per node
   - Use case: Node-level analysis and prediction

2. **Global Dataset**
   - Content: Global features + flattened adjacency
   - Shape: `(seq_len, n_nodes * n_nodes + 6)`
   - Features: Adjacency (n_nodes $^2$) + 6 global features
   - Use case: Graph-level analysis

3. **Combined Dataset**
   - Content: Node features + adjacency information
   - Shape: `(seq_len, n_nodes, n_nodes + 6)`
   - Features: Adjacency (n_nodes) + 6 node features
   - Use case: Combined structural and feature analysis

## Configuration

### Graph Configuration
```yaml
# configs/graph_config.yaml
common:
    nodes: 50            # Number of nodes in each graph
    seq_len: 200         # Length of each sequence
    min_segment: 30      # Minimum length between changes
    min_changes: 1       # Minimum number of change points
    max_changes: 4       # Maximum number of change points

ba:
    edges: 3             # Initial number of edges
    min_edges: 2         # Minimum edges after change
    max_edges: 8         # Maximum edges after change

er:
    p: 0.05             # Initial connection probability
    min_p: 0.03         # Minimum probability
    max_p: 0.15         # Maximum probability

nw:
    k: 4                # Initial nearest neighbors
    p: 0.05             # Initial rewiring probability
    min_k: 2            # Minimum neighbors
    max_k: 8            # Maximum neighbors
    min_p: 0.02         # Minimum rewiring probability
    max_p: 0.20         # Maximum rewiring probability
```

### Dataset Configuration
```yaml
# configs/dataset_config.yaml
dataset:
    num_sequences: 300   # Sequences per graph type
    graph_types: ["BA", "ER", "NW"]
    
    features:
        use_node_features: true   # Use node-level features
        normalize: true           # Normalize features to [0,1]
        types:                    # Feature types to extract
            - degree
            - betweenness
            - eigenvector
            - closeness
            - svd
            - lsvd
    
    split_ratio:              # Dataset split ratios
        train: 0.7
        val: 0.15
        test: 0.15
    
    output_settings:
        output_dir: "dataset"
        save_format: "h5"     # Options: h5, npz
        compression: true     # Use compression
```

## Usage

### 1. Generate Datasets
```bash
# Generate all variants with default settings
python main.py --output datasets

# Generate with custom configurations
python main.py \
    --base-config configs/custom_dataset_config.yaml \
    --graph-config configs/custom_graph_config.yaml \
    --output custom_datasets
```

### 2. Inspect Datasets
```bash
# Basic inspection
python inspect_dataset.py \
    --dataset datasets/node_level/dataset.h5 \
    --output analysis

# This creates:
analysis/
├── statistics/                     # Statistical analysis
│   ├── basic_stats.txt             # Dataset statistics
│   └── change_point_stats.txt      # Change point analysis
├── plots/                          # Visualizations
│   ├── feature_distributions.png   # Feature distributions
│   ├── change_point_distribution.png
│   └── {graph_type}_evolution.png  # Graph evolution
└── element_inspection/             # Detailed inspections
    └── {graph_type}_seq{i}_time{t}_inspection.png
```

### 3. Python API
```python
# Generate dataset
from create_dataset import create_dataset
dataset = create_dataset(
    config_path="configs/dataset_config.yaml",
    graph_config_path="configs/graph_config.yaml"
)

# Inspect dataset
from inspect_dataset import DatasetInspector
inspector = DatasetInspector("datasets/combined/dataset.h5")

# Get statistics
stats = inspector.get_basic_stats()
cp_stats = inspector.analyze_change_points()

# Visualize specific elements
inspector.inspect_sequence_element(
    graph_type="BA",
    seq_idx=0,
    time_idx=50
)
```

## Output Format

### HDF5 Structure
```
dataset.h5
├── BA/                           # Graph type group
│   ├── sequences/                # Sequence data
│   │   ├── seq_0/                # Individual sequence
│   │   │   ├── adjacency         # Shape: (seq_len, n_nodes, n_nodes)
│   │   │   └── features          # Shape depends on variant
│   │   └── ...
│   ├── change_points/            # Change point locations
│   │   ├── seq_0                 # Points for sequence 0
│   │   └── ...
│   └── params/                   # Generation parameters
│       ├── seq_0                 # Parameters for sequence 0
│       └── ...
├── ER/                           # Similar structure for ER
└── NW/                           # Similar structure for NW
```
