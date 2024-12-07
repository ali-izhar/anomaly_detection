# Synthetic Graph Sequence Generator

A framework for generating evolving graph sequences with controlled structural changes for training and evaluating graph-based machine learning models.

## Graph Types
- **Barabási-Albert (BA):** Scale-free networks characterized by hub formation and power-law degree distribution
- **Erdős-Rényi (ER):** Random graphs with binomial degree distribution
- **Newman-Watts (NW):** Small-world networks balancing clustering and path length

## Dataset Structure

### Sequence Properties
- Fixed node count $(n=100)$ to ensure consistent feature dimensionality
- Variable sequence lengths `[150, 200]` timesteps
- 2-3 structural change points per sequence
- Minimum 40 timesteps between changes to establish stable patterns

### Features
1. Node-level Features `(shape: [batch_size, seq_len, num_nodes])`
   - Degree centrality: Direct connections per node
   - Betweenness centrality: Node importance in shortest paths
   - Closeness centrality: Inverse average distance to other nodes
   - Eigenvector centrality: Node influence based on neighbor importance

2. Graph Embeddings
   - Spectral embeddings (SVD): `[batch_size, seq_len, num_nodes, 2]`
   - Laplacian spectral embeddings (LSVD): `[batch_size, seq_len, num_nodes, 16]`

### Data Organization
- Training set (70%): Model parameter learning
- Validation set (15%): Hyperparameter optimization
- Test set (15%): Final model evaluation

## LSTM-GNN Architecture Guidelines

### Feature Processing
1. Input Handling
   - Centrality features: Direct GNN input for spatial patterns
   - Embeddings: Optional dimensionality reduction via attention
   - Per-feature normalization for numerical stability

2. Spatiotemporal Learning
   - GNN component: Node interaction patterns
   - LSTM component: Temporal feature evolution
   - Combined: Joint spatial-temporal prediction

### Training Recommendations

1. Dataset Generation
   - Scale: 1500 sequences (500 per graph type)
   - Distribution: Balanced across graph types
   - Variety: Diverse structural changes via parameter ranges

2. Feature Utilization
   - Core features: Centrality metrics
   - Supplementary: Graph embeddings
   - Normalized independently to preserve relative changes

3. Change Point Handling
   - Detection: Feature pattern shifts
   - Adaptation: Prediction adjustment at changes
   - Forecasting: Post-change evolution

### Dimensions Reference
```python
# Dataset Scale (1500 total sequences)
train_size = 1050  # 70% split
val_size = 225     # 15% split
test_size = 225    # 15% split

# Per-Sequence Dimensions
centrality_shape = (seq_len, num_nodes)        # (199, 100)
svd_shape = (seq_len, num_nodes, 2)            # (199, 100, 2)
lsvd_shape = (seq_len, num_nodes, 16)          # (199, 100, 16)

# Batch Processing Shapes
batch_centrality = (batch_size, seq_len, num_nodes)
batch_svd = (batch_size, seq_len, num_nodes, 2)
batch_lsvd = (batch_size, seq_len, num_nodes, 16)
```

## Configuration

The generator is controlled by three configuration files in `configs/`:
- `graph_config.yaml`: Graph generation parameters and ranges
- `dataset_config.yaml`: Sequence generation and splitting parameters
