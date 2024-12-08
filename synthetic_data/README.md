# Synthetic Graph Sequence Generator

This framework generates time-evolving graph sequences with controlled structural changes, suitable for training and evaluating graph-based forecasting models.

## Graph Types
- **Barabási-Albert (BA):** Scale-free networks with hubs and a power-law degree distribution.
- **Erdős-Rényi (ER):** Random graphs with edges formed independently with a fixed probability.
- **Newman-Watts (NW):** Small-world networks that balance high clustering with short path lengths.

## Dataset Structure

### Sequence Properties
- **Fixed Node Count:** Each graph has a constant number of nodes (e.g., 50), ensuring consistent dimensionality.
- **Fixed Sequence Length:** Each sequence is exactly 200 timesteps long.
- **Controlled Change Points:** Each sequence includes 1-4 structural change points at known timesteps, with a minimum segment length between changes for stable pattern observation.

### Data at Each Timestep
For each timestep in a sequence, the dataset provides:
1. **Adjacency Matrix $(n \times n):$** The full graph connectivity.
2. **Global Feature Vector (6-D):** A single 6-dimensional vector summarizing the entire graph’s properties. These features are:
   - Degree centrality (aggregated)
   - Betweenness centrality (aggregated)
   - Eigenvector centrality (aggregated)
   - Closeness centrality (aggregated)
   - SVD-based embedding (aggregated)
   - LSVD-based embedding (aggregated)

   Each of these node-level metrics is computed for every node, then averaged over nodes, yielding a single scalar per feature per timestep. The result is a 6-feature vector representing the global state of the graph at that instant.

### Normalization and Splits
- **Normalization:**  
  Normalization parameters (mean, std) are computed from the training set only, ensuring no data leakage. This results in the training set features having approximately zero mean and unit variance. Validation and test sets, when normalized with the training statistics, may not perfectly center at zero or have exactly unit variance, reflecting realistic distribution shifts.

- **Data Splits:**  
  The dataset is split into training, validation, and test sets. For example:
  - Training: 70%
  - Validation: 15%
  - Test: 15%

  This allocation ensures proper model training, tuning, and unbiased evaluation.

### Example Shapes
- **Features:** `(num_sequences, 200, 6)`  
  For each sequence (200 timesteps), a 6-dimensional feature vector is available.
  
- **Graphs:** `(num_sequences, 200, n_nodes, n_nodes)`  
  Each sequence has 200 adjacency matrices, each representing the graph’s structure at a given timestep.

### Configuration Files
- `graph_config.yaml`: Parameters for graph generation (e.g., number of nodes, parameters for BA/ER/NW models, range of segment lengths).
- `dataset_config.yaml`: Parameters for sequence generation, including how many sequences to generate, how to introduce change points, and how to split the data.

## Use Cases
The generated dataset can be used to:
- Train forecasting models to predict future global graph features.
- Investigate how structural changes affect the predictive task.
- Develop and benchmark graph-based models that leverage both the graph structure (adjacency) and global features over time.

By providing both the underlying adjacency matrix and the global feature vector per timestep, this dataset supports a wide range of experiments in graph forecasting and anomaly detection tasks.
