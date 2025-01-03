# Network Predictors Documentation

This document provides detailed documentation for each network predictor implementation, including their mathematical foundations, algorithms, and practical considerations.

## Table of Contents
1. [WeightedPredictor](#weightedpredictor)
2. [SpectralPredictor](#spectralpredictor)
3. [EmbeddingPredictor](#embeddingpredictor)
4. [DynamicalPredictor](#dynamicalpredictor)
5. [EnsemblePredictor](#ensemblepredictor)
6. [AdaptivePredictor](#adaptivepredictor)
7. [HybridPredictor](#hybridpredictor)

## WeightedPredictor

### Simple Technical Explanation
To predict the next network state(s), the model:
1. Looks at the last 3 network states (by default)
2. Creates a probability for each possible edge by:
   - Taking each edge's presence/absence from these 3 states
   - Weighing the present state by 0.5, previous by 0.3, and oldest by 0.2
   - For example: if an edge exists in all 3 states, it gets probability = 1.0 (0.5 + 0.3 + 0.2)
3. Targets the exact same average degree as the most recent network state (not from all 3)
   - For example: if the latest network has average degree 4, the prediction will also have average degree 4
4. Builds the predicted network by:
   - Adding edges with highest probabilities first
   - Stopping when it reaches the target average degree
   - Making sure the network stays as one connected piece

For multi-step predictions (m > 1):
- After predicting one step, that prediction becomes the new "present state"
- The process repeats, sliding the 3-state window forward
- Each new prediction uses the previous predictions as part of its history

### Overview
The WeightedPredictor implements a weighted averaging approach for network prediction, maintaining network properties while ensuring connectivity. It uses the most recent network states to predict future states, with configurable weights and history length.

### Hyperparameters

1. **n_history** (int, default=3)
   - Number of historical network states to consider
   - Controls the temporal window for prediction

2. **weights** (np.ndarray, default=[0.5, 0.3, 0.2])
   - Weights for historical points from newest to oldest
   - Automatically normalized to sum to 1
   - Default values favor recent history with decreasing importance

### Prediction Algorithm

The prediction process consists of four main steps:

#### 1. Historical State Processing
```python
# Get most recent n_history networks
last_networks = history[-n_history:]

# Extract target properties from latest network
latest_network = last_networks[-1]["graph"]
target_degrees = sorted([d for _, d in latest_network.degree()])
target_avg_degree = np.mean(target_degrees)
```

#### 2. Weighted Average Computation
```python
# Initialize average matrix
avg_adj = np.zeros_like(adjacency_matrices[0], dtype=float)

# Compute weighted sum
for adj, weight in zip(adjacency_matrices, weights):
    avg_adj += weight * adj.astype(float)

# Normalize to [0,1] range
avg_adj = (avg_adj - avg_adj.min()) / (avg_adj.max() - avg_adj.min() + 1e-10)
```

#### 3. Network Generation
```python
# Calculate target number of edges
n = prob_matrix.shape[0]
target_edges = int((target_avg_degree * n) / 2)

# Get upper triangular probabilities
triu_indices = np.triu_indices(n, k=1)
edge_probs = prob_matrix[triu_indices]
edge_indices = list(zip(triu_indices[0], triu_indices[1]))

# Sort edges by probability
sorted_edges = sorted(zip(edge_probs, edge_indices), reverse=True)

# Add highest probability edges until target density reached
predicted_adj = np.zeros_like(prob_matrix, dtype=int)
for _, (i, j) in sorted_edges[:target_edges]:
    predicted_adj[i, j] = predicted_adj[j, i] = 1
```

#### 4. Connectivity Enforcement
```python
# Check and fix connectivity
g_temp = nx.from_numpy_array(predicted_adj)
components = list(nx.connected_components(g_temp))

if len(components) > 1:
    main_comp = max(components, key=len)
    other_comps = [c for c in components if c != main_comp]
    
    # Connect components using highest probability edges
    for comp in other_comps:
        best_edge = None
        best_prob = -1
        
        for n1 in main_comp:
            for n2 in comp:
                prob = prob_matrix[n1, n2]
                if prob > best_prob:
                    best_prob = prob
                    best_edge = (n1, n2)
                    
        if best_edge:
            i, j = best_edge
            predicted_adj[i, j] = predicted_adj[j, i] = 1
```

### Key Features

1. **Property Preservation**
   - Maintains average degree from the latest network
   - Preserves network connectivity
   - Symmetric adjacency matrix generation

2. **Probability-based Edge Selection**
   - Edges selected based on weighted historical presence
   - Higher weights for recent observations
   - Normalized probability computation

3. **Connectivity Guarantee**
   - Ensures single connected component
   - Uses probability-guided component linking
   - Minimal additional edges for connectivity

### Usage Example
```python
# Initialize with custom history length and weights
predictor = WeightedPredictor(
    n_history=4,
    weights=np.array([0.4, 0.3, 0.2, 0.1])
)

# Make multi-step prediction
future_states = predictor.predict(
    history=network_history,
    horizon=5  # Predict 5 steps ahead
)
```

### Implementation Notes

1. **Memory Management**
   - Creates copy of history for prediction
   - Updates history for multi-step predictions
   - Efficient sparse matrix operations

2. **Numerical Stability**
   - Small epsilon (1e-10) in normalization
   - Integer type for final adjacency matrix
   - Float type for intermediate calculations

3. **Performance Considerations**
   - O(E log E) sorting for edge selection
   - O(C₁C₂) connectivity fixing, where C₁,C₂ are component sizes
   - Matrix operations vectorized using numpy

---

[More predictors to be documented...]
