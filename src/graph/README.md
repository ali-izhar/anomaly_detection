# Graph Generators and Feature Extraction

This module provides functionality for generating dynamic graph sequences and computing graph features for anomaly detection. Built on top of [NetworkX](https://networkx.org/), it supports multiple graph models and feature extraction methods.

## Graph Models

### Barabási-Albert (BA)
- **Description**: Scale-free networks following preferential attachment
- **Parameters**:
  - `n`: Number of nodes
  - `initial_edges`: Number of edges each new node creates ($m$ parameter)
  - `pref_exp`: Preferential attachment exponent ($\alpha$ parameter)
- **Properties**: Power-law degree distribution $P(k) \sim k^{-\alpha}$
- **NetworkX Implementation**: [`barabasi_albert_graph`](https://networkx.org/documentation/stable/reference/generated/networkx.generators.random_graphs.barabasi_albert_graph.html)

### Erdős-Rényi (ER)
- **Description**: Random graphs with fixed edge probability
- **Parameters**:
  - `n`: Number of nodes
  - `initial_prob`: Edge probability ($p$ parameter)
- **Properties**: Poisson degree distribution
- **NetworkX Implementation**: [`erdos_renyi_graph`](https://networkx.org/documentation/stable/reference/generated/networkx.generators.random_graphs.erdos_renyi_graph.html)

### Stochastic Block Model (SBM)
- **Description**: Random graphs with community structure
- **Parameters**:
  - `n`: Number of nodes
  - `num_blocks`: Number of communities
  - `initial_intra_prob`: Edge probability within communities
  - `initial_inter_prob`: Edge probability between communities
- **Properties**: Modular structure with dense intra-community and sparse inter-community connections
- **NetworkX Implementation**: [`stochastic_block_model`](https://networkx.org/documentation/stable/reference/generated/networkx.generators.community.stochastic_block_model.html)

## Feature Extraction

### Centrality Measures
- **Degree Centrality**: Fraction of nodes a node is connected to
  ```python
  nx.degree_centrality(G)
  ```
- **Betweenness Centrality**: Fraction of shortest paths passing through a node
  ```python
  nx.betweenness_centrality(G)
  ```
- **Eigenvector Centrality**: Node importance based on neighbor importance
  ```python
  nx.eigenvector_centrality(G)
  ```

### Graph Embeddings
- **SVD Embeddings**: Singular value decomposition of adjacency matrix
  - Parameters: `n_components`, `use_sparse`
  - Implementation: `sknetwork.embedding.SVD`
  - Properties: Preserves global graph structure

### Graph Statistics
- **Density**: Ratio of actual to possible edges
  ```python
  nx.density(G)
  ```
- **Average Degree**: Mean number of edges per node
  ```python
  np.mean([d for _, d in G.degree()])
  ```
- **Clustering Coefficient**: Fraction of closed triangles
  ```python
  nx.average_clustering(G)
  ```
- **Diameter**: Maximum shortest path length
  ```python
  nx.diameter(G)
  ```

## Testing Framework

### Graph Generation Tests (`TestGraphGeneration`)
1. **Model-Specific Tests**:
   - `test_ba_generation`: Verifies scale-free properties
   - `test_er_generation`: Checks edge probability distribution
   - `test_sbm_generation`: Validates community structure

2. **Change Point Tests**:
   - `test_change_points`: Ensures valid segment lengths and ordering
   - Parameters: `min_segment`, `min_changes`, `max_changes`

### Feature Extraction Tests (`TestFeatureExtraction`)
1. **Centrality Tests**:
   - Verifies value ranges [0,1]
   - Checks computation for multiple measures

2. **Embedding Tests**:
   - Validates orthogonality
   - Checks dimensionality reduction

3. **Graph Statistics Tests**:
   - Verifies basic properties (density, clustering)
   - Ensures valid ranges for metrics

### Performance Tests (`TestPerformance`)
- Tests for large graphs (1000+ nodes)
- Benchmarks parallel computation
- Validates sparse matrix operations

## Usage

### Graph Generation
```python
generator = GraphGenerator()
generator.register_model("BA", nx.barabasi_albert_graph, BAParams)

sequence = generator.generate_sequence(
    model="BA",
    params=BAParams(n=100, initial_edges=3),
    seed=42
)
```

### Feature Extraction
```python
# Compute centralities
centralities = extract_centralities(
    graphs,
    measures=["degree", "betweenness"],
    batch_size=10,
    n_jobs=-1
)

# Generate embeddings
embeddings = compute_embeddings(
    graphs,
    method="svd",
    n_components=10,
    use_sparse=True
)
```

### Example
```python
(venv) PS D:\github\research\anomaly_detection\src\graph> python -m unittest .\main.py
INFO:features:Computing 2 centralities for 10 graphs
INFO:features:Centrality computation complete
.INFO:features:Computing SVD embeddings with 5 components
INFO:features:Successfully computed embeddings for 10 graphs
.INFO:features:Computing statistics for 10 graphs
INFO:features:Statistics computation complete
..INFO:features:Computing SVD embeddings with 5 components
INFO:features:Successfully computed embeddings for 10 graphs
.INFO:graph_generator:Registered model: BA
INFO:graph_generator:Registered model: ER
INFO:graph_generator:Registered model: SBM
INFO:graph_generator:Generated 1 change points at: [83]
INFO:graph_generator:Successfully generated sequence with 100 graphs
.INFO:graph_generator:Registered model: BA
INFO:graph_generator:Registered model: ER
INFO:graph_generator:Registered model: SBM
INFO:graph_generator:Generated 1 change points at: [36]
INFO:graph_generator:Successfully generated sequence with 100 graphs
INFO:main:Change points: [36]
INFO:main:Segment lengths: [36, 64]
INFO:main:Sequence length: 100, Min segment: 10
INFO:main:Min changes: 1, Max changes: 3
.INFO:graph_generator:Registered model: BA
INFO:graph_generator:Registered model: ER
INFO:graph_generator:Registered model: SBM
INFO:graph_generator:Generated 1 change points at: [41]
INFO:graph_generator:Successfully generated sequence with 100 graphs
.INFO:graph_generator:Registered model: BA
INFO:graph_generator:Registered model: ER
INFO:graph_generator:Registered model: SBM
INFO:graph_generator:Generated 3 change points at: [40, 51, 77]
INFO:graph_generator:Successfully generated sequence with 100 graphs
.INFO:features:Computing 1 centralities for 5 graphs
INFO:features:Centrality computation complete
INFO:main:Centrality computation took 0.08 seconds
.INFO:features:Computing SVD embeddings with 10 components
INFO:features:Successfully computed embeddings for 5 graphs
INFO:main:Embedding computation took 0.18 seconds
.
----------------------------------------------------------------------
Ran 11 tests in 1.792s

OK
```

#### Test Output Explanation

1. **Feature Extraction Tests**:
   ```
   INFO:features:Computing 2 centralities for 10 graphs
   INFO:features:Centrality computation complete
   ```
   - Testing centrality computation on 10 test graphs
   - Computing degree and betweenness centralities

2. **Embedding Tests**:
   ```
   INFO:features:Computing SVD embeddings with 5 components
   INFO:features:Successfully computed embeddings for 10 graphs
   ```
   - Testing SVD embedding generation
   - Reducing to 5 dimensions for each graph

3. **Graph Statistics Tests**:
   ```
   INFO:features:Computing statistics for 10 graphs
   INFO:features:Statistics computation complete
   ```
   - Computing density, clustering, and diameter metrics
   - Validating basic graph properties

4. **Model Registration**:
   ```
   INFO:graph_generator:Registered model: BA
   INFO:graph_generator:Registered model: ER
   INFO:graph_generator:Registered model: SBM
   ```
   - Registering three graph models with their parameters
   - Setting up generator functions for each model

5. **Change Point Generation**:
   ```
   INFO:graph_generator:Generated 1 change points at: [83]
   INFO:graph_generator:Successfully generated sequence with 100 graphs
   ```
   - Testing sequence generation with parameter changes
   - Validating change point placement and sequence length

6. **Segment Validation**:
   ```
   INFO:main:Change points: [36]
   INFO:main:Segment lengths: [36, 64]
   INFO:main:Sequence length: 100, Min segment: 10
   INFO:main:Min changes: 1, Max changes: 3
   ```
   - Verifying segment lengths meet minimum requirements
   - Checking change point constraints are satisfied

7. **Performance Tests**:
   ```
   INFO:features:Computing 1 centralities for 5 graphs
   INFO:main:Centrality computation took 0.08 seconds
   INFO:features:Computing SVD embeddings with 10 components
   INFO:main:Embedding computation took 0.18 seconds
   ```
   - Benchmarking centrality computation speed
   - Testing embedding generation performance
   - Validating computation times are within bounds

8. **Test Summary**:
   ```
   Ran 11 tests in 1.792s
   OK
   ```
   - All 11 test cases passed successfully
   - Total execution time under 2 seconds
   - No failures or errors encountered

Each dot (`.`) in the output represents a successfully completed test case. The tests cover feature extraction, graph generation, change point detection, and performance benchmarks.

