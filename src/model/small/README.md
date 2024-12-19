# Small STGCN Model Documentation

## 1. Problem Overview

### Dynamic Graph Data
- Input: Temporal sequence of graphs $\{\mathcal{G}_t\}_{t=1}^T$ where $\mathcal{G}_t = (\mathcal{V}, \mathcal{E}_t)$
- Node features: $\mathbf{X}_t \in \mathbb{R}^{N \times F}$ where $N=30$ nodes, $F=6$ features
- Graph structure: $\mathbf{A}_t \in \{0,1\}^{N \times N}$ (adjacency matrix)
- Edge representation: COO format $\mathbf{E}_t \in \mathbb{R}^{2 \times |\mathcal{E}_t|}$

### Graph Representation
#### COO Format Example
For adjacency matrix:
$$
\mathbf{A} = \begin{bmatrix} 
0 & 1 & 1 \\
1 & 0 & 0 \\
1 & 0 & 0
\end{bmatrix}
$$

Corresponding COO representation:
- edge_index = $\begin{bmatrix} 0 & 0 & 1 & 2 \\ 1 & 2 & 0 & 0 \end{bmatrix}$
- Each column represents (source, target) node indices
- For undirected graphs, each edge appears twice (i→j and j→i)

### Prediction Task
Given window $[\mathcal{G}_{t-9}, ..., \mathcal{G}_t]$, predict $\mathbf{A}_{t+1}$

## 2. Data Processing

### Temporal Window Mechanism
```
Time sequence:
┌─────────────────────────────────┐
│ t0 t1 t2 t3 t4 t5 t6 t7 t8 t9  │ Window 1 → Predict t10
│    t1 t2 t3 t4 t5 t6 t7 t8 t9  │ Window 2 → Predict t11
│       t2 t3 t4 t5 t6 t7 t8 t9  │ Window 3 → Predict t12
└─────────────────────────────────┘
```

### Input Format
1. **Feature Tensor**: $\mathbf{X} \in \mathbb{R}^{B \times T \times N \times F}$
   - $B$: batch size (128)
   - $T$: sequence length (10)
   - $N$: nodes (30)
   - $F$: features (6)

2. **Graph Structure**: $\mathbf{E} \in \mathbb{R}^{2 \times E}$
   - Row 1: source nodes
   - Row 2: target nodes
   - $E$: number of edges

## 3. Model Architecture

### Key Design Choices
1. **Temporal Window Size $(T=10)$**:
   - Short enough to capture local temporal dependencies
   - Long enough to observe meaningful patterns
   - Common in video/sequence processing literature
   - Balances memory usage vs temporal context

2. **Feature Dimension Expansion $(6 \rightarrow 32)$**:
   - Initial features (6) capture basic node properties
   - Expansion to 32 channels allows:
     - More expressive node representations
     - Better separation in feature space
     - Learning hierarchical patterns
   - Similar to CNN channel expansion in computer vision

### STConv Block Implementation
```python
class STConv(nn.Module):
    def __init__(self, num_nodes, in_channels, hidden_channels, out_channels):
        self._temporal_conv1 = TemporalConv(in_channels, hidden_channels)
        self._graph_conv = ChebConv(hidden_channels, hidden_channels, K=2)
        self._temporal_conv2 = TemporalConv(hidden_channels, out_channels)
        self._batch_norm = BatchNorm2d(num_nodes)
```

### Processing Flow
1. **Temporal Convolution 1**:
   - Input: $\mathbf{H}_{\text{in}} \in \mathbb{R}^{B \times T \times N \times C}$
   - Operation: $\text{Conv1D}(k=3)$
   - Output: $\mathbf{H}_{\text{temp}} \in \mathbb{R}^{B \times (T-2) \times N \times C}$

2. **Graph Convolution (ChebConv)**:
   - Chebyshev polynomials up to order $K=2$
   - $\mathbf{H}_{\text{graph}} = \sum_{k=0}^K \mathbf{\Theta}_k T_k(\tilde{\mathbf{L}}) \mathbf{H}_{\text{temp}}$
   - $\tilde{\mathbf{L}} = 2\mathbf{L}/\lambda_{\text{max}} - \mathbf{I}_N$

3. **Dimension Transformations**:
```
Block 1: [B, 10, N, 6] → [B, 6, N, 32]
Block 2: [B, 6, N, 32] → [B, 2, N, 32]
Final: [B, 2, N, 32] → [B, N, N]
```

### Final Projection
1. Flatten: $\mathbf{H}_{\text{flat}} \in \mathbb{R}^{B \times (2N C)}$
2. Dense layers:
   - $\mathbf{H}_1 = \text{ReLU}(\mathbf{W}_1\mathbf{H}_{\text{flat}} + \mathbf{b}_1)$
   - $\mathbf{H}_2 = \mathbf{W}_2\mathbf{H}_1 + \mathbf{b}_2$
3. Reshape: $\mathbf{H}_2 \in \mathbb{R}^{B \times N \times N}$
4. Output: $\hat{\mathbf{A}}_{t+1} = \sigma(\mathbf{H}_2)$

## 4. Training Details

### Loss Function
Binary Cross Entropy:
$\mathcal{L} = -\frac{1}{N^2}\sum_{i,j} [A_{ij}\log(\hat{A}_{ij}) + (1-A_{ij})\log(1-\hat{A}_{ij})]$

### Optimization
```python
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    for batch in train_loader:
        # Get first timestep's graph structure
        edge_index = batch['edge_indices'][0][0]
        
        # Forward pass
        output = model(batch['features'], edge_index)
        loss = criterion(output, batch['targets'])
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### Learning Rate Schedule
```python
scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=5,
    min_lr=1e-6
)
```

## 5. Limitations & Further Reading

### Limitations
1. Fixed sequence length ($T=10$)
2. Quadratic memory in nodes ($\mathcal{O}(N^2)$)
3. Limited receptive field (ChebConv $K=2$)
4. No attention mechanism

### Resources
- [Understanding ChebConv (Video)](https://www.youtube.com/watch?v=Ghw-fp_2HFM)
- [ChebNet Paper](https://arxiv.org/abs/1606.09375)
- [Graph Convolutions Overview](https://distill.pub/2021/understanding-gnns/)
- [Temporal Convolutions in GNNs](https://arxiv.org/abs/2006.10637)
