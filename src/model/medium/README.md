# Medium ASTGCN Model Documentation

## 1. Overview & Motivation

### Why ASTGCN?
- Attention mechanisms for dynamic graph relationships
- Parallel processing of temporal and spatial patterns
- Better handling of complex node interactions

### Comparison with Small Model

```
Aspect          | Small (STGCN)       | Medium (ASTGCN)
----------------|---------------------|------------------
Processing      | Sequential          | Parallel + Attention
Architecture    | Fixed convolutions  | Dynamic attention
Complexity      | ~757K params        | ~53K params
Memory Usage    | Lower               | Higher (attention)
Best For        | Simple patterns     | Complex dynamics
```

## 2. Data & Problem

### Input Structure
- Temporal graph sequence: $\{\mathcal{G}_t\}_{t=1}^T$
- Node features: $\mathbf{X}_t \in \mathbb{R}^{N \times F}$ (30 nodes, 6 features)
- Graph structure: $\mathbf{A}_t \in \{0,1\}^{N \times N}$
- Task: Predict $\mathbf{A}_{t+1}$ from $[t-9, t]$ window

### Data Processing
1. **Input Tensor**: $[B, T, N, F]$ → $[B, N, F, T]$
   - Rearrange for attention operations
   - Preserve batch processing capability
   - Enable parallel computations

2. **Feature Handling**:
   - Normalize node features
   - Convert graph to COO format
   - Prepare attention masks

## 3. Model Architecture

### Core Components
1. **ASTGCN Configuration**:
    ```python
    ASTGCN(
        nb_block=2,              # Number of ASTGCN blocks
        in_channels=6,           # Input features
        K=3,                     # Chebyshev polynomial order
        nb_chev_filter=64,       # Hidden channels
        nb_time_filter=64,       # Output channels
        len_input=10,            # Sequence length
        num_of_vertices=30,      # Number of nodes
        normalization="sym"      # Graph normalization
    )
    ```

2. **ASTGCN Blocks $(L=2)$**
   ```
   Block Structure:
   ├── Temporal Attention
   ├── Spatial Attention
   ├── ChebConv (K=3)
   ├── Time Convolution
   └── Layer Norm + Residual
   ```

### Attention Mechanisms

1. **Temporal Attention**:
   - Input: $\mathbf{H} \in \mathbb{R}^{B \times N \times F \times T}$
   - For each head $h$:
     ```
     Q = W_q^h H    # Query transform
     K = W_k^h H    # Key transform
     V = W_v^h H    # Value transform
     
     Attention = softmax(QK^T/√d_k)V
     ```

2. **Spatial Attention**:
   - Processes node neighborhoods
   - Attention per edge: $\alpha_{ij} = \text{softmax}(e_{ij})$
   - Edge importance: $e_{ij} = \text{LeakyReLU}(\mathbf{a}^T[\mathbf{W}h_i \| \mathbf{W}h_j])$

### Processing Flow
```
Input [B,10,30,6]
   ↓
Permute [B,30,6,10]
   ↓
ASTGCN Block 1 → [B,30,64,10]
   ├── Temporal Attention
   ├── Spatial Attention with ChebConv
   └── Time Convolution + Residual
   ↓
ASTGCN Block 2 → [B,30,64,10]
   ↓
Final Conv → [B,30,1]
   ↓
Linear Projection → [B,30,30]
```

## 4. Training Process

### Loss Function Design
```python
class FocalLoss(nn.Module):
    """Handles imbalanced edge distribution"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, pred, target):
        ce_loss = F.binary_cross_entropy(pred, target, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()
```

### Optimization Strategy
1. **Gradient Control**:
   ```python
   torch.nn.utils.clip_grad_norm_(
       model.parameters(), 
       max_norm=1.0
   )
   ```

2. **Learning Rate**:
   ```python
   scheduler = CosineAnnealingWarmRestarts(
       T_0=5,    # Initial cycle
       T_mult=2  # Cycle multiplier
   )
   ```

### Further Reading
- [ASTGCN Paper](https://ojs.aaai.org/index.php/AAAI/article/view/3881)
- [Attention in GNNs](https://arxiv.org/abs/2009.14794)
