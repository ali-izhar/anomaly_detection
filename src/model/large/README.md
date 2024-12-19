# Large GMAN Documentation

## 1. Overview & Motivation

### Why Custom Attention?
- Flexible attention mechanisms for both spatial and temporal domains
- Parallel multi-head processing
- Deeper architecture with residual connections
- Adapted from GMAN architecture for link prediction

### Comparison with Other Models
```
Aspect          | Small (STGCN)    | Medium (ASTGCN)   | Large (Custom)
----------------|------------------|-------------------|----------------
Processing      | Sequential       | Parallel          | Fully Parallel
Architecture    | Fixed Conv       | ASTGCN Blocks     | Dual Attention
Parameters      | ~757K            | ~53K              | ~390K
Memory Usage    | Low              | Medium            | High
Best For        | Simple Patterns  | Graph Evolution   | Complex Relations
```

## 2. Model Architecture

### Core Components

1. **Spatial Attention Implementation**:
```python
class SpatialAttention(nn.Module):
    def __init__(self, hidden_channels, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = hidden_channels // num_heads
        
        self.query = nn.Linear(hidden_channels, hidden_channels)
        self.key = nn.Linear(hidden_channels, hidden_channels)
        self.value = nn.Linear(hidden_channels, hidden_channels)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_channels)

    def forward(self, x):  # [B, T, N, C]
        B, T, N, C = x.shape
        
        q = self.query(x).view(B, T, N, self.num_heads, -1)
        k = self.key(x).view(B, T, N, self.num_heads, -1)
        v = self.value(x).view(B, T, N, self.num_heads, -1)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / (C ** 0.5)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        x = torch.matmul(attn, v)
        x = x.reshape(B, T, N, C)
        x = self.layer_norm(x)
        return x
```

2. **Main Model Structure**:
```python
class LargeGMAN(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, 
                 spatial_heads, temporal_heads, dropout=0.1):
        # Input projection
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        
        # Attention layers
        self.spatial_attentions = nn.ModuleList([
            SpatialAttention(hidden_channels, spatial_heads, dropout)
            for _ in range(num_layers)
        ])
        self.temporal_attentions = nn.ModuleList([
            TemporalAttention(hidden_channels, temporal_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels * 2),
            nn.LayerNorm(hidden_channels * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 2, num_nodes)
        )
```

3. **Temporal Attention Implementation**:
```python
class TemporalAttention(nn.Module):
    def __init__(self, hidden_channels, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(hidden_channels)
        
    def forward(self, x):
        B, T, N, C = x.shape
        
        # Reshape for multi-head attention
        x = x.reshape(B * N, T, C)
        
        # Self-attention
        x, _ = self.attention(x, x, x)
        x = x.reshape(B, T, N, C)
        
        # Layer norm
        x = self.layer_norm(x)
        return x
```

### Architectural Details

1. **Configuration**:
```python
LargeGMAN(
    in_channels=6,          # Input features
    hidden_channels=128,    # Internal processing
    num_nodes=30,           # Graph size
    window_size=10,         # Sequence length
    num_layers=3,           # Number of attention blocks
    spatial_heads=4,        # Spatial attention heads
    temporal_heads=4,       # Temporal attention heads
    dropout=0.1             # Dropout rate
)
```

2. **Layer Structure**:
```
For each layer:
├── Spatial Attention
│   ├── Multi-head (4 heads)
│   ├── Layer Norm
│   └── Residual Connection
└── Temporal Attention
    ├── Multi-head (4 heads)
    ├── Layer Norm
    └── Residual Connection
```

### Processing Flow

1. **Forward Pass**:
```python
def forward(self, x, edge_index, edge_weight=None):
    # Project input features
    x = self.input_proj(x)  # [batch, time, nodes, hidden]
    
    # Apply attention blocks
    for i in range(self.num_layers):
        # Spatial attention
        x = x + self.spatial_attentions[i](x)
        
        # Temporal attention
        x = x + self.temporal_attentions[i](x)
    
    # Global pooling over time
    x = x.mean(dim=1)  # [batch, nodes, hidden]
    
    # Generate adjacency matrix
    x = self.output_proj(x)  # [batch, nodes, nodes]
    adj_matrix = torch.sigmoid(x)
    
    return adj_matrix
```

2. **Dimension Flow**:
```
Input: [B, T, N, F] = [batch, 10, 30, 6]
   ↓
Projection: [B, T, N, 128]
   ↓
Spatial Attention: [B, T, N, 128] → [B, T, N, 128]
   ↓
Temporal Attention: [B, T, N, 128] → [B, T, N, 128]
   ↓
Time Pooling: [B, N, 128]
   ↓
Output MLP: [B, N, N]
```

## 3. Training Process

### Weighted Focal Loss
```python
class WeightedFocalLoss(nn.Module):
    """Weighted focal loss with class imbalance handling."""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        # Calculate class weights dynamically
        pos_weight = (target == 0).float().sum() / (target == 1).float().sum()

        # Binary cross entropy
        bce = F.binary_cross_entropy(pred, target, reduction="none")

        # Focal term
        pt = torch.exp(-bce)
        focal_term = (1 - pt) ** self.gamma

        # Weight positive examples more heavily
        weights = target * pos_weight + (1 - target)

        return (weights * focal_term * bce).mean()
```

### Training Loop
```python
def train_epoch(model, train_loader, criterion, optimizer, device, scheduler=None):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_targets = []

    with tqdm(train_loader, desc="Training") as pbar:
        for batch in pbar:
            features = batch["features"].to(device)
            edge_index = batch["edge_indices"][0][0].to(device)
            edge_weight = batch["edge_weights"][0][0].to(device) if batch["edge_weights"] else None
            targets = batch["targets"].to(device)

            optimizer.zero_grad()
            output = model(features, edge_index, edge_weight)
            loss = criterion(output, targets)

            # Add regularization loss if available
            if hasattr(model, "regularization_loss"):
                loss += model.regularization_loss

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            if scheduler is not None:
                scheduler.step()
```

### Optimization Setup
```python
# Initialize model
model = LargeGMAN(**model_config).to(device)

# Loss function
criterion = WeightedFocalLoss(alpha=0.25, gamma=2.0)

# Optimizer
optimizer = optim.AdamW(
    model.parameters(),
    lr=training_config["learning_rate"],
    weight_decay=training_config.get("weight_decay", 1e-4)
)
```

### Learning Rate Schedule
```python
# Learning rate scheduler
scheduler = OneCycleLR(
    optimizer,
    max_lr=training_config["learning_rate"],
    epochs=training_config["epochs"],
    steps_per_epoch=len(train_loader),
    pct_start=0.3,    # Warm-up for 30% of training
    div_factor=25.0,  # Initial lr = max_lr/25
    final_div_factor=1e4  # Final lr = max_lr/10000
)
```

### Metrics Calculation
```python
def calculate_metrics(predictions, targets):
    """Calculate various metrics for evaluation."""
    pred_binary = (predictions > 0.5).astype(float)
    
    # Basic metrics
    f1 = f1_score(targets.flatten(), pred_binary.flatten())
    
    # Try to calculate AUC, handle potential errors
    try:
        auc = roc_auc_score(targets.flatten(), predictions.flatten())
    except ValueError:
        auc = 0.0
    
    # Calculate average precision
    ap = average_precision_score(targets.flatten(), predictions.flatten())
    
    # Calculate precision at different thresholds
    precision, recall, _ = precision_recall_curve(targets.flatten(), predictions.flatten())
    
    return {
        'f1_score': f1,
        'auc_score': auc,
        'avg_precision': ap,
        'max_precision': np.max(precision),
        'max_recall': np.max(recall)
    }
```

### Further Reading
- [Attention is All You Need](https://arxiv.org/abs/1706.03762)
- [Graph Attention Networks](https://arxiv.org/abs/1710.10903)
- [Temporal Graph Networks](https://arxiv.org/abs/2006.10637)
