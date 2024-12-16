import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAttention(nn.Module):
    """Spatial attention module from GMAN."""
    def __init__(self, hidden_channels, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = hidden_channels // num_heads
        
        # Multi-head attention layers
        self.query = nn.Linear(hidden_channels, hidden_channels)
        self.key = nn.Linear(hidden_channels, hidden_channels)
        self.value = nn.Linear(hidden_channels, hidden_channels)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_channels)
        
    def forward(self, x):
        B, T, N, C = x.shape  # [batch, time, nodes, channels]
        
        # Multi-head attention
        q = self.query(x).view(B, T, N, self.num_heads, -1)
        k = self.key(x).view(B, T, N, self.num_heads, -1)
        v = self.value(x).view(B, T, N, self.num_heads, -1)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (C ** 0.5)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        x = torch.matmul(attn, v)
        x = x.reshape(B, T, N, C)
        
        # Residual connection and layer norm
        x = self.layer_norm(x)
        return x


class TemporalAttention(nn.Module):
    """Temporal attention module from GMAN."""
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


class LargeGMAN(nn.Module):
    def __init__(
        self,
        in_channels: int = 6,
        hidden_channels: int = 128,
        out_channels: int = 128,
        num_nodes: int = 30,
        window_size: int = 10,
        num_layers: int = 3,
        spatial_heads: int = 4,
        temporal_heads: int = 4,
        dropout: float = 0.1,
    ):
        """
        Large Graph Multi-Attention Network adapted for link prediction.
        Based on GMAN's attention mechanisms but simplified for our task.
        """
        super(LargeGMAN, self).__init__()
        
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        
        # Input projection
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        
        # Stack of attention blocks
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
        
    def forward(self, x, edge_index, edge_weight=None):
        """
        Forward pass of the model.
        
        Args:
            x: Node features [batch_size, sequence_length, num_nodes, features]
            edge_index: Graph connectivity [2, num_edges]
            edge_weight: Edge weights [num_edges], optional
            
        Returns:
            Predicted adjacency matrix [batch_size, num_nodes, num_nodes]
        """
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
