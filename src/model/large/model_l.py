import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import math


class TemporalConvBlock(nn.Module):
    """Temporal convolution block for capturing multi-scale patterns"""
    def __init__(self, channels, kernel_sizes=[3, 5, 7]):
        super().__init__()
        # Ensure total output channels match input channels
        self.out_channels_per_conv = channels // len(kernel_sizes)
        # Adjust last conv to make total channels match input channels
        self.out_channels_last = channels - (self.out_channels_per_conv * (len(kernel_sizes) - 1))
        
        # Create convolutions with adjusted channels
        self.convs = nn.ModuleList()
        for i, k in enumerate(kernel_sizes):
            out_channels = self.out_channels_last if i == len(kernel_sizes)-1 else self.out_channels_per_conv
            self.convs.append(
                nn.Conv2d(channels, out_channels, (k, 1), padding=(k // 2, 0))
            )
        
        # BatchNorm with input channels (since output will match input)
        self.norm = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        # x: [batch, time, nodes, channels]
        x = x.permute(0, 3, 1, 2)  # [batch, channels, time, nodes]
        
        # Apply convolutions and concatenate
        conv_outputs = []
        for conv in self.convs:
            conv_outputs.append(conv(x))
        out = torch.cat(conv_outputs, dim=1)
        
        # Apply batch normalization
        out = self.norm(out)
        
        # Return to original dimension order
        out = out.permute(0, 2, 3, 1)  # [batch, time, nodes, channels]
        return out


class StructuralBlock(nn.Module):
    """Combined GCN and attention for structure learning"""
    def __init__(self, hidden_channels, num_heads):
        super().__init__()
        self.gcn = GCNConv(hidden_channels, hidden_channels)
        self.attn = SpatialAttention(hidden_channels, num_heads)
        self.norm = nn.LayerNorm(hidden_channels)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels * 4),
            nn.ReLU(),
            nn.Linear(hidden_channels * 4, hidden_channels)
        )
        
    def forward(self, x, edge_index):
        # Process each timestep
        batch, time, nodes, channels = x.shape
        
        # Use reshape instead of view for non-contiguous tensors
        x_flat = x.reshape(-1, nodes, channels)
        
        # GCN branch
        gcn_out = []
        for t in range(time):
            t_start = t * batch
            t_end = (t + 1) * batch
            out = self.gcn(x_flat[t_start:t_end], edge_index)
            gcn_out.append(out)
        gcn_out = torch.stack(gcn_out, dim=1)
        
        # Attention branch
        attn_out = self.attn(x)
        
        # Combine
        combined = torch.cat([gcn_out, attn_out], dim=-1)
        out = self.mlp(combined)
        return self.norm(out)


class SpatialAttention(nn.Module):
    """Spatial attention module for learning node relationships"""
    def __init__(self, hidden_channels, num_heads, dropout=0.1):
        super().__init__()
        assert hidden_channels % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_channels // num_heads
        
        # Multi-head attention layers
        self.query = nn.Linear(hidden_channels, hidden_channels)
        self.key = nn.Linear(hidden_channels, hidden_channels)
        self.value = nn.Linear(hidden_channels, hidden_channels)
        
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_channels, hidden_channels)
        
    def forward(self, x):
        B, T, N, C = x.shape  # [batch, time, nodes, channels]
        
        # Reshape for multi-head attention
        q = self.query(x).view(B, T, N, self.num_heads, self.head_dim)
        k = self.key(x).view(B, T, N, self.num_heads, self.head_dim)
        v = self.value(x).view(B, T, N, self.num_heads, self.head_dim)
        
        # Transpose for attention calculation
        q = q.transpose(2, 3)  # [B, T, heads, N, dim]
        k = k.transpose(2, 3)
        v = v.transpose(2, 3)
        
        # Calculate attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, v)  # [B, T, heads, N, dim]
        out = out.transpose(2, 3).contiguous()  # [B, T, N, heads, dim]
        out = out.view(B, T, N, -1)  # [B, T, N, channels]
        
        return self.output(out)


class LargeGMAN(nn.Module):
    def __init__(
        self,
        in_channels: int = 6,
        hidden_channels: int = 256,
        out_channels: int = 256,
        num_nodes: int = 30,
        window_size: int = 10,
        num_layers: int = 4,
        spatial_heads: int = 8,
        temporal_heads: int = 8,
        dropout: float = 0.3,
        l1_lambda: float = 0.01,
    ):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.l1_lambda = l1_lambda
        self.hidden_channels = hidden_channels
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Temporal pattern extraction
        self.temporal_conv = TemporalConvBlock(hidden_channels)
        
        # LSTM for temporal memory
        self.temporal_memory = nn.LSTM(
            input_size=hidden_channels,
            hidden_size=hidden_channels,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )
        
        # Structure learning blocks
        self.structure_blocks = nn.ModuleList([
            StructuralBlock(hidden_channels, spatial_heads)
            for _ in range(num_layers)
        ])
        
        # Cross attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_channels,
            num_heads=temporal_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Final prediction layers with better initialization
        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels * 4),
            nn.LayerNorm(hidden_channels * 4),
            nn.GELU(),  # Change to GELU for better gradients
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 4, hidden_channels * 2),
            nn.LayerNorm(hidden_channels * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 2, 1, bias=False),  # Remove bias for better initialization
        )

        # Initialize the final layer with smaller weights
        with torch.no_grad():
            nn.init.xavier_normal_(self.edge_predictor[-1].weight, gain=0.1)

    def forward(self, x, edge_index, edge_weight=None):
        batch_size = x.size(0)
        
        # Project input
        x = self.input_proj(x)  # [batch, time, nodes, channels]
        
        # Extract temporal patterns
        temporal_features = self.temporal_conv(x)  # [batch, time, nodes, channels]
        
        # Process each node's temporal sequence separately
        batch, time, nodes, channels = temporal_features.shape
        
        # Reshape for per-node LSTM processing
        temporal_features = temporal_features.transpose(1, 2)  # [batch, nodes, time, channels]
        temporal_features = temporal_features.reshape(batch * nodes, time, channels)
        
        # Apply LSTM
        temporal_features, _ = self.temporal_memory(temporal_features)
        
        # Reshape back
        temporal_features = temporal_features.reshape(batch, nodes, time, channels)
        temporal_features = temporal_features.transpose(1, 2)  # [batch, time, nodes, channels]
        
        # Learn structure
        structural_features = temporal_features
        for block in self.structure_blocks:
            structural_features = block(structural_features, edge_index)
        
        # Cross attention between temporal and structural features
        # Reshape for attention: [batch, seq_len, embed_dim]
        temp_flat = temporal_features.mean(dim=2)  # Average over nodes
        struct_flat = structural_features.mean(dim=2)  # Average over nodes
        
        # Apply cross attention
        attended_features, _ = self.cross_attention(
            temp_flat,      # query [batch, time, channels]
            struct_flat,    # key   [batch, time, channels]
            struct_flat     # value [batch, time, channels]
        )
        
        # Combine features from both branches
        final_features = torch.cat([
            attended_features.mean(dim=1),     # [batch, channels]
            structural_features.mean(dim=1).mean(dim=1)  # [batch, channels]
        ], dim=-1)
        
        # Generate adjacency matrix with better initialization
        edge_logits = []
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                node_pair = torch.cat([
                    final_features[:, :self.hidden_channels],
                    final_features[:, self.hidden_channels:],
                ], dim=-1)
                edge_logits.append(self.edge_predictor(node_pair))
        
        logits = torch.stack(edge_logits, dim=1).view(batch_size, self.num_nodes, self.num_nodes)
        
        # Add structure-based regularization with reduced strength
        adj_probs = torch.sigmoid(logits)
        self.regularization_loss = (
            self.l1_lambda * torch.abs(adj_probs).mean() +
            0.005 * torch.abs(adj_probs - torch.matmul(adj_probs, adj_probs)).mean()  # Reduced from 0.01
        )
        
        return logits
