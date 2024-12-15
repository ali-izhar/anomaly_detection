import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv
from ..base_model import BaseTemporalGraphModel

class TemporalConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super(TemporalConv, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv_2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv_3 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        
    def forward(self, X: torch.FloatTensor) -> torch.FloatTensor:
        """
        Args:
            X (torch.FloatTensor): Input of shape [batch_size, time_steps, num_nodes, channels]
        Returns:
            H (torch.FloatTensor): Output of shape [batch_size, time_steps, num_nodes, channels]
        """
        X = X.permute(0, 3, 2, 1)  # [batch_size, channels, num_nodes, time_steps]
        
        # Gated temporal convolution mechanism
        P = self.conv_1(X)
        Q = torch.sigmoid(self.conv_2(X))
        PQ = P * Q
        H = F.relu(PQ + self.conv_3(X))
        
        H = H.permute(0, 3, 2, 1)  # Return to original shape
        return H

class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_nodes, K=2):
        super(STGCNBlock, self).__init__()
        self.temporal_conv1 = TemporalConv(in_channels, out_channels, kernel_size)
        self.graph_conv = ChebConv(out_channels, out_channels, K)
        self.temporal_conv2 = TemporalConv(out_channels, out_channels, kernel_size)
        self.batch_norm = nn.BatchNorm2d(num_nodes)
        
    def forward(self, x, edge_index, edge_weight=None):
        """
        Args:
            x: Input tensor of shape [batch_size, time_steps, num_nodes, channels]
            edge_index: Graph edge indices
            edge_weight: Edge weights (optional)
        """
        # First temporal convolution
        h = self.temporal_conv1(x)
        
        # Graph convolution
        h_graph = torch.zeros_like(h)
        for b in range(h.size(0)):
            for t in range(h.size(1)):
                h_graph[b, t] = self.graph_conv(h[b, t], edge_index, edge_weight)
        
        # Second temporal convolution
        h = F.relu(h_graph)
        h = self.temporal_conv2(h)
        
        # Batch normalization
        h = h.permute(0, 2, 1, 3)
        h = self.batch_norm(h)
        h = h.permute(0, 2, 1, 3)
        
        return h

class STGCN(BaseTemporalGraphModel):
    def __init__(self, num_nodes, num_features, seq_len, hidden_channels=64, kernel_size=3):
        super().__init__(num_nodes, num_features, seq_len)
        
        # Model hyperparameters
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        
        # STGCN blocks
        self.block1 = STGCNBlock(
            in_channels=num_features,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            num_nodes=num_nodes
        )
        
        self.block2 = STGCNBlock(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            num_nodes=num_nodes
        )
        
        # Output projection
        self.out = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, num_nodes)
        )
        
    def forward(self, x, edge_index, edge_weight):
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, num_nodes, features]
            edge_index: Graph connectivity [2, num_edges]
            edge_weight: Edge weights [num_edges]
        Returns:
            out: Predicted adjacency matrix [batch_size, num_nodes, num_nodes]
        """
        # Apply STGCN blocks
        h = self.block1(x, edge_index, edge_weight)
        h = self.block2(h, edge_index, edge_weight)
        
        # Get the last timestep prediction
        h = h[:, -1]  # [batch_size, num_nodes, hidden_channels]
        
        # Output projection
        out = self.out(h)  # [batch_size, num_nodes, num_nodes]
        
        return out