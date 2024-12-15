# src/model/medium/model_m.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric_temporal.attention.astgcn import ASTGCN


class MediumASTGCN(nn.Module):
    def __init__(
        self,
        in_channels: int = 6,
        hidden_channels: int = 64,
        out_channels: int = 64,
        num_nodes: int = 30,
        window_size: int = 10,
        num_heads: int = 4,
        dropout: float = 0.2,
        num_layers: int = 2,
    ):
        """
        Medium complexity ASTGCN model for spatio-temporal graph data.
        
        Args:
            in_channels: Number of input features per node
            hidden_channels: Number of hidden channels
            out_channels: Number of output channels
            num_nodes: Number of nodes in the graph
            window_size: Number of time steps to process
            num_heads: Number of attention heads (not used in ASTGCN)
            dropout: Dropout rate
            num_layers: Number of ASTGCN blocks
        """
        super(MediumASTGCN, self).__init__()
        
        # ASTGCN configuration
        self.astgcn = ASTGCN(
            nb_block=num_layers,
            in_channels=in_channels,
            K=3,  # Order of Chebyshev polynomials
            nb_chev_filter=hidden_channels,
            nb_time_filter=out_channels,
            time_strides=1,
            num_for_predict=1,  # We only predict one adjacency matrix
            len_input=window_size,
            num_of_vertices=num_nodes,
            normalization="sym",
            bias=True,
        )
        
        # Final layers to convert ASTGCN output to adjacency matrix
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_nodes, num_nodes)
        
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
        # ASTGCN expects input shape: [batch_size, num_nodes, features, time_steps]
        x = x.permute(0, 2, 3, 1)
        
        # Apply ASTGCN
        # Output shape will be [batch_size, num_nodes, time_prediction]
        x = self.astgcn(x, edge_index)
        
        # Apply final transformation to get adjacency matrix
        x = self.dropout(x)
        adj_matrix = self.fc(x)
        adj_matrix = torch.sigmoid(adj_matrix)
        
        return adj_matrix
