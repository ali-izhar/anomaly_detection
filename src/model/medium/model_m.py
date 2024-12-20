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
        dropout: float = 0.3,
        num_layers: int = 2,
    ):
        """
        Medium complexity ASTGCN model for spatio-temporal graph data.
        """
        super().__init__()

        self.num_nodes = num_nodes
        self.hidden_channels = hidden_channels
        self.window_size = window_size

        # Improved temporal feature extraction with dilated convolutions
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                hidden_channels,
                kernel_size=(3, 1),
                padding=(1, 0),
                dilation=1,
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=(3, 1),
                padding=(2, 0),
                dilation=2,
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=(3, 1),
                padding=(4, 0),
                dilation=4,
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
        )

        # ASTGCN with modified parameters for sparse graphs
        self.astgcn = ASTGCN(
            nb_block=num_layers,
            in_channels=hidden_channels,
            K=2,
            nb_chev_filter=hidden_channels,
            nb_time_filter=hidden_channels,
            time_strides=1,
            num_for_predict=1,
            len_input=window_size,
            num_of_vertices=num_nodes,
            normalization="rw",
        )

        # Enhanced node feature transformation
        self.node_transform = nn.Sequential(
            nn.Linear(1, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
        )

        # Modify edge predictor for better sparsity
        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels * 2),
            nn.LayerNorm(hidden_channels * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, 1, bias=True),
        )

        # Degree embedding to capture scale-free properties
        self.degree_embedding = nn.Embedding(num_nodes, hidden_channels)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m in self.edge_predictor:
                    if m is self.edge_predictor[-1]:
                        # Initialize around zero
                        nn.init.normal_(m.weight, mean=0.0, std=0.1)
                        nn.init.constant_(m.bias, 0.0)  # Start at zero
                    else:
                        nn.init.xavier_uniform_(m.weight, gain=0.02)
                        nn.init.zeros_(m.bias)
                else:
                    nn.init.xavier_uniform_(m.weight, gain=0.02)
                    nn.init.zeros_(m.bias)

    def compute_degree_features(self, edge_index, batch_size):
        """Compute degree features for scale-free property awareness"""
        degrees = torch.zeros(batch_size, self.num_nodes, device=edge_index.device)
        
        # edge_index is already a tensor of shape [2, num_edges]
        # We need to ensure it's non-negative and compute degrees directly
        degree_count = torch.bincount(
            edge_index[0].long(),  # Ensure long type
            minlength=self.num_nodes
        )
        # Broadcast to all batches since we're using the same graph structure
        degrees = degree_count.unsqueeze(0).expand(batch_size, -1)
        
        return self.degree_embedding(degrees.long())

    def forward(self, x, edge_index, edge_weight=None):
        """
        Forward pass with dimension checks
        Args:
            x: [batch_size, sequence_length, num_nodes, features]
            edge_index: [2, num_edges]
            edge_weight: [num_edges] or None
        """
        batch_size = x.size(0)
        
        # Dimension checks
        assert x.dim() == 4, f"Expected 4D input tensor, got shape {x.shape}"
        assert edge_index.dim() == 2, f"Expected 2D edge_index tensor, got shape {edge_index.shape}"
        if edge_weight is not None:
            assert edge_weight.dim() == 1, f"Expected 1D edge_weight tensor, got shape {edge_weight.shape}"
        
        # Temporal feature extraction
        x = x.permute(0, 3, 1, 2)  # [batch, features, time, nodes]
        temporal_features = self.temporal_conv(x)
        temporal_features = temporal_features.permute(0, 3, 1, 2)  # [batch, nodes, features, time]

        # ASTGCN processing
        graph_features = self.astgcn(temporal_features, edge_index)  # [batch, nodes, 1]
        
        # Compute degree-aware features
        degree_features = self.compute_degree_features(edge_index, batch_size)  # [batch, nodes, hidden]
        
        # Transform node features
        node_features = []
        for i in range(self.num_nodes):
            node_feat = graph_features[:, i, :]  # [batch, 1]
            node_feat = self.node_transform(node_feat)  # [batch, hidden]
            node_feat = node_feat + degree_features[:, i]  # Add degree features
            node_features.append(node_feat)
        
        node_features = torch.stack(node_features, dim=1)  # [batch, nodes, hidden]
        
        # Modified edge prediction
        edge_preds = []
        for i in range(self.num_nodes):
            row_preds = []
            for j in range(self.num_nodes):
                if i != j:
                    node_pair = torch.cat([
                        node_features[:, i],
                        node_features[:, j]
                    ], dim=-1)
                    
                    # Raw prediction
                    edge_pred = self.edge_predictor(node_pair)
                    
                    # Scale predictions to reasonable range
                    edge_pred = edge_pred * 1.5
                    
                    # Add moderate bias based on degrees
                    src_degree = degree_features[:, i].mean(dim=-1, keepdim=True)
                    dst_degree = degree_features[:, j].mean(dim=-1, keepdim=True)
                    degree_factor = torch.sigmoid(src_degree * dst_degree)
                    
                    # Dynamic bias based on degree correlation
                    bias = torch.where(
                        degree_factor > 0.5,
                        torch.zeros_like(degree_factor),
                        -1.0 * (1.0 - degree_factor)
                    )
                    edge_pred = edge_pred + bias
                    
                    row_preds.append(edge_pred)
                else:
                    row_preds.append(torch.full_like(node_features[:, 0, 0].unsqueeze(-1), -2.0))  # Less extreme
            edge_preds.append(torch.cat(row_preds, dim=-1))
        
        adj_matrix = torch.stack(edge_preds, dim=1)
        adj_matrix = 0.5 * (adj_matrix + adj_matrix.transpose(-2, -1))
        
        return adj_matrix  # Return logits without sigmoid
