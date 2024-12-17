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
        """
        super().__init__()

        self.num_nodes = num_nodes
        self.hidden_channels = hidden_channels
        self.window_size = window_size

        # Temporal feature extraction
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=(5, 1), padding=(2, 0)),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
        )

        # ASTGCN - note the output will be [batch, nodes, timesteps]
        self.astgcn = ASTGCN(
            nb_block=num_layers,
            in_channels=hidden_channels,
            K=3,
            nb_chev_filter=hidden_channels,
            nb_time_filter=hidden_channels,
            time_strides=1,
            num_for_predict=1,  # We only want one prediction
            len_input=window_size,
            num_of_vertices=num_nodes,
            normalization="sym",
        )

        # Node feature transformation - input is 1 (from ASTGCN output)
        self.node_transform = nn.Sequential(
            nn.Linear(1, hidden_channels),  # From 1 to hidden_channels
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels),
            nn.LayerNorm(hidden_channels)
        )

        # Edge prediction MLP with better initialization
        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels * 2),
            nn.LayerNorm(hidden_channels * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, 1, bias=True)
        )

        # Initialize with more extreme values to break symmetry
        with torch.no_grad():
            # Last layer initialization
            nn.init.normal_(self.edge_predictor[-1].weight, mean=0.0, std=0.02)
            if hasattr(self.edge_predictor[-1], 'bias'):
                self.edge_predictor[-1].bias.fill_(0.0)  # Start at 0.5 after sigmoid

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

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
        batch_size = x.size(0)

        # Extract temporal patterns [batch, channels, time, nodes]
        x = x.permute(0, 3, 1, 2)  # [batch, features, time, nodes]
        temporal_features = self.temporal_conv(x)

        # Reshape for ASTGCN [batch, nodes, features, time]
        temporal_features = temporal_features.permute(0, 3, 1, 2)

        # Process with ASTGCN - output is [batch, nodes, 1]
        graph_features = self.astgcn(temporal_features, edge_index)

        # Transform node features directly
        node_features = []
        for i in range(self.num_nodes):
            # Take each node's features
            node_feat = graph_features[:, i, :]  # [batch, 1]
            node_feat = self.node_transform(node_feat)  # Transform to hidden dimension
            node_features.append(node_feat)

        # Stack node features
        node_features = torch.stack(node_features, dim=1)  # [batch, nodes, hidden]

        # Generate edge predictions
        edge_preds = []
        for i in range(self.num_nodes):
            row_preds = []
            for j in range(self.num_nodes):
                node_pair = torch.cat([
                    node_features[:, i],
                    node_features[:, j]
                ], dim=-1)
                edge_pred = self.edge_predictor(node_pair)
                row_preds.append(edge_pred)
            edge_preds.append(torch.cat(row_preds, dim=-1))

        # Combine into adjacency matrix
        adj_matrix = torch.stack(edge_preds, dim=1)

        # Make symmetric
        adj_matrix = 0.5 * (adj_matrix + adj_matrix.transpose(-2, -1))

        return adj_matrix
