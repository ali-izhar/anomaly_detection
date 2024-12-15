import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, GCNConv
from torch_geometric_temporal.nn.recurrent import GConvGRU

logger = logging.getLogger(__name__)


class TemporalTransformer(nn.Module):
    """Multi-head temporal transformer for sequence modeling."""

    def __init__(self, hidden_channels, num_heads=4, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            hidden_channels, num_heads, dropout=dropout
        )
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 4, hidden_channels),
        )
        self.norm2 = nn.LayerNorm(hidden_channels)

    def forward(self, x):
        B, T, N, C = x.shape
        x = x.transpose(0, 1).reshape(T, B * N, C)
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x.reshape(T, B, N, C).transpose(0, 1)


class SpatialTransformer(nn.Module):
    """Graph transformer for spatial dependencies."""

    def __init__(self, hidden_channels, heads=4, dropout=0.1):
        super().__init__()
        self.conv = TransformerConv(
            in_channels=hidden_channels,
            out_channels=hidden_channels // heads,
            heads=heads,
            dropout=dropout,
        )
        self.norm = nn.LayerNorm(hidden_channels)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv(x, edge_index, edge_weight=edge_weight)
        return self.norm(x)


class DynamicLinkPredictor(nn.Module):
    def __init__(
        self,
        num_nodes,
        num_features,
        hidden_channels=64,
        num_layers=3,
        temporal_periods=10,
        dropout=0.2,
        graph_types=("BA", "ER", "NW"),
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_channels = hidden_channels
        self.graph_type_embeddings = nn.Embedding(len(graph_types), hidden_channels)

        # Feature projection
        self.feature_proj = nn.Sequential(
            nn.Linear(num_features, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
        )

        # Temporal modeling with GConvGRU
        self.temporal_gru = GConvGRU(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            K=3,  # Spatial neighborhood size
        )

        # Spatial modeling layers
        self.spatial_layers = nn.ModuleList(
            [SpatialTransformer(hidden_channels) for _ in range(num_layers)]
        )
        self.gcn_layers = nn.ModuleList(
            [GCNConv(hidden_channels, hidden_channels) for _ in range(num_layers)]
        )

        # Edge prediction
        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, 1),
        )

        # Auxiliary task: Change-point detection
        self.change_point_predictor = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, 1),
        )

    def forward(
        self, x, edge_index, edge_weight=None, graph_type=None, is_change_point=None
    ):
        # Add graph-type embedding
        if graph_type is not None:
            graph_type_id = torch.tensor([graph_type]).to(x.device)
            graph_emb = self.graph_type_embeddings(graph_type_id).squeeze(0)
            x = x + graph_emb.unsqueeze(0).expand_as(x)

        # Feature projection
        logger.debug(f"Initial x shape: {x.shape}")
        x = self.feature_proj(x)
        logger.debug(f"After feature projection: {x.shape}")

        # Temporal modeling with GConvGRU
        h = torch.zeros(
            (x.size(0), self.num_nodes, self.hidden_channels),
            device=x.device,
        )  # Initial hidden state
        for t in range(x.size(1)):  # Iterate over time steps
            h = self.temporal_gru(h, edge_index, edge_weight)  # Update hidden state

        # Spatial modeling
        for spatial_layer, gcn_layer in zip(self.spatial_layers, self.gcn_layers):
            h = spatial_layer(h, edge_index, edge_weight)
            h = gcn_layer(h, edge_index)

        # Link prediction
        row, col = torch.cartesian_prod(
            torch.arange(self.num_nodes, device=x.device),
            torch.arange(self.num_nodes, device=x.device),
        ).T
        pair_features = torch.cat([h[:, row], h[:, col]], dim=-1)
        adj_pred = self.edge_predictor(pair_features).view(
            -1, self.num_nodes, self.num_nodes
        )

        # Change-point prediction (auxiliary task)
        change_point_pred = self.change_point_predictor(h).squeeze(-1)

        return adj_pred, change_point_pred
