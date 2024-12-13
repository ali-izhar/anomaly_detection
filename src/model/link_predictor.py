# src/model/link_predictor.py

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import DCRNN
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.nn import global_mean_pool

logger = logging.getLogger(__name__)


class DynamicLinkPredictor(nn.Module):
    """
    Enhanced neural network model for dynamic link prediction using DCRNN and GraphSAGE.
    Incorporates skip connections and multi-scale features.
    """

    def __init__(
        self,
        num_nodes: int,
        num_features: int = 6,
        hidden_channels: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        K: int = 3,
        use_edge_weights: bool = True,
        temporal_periods: int = 10,
        batch_size: int = 32,
        pos_weight: float = 5.0,
    ):
        super().__init__()
        
        logger.info(f"Initializing DynamicLinkPredictor with:")
        logger.info(f"  - {num_nodes} nodes")
        logger.info(f"  - {num_features} input features")
        logger.info(f"  - {hidden_channels} hidden channels")
        logger.info(f"  - {num_layers} layers")
        logger.info(f"  - {temporal_periods} temporal periods")
        
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.hidden_channels = hidden_channels
        self.temporal_periods = temporal_periods
        
        # Initial feature projection
        self.feature_proj = nn.Sequential(
            nn.Linear(num_features, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Temporal processing
        self.temporal_proj = nn.Sequential(
            nn.Linear(num_features * temporal_periods, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # DCRNN layers
        self.dcrnn_layers = nn.ModuleList([
            DCRNN(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                K=K,
                bias=True
            ) for _ in range(num_layers)
        ])
        
        # Graph convolution layers for spatial features
        self.graph_conv = nn.ModuleList([
            GCNConv(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                improved=True,
                cached=False,
                add_self_loops=True,
                normalize=True,
            ) for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_channels) for _ in range(num_layers)
        ])
        
        # Node embeddings
        self.node_embeddings = nn.Parameter(torch.randn(num_nodes, hidden_channels))
        
        # Link prediction head
        self.link_predictor = nn.Sequential(
            nn.Linear(hidden_channels * 4, hidden_channels * 2),
            nn.LayerNorm(hidden_channels * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, edge_index, edge_weight=None, hidden_state=None):
        batch_size = x.size(0)
        device = x.device
        
        # Ensure node embeddings are on the correct device
        if self.node_embeddings.device != device:
            self.node_embeddings = self.node_embeddings.to(device)
        
        # Debug input shape
        logger.debug(f"Input x shape: {x.shape}")
        
        # Project current features (using last timestep)
        current_features = x[:, :, :, -1]
        h = self.feature_proj(current_features.reshape(-1, self.num_features))
        h = h.view(batch_size, self.num_nodes, -1)
        
        # Process temporal information
        x_flat = x.reshape(batch_size * self.num_nodes, self.num_features * self.temporal_periods)
        h_temporal = self.temporal_proj(x_flat)
        h_temporal = h_temporal.view(batch_size, self.num_nodes, -1)
        
        # Combine current and temporal features
        h = h + h_temporal
        
        # Add positional node embeddings
        node_emb = self.node_embeddings.unsqueeze(0)
        h = h + node_emb
        
        # Initialize hidden states
        if hidden_state is None:
            hidden_state = [None] * len(self.dcrnn_layers)
        
        # Process through layers
        h_list = [h]
        new_hidden_states = []
        
        for i, (dcrnn, conv, norm) in enumerate(zip(self.dcrnn_layers, self.graph_conv, self.layer_norms)):
            h_flat = h.view(-1, self.hidden_channels)
            
            # Ensure edge tensors are on correct device
            edge_index = edge_index.to(device)
            if edge_weight is not None:
                edge_weight = edge_weight.to(device)
            
            # Apply DCRNN
            h_temp = dcrnn(h_flat, edge_index, edge_weight, hidden_state[i])
            h_temp = h_temp.view(batch_size, self.num_nodes, -1)
            
            # Apply graph convolution
            h_space = conv(h_flat, edge_index)
            h_space = h_space.view(batch_size, self.num_nodes, -1)
            
            h = h_temp + h_space
            h = norm(h)
            h = F.relu(h)
            h = self.dropout(h)
            
            h = h + h_list[-1]
            h_list.append(h)
            new_hidden_states.append(h_temp)
        
        h_final = torch.cat([h_list[-1], h_list[-2]], dim=-1)
        
        # Create node pairs on correct device
        row, col = torch.cartesian_prod(
            torch.arange(self.num_nodes, device=device),
            torch.arange(self.num_nodes, device=device)
        ).T
        
        batch_row = row.unsqueeze(0).expand(batch_size, -1)
        batch_col = col.unsqueeze(0).expand(batch_size, -1)
        
        # Get node pair features
        row_h = h_final[torch.arange(batch_size, device=device).unsqueeze(1), batch_row]
        col_h = h_final[torch.arange(batch_size, device=device).unsqueeze(1), batch_col]
        
        # Predict links
        pair_features = torch.cat([row_h, col_h], dim=-1)
        logits = self.link_predictor(pair_features)
        adj_logits = logits.view(batch_size, self.num_nodes, self.num_nodes)
        
        return adj_logits, tuple(new_hidden_states)

    def predict_links(
        self, adj_pred: torch.Tensor, threshold: float = 0.5
    ) -> torch.Tensor:
        """Convert predicted probabilities to binary adjacency matrix."""
        return (adj_pred > threshold).float()


class ResidualBlock(nn.Module):
    """Residual block for the link predictor."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Linear(in_channels, out_channels)
        self.conv2 = nn.Linear(out_channels, out_channels)
        self.norm1 = nn.LayerNorm(out_channels)
        self.norm2 = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(0.1)

        if in_channels != out_channels:
            self.shortcut = nn.Linear(in_channels, out_channels)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = torch.relu(self.norm1(out))
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.dropout(out)

        return torch.relu(out + identity)
