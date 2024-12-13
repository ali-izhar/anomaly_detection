# src/model/link_predictor.py

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch_geometric_temporal.nn.recurrent import GConvLSTM
from torch_geometric_temporal.nn.recurrent.attentiontemporalgcn import A3TGCN2

logger = logging.getLogger(__name__)


class DynamicLinkPredictor(nn.Module):
    """
    Enhanced neural network model for dynamic link prediction using GConvLSTM
    with A3TGCN2 temporal attention mechanisms.
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
        attention_heads: int = 4,
        attention_dropout: float = 0.1,
        temporal_periods: int = 10,  # Number of time steps to consider
        batch_size: int = 32,
    ):
        super(DynamicLinkPredictor, self).__init__()

        self.num_nodes = num_nodes
        self.num_features = num_features
        self.use_edge_weights = use_edge_weights
        self.hidden_channels = hidden_channels
        self.temporal_periods = temporal_periods

        # Input feature projection
        self.input_proj = nn.Sequential(
            nn.Linear(num_features * temporal_periods, hidden_channels),
            nn.ReLU(),
            nn.LayerNorm(hidden_channels),
            nn.Dropout(dropout),
        )

        # Temporal attention using A3TGCN2
        self.temporal_attention = A3TGCN2(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            periods=1,
            batch_size=batch_size,
            improved=True,
            cached=False,
            add_self_loops=True,
        )

        # GConvLSTM layers
        self.gconv_lstm_layers = nn.ModuleList(
            [
                GConvLSTM(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    K=K,
                    normalization="sym",
                    bias=True,
                )
                for _ in range(num_layers)
            ]
        )

        # Layer normalization
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_channels) for _ in range(num_layers)]
        )

        self.dropout = nn.Dropout(dropout)

        # Node embedding projection
        self.node_proj = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.LayerNorm(hidden_channels),
            nn.Dropout(dropout),
        )

        # Link prediction MLP with residual connections
        self.link_predictor = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.LayerNorm(hidden_channels),
            nn.Dropout(dropout),
            ResidualBlock(hidden_channels, hidden_channels // 2),
            ResidualBlock(hidden_channels // 2, hidden_channels // 2),
            nn.Linear(hidden_channels // 2, 1),
            nn.Sigmoid(),
        )

    def _create_node_pairs(self, batch_size: int, device: torch.device):
        """Create all possible node pairs for link prediction."""
        row, col = torch.cartesian_prod(
            torch.arange(self.num_nodes, device=device),
            torch.arange(self.num_nodes, device=device),
        ).T

        batch_row = row.unsqueeze(0).expand(batch_size, -1)
        batch_col = col.unsqueeze(0).expand(batch_size, -1)

        return batch_row, batch_col

    def forward(
        self,
        x: torch.Tensor,  # Shape: (batch_size, num_nodes, num_features, temporal_periods)
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass with A3TGCN2 temporal attention and GConvLSTM layers."""
        batch_size = x.size(0)
        device = x.device

        # Validate input dimensions
        assert (
            x.dim() == 4
        ), f"Expected 4D input (batch, nodes, features, time), got {x.dim()}D"
        assert (
            x.size(1) == self.num_nodes
        ), f"Expected {self.num_nodes} nodes, got {x.size(1)}"
        assert (
            x.size(2) == self.num_features
        ), f"Expected {self.num_features} features, got {x.size(2)}"
        assert (
            x.size(3) == self.temporal_periods
        ), f"Expected {self.temporal_periods} timesteps, got {x.size(3)}"

        # Reshape input for temporal attention
        # From: (batch_size, num_nodes, num_features, temporal_periods)
        # To: (batch_size, num_nodes, hidden_channels, temporal_periods)
        x = x.permute(
            0, 1, 3, 2
        )  # (batch_size, num_nodes, temporal_periods, num_features)
        x = x.reshape(
            batch_size, self.num_nodes, -1
        )  # Flatten temporal and feature dims
        h = self.input_proj(x)  # Project to hidden size
        h = h.view(
            batch_size, self.num_nodes, self.hidden_channels, 1
        )  # Add temporal dim back

        logger.debug(f"After reshape - h shape: {h.shape}")
        logger.debug(f"Edge index shape: {edge_index.shape}")
        if edge_weight is not None:
            logger.debug(f"Edge weight shape: {edge_weight.shape}")

        # Apply temporal attention
        h = self.temporal_attention(h, edge_index, edge_weight)
        logger.debug(f"After attention - h shape: {h.shape}")

        # Initialize or use provided hidden state
        if hidden_state is None:
            hidden_state = [None] * len(self.gconv_lstm_layers)

        new_hidden_states = []

        # Process through GConvLSTM layers
        for i, (gconv_lstm, norm) in enumerate(
            zip(self.gconv_lstm_layers, self.layer_norms)
        ):
            # Expand edge_index for batch dimension
            batch_edge_index = edge_index.repeat(1, batch_size)
            batch_edge_index[0] += (
                torch.arange(batch_size, device=device).repeat_interleave(
                    edge_index.size(1)
                )
                * self.num_nodes
            )
            batch_edge_index[1] += (
                torch.arange(batch_size, device=device).repeat_interleave(
                    edge_index.size(1)
                )
                * self.num_nodes
            )

            # Expand edge_weight if needed
            batch_edge_weight = (
                edge_weight.repeat(batch_size) if edge_weight is not None else None
            )

            # Reshape h for GConvLSTM
            h_reshaped = h.reshape(
                -1, self.hidden_channels
            )  # Flatten batch and node dimensions

            h_new, new_state = gconv_lstm(
                h_reshaped,
                batch_edge_index,
                batch_edge_weight if self.use_edge_weights else None,
                hidden_state[i],
            )

            # Reshape back
            h_new = h_new.view(batch_size, self.num_nodes, -1)

            # Apply normalization and residual connection
            h_new = norm(h_new)
            h_new = self.dropout(h_new)
            h = h + h_new
            new_hidden_states.append(new_state)

        # Project node embeddings
        h = self.node_proj(h)

        # Create and process node pairs
        batch_row, batch_col = self._create_node_pairs(batch_size, device)
        row_h = h[torch.arange(batch_size).unsqueeze(1), batch_row]
        col_h = h[torch.arange(batch_size).unsqueeze(1), batch_col]

        # Predict links
        pair_features = torch.cat([row_h, col_h], dim=-1)
        link_probs = self.link_predictor(pair_features)
        adj_pred = link_probs.view(batch_size, self.num_nodes, self.num_nodes)

        return adj_pred, tuple(new_hidden_states)

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
