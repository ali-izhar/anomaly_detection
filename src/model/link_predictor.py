# src/model/link_predictor.py

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, TransformerConv, GCNConv
from torch_geometric_temporal.nn.recurrent import DCRNN, GConvGRU, AGCRN
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_geometric.nn import global_mean_pool, global_add_pool

logger = logging.getLogger(__name__)


class TemporalTransformer(nn.Module):
    """Multi-head temporal transformer for sequence modeling."""

    def __init__(self, hidden_channels, num_heads=4, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            hidden_channels, num_heads, dropout=dropout
        )
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.norm2 = nn.LayerNorm(hidden_channels)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 4, hidden_channels),
        )

    def forward(self, x):
        # x shape: [batch, time, nodes, channels]
        B, T, N, C = x.shape
        x = x.transpose(0, 1)  # [time, batch, nodes*channels]
        x = x.reshape(T, B * N, C)

        # Self attention
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_out)

        # FFN
        x = self.norm2(x + self.ffn(x))

        return x.reshape(T, B, N, C).transpose(0, 1)


class SpatialTransformer(nn.Module):
    """Graph transformer for spatial dependencies."""

    def __init__(self, hidden_channels, heads=4, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.hidden_channels = hidden_channels
        self.head_dim = hidden_channels // heads

        # Fix transformer conv dimensions
        self.conv = TransformerConv(
            in_channels=hidden_channels,
            out_channels=self.head_dim,
            heads=heads,
            dropout=dropout,
            edge_dim=self.head_dim,  # Set edge dimension to match head dimension
            beta=True,
        )

        self.norm = nn.LayerNorm(hidden_channels)
        self.dropout = nn.Dropout(dropout)

        # Update edge projection to match head dimensions
        self.edge_proj = nn.Sequential(
            nn.Linear(1, self.head_dim),
            nn.LayerNorm(self.head_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x, edge_index, edge_weight=None):
        # Process edge weights if provided
        if edge_weight is not None:
            # Ensure edge_weight is 2D
            if edge_weight.dim() == 1:
                edge_weight = edge_weight.unsqueeze(-1)

            # Project edge weights to correct dimension
            edge_attr = self.edge_proj(edge_weight)  # [E, head_dim]

            # Debug shapes
            logger.debug(f"Edge weight shape: {edge_weight.shape}")
            logger.debug(f"Edge attr shape: {edge_attr.shape}")
            logger.debug(f"Input x shape: {x.shape}")
            logger.debug(f"Edge index shape: {edge_index.shape}")
        else:
            edge_attr = None

        # Apply transformer convolution
        x = self.conv(x, edge_index, edge_attr=edge_attr)
        x = self.norm(x)
        return self.dropout(x)


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
        num_layers: int = 3,  # Increased layers
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

        # Feature projection with residual
        self.feature_proj = nn.Sequential(
            nn.Linear(num_features, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            ResidualBlock(hidden_channels),
        )

        # Temporal processing
        self.temporal_transformer = TemporalTransformer(hidden_channels)

        # Fix temporal convolution to maintain correct output size
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=(1, temporal_periods),
                stride=1,
                padding=(0, temporal_periods // 2),
                padding_mode="replicate",  # Use replication padding
            ),
            nn.GELU(),
            nn.BatchNorm2d(hidden_channels),
            # Add a final 1x1 conv to ensure correct output size
            nn.Conv2d(
                in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=1
            ),
        )

        # Spatial-Temporal layers
        self.st_layers = nn.ModuleList()
        self.spatial_transformers = nn.ModuleList()
        self.agcrn_layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            self.st_layers.append(
                GConvGRU(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    K=K,
                    normalization="sym",
                    bias=True,
                )
            )

            self.spatial_transformers.append(SpatialTransformer(hidden_channels))

            self.agcrn_layers.append(
                GCNConv(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    improved=True,
                    add_self_loops=True,
                )
            )

            self.norms.append(
                nn.ModuleList(
                    [
                        nn.LayerNorm(hidden_channels),
                        nn.LayerNorm(hidden_channels),
                        nn.LayerNorm(hidden_channels),
                    ]
                )
            )

        # Node embeddings with regularization
        self.node_embeddings = nn.Parameter(torch.randn(num_nodes, hidden_channels))
        self.node_embedding_norm = nn.LayerNorm(hidden_channels)

        # Edge prediction network with correct dimensions
        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels * 2),  # Changed from 4 to 2
            nn.LayerNorm(hidden_channels * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            ResidualBlock(hidden_channels * 2),
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )

        # Regularization
        self.dropout = nn.Dropout(dropout)
        self.feature_dropout = nn.Dropout2d(0.1)  # Structural dropout

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x, edge_index, edge_weight=None):
        batch_size = x.size(0)
        device = x.device

        # Remove self-loops
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

        # Feature projection
        h = self.feature_proj(x.view(-1, self.num_features))
        h = h.view(
            batch_size, self.temporal_periods, self.num_nodes, -1
        )  # [B, T, N, C]
        logger.debug(f"After projection shape: {h.shape}")

        # Temporal processing
        h = self.temporal_transformer(h)  # [B, T, N, C]
        logger.debug(f"After transformer shape: {h.shape}")

        # Reshape for temporal convolution
        h = h.permute(0, 3, 2, 1)  # [B, C, N, T]
        logger.debug(f"Before conv shape: {h.shape}")

        # Apply temporal convolution and ensure correct output size
        h = self.temporal_conv(h)  # [B, C, N, T]
        h = h[..., 0]  # Take first temporal slice
        logger.debug(f"After conv shape: {h.shape}")

        # Rearrange to [batch, nodes, channels]
        h = h.permute(0, 2, 1)  # [B, N, C]
        logger.debug(f"Final temporal shape: {h.shape}")

        # Add regularized node embeddings
        node_emb = self.node_embedding_norm(self.node_embeddings)  # [N, C]
        node_emb = node_emb.unsqueeze(0).expand(batch_size, -1, -1)  # [B, N, C]
        h = h + node_emb

        # Process through layers
        hidden_states = []
        for i, (st_conv, spatial_trans, gcn, norms) in enumerate(
            zip(
                self.st_layers, self.spatial_transformers, self.agcrn_layers, self.norms
            )
        ):
            logger.debug(f"\nLayer {i} processing:")
            logger.debug(f"Input shape: {h.shape}")

            # Flatten batch and nodes for GNN operations
            h_flat = h.reshape(-1, self.hidden_channels)  # [B*N, C]

            # Process edge weights
            if edge_weight is not None:
                # Ensure edge weight is properly shaped
                if edge_weight.dim() == 1:
                    edge_weight = edge_weight.unsqueeze(-1)
                logger.debug(f"Edge weight shape: {edge_weight.shape}")

            # Spatial-temporal convolution with GConvGRU
            h_temp = st_conv(h_flat, edge_index, edge_weight, lambda_max=None)
            h_temp = h_temp.view(batch_size, self.num_nodes, -1)  # [B, N, C]
            h_temp = norms[0](h_temp)
            logger.debug(f"After GConvGRU shape: {h_temp.shape}")

            # Spatial transformer with edge weight processing
            h_struct = spatial_trans(h_flat, edge_index, edge_weight)
            h_struct = h_struct.view(batch_size, self.num_nodes, -1)  # [B, N, C]
            h_struct = norms[1](h_struct)
            logger.debug(f"After transformer shape: {h_struct.shape}")

            # Global features with GCN
            h_global = gcn(h_flat, edge_index, edge_weight)
            h_global = h_global.view(batch_size, self.num_nodes, -1)  # [B, N, C]
            h_global = norms[2](h_global)
            logger.debug(f"After GCN shape: {h_global.shape}")

            # Combine features with gating
            gates = torch.sigmoid(h_temp + h_struct + h_global)
            h = gates * h_temp + (1 - gates) * h_struct + h_global

            # Apply regularization
            h = self.feature_dropout(h)
            hidden_states.append(h)
            logger.debug(f"Layer {i} output shape: {h.shape}")

        # Multi-scale feature aggregation with attention
        weights = F.softmax(
            torch.stack(
                [
                    global_mean_pool(
                        h.view(-1, self.hidden_channels),
                        torch.zeros(
                            h.size(0) * h.size(1), dtype=torch.long, device=device
                        ),
                    )
                    for h in hidden_states
                ],
                dim=0,
            ),
            dim=0,
        )

        h_final = sum(w * h for w, h in zip(weights, hidden_states))

        # Create node pairs (excluding self-loops)
        row, col = torch.cartesian_prod(
            torch.arange(self.num_nodes, device=device),
            torch.arange(self.num_nodes, device=device),
        ).T

        mask = row != col
        row = row[mask]
        col = col[mask]

        batch_row = row.unsqueeze(0).expand(batch_size, -1)
        batch_col = col.unsqueeze(0).expand(batch_size, -1)

        # Get node pair features
        row_h = h_final[
            torch.arange(batch_size, device=device).unsqueeze(1), batch_row
        ]  # [B, E, C]
        col_h = h_final[
            torch.arange(batch_size, device=device).unsqueeze(1), batch_col
        ]  # [B, E, C]

        # Debug shapes and dtypes
        logger.debug(f"Row features shape: {row_h.shape}, dtype: {row_h.dtype}")
        logger.debug(f"Col features shape: {col_h.shape}, dtype: {col_h.dtype}")

        # Concatenate node features
        pair_features = torch.cat([row_h, col_h], dim=-1)  # [B, E, 2C]
        logger.debug(
            f"Pair features shape: {pair_features.shape}, dtype: {pair_features.dtype}"
        )

        # Predict links
        logits = self.edge_predictor(pair_features)  # [B, E, 1]
        logger.debug(f"Logits shape: {logits.shape}, dtype: {logits.dtype}")

        # Create adjacency matrix (without self-loops) with matching dtype
        adj_logits = torch.zeros(
            batch_size,
            self.num_nodes,
            self.num_nodes,
            device=device,
            dtype=logits.dtype,  # Match dtype with logits
        )
        adj_logits[:, row, col] = logits.squeeze(-1)
        logger.debug(
            f"Final adjacency shape: {adj_logits.shape}, dtype: {adj_logits.dtype}"
        )

        return adj_logits, None

    def predict_links(
        self, adj_pred: torch.Tensor, threshold: float = 0.5
    ) -> torch.Tensor:
        """Convert predicted probabilities to binary adjacency matrix."""
        return (adj_pred > threshold).float()


class ResidualBlock(nn.Module):
    """Residual block for the link predictor."""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Linear(channels, channels)
        self.conv2 = nn.Linear(channels, channels)
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        identity = x
        x = self.norm1(x)
        x = F.gelu(x)
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.norm2(x)
        x = F.gelu(x)
        x = self.conv2(x)
        return x + identity
