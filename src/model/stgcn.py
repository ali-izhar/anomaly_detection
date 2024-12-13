import torch
import torch.nn as nn
from torch_geometric_temporal.nn.attention import STConv


class DynamicGraphPredictor(nn.Module):
    """
    Spatio-Temporal Graph Convolutional Network for link prediction.
    Note: DynamicLinkPredictor is recommended over this model for link prediction.
    """

    def __init__(
        self,
        num_nodes: int,
        num_features: int,
        hidden_channels: int = 64,
        temporal_kernel_size: int = 3,
        spatial_kernel_size: int = 2,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super(DynamicGraphPredictor, self).__init__()

        self.num_nodes = num_nodes
        self.num_features = num_features

        # Input projection
        self.input_proj = nn.Linear(num_features, hidden_channels)

        # ST-Conv blocks
        self.st_blocks = nn.ModuleList(
            [
                STConv(
                    num_nodes=num_nodes,
                    in_channels=hidden_channels,
                    hidden_channels=hidden_channels,
                    out_channels=hidden_channels,
                    kernel_size=temporal_kernel_size,
                    K=spatial_kernel_size,
                )
                for _ in range(num_layers)
            ]
        )

        # Layer normalization layers
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_channels) for _ in range(num_layers)]
        )

        self.dropout = nn.Dropout(dropout)

        # Output projection to adjacency matrix
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_channels // 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, edge_index, edge_weight=None):
        """
        Forward pass

        Args:
            x: Node features (batch_size, num_timesteps, num_nodes, num_features)
            edge_index: Edge indices (2, num_edges)
            edge_weight: Optional edge weights (num_edges,)

        Returns:
            Predicted adjacency matrix (batch_size, num_nodes, num_nodes)
        """
        batch_size = x.size(0)

        # Project input
        x = self.input_proj(x)

        # Process through ST-Conv blocks with residual connections
        for block, norm in zip(self.st_blocks, self.layer_norms):
            x_new = block(x, edge_index, edge_weight)
            x_new = norm(x_new)
            x_new = self.dropout(x_new)
            x = x + x_new

        # Generate all possible node pairs
        row, col = torch.cartesian_prod(
            torch.arange(self.num_nodes, device=x.device),
            torch.arange(self.num_nodes, device=x.device),
        ).T

        # Expand for batch dimension
        batch_row = row.unsqueeze(0).expand(batch_size, -1)
        batch_col = col.unsqueeze(0).expand(batch_size, -1)

        # Get final node embeddings
        node_embeddings = x[:, -1]  # Take last timestep

        # Get node pair features
        row_h = node_embeddings[torch.arange(batch_size).unsqueeze(1), batch_row]
        col_h = node_embeddings[torch.arange(batch_size).unsqueeze(1), batch_col]
        pair_features = torch.cat([row_h, col_h], dim=-1)

        # Predict adjacency matrix
        adj_pred = self.output_proj(pair_features).view(
            batch_size, self.num_nodes, self.num_nodes
        )

        return adj_pred
