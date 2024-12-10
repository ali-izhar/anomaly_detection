import torch
import torch.nn as nn
from torch_geometric_temporal.nn.attention import STConv


class DynamicGraphPredictor(nn.Module):
    def __init__(
        self,
        num_nodes,
        num_features,
        hidden_channels=32,
        temporal_kernel_size=3,
        spatial_kernel_size=2,
        num_layers=2,
    ):
        """
        Initialize the STGCN model

        Args:
            num_nodes: int, number of nodes in the graph
            num_features: int, number of features per node
            hidden_channels: int, number of hidden channels
            temporal_kernel_size: int, size of temporal convolution kernel
            spatial_kernel_size: int, size of spatial convolution kernel
            num_layers: int, number of ST-Conv blocks
        """
        super(DynamicGraphPredictor, self).__init__()

        self.num_nodes = num_nodes
        self.num_features = num_features

        # Stack of ST-Conv blocks
        self.st_blocks = nn.ModuleList()

        # First block
        self.st_blocks.append(
            STConv(
                num_nodes=num_nodes,
                in_channels=num_features,
                hidden_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=temporal_kernel_size,
                K=spatial_kernel_size,
            )
        )

        # Middle blocks
        for _ in range(num_layers - 2):
            self.st_blocks.append(
                STConv(
                    num_nodes=num_nodes,
                    in_channels=hidden_channels,
                    hidden_channels=hidden_channels,
                    out_channels=hidden_channels,
                    kernel_size=temporal_kernel_size,
                    K=spatial_kernel_size,
                )
            )

        # Final block
        self.st_blocks.append(
            STConv(
                num_nodes=num_nodes,
                in_channels=hidden_channels,
                hidden_channels=hidden_channels,
                out_channels=num_features,  # Output same number of features as input
                kernel_size=temporal_kernel_size,
                K=spatial_kernel_size,
            )
        )

    def forward(self, x, edge_index, edge_weight):
        """
        Forward pass

        Args:
            x: tensor of shape (batch_size, num_timesteps, num_nodes, num_features)
            edge_index: tensor of shape (2, num_edges)
            edge_weight: tensor of shape (num_edges,)

        Returns:
            Predictions tensor of shape (batch_size, num_nodes, num_features)
        """
        # Pass through ST-Conv blocks
        for block in self.st_blocks:
            x = block(x, edge_index, edge_weight)

        return x
