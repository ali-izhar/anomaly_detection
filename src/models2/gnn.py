# src/models2/gnn.py

import torch
import torch.nn as nn
from typing import Tuple, Optional
from torch import Tensor
from torch_geometric.nn import GCNConv


class TemporalGNN(nn.Module):
    """Temporal Graph Neural Network for processing sequences of graphs.

    Processes a sequence of graphs using a combination of GNN layers for spatial
    dependencies and temporal layers for time dependencies.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_nodes: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        activation: str = "relu",
    ) -> None:
        """Initialize the Temporal GNN.

        Args:
            in_channels: Number of input features per node
            hidden_channels: Number of hidden features per node
            out_channels: Number of output features per node
            num_nodes: Number of nodes in the graph
            num_layers: Number of GNN layers
            dropout: Dropout probability
            activation: Activation function ('relu' or 'gelu')
        """
        super().__init__()
        self.num_nodes = num_nodes
        self.in_channels = in_channels

        # Input projection
        self.input_proj = nn.Linear(in_channels, hidden_channels)

        # GNN layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))

        # Output projection
        self.output_proj = nn.Linear(hidden_channels, out_channels)

        # Other components
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU() if activation == "relu" else nn.GELU()

    def _process_single_timestep(self, x: Tensor, adj: Tensor) -> Tensor:
        """Process a single timestep of the temporal graph.

        Args:
            x: Node features [batch_size, num_features]
            adj: Adjacency matrix [batch_size, N, N]

        Returns:
            Updated node features [batch_size, N, hidden_channels]
        """
        batch_size = x.size(0)

        # Process each batch item separately since GCNConv expects single graphs
        batch_embeddings = []
        for b in range(batch_size):
            # Get single adjacency matrix
            adj_b = adj[b]  # [N, N]
            
            # Reshape features for current batch item
            x_b = x[b].unsqueeze(0).expand(self.num_nodes, -1)  # [N, in_channels]
            
            # Get edge index from adjacency matrix (ensure 2 rows format)
            edge_index = adj_b.nonzero().t()  # [2, num_edges]
            if edge_index.size(0) != 2:
                edge_index = edge_index.t()  # Transpose if needed
            
            # Initial projection
            h = self.input_proj(x_b)  # [N, hidden_channels]
            
            # Apply GNN layers
            for conv, batch_norm in zip(self.convs, self.batch_norms):
                # Apply convolution
                h_conv = conv(h, edge_index)
                # Apply batch norm
                h_conv = batch_norm(h_conv)
                # Apply activation and dropout
                h_conv = self.activation(h_conv)
                h_conv = self.dropout(h_conv)
                # Residual connection
                h = h + h_conv
            
            batch_embeddings.append(h)
        
        # Stack all batch embeddings
        h = torch.stack(batch_embeddings)  # [batch_size, N, hidden_channels]
        
        return h

    def forward(
        self, x: Tensor, adj: Tensor, mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """Forward pass through the temporal GNN.

        Args:
            x: Node features [batch_size, seq_len, num_features]
            adj: Adjacency matrices [batch_size, seq_len, N, N]
            mask: Optional mask [batch_size, seq_len]

        Returns:
            node_embeddings: [batch_size, seq_len, N, out_channels]
            graph_embeddings: [batch_size, seq_len, out_channels]
        """
        batch_size, seq_len = x.size(0), x.size(1)
        
        # Process each timestep
        node_embeddings = []
        graph_embeddings = []
        
        for t in range(seq_len):
            # Get features and adj matrix for current timestep
            xt = x[:, t]  # [batch_size, num_features]
            adjt = adj[:, t]  # [batch_size, N, N]
            
            # Process current timestep
            ht = self._process_single_timestep(xt, adjt)  # [batch_size, N, hidden_channels]
            
            # Project to output dimension
            ht = self.output_proj(ht)  # [batch_size, N, out_channels]
            
            # Get graph-level embeddings using mean pooling
            gt = torch.mean(ht, dim=1)  # [batch_size, out_channels]
            
            node_embeddings.append(ht)
            graph_embeddings.append(gt)
        
        # Stack time steps
        node_embeddings = torch.stack(node_embeddings, dim=1)
        graph_embeddings = torch.stack(graph_embeddings, dim=1)
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(-1)
            graph_embeddings = graph_embeddings * mask
            mask = mask.unsqueeze(-2)
            node_embeddings = node_embeddings * mask
        
        return node_embeddings, graph_embeddings


class TemporalGNNPredictor(nn.Module):
    """Temporal GNN model for time series prediction on graphs."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        forecast_horizon: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        """Initialize the predictor.

        Args:
            in_channels: Number of input features
            hidden_channels: Number of hidden features
            out_channels: Number of output features
            forecast_horizon: Number of future time steps to predict
            num_layers: Number of GNN layers
            dropout: Dropout probability
        """
        super().__init__()

        self.temporal_gnn = TemporalGNN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
            num_layers=num_layers,
            dropout=dropout,
        )

        # MLP for final prediction
        self.predictor = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels * forecast_horizon),
        )

    def forward(self, x: Tensor, adj: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Forward pass.

        Args:
            x: Input features [batch_size, seq_len, num_nodes, in_channels]
            adj: Adjacency matrices [batch_size, seq_len, num_nodes, num_nodes]
            mask: Optional mask for variable length sequences [batch_size, seq_len]

        Returns:
            Predictions for future time steps [batch_size, forecast_horizon, out_channels]
        """
        # Get embeddings from temporal GNN
        _, graph_embeddings = self.temporal_gnn(x, adj, mask)

        # Use the last timestep's embedding for prediction
        final_embedding = graph_embeddings[:, -1]  # [batch_size, hidden_channels]

        # Generate prediction
        pred = self.predictor(
            final_embedding
        )  # [batch_size, out_channels * forecast_horizon]
        batch_size = pred.size(0)
        pred = pred.view(
            batch_size, -1, pred.size(-1)
        )  # [batch_size, forecast_horizon, out_channels]

        return pred
