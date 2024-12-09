# src/models2/gnn.py

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict
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
        num_layers: int = 3,
        dropout: float = 0.3,
        activation: str = "relu",
        config: Optional[Dict] = None,
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
            config: Configuration dictionary
        """
        super().__init__()
        self.num_nodes = num_nodes
        self.in_channels = in_channels

        # Get model config
        if config is None:
            config = {}
        gnn_config = config.get("model", {}).get("gnn", {})
        hw_config = config.get("hardware", {})

        # Use config values with defaults
        self.hidden_dim = gnn_config.get("hidden_dim", hidden_channels)
        self.num_layers = gnn_config.get("num_layers", num_layers)
        self.dropout_rate = gnn_config.get("dropout", dropout)
        self.activation_type = gnn_config.get("activation", activation)

        # Memory optimization settings
        self.use_checkpoint = hw_config.get("gradient_checkpointing", True)
        self.memory_efficient = hw_config.get("memory_efficient_attention", True)

        # Get attention config
        attention_config = hw_config.get("attention", {})

        # Add memory-efficient attention mechanism
        attention_heads = attention_config.get("num_heads", 4)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_channels,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True,
            # Enable memory efficient attention
            **(
                {"attention_probs_dropout_prob": dropout}
                if hasattr(nn.MultiheadAttention, "attention_probs_dropout_prob")
                else {}
            )
        )

        # Add chunking for attention computation
        self.chunk_size = attention_config.get("chunk_size", 128)

        # Define chunk attention function
        def chunk_attention(h):
            chunks = []
            for i in range(0, h.size(0), self.chunk_size):
                chunk = h[i : i + self.chunk_size]
                chunk = chunk.unsqueeze(0)  # Add sequence dimension
                chunk_out, _ = self.attention(chunk, chunk, chunk)
                chunks.append(chunk_out.squeeze(0))
            return torch.cat(chunks, dim=0)

        self.chunk_attention = chunk_attention

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

        # Add layer normalization
        self.layer_norm = nn.LayerNorm(hidden_channels)

        # Add skip connections
        self.use_skip = True if num_layers > 1 else False

        # Add memory-efficient attention mechanism
        attention_heads = (
            config.get("hardware", {}).get("attention", {}).get("num_heads", 4)
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_channels,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True,
            # Enable memory efficient attention
            **(
                {"attention_probs_dropout_prob": dropout}
                if hasattr(nn.MultiheadAttention, "attention_probs_dropout_prob")
                else {}
            )
        )

        # Add chunking for attention computation
        self.chunk_size = (
            config.get("hardware", {}).get("attention", {}).get("chunk_size", 128)
        )

        # Add feature importance tracking
        self.feature_importance = nn.Parameter(torch.ones(in_channels))

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

            # Initial projection with layer norm
            h = self.input_proj(x_b)
            h = self.layer_norm(h)

            # Store initial representation for skip connection
            h_initial = h

            # Apply GNN layers with skip connections
            for conv, batch_norm in zip(self.convs, self.batch_norms):
                h_conv = conv(h, edge_index)
                h_conv = batch_norm(h_conv)

                # Apply chunked attention using the class method
                h_conv = self.chunk_attention(h_conv)

                h_conv = self.activation(h_conv)
                h_conv = self.dropout(h_conv)

                # Skip connection
                if self.use_skip:
                    h = h + h_conv
                else:
                    h = h_conv

            # Final skip connection to initial representation
            if self.use_skip:
                h = h + h_initial

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

            # Apply feature importance
            xt = xt * self.feature_importance

            # Process current timestep
            ht = self._process_single_timestep(
                xt, adjt
            )  # [batch_size, N, hidden_channels]

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
