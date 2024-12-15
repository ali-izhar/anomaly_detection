import torch
import torch.nn as nn
from torch_geometric_temporal.nn.attention import STConv
from torch_geometric_temporal.nn.recurrent import GConvGRU
import logging

logger = logging.getLogger(__name__)


class STGCNLinkPredictor(nn.Module):
    def __init__(
        self,
        num_nodes: int = 30,
        in_channels: int = 6,
        hidden_channels: int = 64,
        num_layers: int = 2,
        kernel_size: int = 3,
        K: int = 3,
        dropout: float = 0.1,
    ):
        super(STGCNLinkPredictor, self).__init__()

        # Stack of ST-Conv blocks
        self.st_blocks = nn.ModuleList()

        # First ST-Conv block
        self.st_blocks.append(
            STConv(
                num_nodes=num_nodes,
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=kernel_size,
                K=K,
            )
        )

        # Additional ST-Conv blocks
        for _ in range(num_layers - 1):
            self.st_blocks.append(
                STConv(
                    num_nodes=num_nodes,
                    in_channels=hidden_channels,
                    hidden_channels=hidden_channels,
                    out_channels=hidden_channels,
                    kernel_size=kernel_size,
                    K=K,
                )
            )

        self.dropout = nn.Dropout(dropout)

        # Link prediction head
        self.link_predictor = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, x, edge_index, edge_weight=None):
        """
        Args:
            x: Node features [batch_size, num_nodes, in_channels, sequence_length]
            edge_index: Graph connectivity [2, num_edges]
            edge_weight: Edge weights [num_edges]
        """
        logger.info(f"Input x shape: {x.shape}")
        logger.info(f"Edge index shape: {edge_index.shape}")
        if edge_weight is not None:
            logger.info(f"Edge weight shape: {edge_weight.shape}")

        # Permute input to match STConv expectations
        x = x.permute(0, 3, 1, 2)
        logger.info(f"After permute x shape: {x.shape}")

        # Process through ST-Conv blocks
        for i, st_block in enumerate(self.st_blocks):
            x = st_block(x, edge_index, edge_weight)
            x = self.dropout(x)
            logger.info(f"After ST-Conv block {i+1} shape: {x.shape}")

        # Get final node embeddings
        node_embeddings = x[:, -1]
        logger.info(f"Node embeddings shape: {node_embeddings.shape}")

        batch_size, num_nodes, hidden_dim = node_embeddings.shape

        # Create all possible node pairs
        source_nodes = node_embeddings.unsqueeze(2).repeat(1, 1, num_nodes, 1)
        target_nodes = node_embeddings.unsqueeze(1).repeat(1, num_nodes, 1, 1)
        logger.info(f"Source nodes shape: {source_nodes.shape}")
        logger.info(f"Target nodes shape: {target_nodes.shape}")

        # Concatenate features
        pair_features = torch.cat([source_nodes, target_nodes], dim=-1)
        logger.info(f"Pair features shape: {pair_features.shape}")

        # Predict links
        link_logits = self.link_predictor(pair_features).squeeze(-1)
        logger.info(f"Output logits shape: {link_logits.shape}")

        # Remove sigmoid activation (will be handled by BCEWithLogitsLoss)
        return link_logits


class GConvGRULinkPredictor(nn.Module):
    def __init__(
        self,
        num_nodes: int = 30,
        in_channels: int = 6,
        hidden_channels: int = 64,
        num_layers: int = 2,
        K: int = 3,
        dropout: float = 0.1,
    ):
        super(GConvGRULinkPredictor, self).__init__()

        # Stack of GConvGRU layers using existing implementation
        self.gru_layers = nn.ModuleList()

        # First layer
        self.gru_layers.append(
            GConvGRU(
                in_channels=in_channels,
                out_channels=hidden_channels,
                K=K,
            )
        )

        # Additional layers
        for _ in range(num_layers - 1):
            self.gru_layers.append(
                GConvGRU(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    K=K,
                )
            )

        self.dropout = nn.Dropout(dropout)

        # Link prediction head
        self.link_predictor = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, x, edge_index, edge_weight=None):
        """
        Args:
            x: Node features [batch_size, sequence_length, num_nodes, in_channels]
            edge_index: Graph connectivity [2, num_edges]
            edge_weight: Edge weights [num_edges]
        """
        batch_size, seq_len, num_nodes, _ = x.shape

        # Process through GRU layers
        h = None  # Initial hidden state

        # Process each timestep through GRU layers
        for t in range(seq_len):
            x_t = x[:, t]  # [batch_size, num_nodes, in_channels]

            # Process through all GRU layers
            for gru in self.gru_layers:
                h = gru(x_t, edge_index, edge_weight, h)
                h = self.dropout(h)

        # Use final hidden state for link prediction
        node_embeddings = h.view(
            batch_size, num_nodes, -1
        )  # [batch_size, num_nodes, hidden_channels]

        # Create all possible node pairs
        source_nodes = node_embeddings.unsqueeze(2).repeat(1, 1, num_nodes, 1)
        target_nodes = node_embeddings.unsqueeze(1).repeat(1, num_nodes, 1, 1)

        # Concatenate source and target node features
        pair_features = torch.cat([source_nodes, target_nodes], dim=-1)

        # Predict links (return logits without sigmoid)
        link_logits = self.link_predictor(pair_features).squeeze(-1)
        return link_logits  # Removed sigmoid activation
