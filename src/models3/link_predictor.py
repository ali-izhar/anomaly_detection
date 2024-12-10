import torch
import torch.nn as nn
from torch_geometric_temporal.nn.recurrent import GCLSTM


class DynamicLinkPredictor(nn.Module):
    """
    A neural network model for dynamic link prediction using GC-LSTM.

    This model combines a Graph Convolutional LSTM with a link prediction
    head to predict the next timestep's adjacency matrix.
    """

    def __init__(
        self, num_nodes, num_features, hidden_channels=32, num_layers=2, dropout=0.1
    ):
        """
        Initialize the model.

        Args:
            num_nodes: int, number of nodes in the graph
            num_features: int, number of features per node
            hidden_channels: int, dimension of hidden representations
            num_layers: int, number of GCLSTM layers
            dropout: float, dropout probability
        """
        super(DynamicLinkPredictor, self).__init__()

        self.num_nodes = num_nodes
        self.num_features = num_features

        # Stack of GC-LSTM layers
        self.gclstm_layers = nn.ModuleList(
            [
                GCLSTM(
                    in_channels=num_features if i == 0 else hidden_channels,
                    out_channels=hidden_channels,
                    K=3,  # Number of filter taps
                )
                for i in range(num_layers)
            ]
        )

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # MLP for link prediction
        self.link_predictor = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, edge_index, edge_weight):
        """
        Forward pass of the model.

        Args:
            x: tensor of node features
            edge_index: tensor of edge indices
            edge_weight: tensor of edge weights

        Returns:
            tensor: predicted adjacency matrix for next timestep
        """
        # Pass through GC-LSTM layers
        h = x
        for gclstm in self.gclstm_layers:
            # GCLSTM returns a tuple of (hidden_state, cell_state)
            h = gclstm(h, edge_index, edge_weight)[0]  # Get only the hidden state
            h = self.dropout(h)

        # Create all possible node pairs
        row, col = torch.cartesian_prod(
            torch.arange(self.num_nodes, device=x.device),
            torch.arange(self.num_nodes, device=x.device),
        ).T

        # Get node pair features
        row_h = h[row]
        col_h = h[col]
        pair_features = torch.cat([row_h, col_h], dim=-1)

        # Predict link probabilities
        link_probs = self.link_predictor(pair_features)

        # Reshape to adjacency matrix
        adj_pred = link_probs.view(self.num_nodes, self.num_nodes)

        return adj_pred
