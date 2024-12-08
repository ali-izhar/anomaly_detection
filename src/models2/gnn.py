# src/models2/gnn.py

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


class GNNEncoder(nn.Module):
    def __init__(
        self,
        num_nodes,
        node_feat_dim=16,
        hidden_dim=64,
        num_layers=2,
        activation="relu",
    ):
        super(GNNEncoder, self).__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(node_feat_dim, hidden_dim))

        for _ in range(1, num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError("Unsupported activation")

    def forward(self, adj_matrices, node_features):
        """
        Args:
            adj_matrices (batch, N, N): Adjacency matrices
            node_features (batch, N, D): Initial node features
        Returns:
            batch, hidden_dim: Graph-level embeddings
        """
        batch_size, num_nodes, _ = adj_matrices.size()
        graph_embeddings = []
        for i in range(batch_size):
            adj = adj_matrices[i]  # (N, N)
            edge_index = adj.nonzero(as_tuple=False).t()  # (2, E)
            x = node_features[i]  # (N, D)
            for conv in self.convs:
                x = conv(x, edge_index)
                x = self.activation(x)
            # Global mean pooling
            graph_emb = torch.mean(x, dim=0)  # (hidden_dim)
            graph_embeddings.append(graph_emb)
        graph_embeddings = torch.stack(graph_embeddings, dim=0)  # (batch, hidden_dim)
        return graph_embeddings
