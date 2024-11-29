# src/models/layers.py

"""Graph neural network layer implementations for spatio-temporal prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvLayer(nn.Module):
    """Graph Convolutional Layer implementing the propagation rule:

    H^(l+1) = sigma(D^(-1/2) * A * D^(-1/2) * H^(l) * W^(l))

    where:
    - H^(l) is in R^(N x d) and is the node feature matrix at layer l
    - A is in R^(N x N) and is the adjacency matrix with self-loops
    - D is the degree matrix where D_ii = sum_j A_ij
    - W^(l) is the trainable weight matrix
    - sigma is a nonlinear activation function
    """

    def __init__(self, in_dim: int, out_dim: int):
        """
        Args:
            in_dim: Input feature dimension d
            out_dim: Output feature dimension d'
        """
        super().__init__()
        self.weights = nn.Parameter(torch.FloatTensor(in_dim, out_dim))
        self.bias = nn.Parameter(torch.FloatTensor(out_dim))
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize weights using Glorot/Xavier initialization."""
        nn.init.xavier_uniform_(self.weights)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features matrix H in R^(B x N x d)
               where B is batch size, N is number of nodes
            adj: Normalized adjacency matrix D^(-1/2) * A * D^(-1/2) in R^(N x N)

        Returns:
            Updated node features H' in R^(B x N x d')
        """
        support = torch.matmul(x, self.weights)  # H * W
        output = torch.matmul(adj, support)  # D^(-1/2) * A * D^(-1/2) * H * W
        return output + self.bias


class TemporalAttention(nn.Module):
    """Temporal self-attention mechanism computing:

    Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V

    where Q, K, V are learned linear transformations of the input.
    This captures temporal dependencies between node features
    across different timesteps.
    """

    def __init__(self, hidden_dim: int):
        """
        Args:
            hidden_dim: Dimension of hidden representations d_model
        """
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor in R^(B x T x N x d) where:
               B is batch size
               T is sequence length
               N is number of nodes
               d is hidden dimension

        Returns:
            Attended features in R^(B x T x N x d)
        """
        q = self.query(x)  # Q = X * W_q
        k = self.key(x)  # K = X * W_k
        v = self.value(x)  # V = X * W_v

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores / torch.sqrt(torch.tensor(x.size(-1), dtype=torch.float32))
        attention = F.softmax(scores, dim=-1)

        return torch.matmul(attention, v)
