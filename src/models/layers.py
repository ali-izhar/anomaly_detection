# src/models/layers.py

"""Graph neural network layer implementations for spatio-temporal prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvLayer(nn.Module):
    """Graph Convolutional Layer implementing the propagation rule:

    H^(l+1) = sigma(D^(-1/2) * A * D^(-1/2) * H^(l) * W^(l))

    where:
    - H^(l) is in R^(N x d) (per sequence element) and is the node feature matrix at layer l
    - A is in R^(N x N) and is the adjacency matrix with self-loops
    - D is the degree matrix where D_ii = sum_j A_ij
    - W^(l) is the trainable weight matrix
    - sigma is a nonlinear activation function
    """

    def __init__(self, in_dim: int, out_dim: int, activation: nn.Module = None):
        """
        Args:
            in_dim: Input feature dimension d
            out_dim: Output feature dimension d'
            activation: Optional nonlinear activation function (e.g., nn.ReLU())
        """
        super().__init__()
        self.weights = nn.Parameter(torch.FloatTensor(in_dim, out_dim))
        self.bias = nn.Parameter(torch.FloatTensor(out_dim))
        self.activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize weights using Glorot/Xavier initialization."""
        nn.init.xavier_uniform_(self.weights)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features matrix H in R^(B x N x d)
               B: batch size (32)
               N: number of nodes (100)
               d: feature dim (varies by feature type)
            adj: Normalized adjacency matrix D^(-1/2) * A * D^(-1/2) 
                Either [N x N] or [B x N x N]
        """
        # (B x N x d) * (d x d') -> (B x N x d')
        support = torch.matmul(x, self.weights)
        
        # Handle batched or single adjacency matrix
        if adj.dim() == 2:
            # Expand adjacency matrix for batch processing
            adj = adj.unsqueeze(0).expand(x.size(0), -1, -1)
        
        # (B x N x N) * (B x N x d') -> (B x N x d')
        output = torch.matmul(adj, support)
        output = output + self.bias
        return output if self.activation is None else self.activation(output)


class TemporalAttention(nn.Module):
    """Temporal self-attention mechanism:

    Attention(Q, K, V) = softmax((Q * K^T) / sqrt(d_k)) * V

    This captures temporal dependencies between node features across timesteps.
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
            x: Input tensor in R^(B x T x N x d)
               B: batch size (32)
               T: sequence length (window_size=20)
               N: number of nodes (100)
               d: feature dimension (hidden_dim)

        Returns:
            Attended features in R^(B x T x N x d)
        """
        # Add shape assertions for validation
        batch_size, seq_len, num_nodes, hidden_dim = x.size()
        assert num_nodes == 100, f"Expected 100 nodes, got {num_nodes}"
        assert seq_len == 20, f"Expected sequence length 20, got {seq_len}"
        
        q = self.query(x)  # (B x T x N x d)
        k = self.key(x)    # (B x T x N x d)
        v = self.value(x)  # (B x T x N x d)

        # Compute attention scores (B x T x N x N)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (hidden_dim ** 0.5)
        attention = F.softmax(scores, dim=-1)
        
        # Weighted sum of values (B x T x N x d)
        output = torch.matmul(attention, v)
        return output
