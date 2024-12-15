import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseTemporalGraphModel(nn.Module, ABC):
    def __init__(self, num_nodes, num_features, seq_len):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.seq_len = seq_len
        
    @abstractmethod
    def forward(self, x, edge_index, edge_weight):
        """
        Forward pass of the model
        Args:
            x: Node features [batch_size, seq_len, num_nodes, num_features]
            edge_index: Graph connectivity [2, num_edges]
            edge_weight: Edge weights [num_edges]
        """
        pass

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad) 