import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax
from torch_geometric.nn.inits import uniform

class GraphGAT(MessagePassing):
    def __init__(self, in_channels, out_channels, dropout=0.2, aggr='add'):
        super(GraphGAT, self).__init__(aggr=aggr)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout

        self.W = Parameter(torch.Tensor(in_channels, out_channels))
        self.a = Parameter(torch.Tensor(2 * out_channels, 1))
        self.bias = Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        uniform(self.in_channels, self.W)
        uniform(2 * self.out_channels, self.a)
        uniform(self.out_channels, self.bias)

    def forward(self, x, edge_index):
        # Linear transformation
        x = torch.matmul(x, self.W)
        return self.propagate(edge_index.detach(), x=x)

    def message(self, x_i, x_j, edge_index_i, size_i):
        # Concatenate for attention: [x_i || x_j]
        cat = torch.cat([x_i, x_j], dim=1)
        alpha = F.leaky_relu(torch.matmul(cat, self.a).squeeze(-1))
        
        # Normalize attention
        alpha = softmax(alpha, edge_index_i, num_nodes=size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training).unsqueeze(-1)
        
        return x_j * alpha
