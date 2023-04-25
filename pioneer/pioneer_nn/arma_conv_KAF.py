import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import inits, MessagePassing, ARMAConv
from torch_geometric.utils import dropout_adj
from .kafnets import KAF

class ARMAConvLayers(torch.nn.Module):
    """
    Auto-regressive moving average convolution layers.
    """
    def __init__(self, input_dim, output_dim, n_layers, drop_prob, D, boundary, kernel):
        """
        Constructs a new `ARMAConvLayers` instance with specified configuration.
        """
        super(ARMAConvLayers, self).__init__()
        self.KAF_act = KAF(output_dim, D = D, conv = False, boundary = boundary, kernel = kernel)
        self.drop_prob = drop_prob
        self.graph_layers = [ARMAConv(input_dim, output_dim, act = None)]
        if n_layers > 1:
            for i in range(n_layers - 1):
                self.graph_layers.append(ARMAConv(output_dim, output_dim, act = None))

        self.graph_layers = torch.nn.ModuleList(self.graph_layers)
    
    def forward(self, x, edge_index):
        """
        Forward pass.
        """
        output_x = x
        for layer in self.graph_layers:
            output_x = self.KAF_act(layer(output_x, edge_index))
        return output_x

class ARMAConvLayersReLu(torch.nn.Module):
    """
    Auto-regressive moving average convolution layers.
    """
    def __init__(self, input_dim, output_dim, n_layers, n_stacks):
        """
        Constructs a new `ARMAConvLayers` instance with specified configuration.
        """
        super(ARMAConvLayersReLu, self).__init__()
        self.graph_layers = ARMAConv(input_dim, output_dim, num_stacks = n_layers, num_layers = n_stacks)

    def forward(self, x, edge_index):
        """
        Forward pass.
        """
        output_x = self.graph_layers(output_x, edge_index)
        return output_x
