import torch
import torch.nn.functional as F
from torch_geometric.utils import dropout_adj
from torch_geometric.nn import inits, MessagePassing
from .kafnets import KAF

class GraphConv(MessagePassing):
    """
    Graph convolutional layer as in Fout, et al. (2017).
    """
    def __init__(self, node_channels, edge_channels, out_channels, drop_prob, D, boundary, kernel):
        super(GraphConv, self).__init__(aggr='mean')
        self.node_channels = node_channels
        self.edge_channels = edge_channels
        self.out_channels = out_channels
         
        self.node_weight = torch.nn.Parameter(torch.FloatTensor(node_channels, out_channels))
        self.edge_weight = torch.nn.Parameter(torch.FloatTensor(edge_channels, out_channels))
        self.neighbor_weight = torch.nn.Parameter(torch.FloatTensor(node_channels, out_channels))
        self.bias = torch.nn.Parameter(torch.FloatTensor(out_channels))
        self.reset_parameters()

        self.KAF_act = KAF(out_channels, D = D, conv = False, boundary = boundary, kernel = kernel)        
        self.drop_prob = drop_prob

    def reset_parameters(self):
        inits.uniform(self.node_channels * self.out_channels, self.node_weight)
        inits.uniform(self.edge_channels * self.out_channels, self.edge_weight)
        inits.uniform(self.node_channels * self.out_channels, self.neighbor_weight)
        inits.zeros(self.bias)
        
    def forward(self, x, edge_index, edge_attr):
        # x: [N, node_channels]
        # edge_index: [2, E]
        # edge_attr: [E, edge_channels]
        # calling propagate function consequently call message and update
        edge_index, edge_attr = dropout_adj(edge_index, edge_attr = edge_attr, p = self.drop_prob)
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
        
    def message(self, x_j, edge_attr):
        # x_j: [E, node_channels]
        # specified how "message" for each node pair (x_i, x_j) is constructed. Since it follows the calls of propagate, it can take any argument passing to propagate
        # performed for each edge_index
        return torch.matmul(edge_attr, self.edge_weight) + torch.matmul(x_j, self.neighbor_weight)
        
    def update(self, aggr_out, x):
        # takes in the aggregated message and other arguments passed into propagate, assigning a new embedding value for each node
        return self.KAF_act(aggr_out + torch.matmul(x, self.node_weight) + self.bias)
    
    def __repr__(self):
        return '{}({}, {}, {})'.format(self.__class__.__name__, self.node_channels, self.edge_channels, self.out_channels)
    
    
class GraphNetLayers(torch.nn.Module):
    """
    Graph neural network layers.
    """
    def __init__(self, input_dim, output_dim, num_layers, drop_prob, D, boundary, kernel):
        """
        Constructs a new `GraphNet` with specified configuration.
        """
        super(GraphNetLayers, self).__init__()
        gcn_layers = [GraphConv(input_dim, 2, output_dim, drop_prob, D, boundary, kernel)]
        
        if num_layers > 1:
            for i in range(num_layers - 1):
                gcn_layers.append(GraphConv(output_dim, 2, output_dim, drop_prob, D, boundary, kernel))

        self.graph_layers = torch.nn.ModuleList(gcn_layers)

    def forward(self, x, edge_index, edge_attr):
        """
        Forward pass.
        """
        output_x = x
        for layer in self.graph_layers:
            output_x = layer(output_x, edge_index, edge_attr)
        return output_x
