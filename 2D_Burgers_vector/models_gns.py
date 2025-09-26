#Import the necessary standard set of libraries

#PyTorch NN essentials
import torch
import torch.nn as nn
import torch.nn.functional as F

#PyTorch Geometric essentials
import torch_geometric.nn as pyg_nn


def build_mlp(input_dim, hidden_dims, output_dim):
    layers = []
    last_dim = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(last_dim, h))
        layers.append(nn.GELU())       #Earlier this was RELU
        last_dim = h
    layers.append(nn.Linear(last_dim, output_dim))
    return layers

class GNSLayer(pyg_nn.MessagePassing):
    def __init__(self, node_dim, edge_dim, hidden_dim, n_mlp_layers=2):
        super().__init__(aggr='mean')  # or 'add'

        # Edge MLP: inputs = [x_i || x_j || edge_attr]
        self.edge_mlp = nn.Sequential(
            *build_mlp(2 * node_dim + edge_dim,
                      [hidden_dim] * n_mlp_layers,
                      edge_dim),
            nn.LayerNorm(edge_dim)
        )

        # Node MLP: inputs = [x || aggregated_edge_features]
        self.node_mlp = nn.Sequential(
            *build_mlp(node_dim + edge_dim,
                      [hidden_dim] * n_mlp_layers,
                      node_dim),
            nn.LayerNorm(node_dim)
        )
        
    def forward(self, x, edge_index, edge_attr):
        x_res = x
        edge_res = edge_attr
        x_updated, edge_attr_updated = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return x_res + x_updated, edge_res + edge_attr_updated

    def message(self, x_i, x_j, edge_attr):
        edge_input = torch.cat([x_i, x_j, edge_attr], dim=-1)
        edge_attr_updated = self.edge_mlp(edge_input)
        return edge_attr_updated  # <-- Return updated edge attributes explicitly

    def update(self, aggr_out, x, edge_attr):
        node_input = torch.cat([x, aggr_out], dim=-1)
        x_updated = self.node_mlp(node_input)
        return x_updated, edge_attr  # <-- Return updated edge_attr here too


class EncodeProcessDecodeGNS(nn.Module):
    def __init__(self, input_dim, hidden_dim, gnn_dim, edge_dim, output_dim, M=6):
        super().__init__()
        
        #--------------Define MLPs for Encoder block (for node and edge)------------#
        self.node_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),          #Earlier, all activations here were LeakyRELU
            nn.Linear(hidden_dim, gnn_dim)
        )

        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, edge_dim)
        )
        #-------------------------------------------------------#
        
        
        #------------------"M" rounds of message passing layers-------#
        self.gns_layers = nn.ModuleList([
            GNSLayer(gnn_dim, edge_dim, hidden_dim) for _ in range(M)
        ])
        #-------------------------------------------------------------#
        
        
        #-------------------Define MLPs for Decoder block (only for node)------------#
        self.decoder = nn.Sequential(
            nn.Linear(gnn_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )
        #----------------------------------------------------------------------------#

        
    def forward(self, x, edge_index, edge_attr):
        
        #Encode
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)
        
        #Process
        for gns in self.gns_layers:
            x, edge_attr = gns(x, edge_index, edge_attr)  # both get updated
        
        #Decode
        x = self.decoder(x)
        
        return x