import torch
import torch.nn as nn
import torch.nn.functional as F
from .GraphGAT import GraphGAT  

class GNN(nn.Module):
    def __init__(self, features, num_user, num_item, dim_id, dim_latent=128):  # Remove edge_index from init
        super(GNN, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.register_buffer('features', features)

        # User preference embeddings
        self.preference = nn.Embedding(num_user, dim_latent)
        nn.init.xavier_normal_(self.preference.weight)

        # Project features to latent space
        self.MLP = nn.Linear(features.size(1), dim_latent)

        # GNN layers
        self.conv1 = GraphGAT(dim_latent, dim_latent)
        self.linear1 = nn.Linear(dim_latent, dim_id)
        self.g_layer1 = nn.Linear(dim_latent, dim_id)

        self.conv2 = GraphGAT(dim_id, dim_id)
        self.linear2 = nn.Linear(dim_id, dim_id)
        self.g_layer2 = nn.Linear(dim_id, dim_id)

    def forward(self, id_embedding, edge_index, features_override=None):
        feats = features_override if features_override is not None else self.features
        temp_features = torch.tanh(self.MLP(feats))
        
        x = torch.cat([self.preference.weight, temp_features], dim=0)
        x = F.normalize(x, dim=1)
        
        h = F.leaky_relu(self.conv1(x, edge_index))
        x_hat = F.leaky_relu(self.linear1(x)) + id_embedding
        x1 = F.leaky_relu(self.g_layer1(h) + x_hat)
        
        h = F.leaky_relu(self.conv2(x1, edge_index))
        x_hat = F.leaky_relu(self.linear2(x1)) + id_embedding
        x2 = F.leaky_relu(self.g_layer2(h) + x_hat)
    
        return torch.cat([x1, x2], dim=1)