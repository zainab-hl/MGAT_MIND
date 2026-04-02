import torch
import torch.nn as nn
from .GNN import GNN  

class MGAT(torch.nn.Module):
    def __init__(self, features_dict, num_user, num_item, num_categories, category_dim, dim_x=64):
        super(MGAT, self).__init__()
        
        self.category_embedding = nn.Embedding(num_categories, category_dim)
        nn.init.xavier_normal_(self.category_embedding.weight)
        
        self.register_buffer('category_ids', features_dict['category_ids'])
        
        self.text_gnn = GNN(features_dict['text_emb'], num_user, num_item, dim_x, dim_latent=128)
        self.entity_gnn = GNN(features_dict['entity_emb'], num_user, num_item, dim_x, dim_latent=64)
        
        num_items = features_dict['text_emb'].size(0)
        dummy_category_feats = torch.zeros(num_items, category_dim)  
        self.category_gnn = GNN(dummy_category_feats, num_user, num_item, dim_x, dim_latent=category_dim)
        
        self.id_embedding = nn.Embedding(num_user + num_item, dim_x)
        nn.init.xavier_normal_(self.id_embedding.weight)

    def forward(self, user_nodes, pos_items, neg_items, edge_index):
        id_emb = self.id_embedding.weight
        
        category_feats = self.category_embedding(self.category_ids)  # [num_items, category_dim]
        
        text_rep = self.text_gnn(id_emb, edge_index)
        entity_rep = self.entity_gnn(id_emb, edge_index)
        category_rep = self.category_gnn(id_emb, edge_index, features_override=category_feats)
        
        representation = (text_rep + entity_rep + category_rep) / 3
        
        user_tensor = representation[user_nodes]
        pos_tensor = representation[pos_items]
        
        neg_items_flat = neg_items.view(-1)
        neg_tensor_flat = representation[neg_items_flat]
        neg_tensor = neg_tensor_flat.view(neg_items.size(0), neg_items.size(1), -1)
        
        pos_scores = torch.sum(user_tensor * pos_tensor, dim=1)
        neg_scores = torch.sum(user_tensor.unsqueeze(1) * neg_tensor, dim=2)
        neg_scores = torch.sum(neg_scores, dim=1)  

        return pos_scores, neg_scores
    

    def loss(self, user, pos_items, neg_items, edge_index):
        pos_scores, neg_scores = self.forward(user, pos_items, neg_items, edge_index)
        return -torch.mean(torch.log2(torch.sigmoid(pos_scores - neg_scores)))  