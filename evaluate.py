import numpy as np
import torch
import torch.nn as nn

def evaluate(model, dev_sequences, num_items, device, edge_index_tensor, topk=10):
    # Get the underlying model if wrapped in DataParallel
    base_model = model.module if isinstance(model, nn.DataParallel) else model
    
    base_model.eval()
    all_precision, all_recall, all_ndcg = [], [], []

    with torch.no_grad():
        id_emb = base_model.id_embedding.weight
        category_feats = base_model.category_embedding(base_model.category_ids)
        text_rep    = base_model.text_gnn(id_emb, edge_index_tensor)
        entity_rep  = base_model.entity_gnn(id_emb, edge_index_tensor)
        category_rep = base_model.category_gnn(id_emb, edge_index_tensor, features_override=category_feats)
        representation = (text_rep + entity_rep + category_rep) / 3

    with torch.no_grad():
        for seq in dev_sequences:
            if len(seq) < 102:
                continue
            user       = seq[0]
            neg_items  = seq[1:101]
            pos_items  = seq[101:]

            user_emb  = representation[user]
            pos_emb   = representation[pos_items]
            neg_emb   = representation[neg_items]

            pos_scores = torch.sum(user_emb * pos_emb, dim=1)
            neg_scores = torch.sum(user_emb * neg_emb, dim=1)

            all_scores = torch.cat([neg_scores, pos_scores])
            _, top_indices = torch.topk(all_scores, min(topk, len(all_scores)))
            top_set = set(top_indices.cpu().numpy())

            num_pos    = len(pos_items)
            pos_indices = set(range(len(neg_items), len(neg_items) + num_pos))

            hits      = len(top_set & pos_indices)
            precision = hits / topk
            recall    = hits / num_pos if num_pos > 0 else 0

            # Vectorized NDCG — no Python for-loop over pos_indices
            pos_idx_tensor = torch.tensor(list(pos_indices), device=device)
            top_tensor     = top_indices  # already on device
            # find which pos_indices appear in top_indices
            matches = (top_tensor.unsqueeze(1) == pos_idx_tensor.unsqueeze(0))  # [k, num_pos]
            matched_ranks = matches.any(dim=1).nonzero(as_tuple=True)[0]        # ranks that hit
            ndcg = (1 / torch.log2(matched_ranks.float() + 2)).sum().item()
            ndcg /= num_pos if num_pos > 0 else 1

            all_precision.append(precision)
            all_recall.append(recall)
            all_ndcg.append(ndcg)

    return np.mean(all_precision), np.mean(all_recall), np.mean(all_ndcg)
