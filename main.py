import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_geometric.utils import remove_self_loops
from collections import defaultdict
from tqdm import tqdm
from configs import Config
from MGAT import MGAT
from dataset import MGATDataset, collate_fn
from evaluate import evaluate
from utils import device


import warnings
warnings.filterwarnings('ignore')
def main():
   
    config=Config()

    full_edge_index = np.load(os.path.join(config.EDGE_PATH, 'edge_index.npy'))
    full_adj_dict   = np.load(os.path.join(config.EDGE_PATH, 'adj_dict.npy'), allow_pickle=True).item()
    train_sequences = np.load(os.path.join(config.EDGE_PATH, 'train_sequences.npy'))
    dev_sequences   = np.load(os.path.join(config.EDGE_PATH, 'dev_sequences.npy'), allow_pickle=True)

    with open(os.path.join(config.EDGE_PATH, 'metadata.json'), 'r') as f:
        metadata = json.load(f)
    num_users, num_items = metadata['num_users'], metadata['num_items']

    train_articles = pd.read_parquet(os.path.join(config.DATA_PATH, 'train/article_features.parquet'))
    text_emb   = np.stack(train_articles['text_embedding'].values).astype(np.float32)
    entity_emb = np.stack(train_articles['entity_embedding'].values).astype(np.float32)
    category_ids      = train_articles['category_id'].values
    num_unique_categories = len(np.unique(category_ids))
    features = {
        'text_emb':     torch.tensor(text_emb,   device=device),
        'entity_emb':   torch.tensor(entity_emb, device=device),
        'category_ids': torch.tensor(category_ids, device=device, dtype=torch.long),
    }

    train_portion        = 0.1
    num_train_samples    = int(len(train_sequences) * train_portion)
    train_sequences_subset = train_sequences[:num_train_samples]

    train_pairs = {(u, i) for u, i in train_sequences_subset}
    edge_list   = [[u, i + num_users] for u, i in train_pairs]   
    edge_index  = np.array(edge_list).T
    print(f"Built edge_index with {edge_index.shape[1]} edges")

    edge_index_tensor = torch.LongTensor(edge_index).to(device)
    edge_index_tensor, _ = remove_self_loops(edge_index_tensor)
    print(f"Edge index shape after removing self-loops: {edge_index_tensor.shape}")

    adj_dict = defaultdict(set)
    for u, i in train_pairs:
        adj_dict[u].add(i)
    adj_dict = dict(adj_dict)

    dev_portion          = 0.1
    num_dev_samples      = int(len(dev_sequences) * dev_portion)
    dev_sequences_subset = dev_sequences[:num_dev_samples]
    dataset    = MGATDataset(train_sequences_subset, adj_dict, num_users, num_items, config.NUM_NEGATIVES)
    dataloader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,               # faster CPU→GPU transfer
        prefetch_factor=config.PREFETCH_FACTOR,
        persistent_workers=True,       # keep workers alive between epochs
    )

    print(f"Number of nodes : {num_users + num_items}")
    print(f"Training samples: {len(train_sequences_subset)}")
    print(f"Dev samples     : {len(dev_sequences_subset)}")

    model = MGAT(features, num_users, num_items, num_unique_categories,
                 config.CATEGORY_DIM, config.ID_DIM).to(device)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)   # splits batches across both T4s

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE,
                                 weight_decay=config.WEIGHT_DECAY)

    scaler = torch.cuda.amp.GradScaler()

    best_precision = 0
    patience       = 0
    for epoch in range(config.NUM_EPOCHS):
        print(f"\n{'='*60}\nEpoch {epoch+1}/{config.NUM_EPOCHS}\n{'='*60}")
        model.train()
        total_loss = 0

        for users, pos, neg in tqdm(dataloader, desc="Training"):
            ### non_blocking=True overlaps data transfer with computation
            users = users.to(device, non_blocking=True)
            pos   = pos.to(device,   non_blocking=True)
            neg   = neg.to(device,   non_blocking=True)

            optimizer.zero_grad(set_to_none=True)   

            with torch.cuda.amp.autocast():
                loss = model.module.loss(users, pos, neg, edge_index_tensor) \
                       if isinstance(model, nn.DataParallel) \
                       else model.loss(users, pos, neg, edge_index_tensor)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Training Loss: {avg_loss:.4f}")

        precision, recall, ndcg = evaluate(
            model, dev_sequences_subset, num_items, device, edge_index_tensor
        )
        print(f"Dev - Precision@10: {precision:.4f}, Recall@10: {recall:.4f}, NDCG@10: {ndcg:.4f}")

        if precision > best_precision:
            best_precision = precision
            patience       = 0
            state = model.module.state_dict() if isinstance(model, nn.DataParallel) \
                    else model.state_dict()
            torch.save(state, 'best_mgat_model.pth')
            print(" New best model saved!")
        else:
            patience += 1
            if patience >= config.EARLY_STOPPING_PATIENCE:
                print("  Early stopping triggered!")
                break

        torch.cuda.empty_cache()

    print(f"\n Training complete. Best Precision@10: {best_precision:.4f}")


if __name__ == "__main__":
    main()