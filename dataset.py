import numpy as np
import torch
from torch.utils.data import Dataset

class MGATDataset(Dataset):
    def __init__(self, train_sequences, adj_dict, num_users, num_items, num_negatives=4):
        self.train_sequences = train_sequences
        self.adj_dict = adj_dict
        self.num_users = num_users
        self.num_items = num_items
        self.num_negatives = num_negatives
        self.all_items = set(range(num_items))

    def __len__(self):
        return len(self.train_sequences)

    def __getitem__(self, idx):
        user, pos_item = self.train_sequences[idx]
        interacted = set(self.adj_dict.get(user, []))
        candidates = list(self.all_items - interacted - {pos_item})

        if len(candidates) >= self.num_negatives:
            neg_items = np.random.choice(candidates, self.num_negatives, replace=False)
        else:
            neg_items = np.random.choice(list(self.all_items - {pos_item}), self.num_negatives, replace=True)

        return user, pos_item, neg_items

def collate_fn(batch):
    users = torch.LongTensor([b[0] for b in batch])
    pos = torch.LongTensor([b[1] for b in batch])
    neg = torch.LongTensor(np.array([b[2] for b in batch]))
    return users, pos, neg