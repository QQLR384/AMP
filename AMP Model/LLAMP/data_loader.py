import numpy as np
import torch
from torch.utils.data import Dataset
# Import sequence_to_input from your existing dataset module
from dataset import sequence_to_input

# ==========================================
# 2. Dynamic Dataset Class
# ==========================================
class QMAP_LLAMP_Dataset(Dataset):
    def __init__(self, df, genome_feat):
        self.seqs = df['SEQUENCE'].tolist()
        self.labels = np.log10(df['TARGET ACTIVITY - CONCENTRATION - PROCED'].values).astype(np.float32)
        self.input_ids, self.attention_mask = sequence_to_input(self.seqs)
        self.genome_feats = genome_feat.unsqueeze(0).repeat(len(self.seqs), 1)

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'genome_feat': self.genome_feats[idx],
            'labels': torch.tensor(self.labels[idx], dtype=torch.float32)
        }