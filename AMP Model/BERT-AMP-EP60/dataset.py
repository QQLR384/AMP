import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

# ==========================================
# 2. Dynamic Dataset Class (Adapts to HuggingFace Cached Online IDs)
# ==========================================
class ProtBERT_Dynamic_Dataset(Dataset):
    def __init__(self, df, tokenizer_name="Rostlab/prot_bert", max_length=100):
        self.df = df.reset_index(drop=True)
        # Directly use the online ID with local_files_only=True to automatically use local cache
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name, do_lower_case=False, local_files_only=True)
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        seq = self.df['SEQUENCE'].iloc[idx]
        seq_spaced = " ".join(list(seq))
        target_raw = self.df['TARGET ACTIVITY - CONCENTRATION - PROCED'].iloc[idx]
        target_log = np.log10(target_raw)

        inputs = self.tokenizer(
            seq_spaced,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'labels': torch.tensor(target_log, dtype=torch.float32)
        }