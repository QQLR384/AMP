import os
import math
import torch
import pandas as pd
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from config import TENSORS_DIR, HIGH_POTENCY_THRESHOLD_LOG

# ==========================================
# 2. Geometry Tensor Loading and Collate
# ==========================================
class QMAP_Tensor_Dataset(Dataset):
    def __init__(self, df, mapping_df):
        self.seq_to_pt = dict(zip(mapping_df['sequence'], mapping_df['hash_filename']))
        self.seq_to_pt = {k: v.replace('.pdb', '.pt') for k, v in self.seq_to_pt.items()}
        valid_rows = df['SEQUENCE'].apply(lambda seq: seq in self.seq_to_pt)
        self.data = df[valid_rows].reset_index(drop=True)
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        sequence = row['SEQUENCE']
        mic_value = float(row['TARGET ACTIVITY - CONCENTRATION - PROCED'])
        
        tensor_path = os.path.join(TENSORS_DIR, self.seq_to_pt[sequence])
        geom_data = torch.load(tensor_path, weights_only=True)
        
        # Ultimate verification: Completely ignore file content, force coordinates to all zeros
        coords = torch.zeros(len(sequence), 3).float()
        
        vec_sidechain = geom_data['vec_sidechain'].float()
        dihedrals = geom_data['dihedrals'].float()
        
        analyzer = ProteinAnalysis(sequence)
        mic_log = math.log10(mic_value)
        is_hp = 1.0 if mic_log < HIGH_POTENCY_THRESHOLD_LOG else 0.0
        
        return {
            'sequence': sequence,
            'coords': coords,
            'vec_sidechain': vec_sidechain,
            'dihedrals': dihedrals,
            'mic_log': torch.tensor([mic_log], dtype=torch.float32),
            'is_hp': torch.tensor([is_hp], dtype=torch.float32),
            'charge': torch.tensor([analyzer.charge_at_pH(7.0)], dtype=torch.float32),
            'gravy': torch.tensor([analyzer.gravy()], dtype=torch.float32)
        }

def pad_geom_tensor(tensor_list, max_len):
    padded_list = []
    for t in tensor_list:
        dim = t.shape[-1]
        t_2d = t.view(-1, dim)
        t_padded = torch.cat([torch.zeros(1, dim), t_2d, torch.zeros(1, dim)], dim=0)
        padded_list.append(t_padded)
        
    padded = pad_sequence(padded_list, batch_first=True, padding_value=0.0)
    b, l, d = padded.shape
    if l < max_len:
        pad_tensor = torch.zeros(b, max_len - l, d)
        padded = torch.cat([padded, pad_tensor], dim=1)
    else:
        padded = padded[:, :max_len, :]
    return padded

def collate_fn_tensors(batch, tokenizer):
    sequences = [item['sequence'] for item in batch]
    tokens = tokenizer(sequences, return_tensors="pt", padding=True, truncation=True, max_length=128)
    max_len = tokens['input_ids'].shape[1]
    
    coords_padded = pad_geom_tensor([item['coords'] for item in batch], max_len)
    vec_padded = pad_geom_tensor([item['vec_sidechain'] for item in batch], max_len)
    dih_padded = pad_geom_tensor([item['dihedrals'] for item in batch], max_len)
    
    return {
        'input_ids': tokens['input_ids'],
        'attention_mask': tokens['attention_mask'],
        'coords': coords_padded,
        'vec_sidechain': vec_padded,
        'dihedrals': dih_padded,
        'mic_log': torch.stack([item['mic_log'] for item in batch]),
        'is_hp': torch.stack([item['is_hp'] for item in batch]),
        'charge': torch.stack([item['charge'] for item in batch]),
        'gravy': torch.stack([item['gravy'] for item in batch])
    }