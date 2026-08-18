import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from config import BEST_14_FEATURES

# ==========================================
# 3. Dynamic Physiochemical Dataset Class
# ==========================================
class QMAP_PhysChem_Dataset(Dataset):
    def __init__(self, df, tensor_dir, global_p_dims=None, target_mean=None, target_std=None):
        self.tensor_dir = tensor_dir
        
        valid_indices = []
        for idx, row in df.iterrows():
            seq = row['SEQUENCE']
            seq_id = f"{seq[:10]}_{len(seq)}" 
            tensor_path = os.path.join(self.tensor_dir, f"{seq_id}.pt")
            if os.path.exists(tensor_path):
                valid_indices.append(idx)
                
        self.df = df.loc[valid_indices].reset_index(drop=True)
        
        raw_mics = self.df['TARGET ACTIVITY - CONCENTRATION - PROCED'].values
        log_mics = [math.log10(m) for m in raw_mics]
        
        self.target_mean = np.mean(log_mics) if target_mean is None else target_mean
        self.target_std = np.std(log_mics) if target_std is None else target_std
        
        if global_p_dims is None:
            self.global_p_dims = [1] * len(BEST_14_FEATURES) 
            for idx in range(len(self.df)):
                seq = self.df.iloc[idx]['SEQUENCE']
                seq_id = f"{seq[:10]}_{len(seq)}"
                tensor_path = os.path.join(self.tensor_dir, f"{seq_id}.pt")
                tensor_dict = torch.load(tensor_path, weights_only=True, map_location='cpu')
                for i, ft_name in enumerate(BEST_14_FEATURES):
                    p = tensor_dict[ft_name].view(-1).shape[0]
                    if p > self.global_p_dims[i]:
                        self.global_p_dims[i] = p
        else:
            self.global_p_dims = global_p_dims

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        seq = row['SEQUENCE']
        mic = float(row['TARGET ACTIVITY - CONCENTRATION - PROCED'])
        log10_mic = math.log10(mic)
        z_mic = (log10_mic - self.target_mean) / (self.target_std + 1e-8)
        
        seq_id = f"{seq[:10]}_{len(seq)}" 
        tensor_path = os.path.join(self.tensor_dir, f"{seq_id}.pt")
        tensor_dict = torch.load(tensor_path, weights_only=True, map_location='cpu')
        
        features_list = []
        for i, ft_name in enumerate(BEST_14_FEATURES):
            ft = tensor_dict[ft_name].view(-1).float()
            ft = torch.nan_to_num(ft, nan=0.0) 
            
            p = ft.shape[0]
            target_p = self.global_p_dims[i]
            
            if p == 0:
                ft = torch.zeros(target_p, dtype=torch.float32)
            else:
                if p < target_p:
                    ft = F.pad(ft, (0, target_p - p))
                elif p > target_p:
                    ft = ft[:target_p]
            features_list.append(ft)
        
        return {
            'features': features_list,
            'z_mic': torch.tensor(z_mic, dtype=torch.float32),          
            'log_mic': torch.tensor(log10_mic, dtype=torch.float32),    
            'raw_mic': torch.tensor(mic, dtype=torch.float32)           
        }