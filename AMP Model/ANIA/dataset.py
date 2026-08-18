import numpy as np
import torch
from torch.utils.data import Dataset

# ==========================================
# 3. Dataset Loading and Offline Feature Reshape
# ==========================================
class ANIADataset(Dataset):
    def __init__(self, df):
        # 1. Extract labels and calculate log10
        raw_mics = df['TARGET ACTIVITY - CONCENTRATION - PROCED'].values.astype(np.float32)
        
        # Protection mechanism: Prevent extremely small or anomalous values (<=0) from causing log10 to throw NaN.
        valid_mics = np.where(raw_mics > 0, raw_mics, 1e-6)
        self.mic_logs = torch.tensor(np.log10(valid_mics), dtype=torch.float32).unsqueeze(1)
        
        # 2. Extract 2816-dimensional features
        # Find column names containing CGR or PROP keywords
        feature_cols = [c for c in df.columns if 'CGR' in c or 'PROP' in c]
        if len(feature_cols) != 2816:
            # Fallback mechanism: If column name matching fails, forcibly slice the last 2816 columns
            feature_cols = df.columns[-2816:]
            
        features = df[feature_cols].values.astype(np.float32)
        
        # 3. Reshape the (num_samples, 2816) flattened matrix into a (num_samples, 11, 16, 16) Tensor
        # This strictly aligns with the official extract_cgr_features_and_target_for_dl function
        self.fcgrs = torch.tensor(features).view(-1, 11, 16, 16)

    def __len__(self):
        return len(self.mic_logs)

    def __getitem__(self, idx):
        # __getitem__ has no time-consuming feature calculations, returns tensor slices directly
        return {
            'fcgr': self.fcgrs[idx], 
            'mic_log': self.mic_logs[idx]
        }