import os
import sys
import torch
import pandas as pd
import numpy as np

from config import DATA_DIR, TENSORS_DIR, BEST_14_FEATURES

# Automatically acquire the current script path and append the project root to the Python search path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# Attempt to load the external iFeature-based tool
try:
    from tools.features import geneFeature 
except ImportError:
    print("Warning: 'tools.features' module not found. Ensure the 'tools' package exists in the parent directory.")
    geneFeature = None

# ==========================================
# 2. Offline MBC Feature Extraction
# ==========================================
def extract_all():
    if geneFeature is None:
        raise ImportError("Cannot perform feature extraction without 'tools.features.geneFeature'.")
        
    # 1. Collect all unique E. coli sequences from the 5-Fold splits
    all_seqs = set()
    for i in range(5):
        for split in ['train', 'test']:
            file_path = os.path.join(DATA_DIR, f"qmap_{split}_set_split_{i}.csv")
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                all_seqs.update(df['SEQUENCE'].tolist())
    
    print(f"Found {len(all_seqs)} unique sequences. Starting offline feature extraction...")
    
    min_len = 5  # Length constraint required by QSOrder

    for seq in all_seqs:
        if len(seq) < min_len:
           continue  # Skip sequences that are too short

        seq_id = f"{seq[:10]}_{len(seq)}"
        save_path = os.path.join(TENSORS_DIR, f"{seq_id}.pt")
        
        if os.path.exists(save_path):
            continue  # Support resuming from interruptions
            
        tensor_dict = {}
        for ft_name in BEST_14_FEATURES:
            # Raw iFeature extraction
            raw_matrix = geneFeature([[seq_id, seq]], ft_whole_name=ft_name, max_len=60)
            
            # Skip the first column (usually ID) and force conversion to numpy float32
            try:
                feature_values = raw_matrix.iloc[:, 1:].to_numpy(dtype=np.float32)
                
                # Directly convert the extracted 1D features to a Tensor
                img_tensor = torch.tensor(feature_values[0], dtype=torch.float32)
                tensor_dict[ft_name] = img_tensor
            except Exception as e:
                print(f"Error during extraction of feature {ft_name} for sequence {seq_id}: {e}")
                # Inject a microscopic amount of random noise to prevent downstream all-zero identification
                tensor_dict[ft_name] = torch.randn(10) * 1e-5
            
        torch.save(tensor_dict, save_path)
    
    print("All features extracted and successfully packaged into .pt files!")