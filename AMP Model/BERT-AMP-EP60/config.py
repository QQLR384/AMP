import os
import random
import numpy as np
import torch
from transformers import set_seed as transformers_set_seed

print("Start BERT-AmPEP60 (ProtBERT) 5-Fold Cross-Validation (Max 30 Epochs + Early Stopping)")

# ==========================================
# 1. Environment Configuration and Random Seed Locking
# ==========================================
# Core: Environment variables must be at the top to block network requests
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

def seed_everything(seed=42):
    print(f"Locking global random seeds (Seed: {seed})...")
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    
    transformers_set_seed(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Compute device: {device}")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
MAX_EPOCHS = 30
PATIENCE = 5  # Trigger early stopping if validation loss doesn't improve for 5 consecutive rounds