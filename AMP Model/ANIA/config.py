import os
import random
import numpy as np
import torch

# ==========================================
# 1. Environment Configuration and Path Definitions
# ==========================================
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

def seed_everything(seed=42):
    print(f"Locking global random seeds (Seed: {seed})...")
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
CONFIG_DIR = os.path.join(SCRIPT_DIR, "configs")

os.makedirs(RESULTS_DIR, exist_ok=True)