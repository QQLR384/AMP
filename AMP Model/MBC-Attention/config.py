import os
import random
import numpy as np
import torch

# ==========================================
# 1. Environment Configuration and Path Definitions
# ==========================================
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

seed_everything(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Compute device selected: {DEVICE}")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
TENSORS_DIR = os.path.join(DATA_DIR, "mbc_tensors")
MODEL_SAVE_DIR = os.path.join(SCRIPT_DIR, "saved_models_mbc")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

os.makedirs(TENSORS_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# The best 14 features reported in the original paper (Best-14 combination)
BEST_14_FEATURES = [
    'type8raac9glmd3lambda-correlation', 'type8raac7glmd3lambda-correlation', 
    'QSOrder_lmd4', 'QSOrder_lmd3', 'QSOrder_lmd2', 'QSOrder_lmd1', 'QSOrder_lmd0', 
    'type5raac15glmd4lambda-correlation', 'type7raac10glmd3lambda-correlation',
    'type5raac8glmd2lambda-correlation', 'type3Braac9glmd3lambda-correlation', 
    'type2raac15glmd4lambda-correlation', 'type2raac8glmd2lambda-correlation', 
    'type8raac14glmd1lambda-correlation'
]