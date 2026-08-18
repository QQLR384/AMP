import os
import random
import numpy as np
import torch

# ==========================================
# 1. Ultimate Environment Configuration
# ==========================================
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" 
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
# Force Python hash seed (must be set before importing modules that use it)
os.environ['PYTHONHASHSEED'] = '42'

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Keep warn_only=True to prevent direct errors from certain Transformer operators,
    # relying on fold-level resets for macro-reproducibility.
    torch.use_deterministic_algorithms(True, warn_only=True)

# DataLoader worker seed generator
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Start LLAMP (2025) 5-Fold Cross-Validation (Compute device: {device})")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
GENOME_FEAT_PATH = os.path.join(SCRIPT_DIR, 'genome_features.pt')
LOCAL_ESM2_PATH = os.path.join(SCRIPT_DIR, "esm2_local")

MAX_EPOCHS = 30
PATIENCE = 5  

if not os.path.exists(GENOME_FEAT_PATH):
    print(f"Fatal Error: Cannot find genome features file -> {GENOME_FEAT_PATH}")
    exit()