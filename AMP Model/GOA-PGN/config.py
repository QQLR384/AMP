import os
import random
import numpy as np
import torch

# ==========================================
# 1. Environment Configuration and Random Seed Locking
# ==========================================
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



print("Start [Original Code Verification: Forced Full 0 Coordinates]")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
TENSORS_DIR = os.path.join(DATA_DIR, "tensors")
MODEL_PATH = os.path.join(SCRIPT_DIR, "esm2_local")

PHASE1_EPOCHS = 10
PHASE2_EPOCHS = 30
TOTAL_EPOCHS = PHASE1_EPOCHS + PHASE2_EPOCHS
PATIENCE = 7

HIGH_POTENCY_THRESHOLD_LOG = 1.0
HP_LOSS_WEIGHT = 5.0