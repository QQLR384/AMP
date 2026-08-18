import os
import random
import numpy as np
import torch

# ==========================================
# 1. Environment Configuration and Random Seed Locking
# ==========================================
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

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Start Official Linear Baseline 5-Fold Reproduction (Local Mode & Automatic Feature Caching)")

# Directory definitions
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_ESM2_PATH = os.path.join(SCRIPT_DIR, "esm2_local")  # Ensure consistent dynamic path
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
CACHE_DIR = os.path.join(DATA_DIR, "embeddings_cache")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

if not os.path.exists(LOCAL_ESM2_PATH):
    print(f"\nError: Cannot find model folder -> {LOCAL_ESM2_PATH}")
    print("Please make sure you have downloaded the ESM2 model to the 'esm2_local' folder.")
    exit()

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)