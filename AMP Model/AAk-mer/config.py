import os
import random
import numpy as np

# ==========================================
# 1. Environment Configuration and Path Definitions
# ==========================================
def seed_everything(seed=42):
    print(f"Locking global random seeds (Seed: {seed})...")
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(42)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

# Algorithm hyperparameters
K_MER = 3
HIGH_POTENCY_THRESHOLD_RAW = 10.0