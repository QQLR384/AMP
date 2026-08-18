import os
import sys
import random
import numpy as np

print("Start J. Witten (2019) Classic CNN Baseline 5-Fold Reproduction")

# ==========================================
# Critical modification: System environment variables must be set before importing tensorflow
# ==========================================
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

# Pre-lock Python and Numpy seeds
random.seed(SEED)
np.random.seed(SEED)

# ==========================================
# Check environment and import TensorFlow
# ==========================================
try:
    import tensorflow as tf
    
    # Core fix: Use Keras official unified seed setting method to fully take over internal operations
    tf.keras.utils.set_random_seed(SEED)
    
    try:
        tf.config.experimental.enable_op_determinism()
    except AttributeError:
        pass # Compatible with older TF versions
        
except ImportError:
    print("\nFatal Error: TensorFlow is not installed in your environment!")
    print("Please run in the terminal first: pip install tensorflow")
    sys.exit(1)

# Base directory configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

# One-hot encoding dictionary
CHARACTER_DICT = [
    'A', 'C', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L',
    'N', 'Q', 'P', 'S', 'R', 'T', 'W', 'V', 'Y'
]
MAX_SEQUENCE_LENGTH = 100