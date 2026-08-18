import numpy as np
from config import CHARACTER_DICT, MAX_SEQUENCE_LENGTH

# ==========================================
# 1. One-hot Encoding Logic (Witten 2019)
# ==========================================
character_to_index = {char: i for i, char in enumerate(CHARACTER_DICT)}

def sequence_to_vector(sequence):
    # Build 100 x 21 One-hot matrix
    vector = np.zeros([MAX_SEQUENCE_LENGTH, len(character_to_index) + 1])
    for i, character in enumerate(sequence[:MAX_SEQUENCE_LENGTH]):
        if character in character_to_index:
            vector[i][character_to_index[character]] = 1
    return vector