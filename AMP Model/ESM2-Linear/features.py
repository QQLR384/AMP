import os
import numpy as np
import torch
from tqdm import tqdm
from transformers import EsmTokenizer, EsmModel
from config import LOCAL_ESM2_PATH, CACHE_DIR, device

# ==========================================
# 2. ESM2 Feature Extraction Logic
# ==========================================
def generate_esm2_embeddings_native(protein_sequences, model_name, device, batch_size=8):
    tokenizer = EsmTokenizer.from_pretrained(model_name, local_files_only=True)
    model = EsmModel.from_pretrained(model_name, local_files_only=True).to(device)
    model.eval()

    all_embeddings = []
    
    for i in tqdm(range(0, len(protein_sequences), batch_size), desc="Extracting Features"):
        batch_sequences = protein_sequences[i:i + batch_size]
        
        inputs = tokenizer(batch_sequences, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            hidden_states = outputs.last_hidden_state

        for j in range(len(batch_sequences)):
            attention_mask = inputs['attention_mask'][j]
            seq_embeddings = hidden_states[j][attention_mask.bool()]
            seq_embeddings_no_special = seq_embeddings[1:-1]
            seq_embedding = seq_embeddings_no_special.mean(dim=0)
            all_embeddings.append(seq_embedding.cpu().numpy())

    return np.array(all_embeddings)

def get_embeddings_with_cache(seqs, split_idx, mode="train"):
    cache_path = os.path.join(CACHE_DIR, f"esm2_650m_{mode}_split_{split_idx}.npy")
    if os.path.exists(cache_path):
        print(f"Cache detected, loading {mode} set features directly (Split {split_idx})...")
        return np.load(cache_path)
    else:
        print(f"Cache miss, extracting {mode} set features (Split {split_idx})...")
        emb = generate_esm2_embeddings_native(seqs, model_name=LOCAL_ESM2_PATH, device=device, batch_size=8)
        np.save(cache_path, emb)
        return emb