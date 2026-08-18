import numpy as np
from tqdm import tqdm

# ==========================================
# 2. Pure Python High-Speed K-mer Extraction
# ==========================================
def extract_aakmer_features(sequences, prefix, k_mer=3):
    """
    Uses a pure Python sliding window to extract k-mers.
    Mathematically equivalent to using mercat but entirely in-memory,
    avoiding all disk I/O and vastly improving execution speed.
    """
    print(f"[{prefix}] Extracting in-memory {k_mer}-mer features...")
    all_kmers = set()
    sample_kmer_dicts = []
    
    for seq in tqdm(sequences, desc=f"[{prefix}] Scanning peptide sequences"):
        kmer_counts = {}
        seq_len = len(seq)
        
        # Ensure the sequence length is at least K to extract features
        if seq_len >= k_mer:
            for i in range(seq_len - k_mer + 1):
                kmer = seq[i:i + k_mer]
                kmer_counts[kmer] = kmer_counts.get(kmer, 0) + 1
                all_kmers.add(kmer)
                
        sample_kmer_dicts.append(kmer_counts)
        
    return sample_kmer_dicts, list(all_kmers)

def build_feature_matrix(sample_kmer_dicts, global_kmers):
    """
    Constructs a 2D feature matrix suitable for XGBoost 
    based on the global k-mer dictionary.
    """
    X = np.zeros((len(sample_kmer_dicts), len(global_kmers)))
    kmer_to_idx = {kmer: i for i, kmer in enumerate(global_kmers)}
    
    for i, kmer_counts in enumerate(sample_kmer_dicts):
        for kmer, count in kmer_counts.items():
            if kmer in kmer_to_idx:
                X[i, kmer_to_idx[kmer]] = count
    return X