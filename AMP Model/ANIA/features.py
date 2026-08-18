import os
import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from config import DATA_DIR, CONFIG_DIR

# ==========================================
# 2. [Official Aligned] 11-Channel 3-mer FCGR Feature Generator
# ==========================================
def generate_aligned_cgr():
    print("Start [Official Aligned] 11-Channel 3-mer FCGR Feature Generator...")
    
    aa_props_path = os.path.join(CONFIG_DIR, "AAindex_properties.csv")
    
    if not os.path.exists(aa_props_path):
        raise FileNotFoundError(f"Error: Physicochemical properties file not found: {aa_props_path}")
        
    # 1. Load physicochemical properties (strictly no Min-Max normalization to preserve official value distribution)
    aa_props = pd.read_csv(aa_props_path, index_col='AminoAcid')
    
    # Official 10 specific AAindex properties
    target_props = [
        "ARGP820101", "CHAM830107", "FAUJ880103", "GRAR740102", "JANJ780101",
        "KYTJ820101", "NAKH920104", "ROSM880102", "WERD780104", "ZIMJ680101"
    ]
    
    # 2. CGR 4-quadrant mapping (Standard Chaos Game Representation coordinate system)
    corners = {
        'A': (0,0), 'C': (0,0), 'G': (0,0), 'I': (0,0), 'L': (0,0), 'M': (0,0), 'F': (0,0), 'P': (0,0), 'W': (0,0), 'V': (0,0),
        'N': (1,0), 'Q': (1,0), 'S': (1,0), 'T': (1,0), 'Y': (1,0),
        'R': (1,1), 'H': (1,1), 'K': (1,1),
        'D': (0,1), 'E': (0,1)
    }

    def calc_official_3mer_cgr(seq, res=16, k=3):
        seq = seq.upper()
        L = len(seq)
        grids = np.zeros((11, res, res), dtype=np.float32)
        
        if L == 0:
            return grids.flatten()

        # Step A: Track the landing point trajectory of each amino acid on CGR
        pixels = []
        x, y = 0.5, 0.5
        for aa in seq:
            cx, cy = corners.get(aa, (0.5, 0.5))
            x = x + 0.86 * (cx - x)
            y = y + 0.86 * (cy - y)
            
            col = min(int(x * res), res - 1)
            row = min(int(y * res), res - 1)
            pixels.append((row, col))
            
            # Channel 0: Landing point frequency matrix for the entire sequence (Base CGR)
            grids[0, row, col] += 1.0

        if L < k:
            # If sequence is too short to extract 3-mers, return only the base frequency map
            return grids.flatten()

        # Step B: Build pixel-level 3-mer mapping
        pixel_kmers = defaultdict(list)
        for i in range(L - k + 1):
            kmer = seq[i : i + k]
            # Official logic: The landing point of a 3-mer is determined by its last amino acid
            r, c = pixels[i + k - 1]
            pixel_kmers[(r, c)].append(kmer)

        # Step C: Calculate 3-mer combination features for the 10 physicochemical channels
        for p_idx, prop_name in enumerate(target_props):
            channel_idx = p_idx + 1  # Corresponds to Channel 1 to 10
            
            for (r, c), kmer_list in pixel_kmers.items():
                # Official deduplication logic: Identical 3-mers on the same pixel are calculated only once
                unique_kmers = set(kmer_list)
                kmer_sums = []
                
                for kmer in unique_kmers:
                    # Sum the attribute values of the 3 amino acids within the 3-mer
                    try:
                        aa_sum = sum(aa_props.loc[aa, prop_name] for aa in kmer if aa in aa_props.index)
                        kmer_sums.append(aa_sum)
                    except KeyError:
                        continue
                
                # Official aggregation logic: Average the attribute sums of all unique 3-mers on that pixel
                if kmer_sums:
                    grids[channel_idx, r, c] = np.mean(kmer_sums)

        # Flatten to a 2816-dimensional vector
        return grids.flatten()

    # 3. Batch process QMAP's 5-fold files
    for fold in range(5):
        for split in ['train', 'test']:
            file_name = f'qmap_{split}_set_split_{fold}.csv'
            file_path = os.path.join(DATA_DIR, file_name)
            
            if not os.path.exists(file_path):
                continue
                
            print(f"Processing: {file_name}")
            df = pd.read_csv(file_path)
            
            # Extract features
            features = []
            for seq in tqdm(df['SEQUENCE'], desc=f"Calculating 3-mer FCGR"):
                features.append(calc_official_3mer_cgr(seq))
                
            # Convert 2816-dimensional features to DataFrame
            feat_columns = [f'CGR(resolution=16) | {i+1}-{j+1}' for i in range(16) for j in range(16)]
            for prop in target_props:
                feat_columns.extend([f'PROP({prop}) | {i+1}-{j+1}' for i in range(16) for j in range(16)])
            
            feat_df = pd.DataFrame(features, columns=feat_columns)
            
            # Concatenate and save as a new file with features
            merged_df = pd.concat([df, feat_df], axis=1)
            out_path = os.path.join(DATA_DIR, f'qmap_{split}_set_split_{fold}_cgr.csv')
            merged_df.to_csv(out_path, index=False)
            
    print("All datasets' 11-channel 3-mer features have been strictly aligned and generated successfully!")

if __name__ == "__main__":
    generate_aligned_cgr()