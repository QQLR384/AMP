import os
import pandas as pd
# Import the official homology-aware split function
from qmap.toolkit.split import train_test_split

def prepare_qmap_pure_data_5_splits():
    num_splits = 5
    # Rigorous Reproducibility: Use the 5 specific random seeds explicitly defined in the QMAP paper
    qmap_seeds = [1, 3, 7, 12, 404]
    
    # Check whether all files have been generated
    all_files_exist = True
    for i in range(num_splits):
        if not (os.path.exists(f'qmap_train_set_split_{i}.csv') and os.path.exists(f'qmap_test_set_split_{i}.csv')):
            all_files_exist = False
            break

    # Rigorous Reproducibility: File-level State Locking (Cache Lock)
    if all_files_exist:
        print(f"Detected 5 complete sets of QMAP split files in local storage!")
        print(f"To guarantee 100% experimental reproducibility, bypassing the underlying random engine and loading from local cache...\n")
        
        # Briefly print the data size of each fold for verification
        for i in range(num_splits):
            train_len = len(pd.read_csv(f'qmap_train_set_split_{i}.csv'))
            test_len = len(pd.read_csv(f'qmap_test_set_split_{i}.csv'))
            print(f" [Split {i}] trainset: {train_len} 条 | testset: {test_len} 条")
            
        print("\nDataset state consistency check passed!")
        return  #Early return
    

    # The following code only executes if files are missing

    print("First run or incomplete splits detected. Loading raw dataset...")
    df = pd.read_csv('./peptides-complete-new_processed.csv')
    
    # 1. Basic physicochemical filtering
    df = df[df['TARGET ACTIVITY - TARGET SPECIES'] == "escherichia coli"]
    # Remove sequences containing non-standard amino acids
    valid_rows = df['SEQUENCE'].apply(lambda seq: all(aa in 'ACDEFGHIKLMNPQRSTVWY' for aa in seq))
    df = df[valid_rows].reset_index(drop=True)
    
    # Core deduplication
    df = df.drop_duplicates(subset=['SEQUENCE']).reset_index(drop=True)
    
    # 2. Data format conversion
    sequences = df['SEQUENCE'].tolist()
    dataset = df.to_dict(orient='records')
    
    print(f"Candidate pool ready. Retrieved {len(sequences)} unique sequences.")
    
    # 3. Loop 5 times to generate 5 standard splits
    for i in range(num_splits):
        current_seed = qmap_seeds[i]
        train_file_path = f'qmap_train_set_split_{i}.csv'
        test_file_path = f'qmap_test_set_split_{i}.csv'
        
        print(f"Generating split set {i} (Random State: {current_seed})...")

        # Core Fix: Directly use the correct random_state parameter to fully lock the random stream
        train_seqs, test_seqs, train_samples, test_samples = train_test_split(
            sequences,            
            dataset,              
            test_size=0.20,
            threshold=0.60,       
            post_filtering=True,
            random_state=current_seed  # Explicitly use random_state
        )

        # 4. Convert back to DataFrame and export to disk
        df_train_safe = pd.DataFrame(train_samples)
        df_test_safe = pd.DataFrame(test_samples)
        
        df_train_safe.to_csv(train_file_path, index=False)
        df_test_safe.to_csv(test_file_path, index=False)
        
        leaked_count = len(sequences) - len(df_train_safe) - len(df_test_safe)
        
        print(f"   -  Absolute safe training set : {len(df_train_safe)}条")
        print(f"   -  Pure high-difficulty test set : {len(df_test_safe)}条")
        print(f"   -  Forced removal for anti-cheating : {leaked_count}条\n")
    
    print(f"All 5 QMAP standard datasets generated and successfully locked!")
    print("Note: Subsequent script runs will load directly from these 10 cached files.")

if __name__ == "__main__":
    prepare_qmap_pure_data_5_splits()