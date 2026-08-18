import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from config import DATA_DIR, RESULTS_DIR
from features import get_embeddings_with_cache
from utils import compute_metrics, print_metrics, save_and_print_final_summary

# ==========================================
# 4. 5-Fold Cross-Validation Main Loop
# ==========================================
def main():
    all_full_metrics = []
    all_hp_metrics = []

    for split_idx in range(5):
        print(f"\n{'='*60}")
        print(f"Start evaluating Split {split_idx}")
        print("="*60)
        
        train_csv = os.path.join(DATA_DIR, f"qmap_train_set_split_{split_idx}.csv")
        test_csv = os.path.join(DATA_DIR, f"qmap_test_set_split_{split_idx}.csv")
        
        train_df = pd.read_csv(train_csv)
        test_df = pd.read_csv(test_csv)

        train_seqs = train_df['SEQUENCE'].tolist()
        test_seqs = test_df['SEQUENCE'].tolist()

        y_train_log = np.log10(train_df['TARGET ACTIVITY - CONCENTRATION - PROCED'].values)
        y_test_log = np.log10(test_df['TARGET ACTIVITY - CONCENTRATION - PROCED'].values)
        y_test_raw = test_df['TARGET ACTIVITY - CONCENTRATION - PROCED'].values

        X_train = get_embeddings_with_cache(train_seqs, split_idx, mode="train")
        X_test = get_embeddings_with_cache(test_seqs, split_idx, mode="test")

        print("Training Official Linear Regression Baseline...")
        model = LinearRegression()
        model.fit(X_train, y_train_log)

        print("Running blind test on the test set...")
        y_pred_log = model.predict(X_test)

        full_metrics = compute_metrics(y_test_log, y_pred_log)
        all_full_metrics.append(full_metrics)
        print_metrics(full_metrics, f"Split {split_idx} - Full Test Set")

        hp_mask = y_test_raw < 10.0
        if np.sum(hp_mask) > 0:
            hp_metrics = compute_metrics(y_test_log[hp_mask], y_pred_log[hp_mask])
            all_hp_metrics.append(hp_metrics)
            print_metrics(hp_metrics, f"Split {split_idx} - High Potency Subset (MIC < 10)")
        else:
            print("\nWarning: No data with MIC < 10 found in the current test set.")

    # Execute summary and save
    save_and_print_final_summary(
        all_full_metrics, 
        "Official Linear Baseline - Full Test Set (Log10 Space)", 
        os.path.join(RESULTS_DIR, "baseline_full_dataset_summary.csv")
    )

    save_and_print_final_summary(
        all_hp_metrics, 
        "Official Linear Baseline - High Potency Subset (MIC < 10)", 
        os.path.join(RESULTS_DIR, "baseline_high_potency_summary.csv")
    )

if __name__ == '__main__':
    main()