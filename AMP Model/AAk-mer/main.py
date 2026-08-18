import os
import gc
import numpy as np
import pandas as pd
import xgboost as xgb

from config import DATA_DIR, RESULTS_DIR, K_MER, HIGH_POTENCY_THRESHOLD_RAW
from features import extract_aakmer_features, build_feature_matrix
from utils import compute_metrics, save_and_print_final_summary

# ==========================================
# 4. 5-Fold Cross-Validation Main Loop
# ==========================================
def main():
    all_full_metrics = []
    all_hp_metrics = []

    for split_idx in range(5):
        print(f"\n{'='*60}\nStart evaluating AAk-mer Baseline - Split {split_idx}\n{'='*60}")
        
        # Read dataset
        train_path = os.path.join(DATA_DIR, f"qmap_train_set_split_{split_idx}.csv")
        test_path = os.path.join(DATA_DIR, f"qmap_test_set_split_{split_idx}.csv")
        
        if not os.path.exists(train_path) or not os.path.exists(test_path):
            print(f"Warning: Fold {split_idx} data files not found. Skipping...")
            continue
            
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        y_train_raw = train_df['TARGET ACTIVITY - CONCENTRATION - PROCED'].values
        y_test_raw = test_df['TARGET ACTIVITY - CONCENTRATION - PROCED'].values
        y_train_log = np.log10(y_train_raw)
        y_test_log = np.log10(y_test_raw)
        
        # Extract training features and construct the global K-mer dictionary
        train_dicts, train_kmers = extract_aakmer_features(train_df['SEQUENCE'].values, f"train_{split_idx}", K_MER)
        X_train = build_feature_matrix(train_dicts, train_kmers)
        
        # Extract testing features (strictly mapped to the training feature space to prevent data leakage)
        test_dicts, _ = extract_aakmer_features(test_df['SEQUENCE'].values, f"test_{split_idx}", K_MER)
        X_test = build_feature_matrix(test_dicts, train_kmers)
        
        print(f"Feature matrix constructed! Dimensions: {X_train.shape[1]}")
        
        # XGBoost Training
        print("Training XGBoost Regressor...")
        xgb_model = xgb.XGBRegressor(
            n_estimators=500, 
            max_depth=6, 
            learning_rate=0.05,
            subsample=0.8, 
            colsample_bytree=0.8, 
            random_state=42, 
            n_jobs=-1,
            tree_method='gpu_hist'  
        )
        xgb_model.fit(X_train, y_train_log)
        
        print("Running blind test on the test set...")
        y_pred_log = xgb_model.predict(X_test)
        
        # Evaluation and Statistics
        full_metrics = compute_metrics(y_test_log, y_pred_log)
        all_full_metrics.append(full_metrics)
        
        hp_mask = y_test_raw < HIGH_POTENCY_THRESHOLD_RAW
        if np.sum(hp_mask) > 1: 
            all_hp_metrics.append(compute_metrics(y_test_log[hp_mask], y_pred_log[hp_mask]))
            
        # Clear memory
        del xgb_model, X_train, X_test, train_dicts, test_dicts
        gc.collect()

    # ==========================================
    # 5. Export Summary
    # ==========================================
    save_and_print_final_summary(
        all_full_metrics, 
        "AAk-mer + XGBoost - Full Test Set", 
        os.path.join(RESULTS_DIR, "aakmer_xgb_full_dataset_summary.csv")
    )
    save_and_print_final_summary(
        all_hp_metrics, 
        "AAk-mer + XGBoost - High Potency Subset (MIC < 10)", 
        os.path.join(RESULTS_DIR, "aakmer_xgb_high_potency_summary.csv")
    )

if __name__ == '__main__':
    main()