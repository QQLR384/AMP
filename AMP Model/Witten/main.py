import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

# Note: Importing config automatically sets up the TF environment variables and seeds
from config import DATA_DIR, RESULTS_DIR
from encoder import sequence_to_vector
from models import build_witten_model
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
        
        # Read data
        train_csv = os.path.join(DATA_DIR, f"qmap_train_set_split_{split_idx}.csv")
        test_csv = os.path.join(DATA_DIR, f"qmap_test_set_split_{split_idx}.csv")
        
        train_df = pd.read_csv(train_csv)
        test_df = pd.read_csv(test_csv)

        y_train_log = np.log10(train_df['TARGET ACTIVITY - CONCENTRATION - PROCED'].values)
        y_test_log = np.log10(test_df['TARGET ACTIVITY - CONCENTRATION - PROCED'].values)
        y_test_raw = test_df['TARGET ACTIVITY - CONCENTRATION - PROCED'].values

        # Feature extraction (One-hot encoding is very fast, perform real-time conversion)
        print("Performing traditional One-hot matrix encoding...")
        X_train = np.array([sequence_to_vector(seq) for seq in train_df['SEQUENCE'].tolist()])
        X_test = np.array([sequence_to_vector(seq) for seq in test_df['SEQUENCE'].tolist()])

        # Train model
        print("Training J. Witten (2019) CNN...")
        model = build_witten_model()
        
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        model.fit(
            X_train, y_train_log, 
            epochs=50, 
            batch_size=32, 
            validation_split=0.1, 
            callbacks=[early_stop],
            verbose=1
        )

        print("Training complete! Running blind test on the test set...")
        y_pred_log = model.predict(X_test, verbose=0).flatten()

        # Evaluate full test set
        full_metrics = compute_metrics(y_test_log, y_pred_log)
        all_full_metrics.append(full_metrics)
        print_metrics(full_metrics, f"Split {split_idx} - Full Test Set")

        # Evaluate high potency subset (MIC < 10)
        hp_mask = y_test_raw < 10.0
        if np.sum(hp_mask) > 0:
            hp_metrics = compute_metrics(y_test_log[hp_mask], y_pred_log[hp_mask])
            all_hp_metrics.append(hp_metrics)
            print_metrics(hp_metrics, f"Split {split_idx} - High Potency Subset (MIC < 10)")
        else:
            print("\nWarning: No data with MIC < 10 found in the current test set.")
            
        # Core: Clear TensorFlow session, release memory to prevent OOM in the next fold
        tf.keras.backend.clear_session()

    # ==========================================
    # 5. Execute Summary and Save
    # ==========================================
    save_and_print_final_summary(
        all_full_metrics, 
        "J. Witten (2019) CNN - Full Test Set (Log10 Space)", 
        os.path.join(RESULTS_DIR, "witten2019_full_dataset_summary.csv")
    )

    save_and_print_final_summary(
        all_hp_metrics, 
        "J. Witten (2019) CNN - High Potency Subset (MIC < 10)", 
        os.path.join(RESULTS_DIR, "witten2019_high_potency_summary.csv")
    )

if __name__ == '__main__':
    main()