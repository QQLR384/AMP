import os
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from config import (
    device, SCRIPT_DIR, DATA_DIR, RESULTS_DIR, GENOME_FEAT_PATH, LOCAL_ESM2_PATH,
    MAX_EPOCHS, PATIENCE, seed_everything, seed_worker
)
from data_loader import QMAP_LLAMP_Dataset
from utils import compute_metrics, save_and_print_final_summary
from model import LLAMP

# ==========================================
# 4. Main 5-Fold Training Loop (Early Stopping Mode)
# ==========================================
def main():
    print("Extracting actual E. coli genome features...")
    genome_features_dict = torch.load(GENOME_FEAT_PATH, weights_only=False)
    raw_feat = genome_features_dict['Escherichia coli'][0]
    e_coli_genome_feat = torch.tensor(np.array(raw_feat), dtype=torch.float32).view(-1)

    all_full_metrics = []
    all_hp_metrics = []

    for split_idx in range(5):
        print(f"\n{'='*60}\nStart evaluating Split {split_idx}\n{'='*60}")
        
        # Core fix 1: Re-lock random seed at the start of each fold.
        # Add split_idx to prevent identical data splits and model initializations.
        current_seed = 42 + split_idx
        seed_everything(current_seed)
        
        # Core fix 2: Instantiate Generator inside each fold to ensure isolation
        # from the previous fold's DataLoader iteration count.
        g = torch.Generator()
        g.manual_seed(current_seed)
        
        # 1. Read data and split 10% as early stopping validation set
        train_full_df = pd.read_csv(os.path.join(DATA_DIR, f"qmap_train_set_split_{split_idx}.csv"))
        test_df = pd.read_csv(os.path.join(DATA_DIR, f"qmap_test_set_split_{split_idx}.csv"))
        
        train_df, val_df = train_test_split(train_full_df, test_size=0.1, random_state=42)
        
        train_loader = DataLoader(
            QMAP_LLAMP_Dataset(train_df, e_coli_genome_feat), 
            batch_size=32, 
            shuffle=True, 
            worker_init_fn=seed_worker, 
            generator=g
        )
        val_loader = DataLoader(QMAP_LLAMP_Dataset(val_df, e_coli_genome_feat), batch_size=32, shuffle=False)
        test_loader = DataLoader(QMAP_LLAMP_Dataset(test_df, e_coli_genome_feat), batch_size=32, shuffle=False)
        
        y_test_raw = test_df['TARGET ACTIVITY - CONCENTRATION - PROCED'].values
        y_test_log = np.log10(y_test_raw)

        # 2. Initialize LLAMP model (under strictly fixed RNG state)
        print("Mounting LLAMP architecture (loading ESM-2 pretrained base and genome fusion layer)...")
        model = LLAMP(pooling='mean', pretrained_model=LOCAL_ESM2_PATH).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
        mse_loss = nn.MSELoss()
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_path = os.path.join(SCRIPT_DIR, f"temp_llamp_split_{split_idx}_best.pth")
        
        # 3. Training Loop
        for epoch in range(MAX_EPOCHS):
            model.train()
            total_train_loss = 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS} [Train]")
            for batch in pbar:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                genome_feat = batch['genome_feat'].to(device)
                labels = batch['labels'].to(device)
                
                optimizer.zero_grad()
                preds = model(input_ids, attention_mask, genome_feat)
                
                preds = preds.view(-1)
                labels = labels.view(-1)
                
                loss = mse_loss(preds, labels)
                loss.backward()
                optimizer.step()
                
                total_train_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                
            # Validation loop
            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    genome_feat = batch['genome_feat'].to(device)
                    labels = batch['labels'].to(device)
                    
                    preds = model(input_ids, attention_mask, genome_feat)
                    preds = preds.view(-1)
                    labels = labels.view(-1)
                    
                    total_val_loss += mse_loss(preds, labels).item()
                    
            avg_val_loss = total_val_loss / len(val_loader)
            print(f" Epoch {epoch+1} Complete | Val Loss: {avg_val_loss:.4f}")
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), best_model_path)
                patience_counter = 0
                print(f" Saved best model weights (Val Loss: {best_val_loss:.4f})")
            else:
                patience_counter += 1
                print(f" Validation Loss did not improve, Patience: {patience_counter}/{PATIENCE}")
                if patience_counter >= PATIENCE:
                    print(" Triggered Early Stopping, training for this fold ended early!")
                    break
                    
        # 4. Ultimate Blind Test
        print("Loading best weights for this fold, performing strict QMAP blind test...")
        model.load_state_dict(torch.load(best_model_path))
        model.eval()
        
        y_pred_log = []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Predicting"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                genome_feat = batch['genome_feat'].to(device)
                
                preds = model(input_ids, attention_mask, genome_feat)
                y_pred_log.extend(preds.view(-1).cpu().numpy())
                
        y_pred_log = np.array(y_pred_log)

        # 5. Metric Collection
        full_metrics = compute_metrics(y_test_log, y_pred_log)
        all_full_metrics.append(full_metrics)
        
        hp_mask = y_test_raw < 10.0
        if np.sum(hp_mask) > 0:
            hp_metrics = compute_metrics(y_test_log[hp_mask], y_pred_log[hp_mask])
            all_hp_metrics.append(hp_metrics)

        # 6. Clean Up
        del model, optimizer, train_loader, val_loader, test_loader
        torch.cuda.empty_cache()
        gc.collect()
        if os.path.exists(best_model_path):
            os.remove(best_model_path)

    # ==========================================
    # 7. Summary Export
    # ==========================================
    save_and_print_final_summary(
        all_full_metrics, 
        "LLAMP (2025) - Full Test Set", 
        os.path.join(RESULTS_DIR, "llamp_2025_full_dataset_summary.csv")
    )
    save_and_print_final_summary(
        all_hp_metrics, 
        "LLAMP (2025) - High Potency Subset (MIC < 10)", 
        os.path.join(RESULTS_DIR, "llamp_2025_high_potency_summary.csv")
    )

if __name__ == '__main__':
    main()