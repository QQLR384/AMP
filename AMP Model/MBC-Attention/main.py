import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from config import DATA_DIR, TENSORS_DIR, MODEL_SAVE_DIR, RESULTS_DIR, DEVICE, seed_everything
from dataset import QMAP_PhysChem_Dataset
from models import MBC_Attention
from utils import compute_metrics, save_and_print_final_summary
from features import extract_all

# ==========================================
# 6. Main Training and Evaluation Loop
# ==========================================
def train_and_evaluate():
    # 1. Optionally trigger offline feature extraction if the directory is empty
    if not os.path.exists(TENSORS_DIR) or len(os.listdir(TENSORS_DIR)) == 0:
        print("Tensors directory is empty. Running offline feature extraction...")
        extract_all()

    all_full_metrics = []
    all_hp_metrics = []
    criterion = nn.SmoothL1Loss() 

    for fold in range(5):
        print(f"\n{'='*40}")
        print(f"Start Training QMAP Split Fold {fold}")
        print(f"{'='*40}")
        
        train_val_df = pd.read_csv(os.path.join(DATA_DIR, f"qmap_train_set_split_{fold}.csv"))
        test_df = pd.read_csv(os.path.join(DATA_DIR, f"qmap_test_set_split_{fold}.csv"))
        
        train_df, val_df = train_test_split(train_val_df, test_size=0.1, random_state=42)
        
        train_dataset = QMAP_PhysChem_Dataset(train_df, TENSORS_DIR)
        t_mean, t_std = train_dataset.target_mean, train_dataset.target_std
        
        val_dataset = QMAP_PhysChem_Dataset(val_df, TENSORS_DIR, train_dataset.global_p_dims, t_mean, t_std)
        test_dataset = QMAP_PhysChem_Dataset(test_df, TENSORS_DIR, train_dataset.global_p_dims, t_mean, t_std)
        
        batch_sz = 32
        train_loader = DataLoader(train_dataset, batch_size=batch_sz, shuffle=True, drop_last=True) 
        val_loader = DataLoader(val_dataset, batch_size=batch_sz, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_sz, shuffle=False)
        
        model = MBC_Attention(p_dims=train_dataset.global_p_dims).to(DEVICE)
        
        # Use a gentler learning rate to prevent early divergence
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        epochs = 100
        patience = 20
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_path = os.path.join(MODEL_SAVE_DIR, f"mbc_best_fold_{fold}.pt")
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            
            for batch_idx, batch in enumerate(train_loader):
                features = [f.to(DEVICE) for f in batch['features']]
                targets = batch['z_mic'].to(DEVICE) 
                
                # Data Health Probe (Print only for the first batch of Epoch 0)
                if epoch == 0 and batch_idx == 0:
                    ft_sum = sum([f.abs().sum().item() for f in features])
                    print(f"[Health Probe] First batch feature activity (sum of absolutes): {ft_sum:.2f}")
                    if ft_sum < 1e-5:
                        print("Fatal Error: Your feature inputs are all zeros! The model cannot learn anything.")
                        print("Solution: Please verify that the .pt files in TENSORS_DIR were extracted successfully.")
                        sys.exit(1)
                
                optimizer.zero_grad()
                preds = model(features)
                loss = criterion(preds, targets)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()
                
            model.eval()
            val_loss = 0
            val_preds_list = []
            
            with torch.no_grad():
                for batch in val_loader:
                    features = [f.to(DEVICE) for f in batch['features']]
                    targets = batch['z_mic'].to(DEVICE)
                    preds = model(features)
                    val_loss += criterion(preds, targets).item()
                    val_preds_list.extend(preds.cpu().numpy())
                    
            val_loss /= len(val_loader) if len(val_loader) > 0 else 1
            scheduler.step(val_loss)
            
            if epoch == 0 or epoch == 1:
                val_preds_std = np.std(val_preds_list)
                if val_preds_std < 1e-6:
                    print(f"\n[Early Stop Radar] Model collapsed at Epoch {epoch+1}, Standard Deviation: {val_preds_std:.6f}")
                    sys.exit(1)
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:03d} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), best_model_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Normal Early Stopping Triggered (Patience={patience}). Best Val Loss: {best_val_loss:.4f}")
                    break
                    
        model.load_state_dict(torch.load(best_model_path, weights_only=True))
        model.eval()
        
        y_pred_log, y_test_log, y_test_raw = [], [], []
        with torch.no_grad():
            for batch in test_loader:
                features = [f.to(DEVICE) for f in batch['features']]
                preds_z = model(features).cpu().numpy()
                
                preds_log_mic = preds_z * t_std + t_mean
                
                y_pred_log.extend(preds_log_mic)
                y_test_log.extend(batch['log_mic'].numpy())
                y_test_raw.extend(batch['raw_mic'].numpy())
                
        y_pred_log = np.array(y_pred_log)
        y_test_log = np.array(y_test_log)
        y_test_raw = np.array(y_test_raw)
        
        full_metrics = compute_metrics(y_test_log, y_pred_log)
        all_full_metrics.append(full_metrics)
        
        hp_mask = y_test_raw < 10.0
        if np.sum(hp_mask) > 1:
            hp_metrics = compute_metrics(y_test_log[hp_mask], y_pred_log[hp_mask])
            all_hp_metrics.append(hp_metrics)

    # ==========================================
    # 7. Execute Summary and Save
    # ==========================================
    save_and_print_final_summary(
        all_full_metrics, 
        "MBC-Attention Baseline - Full Test Set (Log10 Space)", 
        os.path.join(RESULTS_DIR, "mbc_attention_full_dataset_summary.csv")
    )

    save_and_print_final_summary(
        all_hp_metrics, 
        "MBC-Attention Baseline - High Potency Subset (MIC < 10)", 
        os.path.join(RESULTS_DIR, "mbc_attention_high_potency_summary.csv")
    )

if __name__ == "__main__":
    train_and_evaluate()