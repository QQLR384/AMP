import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from config import SCRIPT_DIR, DATA_DIR, RESULTS_DIR, seed_everything
from dataset import ANIADataset
from models import ANIA
from utils import compute_metrics, save_and_print_final_summary
from features import generate_aligned_cgr

# ==========================================
# 6. 5-Fold Main Training Loop
# ==========================================
def main():
    seed_everything(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Start ANIA (Offline High-Speed Feature Reading) 5-Fold Cross-Validation")
    print(f"Compute device: {device}")
    
    # 1. Optionally generate features if they don't exist
    print("\nChecking for CGR features...")
    features_exist = True
    for fold in range(5):
        if not os.path.exists(os.path.join(DATA_DIR, f"qmap_train_set_split_{fold}_cgr.csv")):
            features_exist = False
            break
            
    if not features_exist:
        print("Features missing. Generating now...")
        generate_aligned_cgr()
    else:
        print("CGR features found. Proceeding to training.")
    
    all_full_metrics = []
    all_hp_metrics = []
    TOTAL_EPOCHS = 50
    PATIENCE = 10

    for fold in range(5):
        print(f"\n{'='*60}\nStart evaluating Fold {fold+1}/5\n{'='*60}")
        
        # Load the _cgr.csv file containing the features
        train_path = os.path.join(DATA_DIR, f"qmap_train_set_split_{fold}_cgr.csv")
        test_path = os.path.join(DATA_DIR, f"qmap_test_set_split_{fold}_cgr.csv")
        
        if not os.path.exists(train_path) or not os.path.exists(test_path):
            print(f"Warning: Feature file for Fold {fold} not found, please check file path!")
            continue
            
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        # Split 10% from the training set for validation
        train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)
        
        # Dataset initialization automatically converts 2816-dim features to Tensor
        train_loader = DataLoader(ANIADataset(train_df), batch_size=32, shuffle=True)
        val_loader = DataLoader(ANIADataset(val_df), batch_size=32, shuffle=False)
        test_loader = DataLoader(ANIADataset(test_df), batch_size=32, shuffle=False)

        print("Initializing ANIA model...")
        model = ANIA().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.MSELoss()
        
        best_loss = float('inf')
        patience_counter = 0
        best_path = os.path.join(SCRIPT_DIR, f"ania_fold_{fold}_best.pt")

        for epoch in range(TOTAL_EPOCHS):
            # --- Training Phase ---
            model.train()
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{TOTAL_EPOCHS} [Train]")
            for b in pbar:
                fcgr = b['fcgr'].to(device)
                mic_log = b['mic_log'].to(device)
                
                optimizer.zero_grad()
                pred_mic = model(fcgr)
                loss = criterion(pred_mic, mic_log)
                loss.backward()
                optimizer.step()
                
                pbar.set_postfix({"MIC_Loss": f"{loss.item():.4f}"})
            
            # --- Validation Phase ---
            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for b in val_loader:
                    fcgr = b['fcgr'].to(device)
                    mic_log = b['mic_log'].to(device)
                    pred_mic = model(fcgr)
                    total_val_loss += criterion(pred_mic, mic_log).item()
                    
            avg_val_loss = total_val_loss / len(val_loader)
            print(f" Epoch {epoch+1} Complete | Val MIC Loss: {avg_val_loss:.4f}")
            
            # --- Early Stopping and Save ---
            if avg_val_loss < best_loss:
                print(f" Saved best model weights (Val Loss: {best_loss:.4f} -> {avg_val_loss:.4f})")
                best_loss = avg_val_loss
                torch.save(model.state_dict(), best_path)
                patience_counter = 0
            else:
                patience_counter += 1
                print(f" Validation Loss did not improve, Patience: {patience_counter}/{PATIENCE}")
                if patience_counter >= PATIENCE:
                    print(f" Triggered Early Stopping, training for this fold ended early!")
                    break

        # --- Blind Test Phase ---
        print("Loading best weights for QMAP strict blind test...")
        model.load_state_dict(torch.load(best_path))
        model.eval()
        
        y_true_raw = test_df['TARGET ACTIVITY - CONCENTRATION - PROCED'].values
        y_true_log = np.log10(y_true_raw)
        y_pred_log = []
        
        with torch.no_grad():
            for b in tqdm(test_loader, desc="Predicting"):
                fcgr = b['fcgr'].to(device)
                preds = model(fcgr).view(-1).cpu().numpy()
                y_pred_log.extend(preds)
        
        y_pred_log = np.array(y_pred_log)
        
        # Metric Collection
        all_full_metrics.append(compute_metrics(y_true_log, y_pred_log))
        hp_mask = y_true_raw < 10.0
        if hp_mask.any(): 
            all_hp_metrics.append(compute_metrics(y_true_log[hp_mask], y_pred_log[hp_mask]))
            
        if os.path.exists(best_path): os.remove(best_path)

    # ==========================================
    # 7. Summary Export
    # ==========================================
    save_and_print_final_summary(
        all_full_metrics, 
        "ANIA - Full Test Set", 
        os.path.join(RESULTS_DIR, "ania_full_dataset_summary.csv")
    )
    save_and_print_final_summary(
        all_hp_metrics, 
        "ANIA - High Potency Subset (MIC < 10)", 
        os.path.join(RESULTS_DIR, "ania_high_potency_summary.csv")
    )

if __name__ == "__main__": 
    main()