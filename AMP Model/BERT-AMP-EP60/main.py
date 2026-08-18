import os
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from config import DATA_DIR, RESULTS_DIR, MAX_EPOCHS, PATIENCE, device
from dataset import ProtBERT_Dynamic_Dataset
from utils import compute_metrics, save_and_print_final_summary
from model_def import REG  # Reusing your defined model

# ==========================================
# 4. 5-Fold Main Training Loop (Early Stopping Mode)
# ==========================================
def main():
    all_full_metrics = []
    all_hp_metrics = []

    for split_idx in range(5):
        print(f"\n{'='*60}\nStart evaluating Split {split_idx}\n{'='*60}")
        
        # 1. Read data
        train_full_df = pd.read_csv(os.path.join(DATA_DIR, f"qmap_train_set_split_{split_idx}.csv"))
        test_df = pd.read_csv(os.path.join(DATA_DIR, f"qmap_test_set_split_{split_idx}.csv"))
        
        train_df, val_df = train_test_split(train_full_df, test_size=0.1, random_state=42)
        
        train_loader = DataLoader(ProtBERT_Dynamic_Dataset(train_df), batch_size=16, shuffle=True)
        val_loader = DataLoader(ProtBERT_Dynamic_Dataset(val_df), batch_size=16, shuffle=False)
        test_loader = DataLoader(ProtBERT_Dynamic_Dataset(test_df), batch_size=16, shuffle=False)
        
        y_test_raw = test_df['TARGET ACTIVITY - CONCENTRATION - PROCED'].values
        y_test_log = np.log10(y_test_raw)

        # 2. Initialize official model
        print("Mounting official REG model...")
        model = REG().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        mse_loss = nn.MSELoss()
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_path = f"./temp_bert_split_{split_idx}_best.pth"
        
        # 3. Training loop with early stopping mechanism
        for epoch in range(MAX_EPOCHS):
            # --- Training Phase ---
            model.train()
            total_train_loss = 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS} [Train]")
            for batch in pbar:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].unsqueeze(1).to(device).float()
                
                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask)
                
                # Crucial fix: Handle tuple returned in model_def.py
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                    
                loss = mse_loss(outputs, labels)
                loss.backward()
                optimizer.step()
                
                total_train_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                
            # --- Validation Phase ---
            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].unsqueeze(1).to(device).float()
                    
                    outputs = model(input_ids, attention_mask)
                    
                    # Crucial fix: Unpack validation set as well
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                        
                    total_val_loss += mse_loss(outputs, labels).item()
                    
            avg_val_loss = total_val_loss / len(val_loader)
            print(f" Epoch {epoch+1} Complete | Val Loss: {avg_val_loss:.4f}")
            
            # --- Early Stopping Check ---
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
                    
        # 4. Ultimate Blind Test (Load best weights of this fold to test real QMAP isolated data)
        print("Loading best weights for this fold, performing strict QMAP blind test...")
        model.load_state_dict(torch.load(best_model_path))
        model.eval()
        
        y_pred_log = []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Predicting"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                outputs = model(input_ids, attention_mask)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                    
                y_pred_log.extend(outputs.cpu().numpy().flatten())
                
        y_pred_log = np.array(y_pred_log)

        # 5. Metric Collection
        full_metrics = compute_metrics(y_test_log, y_pred_log)
        all_full_metrics.append(full_metrics)
        
        hp_mask = y_test_raw < 10.0
        if np.sum(hp_mask) > 0:
            hp_metrics = compute_metrics(y_test_log[hp_mask], y_pred_log[hp_mask])
            all_hp_metrics.append(hp_metrics)

        # Release GPU memory and delete temporary weight files
        del model, optimizer, train_loader, val_loader, test_loader
        torch.cuda.empty_cache()
        gc.collect()
        if os.path.exists(best_model_path):
            os.remove(best_model_path)

    # ==========================================
    # 5. Export Summary
    # ==========================================
    save_and_print_final_summary(
        all_full_metrics, 
        f"BERT-AmPEP60 (Max {MAX_EPOCHS} Epochs + Early Stop) - Full Test Set", 
        os.path.join(RESULTS_DIR, "bert_ampep60_es_full_dataset_summary.csv")
    )
    save_and_print_final_summary(
        all_hp_metrics, 
        f"BERT-AmPEP60 (Max {MAX_EPOCHS} Epochs + Early Stop) - High Potency Subset (MIC < 10)", 
        os.path.join(RESULTS_DIR, "bert_ampep60_es_high_potency_summary.csv")
    )

if __name__ == '__main__':
    main()