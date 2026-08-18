import os
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from transformers import EsmModel, EsmTokenizer
from tqdm import tqdm

from config import (
    device, MODEL_PATH, DATA_DIR, RESULTS_DIR, SCRIPT_DIR,
    PHASE1_EPOCHS, PHASE2_EPOCHS, TOTAL_EPOCHS, PATIENCE,
    HIGH_POTENCY_THRESHOLD_LOG, HP_LOSS_WEIGHT,
    seed_worker, g
)
from dataset import QMAP_Tensor_Dataset, collate_fn_tensors
from models import WeightedMSELoss, DIR_ZeroParamPhys_Adapter
from utils import compute_metrics, save_and_print_final_summary


# ==========================================
# 6. 5-Fold Main Training Loop
# ==========================================
def main():
    all_full_metrics = []
    all_hp_metrics = []
    
    tokenizer = EsmTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    mapping_df = pd.read_csv(os.path.join(DATA_DIR, "sequence_to_tensor_mapping.csv"))

    for split_idx in range(5):
        print(f"\n{'='*60}\n Start evaluating Split {split_idx}\n{'='*60}")
        
        train_full_df = pd.read_csv(os.path.join(DATA_DIR, f"qmap_train_set_split_{split_idx}.csv"))
        test_df = pd.read_csv(os.path.join(DATA_DIR, f"qmap_test_set_split_{split_idx}.csv"))
        
        train_df, val_df = train_test_split(train_full_df, test_size=0.1, random_state=42)
        
        train_loader = DataLoader(
            QMAP_Tensor_Dataset(train_df, mapping_df), batch_size=32, shuffle=True,
            collate_fn=lambda x: collate_fn_tensors(x, tokenizer),
            worker_init_fn=seed_worker, generator=g
        )
        val_loader = DataLoader(
            QMAP_Tensor_Dataset(val_df, mapping_df), batch_size=32, shuffle=False,
            collate_fn=lambda x: collate_fn_tensors(x, tokenizer)
        )
        test_loader = DataLoader(
            QMAP_Tensor_Dataset(test_df, mapping_df), batch_size=32, shuffle=False,
            collate_fn=lambda x: collate_fn_tensors(x, tokenizer)
        )
        
        y_test_raw = test_loader.dataset.data['TARGET ACTIVITY - CONCENTRATION - PROCED'].values
        y_test_log = np.log10(y_test_raw)

        esm_model = EsmModel.from_pretrained(MODEL_PATH, local_files_only=True).to(device)
        adapter = DIR_ZeroParamPhys_Adapter(input_dim=480).to(device)
        
        weighted_mse_loss = WeightedMSELoss(hp_threshold=HIGH_POTENCY_THRESHOLD_LOG, hp_weight=HP_LOSS_WEIGHT)
        bce_loss = nn.BCELoss()
        mse_aux_loss = nn.MSELoss()
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_esm_path = os.path.join(SCRIPT_DIR, f"temp_esm_split_{split_idx}_best.pth")
        best_adapter_path = os.path.join(SCRIPT_DIR, f"temp_adapter_split_{split_idx}_best.pth")
        
        for epoch in range(TOTAL_EPOCHS):
            if epoch == 0:
                for param in esm_model.parameters(): param.requires_grad = False
                optimizer = torch.optim.AdamW(adapter.parameters(), lr=5e-4)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=PHASE1_EPOCHS)
            elif epoch == PHASE1_EPOCHS:
                for param in esm_model.parameters(): param.requires_grad = True
                optimizer = torch.optim.AdamW([{'params': esm_model.parameters(), 'lr': 2e-5},
                                               {'params': adapter.parameters(), 'lr': 1e-4}], weight_decay=1e-4)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=PHASE2_EPOCHS)

            esm_model.train() if epoch >= PHASE1_EPOCHS else esm_model.eval()
            adapter.train()
            
            total_train_loss = 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{TOTAL_EPOCHS} [Train]")
            for batch in pbar:
                input_ids, mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
                coords = batch['coords'].to(device)
                vec = batch['vec_sidechain'].to(device)
                dih = batch['dihedrals'].to(device)
                mic_true, is_hp_true = batch['mic_log'].to(device), batch['is_hp'].to(device)
                charge_true, gravy_true = batch['charge'].to(device), batch['gravy'].to(device)
                
                outputs = esm_model(input_ids=input_ids, attention_mask=mask)
                pred_mic, pred_cls, pred_charge, pred_gravy = adapter(outputs.last_hidden_state, coords, vec, dih, mask)
                
                loss_reg = weighted_mse_loss(pred_mic.view(-1), mic_true.view(-1))
                loss_cls = bce_loss(pred_cls.view(-1), is_hp_true.view(-1))
                loss_charge = mse_aux_loss(pred_charge.view(-1), charge_true.view(-1))
                loss_gravy = mse_aux_loss(pred_gravy.view(-1), gravy_true.view(-1))
                
                loss = loss_reg + 0.5 * loss_cls + 0.1 * (loss_charge + loss_gravy)
                
                optimizer.zero_grad()
                loss.backward()
                
                if epoch >= PHASE1_EPOCHS: torch.nn.utils.clip_grad_norm_(esm_model.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_train_loss += loss_reg.item()
                pbar.set_postfix({"Reg": f"{loss_reg.item():.3f}"})
                
            scheduler.step()
            
            esm_model.eval()
            adapter.eval()
            total_val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids, mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
                    coords = batch['coords'].to(device)
                    vec = batch['vec_sidechain'].to(device)
                    dih = batch['dihedrals'].to(device)
                    mic_true = batch['mic_log'].to(device)
                    
                    outputs = esm_model(input_ids=input_ids, attention_mask=mask)
                    pred_mic, _, _, _ = adapter(outputs.last_hidden_state, coords, vec, dih, mask)
                    total_val_loss += mse_aux_loss(pred_mic.view(-1), mic_true.view(-1)).item()
                    
            avg_val_loss = total_val_loss / len(val_loader) if len(val_loader) > 0 else float('inf')
            print(f" Epoch {epoch+1} End | Val MSE Loss: {avg_val_loss:.4f}")
            
            if avg_val_loss < best_val_loss:
                best_val_loss, patience_counter = avg_val_loss, 0
                torch.save(esm_model.state_dict(), best_esm_path)
                torch.save(adapter.state_dict(), best_adapter_path)
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE: break
                    
        if os.path.exists(best_esm_path) and os.path.exists(best_adapter_path):
            esm_model.load_state_dict(torch.load(best_esm_path, weights_only=True))
            adapter.load_state_dict(torch.load(best_adapter_path, weights_only=True))
        
        esm_model.eval()
        adapter.eval()
        y_pred_log = []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Predicting"):
                input_ids, mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
                coords = batch['coords'].to(device)
                vec = batch['vec_sidechain'].to(device)
                dih = batch['dihedrals'].to(device)
                outputs = esm_model(input_ids=input_ids, attention_mask=mask)
                
                pred_mic, _, _, _ = adapter(outputs.last_hidden_state, coords, vec, dih, mask)
                y_pred_log.extend(pred_mic.view(-1).cpu().numpy())
                
        y_pred_log = np.array(y_pred_log)
        all_full_metrics.append(compute_metrics(y_test_log, y_pred_log))
        
        hp_mask = y_test_raw < 10.0
        if np.sum(hp_mask) > 1:
            all_hp_metrics.append(compute_metrics(y_test_log[hp_mask], y_pred_log[hp_mask]))
        else:
            print(f"Warning: Insufficient high potency samples in the current test set (N={np.sum(hp_mask)}). Skipping metrics calculation for this subset.")

        del esm_model, adapter, optimizer, train_loader, val_loader, test_loader
        torch.cuda.empty_cache(); gc.collect()

    save_and_print_final_summary(all_full_metrics, "DIR-VerifyZeroCoords-PConv - Full", os.path.join(RESULTS_DIR, "dir_verifyzerocoords_full.csv"))
    save_and_print_final_summary(all_hp_metrics, "DIR-VerifyZeroCoords-PConv - High Potency", os.path.join(RESULTS_DIR, "dir_verifyzerocoords_hp.csv"))

if __name__ == '__main__':
    main()