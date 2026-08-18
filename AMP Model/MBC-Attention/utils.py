import os
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats

# ==========================================
# 5. Evaluation and Logging Tools
# ==========================================
def compute_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if np.std(y_pred) < 1e-6 or np.std(y_true) < 1e-6:
            pearson_r, spearman_r, kendall_tau = 0.0, 0.0, 0.0
        else:
            pearson_r, _ = stats.pearsonr(y_true, y_pred)
            spearman_r, _ = stats.spearmanr(y_true, y_pred)
            kendall_tau, _ = stats.kendalltau(y_true, y_pred)
            
    return {
        'RMSE': rmse, 'MSE': mse, 'MAE': mae, 'R2': r2,
        'Pearson': pearson_r, 'Spearman': spearman_r, 'Kendall': kendall_tau
    }

def save_and_print_final_summary(metrics_list, title, save_path):
    if not metrics_list: return
    df_metrics = pd.DataFrame(metrics_list)
    print(f"\n\n{'*'*70}\nFinal 5-Fold Summary Report: {title}\n{'*'*70}")
    
    export_data = []
    for col in df_metrics.columns:
        mean_val = df_metrics[col].mean()
        std_val = df_metrics[col].std()
        median_val = df_metrics[col].median()
        
        arrow = "↓" if col in ["RMSE", "MSE", "MAE"] else "↑"
        mean_std_str = f"{mean_val:.4f} ± {std_val:.4f}"
        print(f" {col:<10} {arrow} : {mean_std_str:<15}  |  Median: {median_val:.4f}")
        
        export_data.append({
            "Metric": col, "Direction": arrow, "Mean": round(mean_val, 4),
            "Std": round(std_val, 4), "Mean ± Std": mean_std_str, "Median": round(median_val, 4)
        })
    print("*"*70)
    
    df_export = pd.DataFrame(export_data)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df_export.to_csv(save_path, index=False, encoding='utf_8_sig')