import os
import sys
sys.path.insert(0, os.path.abspath("."))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_curve,
    precision_recall_curve
)
from sklearn.calibration import calibration_curve

from src.classical_models import get_classical_models
from src.quantum_models import get_quantum_models

def calculate_ece(y_true, y_prob, n_bins=5):
    """
    Computes Expected Calibration Error (ECE) across n_bins.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        mask = (y_prob >= bin_lower) & (y_prob < bin_upper) if i < n_bins - 1 else (y_prob >= bin_lower) & (y_prob <= bin_upper)
        bin_size = np.sum(mask)
        if bin_size > 0:
            prop_true = np.mean(y_true[mask])
            avg_prob = np.mean(y_prob[mask])
            ece += (bin_size / len(y_true)) * np.abs(prop_true - avg_prob)
    return ece

def evaluate_models_cv(train_csv_path="data/Stage1_Train.csv", n_splits=5, random_state=42):
    """
    Runs 5-fold Stratified Cross-Validation on Stage1_Train ONLY.
    Strictly fits all transformers, selectors, and classifiers on training folds.
    """
    df_train = pd.read_csv(train_csv_path, dtype=str, keep_default_na=False)
    X = df_train.drop(columns=['customer_id', 'y'])
    y = (df_train['y'] == 'yes').astype(int).values
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Collect models
    all_models = {}
    all_models.update(get_classical_models(random_state=random_state))
    all_models.update(get_quantum_models(random_state=random_state))
    
    results = []
    oof_predictions = {name: np.zeros(len(y)) for name in all_models}
    
    print(f"Starting {n_splits}-Fold Stratified Cross-Validation on {len(df_train)} training samples...")
    print(f"Total Positives (y=1): {y.sum()} / {len(y)} ({y.mean()*100:.2f}%)\n")
    
    for model_name, pipeline in all_models.items():
        print(f"--> Evaluating: {model_name}...")
        fold_metrics = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            X_tr, y_tr = X.iloc[train_idx], y[train_idx]
            X_val, y_val = X.iloc[val_idx], y[val_idx]
            
            # Fit strictly on train fold
            pipeline.fit(X_tr, y_tr)
            
            # Predict validation probabilities
            val_probs = pipeline.predict_proba(X_val)[:, 1]
            oof_predictions[model_name][val_idx] = val_probs
            
            # Compute fold metrics
            roc = roc_auc_score(y_val, val_probs)
            pr = average_precision_score(y_val, val_probs)
            brier = brier_score_loss(y_val, val_probs)
            loss = log_loss(y_val, val_probs)
            ece = calculate_ece(y_val, val_probs)
            
            # Precision@10 and Recall@10 on validation fold
            val_order = np.argsort(val_probs)[::-1]
            top10_idx = val_order[:10]
            positives_in_top10 = np.sum(y_val[top10_idx])
            total_val_positives = np.sum(y_val)
            
            p_at_10 = positives_in_top10 / 10.0
            r_at_10 = positives_in_top10 / total_val_positives if total_val_positives > 0 else 0.0
            
            fold_metrics.append({
                'roc_auc': roc,
                'pr_auc': pr,
                'precision_at_10': p_at_10,
                'recall_at_10': r_at_10,
                'brier_score': brier,
                'log_loss': loss,
                'ece': ece
            })
            
        df_fold = pd.DataFrame(fold_metrics)
        summary = {
            'Model': model_name,
            'ROC-AUC (mean)': df_fold['roc_auc'].mean(),
            'ROC-AUC (std)': df_fold['roc_auc'].std(),
            'PR-AUC (mean)': df_fold['pr_auc'].mean(),
            'PR-AUC (std)': df_fold['pr_auc'].std(),
            'Precision@10 (mean)': df_fold['precision_at_10'].mean(),
            'Precision@10 (std)': df_fold['precision_at_10'].std(),
            'Recall@10 (mean)': df_fold['recall_at_10'].mean(),
            'Recall@10 (std)': df_fold['recall_at_10'].std(),
            'Brier Score (mean)': df_fold['brier_score'].mean(),
            'Brier Score (std)': df_fold['brier_score'].std(),
            'Log Loss (mean)': df_fold['log_loss'].mean(),
            'Log Loss (std)': df_fold['log_loss'].std(),
            'ECE (mean)': df_fold['ece'].mean(),
            'ECE (std)': df_fold['ece'].std()
        }
        results.append(summary)
        print(f"    PR-AUC: {summary['PR-AUC (mean)']:.4f} ± {summary['PR-AUC (std)']:.4f} | "
              f"ROC-AUC: {summary['ROC-AUC (mean)']:.4f} ± {summary['ROC-AUC (std)']:.4f} | "
              f"Brier: {summary['Brier Score (mean)']:.4f}")

    df_results = pd.DataFrame(results)
    os.makedirs("outputs", exist_ok=True)
    df_results.to_csv("outputs/stage1_cv_metrics.csv", index=False)
    print("\n=== STAGE 1 CV EVALUATION COMPLETED ===")
    print(df_results.to_string())
    
    # Generate diagnostic plots
    _plot_diagnostics(y, oof_predictions)
    
    return df_results

def _plot_diagnostics(y_true, oof_predictions):
    """
    Plots ROC, PR, and Calibration curves for all models using Out-Of-Fold predictions.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. ROC Curves
    ax_roc = axes[0]
    ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random (0.50)')
    for name, probs in oof_predictions.items():
        fpr, tpr, _ = roc_curve(y_true, probs)
        score = roc_auc_score(y_true, probs)
        ax_roc.plot(fpr, tpr, label=f"{name} ({score:.3f})")
    ax_roc.set_title("ROC Curves (5-Fold OOF)")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.legend(fontsize=8, loc='lower right')
    ax_roc.grid(True, alpha=0.3)
    
    # 2. Precision-Recall Curves
    ax_pr = axes[1]
    baseline_pr = y_true.mean()
    ax_pr.axhline(baseline_pr, color='k', linestyle='--', alpha=0.5, label=f'Baseline ({baseline_pr:.3f})')
    for name, probs in oof_predictions.items():
        precision, recall, _ = precision_recall_curve(y_true, probs)
        score = average_precision_score(y_true, probs)
        ax_pr.plot(recall, precision, label=f"{name} ({score:.3f})")
    ax_pr.set_title("Precision-Recall Curves (5-Fold OOF)")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.legend(fontsize=8, loc='upper right')
    ax_pr.grid(True, alpha=0.3)
    
    # 3. Calibration Curves
    ax_cal = axes[2]
    ax_cal.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Calibration')
    for name, probs in oof_predictions.items():
        prob_true, prob_pred = calibration_curve(y_true, probs, n_bins=5)
        brier = brier_score_loss(y_true, probs)
        ax_cal.plot(prob_pred, prob_true, marker='o', label=f"{name} (Brier: {brier:.3f})")
    ax_cal.set_title("Calibration Curves (5-Fold OOF)")
    ax_cal.set_xlabel("Mean Predicted Probability")
    ax_cal.set_ylabel("Fraction of Positives")
    ax_cal.legend(fontsize=8, loc='upper left')
    ax_cal.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("outputs/stage1_cv_curves.png", dpi=300)
    plt.close()
    print("Saved diagnostic plots to outputs/stage1_cv_curves.png")

if __name__ == "__main__":
    evaluate_models_cv()
