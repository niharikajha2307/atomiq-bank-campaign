import os
import sys

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(WORKSPACE)
sys.path.insert(0, WORKSPACE)

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
)
from src.preprocessing import build_classical_preprocessor
from src.evaluate import calculate_ece

def run_job1():
    df = pd.read_csv("data/Stage1_Train.csv")
    X = df.drop(columns=["customer_id", "y"])
    y = (df["y"] == "yes").astype(int).values
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # 1. Constant-prior baseline
    b_p, l_p, e_p = [], [], []
    for tr, val in skf.split(X, y):
        pi_tr = float(y[tr].mean())
        pred = np.full(len(val), pi_tr)
        b_p.append(brier_score_loss(y[val], pred))
        l_p.append(log_loss(y[val], pred))
        e_p.append(calculate_ece(y[val], pred))
    print(f"Constant_Prior: Brier={np.mean(b_p):.4f}+-{np.std(b_p):.4f}, LogLoss={np.mean(l_p):.4f}+-{np.std(l_p):.4f}, ECE={np.mean(e_p):.4f}+-{np.std(e_p):.4f}", flush=True)

    # 2. Unweighted Logistic Regression with C in {0.03, 0.1, 0.3, 1}
    for C_val in [0.03, 0.1, 0.3, 1.0]:
        rocs, prs, p10s, r10s, briers, logs, eces = [], [], [], [], [], [], []
        for tr, val in skf.split(X, y):
            pipe = build_classical_preprocessor()
            X_tr = pipe.fit_transform(X.iloc[tr], y[tr])
            X_val = pipe.transform(X.iloc[val])
            clf = LogisticRegression(C=C_val, penalty="l2", solver="lbfgs", max_iter=1000, random_state=42)
            clf.fit(X_tr, y[tr])
            prob = clf.predict_proba(X_val)[:, 1]
            
            rocs.append(roc_auc_score(y[val], prob))
            prs.append(average_precision_score(y[val], prob))
            briers.append(brier_score_loss(y[val], prob))
            logs.append(log_loss(y[val], prob))
            eces.append(calculate_ece(y[val], prob))
            top10_idx = np.argsort(prob)[::-1][:int(len(val) * 0.1)]
            p10s.append(np.mean(y[val][top10_idx]))
            r10s.append(np.sum(y[val][top10_idx]) / np.sum(y[val]))
        print(f"LR_Unweighted_C{C_val}: ROC={np.mean(rocs):.4f}+-{np.std(rocs):.4f}, PR={np.mean(prs):.4f}+-{np.std(prs):.4f}, P@10={np.mean(p10s):.4f}+-{np.std(p10s):.4f}, R@10={np.mean(r10s):.4f}+-{np.std(r10s):.4f}, Brier={np.mean(briers):.4f}+-{np.std(briers):.4f}, LogLoss={np.mean(logs):.4f}+-{np.std(logs):.4f}, ECE={np.mean(eces):.4f}+-{np.std(eces):.4f}", flush=True)

    # 3. LogisticRegression_Balanced with prior-shift correction
    rocs, prs, p10s, r10s, briers, logs, eces = [], [], [], [], [], [], []
    for tr, val in skf.split(X, y):
        pipe = build_classical_preprocessor()
        X_tr = pipe.fit_transform(X.iloc[tr], y[tr])
        X_val = pipe.transform(X.iloc[val])
        clf = LogisticRegression(class_weight="balanced", solver="lbfgs", max_iter=1000, random_state=42)
        clf.fit(X_tr, y[tr])
        
        pi_tr = float(y[tr].mean())
        shift = np.log(pi_tr / (1.0 - pi_tr))
        decision = clf.decision_function(X_val) + shift
        prob = 1.0 / (1.0 + np.exp(-decision))
        
        rocs.append(roc_auc_score(y[val], prob))
        prs.append(average_precision_score(y[val], prob))
        briers.append(brier_score_loss(y[val], prob))
        logs.append(log_loss(y[val], prob))
        eces.append(calculate_ece(y[val], prob))
        top10_idx = np.argsort(prob)[::-1][:int(len(val) * 0.1)]
        p10s.append(np.mean(y[val][top10_idx]))
        r10s.append(np.sum(y[val][top10_idx]) / np.sum(y[val]))
    print(f"LR_Balanced_PriorShift: ROC={np.mean(rocs):.4f}+-{np.std(rocs):.4f}, PR={np.mean(prs):.4f}+-{np.std(prs):.4f}, P@10={np.mean(p10s):.4f}+-{np.std(p10s):.4f}, R@10={np.mean(r10s):.4f}+-{np.std(r10s):.4f}, Brier={np.mean(briers):.4f}+-{np.std(briers):.4f}, LogLoss={np.mean(logs):.4f}+-{np.std(logs):.4f}, ECE={np.mean(eces):.4f}+-{np.std(eces):.4f}", flush=True)

if __name__ == "__main__":
    run_job1()
