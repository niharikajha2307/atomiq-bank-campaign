"""
scripts/run_final_stage1.py
===========================
Executes Stage 1 final prediction:
- LogisticRegression(C=0.3, penalty='l2', solver='lbfgs', max_iter=1000, random_state=42)
- Preprocessing inside scikit-learn Pipeline, fitted on all of Stage1_Train only.
- Predicts Stage1_Test exactly once.
- customer_id is strictly a key and never enters model features.
- Saves top-level stage1_results.csv.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(WORKSPACE)
sys.path.insert(0, WORKSPACE)

from src.preprocessing import build_classical_preprocessor

def main():
    train_df = pd.read_csv("data/Stage1_Train.csv", dtype=str, keep_default_na=False)
    test_df = pd.read_csv("data/Stage1_Test.csv", dtype=str, keep_default_na=False)

    X_train = train_df.drop(columns=["customer_id", "y"])
    y_train = (train_df["y"] == "yes").astype(int).values

    test_ids = test_df["customer_id"].tolist()
    X_test = test_df.drop(columns=["customer_id"])

    # Build and fit preprocessing pipeline on Stage1_Train ONLY
    pipe = build_classical_preprocessor()
    X_train_proc = pipe.fit_transform(X_train, y_train)
    X_test_proc = pipe.transform(X_test)

    # Train model
    clf = LogisticRegression(C=0.3, penalty="l2", solver="lbfgs", max_iter=1000, random_state=42)
    clf.fit(X_train_proc, y_train)

    # Predict test probabilities
    test_probs = clf.predict_proba(X_test_proc)[:, 1]

    # Build results DataFrame
    results_df = pd.DataFrame({
        "customer_id": test_ids,
        "predicted_probability": test_probs
    })

    # Sort descending by probability, tie-break by customer_id ascending
    results_df = results_df.sort_values(
        by=["predicted_probability", "customer_id"],
        ascending=[False, True]
    ).reset_index(drop=True)

    results_df["rank"] = np.arange(1, len(results_df) + 1)
    results_df["is_top_10"] = (results_df["rank"] <= 10).astype(int)

    out_path = "stage1_results.csv"
    results_df.to_csv(out_path, index=False)
    print(f"Saved {out_path} with {len(results_df)} rows.")

    top10 = results_df.head(10)
    print("\nTop 10 Customers:")
    for _, r in top10.iterrows():
        print(f"  Rank {int(r['rank'])}: {r['customer_id']} -> Prob = {r['predicted_probability']:.6f}")

    print(f"\nStats: Min={test_probs.min():.6f}, Max={test_probs.max():.6f}, Mean={test_probs.mean():.6f}")

if __name__ == "__main__":
    main()
