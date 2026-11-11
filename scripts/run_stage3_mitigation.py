"""
scripts/run_stage3_mitigation.py
================================
Stage 3 Readout-Error Mitigation:
- Reuses the existing Stage 3 6-qubit circuit (same qubit order, same angles)
- Noise model: 0.1% 1-qubit depolarizing error + 1.5% symmetric readout error
- 8192 shots with seed_simulator=42 (reproduces unmitigated 0.207764 exactly)
- Mitigation method: Per-qubit confusion matrix inversion:
    A = [[1-e, e], [e, 1-e]] with e = 0.015
    Full 6-qubit matrix is tensor product: A^-1_6q = (A^-1)^⊗6
- Evaluates raw-inverted and clipped-renormalized probability distributions
- Updates stage3_simulation.json and stage3_simulation.csv with mitigation fields
"""

import os
import sys
import json
import numpy as np
import pandas as pd

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(WORKSPACE)
sys.path.insert(0, WORKSPACE)

def main():
    print("=" * 70)
    print("STAGE 3 READOUT ERROR MITIGATION")
    print("=" * 70)

    # 1. Load selected customers and probabilities
    stage2_df = pd.read_csv("stage2_selection.csv")
    s3_probs = stage2_df["predicted_probability"].values
    s3_ids = stage2_df["customer_id"].tolist()

    # Exact Poisson-Binomial DP
    def exact_poisson_binomial(probs_arr, k_thresh=3):
        dp = np.zeros(len(probs_arr) + 1)
        dp[0] = 1.0
        for p_val in probs_arr:
            dp_next = np.zeros(len(probs_arr) + 1)
            dp_next[0] = dp[0] * (1.0 - p_val)
            for j in range(1, len(probs_arr) + 1):
                dp_next[j] = dp[j] * (1.0 - p_val) + dp[j - 1] * p_val
            dp = dp_next
        return float(dp[k_thresh:].sum()), dp

    p_exact_ge3, _ = exact_poisson_binomial(s3_probs, k_thresh=3)

    # Build 6-qubit circuit
    qc_s3 = QuantumCircuit(6)
    ry_angles = [2.0 * np.arcsin(np.sqrt(p_val)) for p_val in s3_probs]
    for q, ang in enumerate(ry_angles):
        qc_s3.ry(ang, q)
    qc_s3.measure_all()

    # Setup noise model
    noise_model = NoiseModel()
    error_1q = depolarizing_error(0.001, 1)
    noise_model.add_all_qubit_quantum_error(error_1q, ['ry', 'u', 'sx'])
    e = 0.015
    readout_err = ReadoutError([[1.0 - e, e], [e, 1.0 - e]])
    noise_model.add_all_qubit_readout_error(readout_err)

    # Run noisy simulation with seed=42 and 8192 shots
    noisy_sim = AerSimulator(noise_model=noise_model)
    job_noisy = noisy_sim.run(transpile(qc_s3, noisy_sim), shots=8192, seed_simulator=42)
    counts_noisy = job_noisy.result().get_counts()

    # Observed probability vector over 64 states
    p_obs = np.zeros(64)
    for i in range(64):
        bs = format(i, '06b')
        p_obs[i] = counts_noisy.get(bs, 0) / 8192.0

    # Success indicator: states with Hamming weight >= 3
    success_mask = np.array([bin(i).count('1') >= 3 for i in range(64)], dtype=float)

    # Unmitigated probability
    p_unmit = float((p_obs * success_mask).sum())
    unmit_err = abs(p_unmit - p_exact_ge3)
    se_unmit = float(np.sqrt(p_unmit * (1.0 - p_unmit) / 8192.0))

    # Readout error mitigation via tensor-product matrix inversion
    A_1q = np.array([[1.0 - e, e], [e, 1.0 - e]])
    A_inv_1q = np.linalg.inv(A_1q)

    # 6-qubit tensor product inverse
    A_inv_6q = A_inv_1q
    for _ in range(5):
        A_inv_6q = np.kron(A_inv_6q, A_inv_1q)

    # Raw inverted probability distribution
    p_mit_raw = A_inv_6q @ p_obs
    p_mit_raw_val = float((p_mit_raw * success_mask).sum())

    # Clipped and renormalized probability distribution
    p_mit_clipped = np.clip(p_mit_raw, 0.0, None)
    p_mit_clipped /= p_mit_clipped.sum()
    p_mit_clipped_val = float((p_mit_clipped * success_mask).sum())

    mit_err = abs(p_mit_clipped_val - p_exact_ge3)

    # Standard error for mitigated estimator
    c = success_mask @ A_inv_6q
    var_mit = (np.sum(p_obs * (c ** 2)) - (np.sum(p_obs * c) ** 2)) / 8192.0
    se_mit = float(np.sqrt(var_mit))

    print(f"Exact Analytical P(>=3)       : {p_exact_ge3:.6f}")
    print(f"Noisy Unmitigated P(>=3)      : {p_unmit:.6f} | Error: {unmit_err:.6f} | SE: {se_unmit:.6f}")
    print(f"Noisy Mitigated (Raw Inversion): {p_mit_raw_val:.6f} | Min Prob in Dist: {p_mit_raw.min():.6e}")
    print(f"Noisy Mitigated (Clipped)     : {p_mit_clipped_val:.6f} | Error: {mit_err:.6f} | SE: {se_mit:.6f}")

    # 2. Update stage3_simulation.json
    json_path = "stage3_simulation.json"
    with open(json_path, "r") as f:
        data = json.load(f)

    data["noisy_simulation"]["mitigation_method"] = "tensor-product per-qubit confusion matrix inversion (A^-1)^⊗6"
    data["noisy_simulation"]["readout_error_assumed"] = e
    data["noisy_simulation"]["unmitigated_estimate"] = round(p_unmit, 6)
    data["noisy_simulation"]["unmitigated_error_vs_exact"] = round(unmit_err, 6)
    data["noisy_simulation"]["unmitigated_standard_error"] = round(se_unmit, 6)
    data["noisy_simulation"]["mitigated_estimate_raw"] = round(p_mit_raw_val, 6)
    data["noisy_simulation"]["mitigated_estimate"] = round(p_mit_clipped_val, 6)
    data["noisy_simulation"]["mitigated_error_vs_exact"] = round(mit_err, 6)
    data["noisy_simulation"]["mitigated_standard_error"] = round(se_mit, 6)

    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Updated {json_path}.")

    # 3. Update stage3_simulation.csv
    csv_path = "stage3_simulation.csv"
    df_s3 = pd.read_csv(csv_path)

    # Add mitigation columns while preserving existing rows and columns
    df_s3["mitigation_method"] = "None"
    df_s3["readout_error_assumed"] = 0.0
    df_s3["unmitigated_estimate"] = df_s3["p_ge3_estimate"]
    df_s3["unmitigated_error_vs_exact"] = df_s3["absolute_error"]
    df_s3["mitigated_estimate"] = df_s3["p_ge3_estimate"]
    df_s3["mitigated_error_vs_exact"] = df_s3["absolute_error"]

    # Append noisy unmitigated and noisy mitigated rows
    noisy_unmit_row = {
        "shots": 8192,
        "seed": 42,
        "p_ge3_estimate": round(p_unmit, 6),
        "exact_p_ge3": round(p_exact_ge3, 6),
        "absolute_error": round(unmit_err, 6),
        "mitigation_method": "Unmitigated",
        "readout_error_assumed": e,
        "unmitigated_estimate": round(p_unmit, 6),
        "unmitigated_error_vs_exact": round(unmit_err, 6),
        "mitigated_estimate": round(p_unmit, 6),
        "mitigated_error_vs_exact": round(unmit_err, 6)
    }

    noisy_mit_row = {
        "shots": 8192,
        "seed": 42,
        "p_ge3_estimate": round(p_mit_clipped_val, 6),
        "exact_p_ge3": round(p_exact_ge3, 6),
        "absolute_error": round(mit_err, 6),
        "mitigation_method": "Per-Qubit Confusion Matrix Inversion",
        "readout_error_assumed": e,
        "unmitigated_estimate": round(p_unmit, 6),
        "unmitigated_error_vs_exact": round(unmit_err, 6),
        "mitigated_estimate": round(p_mit_clipped_val, 6),
        "mitigated_error_vs_exact": round(mit_err, 6)
    }

    df_s3 = pd.concat([df_s3, pd.DataFrame([noisy_unmit_row, noisy_mit_row])], ignore_index=True)
    df_s3.to_csv(csv_path, index=False)
    print(f"Updated {csv_path}.")

if __name__ == "__main__":
    main()
