"""
src/classical_benchmark.py
==========================
Stage 2 classical brute-force benchmark (standalone — no quantum imports).

Exhaustively evaluates all C(10,6) = 210 combinations of 10 candidates and
finds the feasible combination that maximises expected net value (ENV) subject
to budget and capacity constraints.

WARNING: All inputs here are DUMMY values for a dry run.  Final run waits for
real Stage 1 probabilities and confirmed customer_id mapping.
"""

import os
import sys
import json
import itertools

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath("."))

# Resolve workspace root (parent of src/)
WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(WORKSPACE)

# ─── DUMMY DATA (must match quantum_stage2.py exactly) ────────────────────────
DUMMY_IDS = [
    "C0005", "C0007", "C0009", "C0012", "C0019",
    "C0020", "C0022", "C0025", "C0028", "C0031",
]
rng = np.random.RandomState(42)
DUMMY_PROBS = np.round(rng.uniform(0.05, 0.60, 10), 4)

s2 = pd.read_csv("data/Stage2_Data.csv", dtype=str, keep_default_na=False)
s2_subset = s2[s2["customer_id"].isin(DUMMY_IDS)].copy()
s2_subset = s2_subset.set_index("customer_id").loc[DUMMY_IDS].reset_index()

COSTS    = s2_subset["contact_cost"].astype(float).values
DEPOSITS = s2_subset["expected_deposit_value"].astype(float).values
CAPS     = s2_subset["capacity_units"].astype(float).values
N        = len(DUMMY_IDS)
K        = 6
BUDGET   = 25.0
CAPACITY = 22.0
ENV      = DUMMY_PROBS * DEPOSITS - COSTS


# ─── BRUTE-FORCE ──────────────────────────────────────────────────────────────
def brute_force_all(env, costs, caps, budget, capacity, k=6):
    """
    Enumerate all C(N,k) combinations.
    Returns:
        best_obj   – best feasible objective value
        best_sel   – list of indices of the best combination
        all_valid  – sorted list of (obj, combo) for all feasible combos
        invalid    – count of infeasible combos
    """
    best_obj  = -np.inf
    best_sel  = None
    all_valid = []
    invalid   = 0

    for combo in itertools.combinations(range(len(env)), k):
        idx = list(combo)
        cost_used = costs[idx].sum()
        cap_used  = caps[idx].sum()

        if cost_used <= budget and cap_used <= capacity:
            obj = env[idx].sum()
            all_valid.append((round(float(obj), 6), idx))
            if obj > best_obj:
                best_obj = obj
                best_sel = idx
        else:
            invalid += 1

    all_valid.sort(key=lambda x: x[0], reverse=True)
    return float(best_obj), best_sel, all_valid, invalid


def main():
    print("=" * 60)
    print("Classical brute-force benchmark (Stage 2 dry run)")
    print("=" * 60)
    print(f"\nCandidates  : {N}")
    print(f"Select      : {K}")
    print(f"Budget limit: {BUDGET}")
    print(f"Capacity lim: {CAPACITY}")
    print()

    # ── Print candidate table ────────────────────────────────────────────────
    print(f"{'ID':<8} {'p':>6} {'cost':>6} {'deposit':>8} {'cap':>5} {'ENV':>8}")
    print("-" * 50)
    for i, cid in enumerate(DUMMY_IDS):
        print(f"{cid:<8} {DUMMY_PROBS[i]:>6.4f} {COSTS[i]:>6.2f} "
              f"{DEPOSITS[i]:>8.2f} {CAPS[i]:>5.2f} {ENV[i]:>8.4f}")
    print()

    # ── Run brute force ──────────────────────────────────────────────────────
    best_obj, best_sel, all_valid, invalid = brute_force_all(
        ENV, COSTS, CAPS, BUDGET, CAPACITY, K)

    best_ids  = [DUMMY_IDS[i] for i in best_sel]
    cost_used = COSTS[best_sel].sum()
    cap_used  = CAPS[best_sel].sum()

    print(f"Total combinations     : 210")
    print(f"Feasible combinations  : {len(all_valid)}")
    print(f"Infeasible combinations: {invalid}")
    print()
    print(f"Best objective value : {best_obj:.6f}")
    print(f"Selected indices     : {best_sel}")
    print(f"Selected IDs         : {best_ids}")
    print(f"Budget used          : {cost_used:.2f} / {BUDGET}")
    print(f"Capacity used        : {cap_used:.2f} / {CAPACITY}")

    # ── Top-5 feasible combos ────────────────────────────────────────────────
    print("\nTop 5 feasible combinations:")
    for rank, (obj, idx) in enumerate(all_valid[:5], 1):
        ids = [DUMMY_IDS[i] for i in idx]
        print(f"  #{rank}: obj={obj:.4f}  ids={ids}")

    # ── Compare with QAOA if dryrun_stage2.json exists ───────────────────────
    stage2_path = "outputs/dryrun/dryrun_stage2.json"
    comparison = {}
    if os.path.exists(stage2_path):
        with open(stage2_path) as f:
            s2_data = json.load(f)

        print("\n--- Comparison with QAOA ---")
        for key in ["qaoa_reps1", "qaoa_reps2"]:
            if key in s2_data:
                q = s2_data[key]
                if "error" in q:
                    print(f"  {key}: FAILED — {q['error']}")
                    comparison[key] = {"status": "failed", "error": q["error"]}
                else:
                    q_obj   = q.get("objective_value", None)
                    q_valid = q.get("valid", False)
                    q_ids   = q.get("selected_ids", [])
                    q_time  = q.get("elapsed_s", None)
                    gap     = round(best_obj - q_obj, 6) if (q_valid and q_obj is not None) else None
                    matched = (sorted(q_ids) == sorted(best_ids))

                    print(f"\n  {key}:")
                    print(f"    Valid?             : {q_valid}")
                    print(f"    QAOA obj           : {q_obj}")
                    print(f"    Classical optimum  : {best_obj:.6f}")
                    print(f"    Gap (classic-QAOA) : {gap}")
                    print(f"    Matched optimal?   : {matched}")
                    print(f"    Time (s)           : {q_time}")

                    comparison[key] = {
                        "qaoa_valid":        q_valid,
                        "qaoa_objective":    q_obj,
                        "classical_optimum": round(best_obj, 6),
                        "gap":               gap,
                        "matched_optimal":   matched,
                        "elapsed_s":         q_time,
                    }
    else:
        print(f"\n(dryrun_stage2.json not found — QAOA results not yet available)")

    # ── Save output ───────────────────────────────────────────────────────────
    out = {
        "NOTE": "DUMMY DRY RUN — not real Stage 1 probabilities",
        "total_combinations": 210,
        "feasible_combinations": len(all_valid),
        "infeasible_combinations": invalid,
        "classical_optimum": {
            "objective_value": round(best_obj, 6),
            "selected_ids":    best_ids,
            "selected_indices": best_sel,
            "budget_used":     round(float(cost_used), 4),
            "capacity_used":   round(float(cap_used), 4),
        },
        "top5_feasible": [
            {"rank": r+1, "objective": o, "ids": [DUMMY_IDS[i] for i in idx]}
            for r, (o, idx) in enumerate(all_valid[:5])
        ],
        "comparison_with_qaoa": comparison,
        "candidates": [
            {
                "customer_id": DUMMY_IDS[i],
                "dummy_probability": float(DUMMY_PROBS[i]),
                "contact_cost": float(COSTS[i]),
                "expected_deposit_value": float(DEPOSITS[i]),
                "capacity_units": float(CAPS[i]),
                "expected_net_value": round(float(ENV[i]), 4),
            }
            for i in range(N)
        ],
    }

    os.makedirs("outputs/dryrun", exist_ok=True)
    out_path = "outputs/dryrun/dryrun_classical_benchmark.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
