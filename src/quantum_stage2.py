"""
src/quantum_stage2.py
=====================
Stage 2: Quantum Binary Optimization (DRY RUN on DUMMY inputs only).

Decision variables: x_i in {0,1} for i=0..9 (10 candidates)
Objective: maximize sum(ENV_i * x_i)
Constraints:
  (C1) sum(x_i) = 6
  (C2) sum(cost_i * x_i) <= 25 (Budget)
  (C3) sum(cap_i * x_i) <= 22  (Capacity)

QUBO formulation:
  - 10 decision variables x0..x9
  - 5 slack binary variables for Budget inequality (budget@int_slack@0..4)
  - 5 slack binary variables for Capacity inequality (capacity@int_slack@0..4)
  - Total QUBO variables / qubits: 20 qubits.

Solvers compared:
  1. Classical brute force (all C(10,6)=210 combinations)
  2. NumPyMinimumEigensolver (exact QUBO matrix diagonalization)
  3. QAOA reps=1 and reps=2 using Aer SamplerV2 with COBYLA (50 maxiter, seed 42)
"""

import os
import sys
import json
import time
import warnings
import itertools
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(WORKSPACE)
sys.path.insert(0, WORKSPACE)

def p(*args, **kwargs):
    print(*args, **kwargs, flush=True)

# ── Qiskit imports ────────────────────────────────────────────────────────────
from qiskit_aer.primitives import SamplerV2
from qiskit_algorithms import SamplingVQE, NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import COBYLA
from qiskit.circuit.library import QAOAAnsatz
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.converters import QuadraticProgramToQubo

# ═════════════════════════════════════════════════════════════════════════════
# DUMMY DATA (seed 42)
# ═════════════════════════════════════════════════════════════════════════════
DUMMY_IDS = [
    "C0005", "C0007", "C0009", "C0012", "C0019",
    "C0020", "C0022", "C0025", "C0028", "C0031",
]
rng = np.random.RandomState(42)
DUMMY_PROBS = np.round(rng.uniform(0.05, 0.60, 10), 4)

s2 = pd.read_csv("data/Stage2_Data.csv", dtype=str, keep_default_na=False)
s2_sub = s2[s2["customer_id"].isin(DUMMY_IDS)].copy()
s2_sub = s2_sub.set_index("customer_id").loc[DUMMY_IDS].reset_index()

COSTS    = s2_sub["contact_cost"].astype(float).values
DEPOSITS = s2_sub["expected_deposit_value"].astype(float).values
CAPS     = s2_sub["capacity_units"].astype(float).values
N, K, BUDGET, CAPACITY = 10, 6, 25.0, 22.0
ENV = DUMMY_PROBS * DEPOSITS - COSTS

def classical_brute_force():
    best_val, best_sel = -np.inf, None
    valid = []
    for combo in itertools.combinations(range(N), K):
        idx = list(combo)
        if COSTS[idx].sum() <= BUDGET and CAPS[idx].sum() <= CAPACITY:
            obj = ENV[idx].sum()
            valid.append((round(float(obj), 6), idx))
            if obj > best_val:
                best_val, best_sel = obj, idx
    valid.sort(key=lambda x: x[0], reverse=True)
    return float(best_val), best_sel, valid

def build_qp():
    qp = QuadraticProgram("stage2_selection")
    for i in range(N):
        qp.binary_var(f"x{i}")
    qp.maximize(linear={f"x{i}": float(ENV[i]) for i in range(N)})
    qp.linear_constraint({f"x{i}": 1.0 for i in range(N)}, "==", K, "select_k")
    qp.linear_constraint({f"x{i}": float(COSTS[i]) for i in range(N)}, "<=", BUDGET, "budget")
    qp.linear_constraint({f"x{i}": float(CAPS[i]) for i in range(N)}, "<=", CAPACITY, "capacity")
    return qp

def is_valid(x_bits):
    b = np.array(x_bits[:N])
    return (int(b.sum()) == K and
            float(COSTS @ b) <= BUDGET and
            float(CAPS @ b) <= CAPACITY)

def solve_qaoa_aer(op, qp, reps=1, maxiter=50, seed=42):
    t0 = time.time()
    # Decompose ansatz to primitive single & two qubit gates for Aer
    ansatz = QAOAAnsatz(cost_operator=op, reps=reps).decompose(reps=3)
    sampler = SamplerV2(seed=seed)
    optimizer = COBYLA(maxiter=maxiter, rhobeg=0.5)
    vqe = SamplingVQE(sampler=sampler, ansatz=ansatz, optimizer=optimizer)
    
    # Compute ground state of Ising Hamiltonian
    vqe_result = vqe.compute_minimum_eigenvalue(op)
    elapsed = time.time() - t0
    
    # Extract bitstring from best measurement (Qiskit right-to-left endianness)
    best_bs = vqe_result.best_measurement["bitstring"]
    # The first 10 variables are x0..x9 (rightmost 10 bits in reverse order)
    x_sol = [int(best_bs[-(i + 1)]) for i in range(N)]
    valid = is_valid(x_sol)
    
    obj_val = float(ENV @ np.array(x_sol))
    sel_indices = [i for i, v in enumerate(x_sol) if v == 1]
    sel_ids = [DUMMY_IDS[i] for i in sel_indices]
    
    return {
        "reps": reps,
        "elapsed_s": round(elapsed, 2),
        "bitstring": best_bs,
        "x_solution": x_sol,
        "valid": valid,
        "selected_count": int(sum(x_sol)),
        "objective_value": round(obj_val, 6),
        "selected_ids": sel_ids,
        "selected_indices": sel_indices,
        "budget_used": round(float(COSTS[sel_indices].sum()), 4) if sel_indices else 0.0,
        "capacity_used": round(float(CAPS[sel_indices].sum()), 4) if sel_indices else 0.0,
        "optimizer": "COBYLA",
        "max_iter": maxiter,
        "seed": seed,
        "vqe_eigenvalue": round(float(np.real(vqe_result.eigenvalue)), 4),
    }

def main():
    os.makedirs("outputs/dryrun", exist_ok=True)
    out = {"NOTE": "DUMMY DRY RUN — not real Stage 1 probabilities"}

    p("=" * 60)
    p("Stage 2 Dry Run: Quantum Binary Optimization (QAOA on Aer)")
    p("=" * 60)

    # 1. Classical Brute Force
    bf_val, bf_sel, bf_valid = classical_brute_force()
    bf_ids = [DUMMY_IDS[i] for i in bf_sel]
    p(f"1. Classical Brute Force:")
    p(f"   Best Objective: {bf_val:.4f}")
    p(f"   Selected IDs  : {bf_ids}")
    p(f"   Budget Used   : {COSTS[bf_sel].sum():.2f}/{BUDGET}")
    p(f"   Capacity Used : {CAPS[bf_sel].sum():.2f}/{CAPACITY}")
    p(f"   Valid Combos  : {len(bf_valid)}/210")

    out["classical"] = {
        "objective_value": round(bf_val, 6),
        "selected_ids": bf_ids,
        "selected_indices": bf_sel,
        "budget_used": round(float(COSTS[bf_sel].sum()), 4),
        "capacity_used": round(float(CAPS[bf_sel].sum()), 4),
        "valid_combos": len(bf_valid),
    }

    # 2. Build QP and convert to QUBO
    qp = build_qp()
    conv = QuadraticProgramToQubo()
    qubo = conv.convert(qp)
    op, offset = qubo.to_ising()
    n_vars = qubo.get_num_vars()
    
    p(f"\n2. Problem Encoding:")
    p(f"   Decision variables : 10 (x0..x9)")
    p(f"   Slack variables    : {n_vars - 10} (5 budget slack bits + 5 capacity slack bits)")
    p(f"   Total QUBO Qubits  : {n_vars} qubits")
    p(f"   Ising Hamiltonian  : {len(op)} Pauli terms")
    
    out["qubo_num_variables"] = n_vars
    out["slack_variables"] = n_vars - 10
    out["pauli_terms"] = len(op)

    # 3. NumPy Minimum Eigensolver (Exact QUBO optimum)
    p("\n3. NumPyMinimumEigensolver (Exact QUBO Verification):")
    t0 = time.time()
    np_solver = MinimumEigenOptimizer(NumPyMinimumEigensolver())
    np_res = np_solver.solve(qp)
    np_elapsed = time.time() - t0
    np_x = [int(v) for v in np_res.x[:N]]
    np_valid = is_valid(np_x)
    np_ids = [DUMMY_IDS[i] for i, v in enumerate(np_x) if v == 1]
    p(f"   Exact QUBO Obj : {np_res.fval:.4f} (Elapsed: {np_elapsed:.2f}s)")
    p(f"   Selected IDs   : {np_ids}")
    p(f"   Matches Classic: {sorted(np_ids) == sorted(bf_ids)}")
    
    out["numpy_exact"] = {
        "elapsed_s": round(np_elapsed, 2),
        "valid": np_valid,
        "objective_value": round(float(np_res.fval), 6),
        "selected_ids": np_ids,
        "matches_brute_force": sorted(np_ids) == sorted(bf_ids),
    }

    # 4. QAOA Solvers (reps=1 and reps=2)
    for reps in [1, 2]:
        p(f"\n4. QAOA (reps={reps}, COBYLA maxiter=50, AerSimulator SamplerV2):")
        try:
            q_res = solve_qaoa_aer(op, qp, reps=reps, maxiter=50, seed=42)
            gap = round(bf_val - q_res["objective_value"], 6) if q_res["valid"] else None
            q_res["gap_vs_classical"] = gap
            q_res["classical_optimum"] = round(bf_val, 6)
            
            p(f"   Elapsed        : {q_res['elapsed_s']}s")
            p(f"   Valid Solution : {q_res['valid']} (Selected: {q_res['selected_count']})")
            p(f"   Objective Value: {q_res['objective_value']:.4f}")
            p(f"   Selected IDs   : {q_res['selected_ids']}")
            p(f"   Gap vs Classic : {gap}")
            out[f"qaoa_reps{reps}"] = q_res
        except Exception as e:
            p(f"   QAOA reps={reps} failed: {e}")
            out[f"qaoa_reps{reps}"] = {"error": str(e)}

    # 5. Candidate details
    out["candidates"] = [
        {
            "customer_id": cid,
            "dummy_probability": float(DUMMY_PROBS[i]),
            "contact_cost": float(COSTS[i]),
            "expected_deposit_value": float(DEPOSITS[i]),
            "capacity_units": float(CAPS[i]),
            "expected_net_value": round(float(ENV[i]), 4),
        }
        for i, cid in enumerate(DUMMY_IDS)
    ]

    out_file = "outputs/dryrun/dryrun_stage2.json"
    with open(out_file, "w") as f:
        json.dump(out, f, indent=2)
    p(f"\nSaved Stage 2 dry run output to {out_file}")

if __name__ == "__main__":
    main()
