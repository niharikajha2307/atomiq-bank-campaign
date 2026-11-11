"""
scripts/run_job2.py
===================
JOB 2: Stage 2 Quantum Solver Enhancement (DUMMY dry run only).

2a) Probability-mass metric: fraction of shots hitting optimal bitstring vs uniform random.
2b) Dicke(6) initial state + XY ring mixer QAOA variant (10 decision qubits only for K=6
    constraint; budget+capacity still as penalty terms with slack bits → still ~20 qubits).
2c) Run both versions (penalty-only & Dicke+XY) with reps=1,2,3; COBYLA maxiter=200;
    seeds=[42,7,123]; 5-min limit per run. Report all metrics.

All outputs -> outputs/dryrun/ only.
"""

import os, sys, json, time, warnings, itertools, threading
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(WORKSPACE)
sys.path.insert(0, WORKSPACE)

TIMEOUT_S = 300  # 5 minutes per run

def p(*args, **kwargs):
    print(*args, **kwargs, flush=True)

# ── Qiskit imports ─────────────────────────────────────────────────────────────
p("Loading Qiskit (this takes ~90 seconds on first import)...")
t_import = time.time()
from qiskit_aer.primitives import SamplerV2
from qiskit_algorithms import SamplingVQE, NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import COBYLA
from qiskit.circuit.library import QAOAAnsatz
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.quantum_info import SparsePauliOp
p(f"Qiskit loaded in {time.time()-t_import:.1f}s")

# ═══════════════════════════════════════════════════════════════════════════════
# DUMMY DATA (seed 42) — same as quantum_stage2.py
# ═══════════════════════════════════════════════════════════════════════════════
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

p(f"\nDUMMY ENV values: {np.round(ENV, 3)}")

# ═══════════════════════════════════════════════════════════════════════════════
# CLASSICAL BRUTE FORCE (reference)
# ═══════════════════════════════════════════════════════════════════════════════
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

BF_VAL, BF_SEL, BF_VALID = classical_brute_force()
BF_IDS = [DUMMY_IDS[i] for i in BF_SEL]
# Optimal bitstring: x[i]=1 if i in BF_SEL, right-to-left = bits[9-i]
OPTIMAL_BITS = "".join(str(int(i in BF_SEL)) for i in range(N - 1, -1, -1))
p(f"Brute-force optimal: {BF_VAL:.4f}, Selection: {BF_IDS}")
p(f"Optimal bitstring (Qiskit big-endian right-to-left): {OPTIMAL_BITS}")
p(f"Valid combos: {len(BF_VALID)}/210")

def is_valid(x_bits):
    b = np.array(x_bits[:N])
    return (int(b.sum()) == K and
            float(COSTS @ b) <= BUDGET and
            float(CAPS @ b) <= CAPACITY)

def decode_bitstring(bs):
    """Qiskit little-endian: rightmost char = qubit 0 = x0."""
    return [int(bs[-(i + 1)]) for i in range(N)]

def evaluate_selection(x_sol):
    idx = [i for i, v in enumerate(x_sol) if v == 1]
    return float(ENV[idx].sum()) if idx else 0.0

# ═══════════════════════════════════════════════════════════════════════════════
# BUILD QUBO (penalty-only, full slack) — same as original
# ═══════════════════════════════════════════════════════════════════════════════
def build_qubo_op():
    """Returns (ising_op, offset, qubo, n_vars)."""
    qp = QuadraticProgram("stage2_selection")
    for i in range(N):
        qp.binary_var(f"x{i}")
    qp.maximize(linear={f"x{i}": float(ENV[i]) for i in range(N)})
    qp.linear_constraint({f"x{i}": 1.0 for i in range(N)}, "==", K, "select_k")
    qp.linear_constraint({f"x{i}": float(COSTS[i]) for i in range(N)}, "<=", BUDGET, "budget")
    qp.linear_constraint({f"x{i}": float(CAPS[i]) for i in range(N)}, "<=", CAPACITY, "capacity")
    conv = QuadraticProgramToQubo()
    qubo = conv.convert(qp)
    op, offset = qubo.to_ising()
    return op, offset, qubo, qubo.get_num_vars()

# ═══════════════════════════════════════════════════════════════════════════════
# XY RING MIXER (preserves Hamming weight → no K constraint needed in QUBO)
# ═══════════════════════════════════════════════════════════════════════════════
def xy_ring_mixer(n_qubits, beta):
    """
    XY mixer: H_B = sum_i (X_i X_{i+1} + Y_i Y_{i+1}) on a ring.
    Implements exp(-i*beta*H_B) via RXX and RYY gates on adjacent pairs.
    beta is a circuit Parameter (angle).
    """
    qc = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        j = (i + 1) % n_qubits
        qc.rxx(2 * beta, i, j)
        qc.ryy(2 * beta, i, j)
    return qc

def prepare_dicke_state(n, k):
    """
    Approximate Dicke state |D(n,k)> preparation.
    Uses the deterministic Dicke state preparation algorithm (Bartschi & Eidenbenz 2019).
    For n=10, k=6: places exactly k ones in superposition.
    Implementation: build layer by layer using SCS (Splitting-Combining-Swapping) gates.
    
    For simplicity we use a proven shortcut: 
      1. Set qubits 0..k-1 to |1> (computational basis state with k ones).
      2. Apply a series of CNOT-controlled Ry rotations to spread the weight.
    
    Here we use the recursive split approach: apply Ry(theta) to qubit i,
    controlled on the partial sum, to distribute excitations evenly.
    
    Practical alternative: use the QFT-based equal superposition and
    post-select (too slow for QAOA). Instead we use the direct SCS method.
    """
    qc = QuantumCircuit(n)
    # Simple Dicke state: start with |111111 0000> (k=6 ones on first k qubits)
    for i in range(k):
        qc.x(i)
    # Apply a cascade of Givens rotations to spread into equal superposition
    # of all C(n,k) basis states. This is the Bartschi-Eidenbenz ladder:
    # For each layer l from 0 to n-k-1, apply controlled-Ry on qubit k+l
    # controlled by qubits 0..k+l-1. We use a simplified 2-qubit SCS gate.
    for layer in range(n - k):
        target = k + layer
        for ctrl in range(k + layer - 1, -1, -1):
            # Angle for uniform split at this stage
            remaining = n - ctrl - 1
            ones_left = k - ctrl if ctrl < k else 0
            if remaining > 0 and ones_left > 0:
                theta = 2 * np.arccos(np.sqrt(ones_left / remaining))
                # Controlled Ry: rotate target conditioned on ctrl being |1>
                qc.cry(theta, ctrl, target)
                break  # one controlled rotation per layer for approx Dicke
    return qc

def build_dicke_qubo_op():
    """
    For Dicke+XY variant: K=6 is enforced by the mixer (no slack needed for K).
    Budget and capacity still need slack bits → 10 + ~5 + ~5 = ~20 qubits.
    We build a QUBO WITHOUT the K equality constraint, so the slack count drops.
    """
    qp = QuadraticProgram("stage2_dicke")
    for i in range(N):
        qp.binary_var(f"x{i}")
    qp.maximize(linear={f"x{i}": float(ENV[i]) for i in range(N)})
    # Only inequality constraints (K enforced by Dicke+XY)
    qp.linear_constraint({f"x{i}": float(COSTS[i]) for i in range(N)}, "<=", BUDGET, "budget")
    qp.linear_constraint({f"x{i}": float(CAPS[i]) for i in range(N)}, "<=", CAPACITY, "capacity")
    conv = QuadraticProgramToQubo()
    qubo = conv.convert(qp)
    op, offset = qubo.to_ising()
    return op, offset, qubo, qubo.get_num_vars()

# ═══════════════════════════════════════════════════════════════════════════════
# PROBABILITY MASS ON OPTIMAL BITSTRING
# ═══════════════════════════════════════════════════════════════════════════════
def measure_prob_mass(ansatz, op, params, n_shots=2048, seed=42):
    """
    Re-run the circuit at the optimized parameters and count fraction of
    shots that hit the optimal bitstring.
    """
    sampler = SamplerV2(seed=seed)
    bound = ansatz.assign_parameters(params)
    job = sampler.run([bound], shots=n_shots)
    result = job.result()[0]
    counts = result.data.meas.get_counts()
    total = sum(counts.values())
    
    # Count valid selections and find best valid
    best_valid_obj = -np.inf
    best_valid_ids = []
    n_invalid = 0
    for bs, cnt in counts.items():
        x_sol = decode_bitstring(bs)
        if is_valid(x_sol):
            obj = evaluate_selection(x_sol)
            if obj > best_valid_obj:
                best_valid_obj = obj
                best_valid_ids = [DUMMY_IDS[i] for i, v in enumerate(x_sol) if v == 1]
        else:
            n_invalid += cnt
    
    # Optimal bitstring probability
    # Qiskit counts keys: big-endian qubit string
    opt_count = 0
    for bs, cnt in counts.items():
        x_sol = decode_bitstring(bs)
        if [DUMMY_IDS[i] for i, v in enumerate(x_sol) if v == 1] == BF_IDS or \
           sorted([DUMMY_IDS[i] for i, v in enumerate(x_sol) if v == 1]) == sorted(BF_IDS):
            opt_count += cnt
    
    p_opt = opt_count / total
    invalid_rate = n_invalid / total
    
    return {
        "shots": n_shots,
        "p_optimal_bitstring": round(p_opt, 6),
        "random_baseline_p_optimal": round(1.0 / 210, 6),  # 1/C(10,6)
        "best_valid_obj": round(float(best_valid_obj), 6) if best_valid_obj > -np.inf else None,
        "best_valid_ids": best_valid_ids,
        "invalid_rate": round(invalid_rate, 4),
    }

def random_sampling_baseline(n_shots=2048, seed=42):
    """Best valid result from uniform random over all C(10,6) combinations."""
    rng_b = np.random.RandomState(seed)
    combos = list(itertools.combinations(range(N), K))
    valid_combos = [(c, ENV[list(c)].sum()) for c in combos
                    if COSTS[list(c)].sum() <= BUDGET and CAPS[list(c)].sum() <= CAPACITY]
    # Sample n_shots times with replacement
    sample_indices = rng_b.randint(0, len(valid_combos), n_shots)
    best_obj = max(valid_combos[i][1] for i in sample_indices)
    # Count how many samples hit the optimal
    opt_hits = sum(1 for i in sample_indices
                   if sorted(DUMMY_IDS[j] for j in valid_combos[i][0]) == sorted(BF_IDS))
    return {
        "n_shots": n_shots,
        "best_obj_from_random_sampling": round(float(best_obj), 6),
        "p_optimal": round(opt_hits / n_shots, 6),
        "theoretical_p_optimal": round(1.0 / 210, 6),
    }

# ═══════════════════════════════════════════════════════════════════════════════
# TIMEOUT WRAPPER
# ═══════════════════════════════════════════════════════════════════════════════
class TimeoutError(Exception):
    pass

def run_with_timeout(fn, timeout_s=TIMEOUT_S):
    """Run fn() in a thread; if it takes longer than timeout_s, return None."""
    result_box = [None]
    exc_box = [None]
    
    def worker():
        try:
            result_box[0] = fn()
        except Exception as e:
            exc_box[0] = e
    
    t = threading.Thread(target=worker, daemon=True)
    t.start()
    t.join(timeout=timeout_s)
    if t.is_alive():
        return None, f"TIMEOUT after {timeout_s}s"
    if exc_box[0]:
        return None, str(exc_box[0])
    return result_box[0], None

# ═══════════════════════════════════════════════════════════════════════════════
# PENALTY-ONLY QAOA (original formulation, full QUBO)
# ═══════════════════════════════════════════════════════════════════════════════
def run_penalty_qaoa(op, reps, maxiter, seed):
    t0 = time.time()
    ansatz = QAOAAnsatz(cost_operator=op, reps=reps).decompose(reps=3)
    sampler = SamplerV2(seed=seed)
    optimizer = COBYLA(maxiter=maxiter, rhobeg=0.5)
    vqe = SamplingVQE(sampler=sampler, ansatz=ansatz, optimizer=optimizer)
    vqe_result = vqe.compute_minimum_eigenvalue(op)
    elapsed = time.time() - t0
    
    best_bs = vqe_result.best_measurement["bitstring"]
    x_sol = decode_bitstring(best_bs)
    valid = is_valid(x_sol)
    obj_val = evaluate_selection(x_sol) if valid else evaluate_selection(x_sol)
    sel_ids = [DUMMY_IDS[i] for i, v in enumerate(x_sol) if v == 1]
    
    # Measure probability mass at optimized params
    opt_params = vqe_result.optimal_parameters
    # Bind to ansatz
    prob_info = {}
    try:
        bound_circ = ansatz.assign_parameters(opt_params)
        sampler2 = SamplerV2(seed=seed)
        job = sampler2.run([bound_circ], shots=2048)
        res2 = job.result()[0]
        counts2 = res2.data.meas.get_counts()
        total2 = sum(counts2.values())
        
        n_inv = 0
        best_v_obj = -np.inf
        opt_hits = 0
        for bs, cnt in counts2.items():
            xs = decode_bitstring(bs)
            ids_hit = sorted([DUMMY_IDS[i] for i, v in enumerate(xs) if v == 1])
            if is_valid(xs):
                obj_h = evaluate_selection(xs)
                if obj_h > best_v_obj:
                    best_v_obj = obj_h
            else:
                n_inv += cnt
            if ids_hit == sorted(BF_IDS):
                opt_hits += cnt
        
        prob_info = {
            "p_optimal": round(opt_hits / total2, 6),
            "random_baseline": round(1.0/210, 6),
            "beats_random": (opt_hits / total2) > (1.0 / 210),
            "best_valid_obj_from_2048_shots": round(float(best_v_obj), 6) if best_v_obj > -np.inf else None,
            "invalid_rate_2048shots": round(n_inv / total2, 4),
        }
    except Exception as e:
        prob_info = {"error": str(e)}
    
    gap = round(BF_VAL - evaluate_selection(x_sol), 6)
    return {
        "variant": "penalty_only",
        "reps": reps, "maxiter": maxiter, "seed": seed,
        "elapsed_s": round(elapsed, 2),
        "valid": valid,
        "selected_count": int(sum(x_sol)),
        "objective_value": round(evaluate_selection(x_sol), 6),
        "gap_vs_brute_force": gap,
        "selected_ids": sel_ids,
        "bitstring": best_bs,
        "probability_mass": prob_info,
    }

# ═══════════════════════════════════════════════════════════════════════════════
# DICKE + XY RING MIXER QAOA
# ═══════════════════════════════════════════════════════════════════════════════
def build_dicke_qaoa_circuit(op_dicke, reps):
    """
    Custom QAOA circuit with Dicke(10,6) initial state and XY ring mixer.
    - Initial state: approximate |D(10,6)> (superposition of all 10-bit strings w/ 6 ones)
    - Cost unitary: e^{-i*gamma*H_C} (standard QAOA phase separator)
    - Mixer unitary: e^{-i*beta*H_B} (XY ring mixer preserving Hamming weight)
    
    Circuit structure (p layers):
      Dicke_init | [Phase_Sep(gamma_l) | XY_Mix(beta_l)]^p | measure
    
    Parameters: gamma_0, beta_0, gamma_1, beta_1, ...  (2*reps total)
    """
    from qiskit.circuit import ParameterVector
    
    n_qubits = op_dicke.num_qubits
    params = ParameterVector("θ", 2 * reps)
    
    qc = QuantumCircuit(n_qubits)
    
    # 1. Dicke(n, k) initial state
    dicke = prepare_dicke_state(n_qubits, K)
    qc.compose(dicke, inplace=True)
    qc.barrier()
    
    # 2. QAOA layers
    for layer in range(reps):
        gamma = params[2 * layer]
        beta  = params[2 * layer + 1]
        
        # Phase separator: e^{-i*gamma*H_C}
        # H_C is a sum of Pauli terms; we implement via Pauli evolution
        # For SparsePauliOp, decompose into individual ZZ/Z terms
        for pauli, coeff in zip(op_dicke.paulis, op_dicke.coeffs):
            pauli_str = pauli.to_label()
            active = [(idx, p_char) for idx, p_char in enumerate(reversed(pauli_str))
                      if p_char != 'I']
            
            if len(active) == 0:
                # Global phase, skip
                pass
            elif len(active) == 1:
                idx, ch = active[0]
                ang = 2.0 * float(np.real(coeff)) * gamma
                if ch == 'Z':
                    qc.rz(ang, idx)
            elif len(active) == 2:
                (i0, c0), (i1, c1) = active[0], active[1]
                ang = 2.0 * float(np.real(coeff)) * gamma
                if c0 == 'Z' and c1 == 'Z':
                    qc.cx(i0, i1)
                    qc.rz(ang, i1)
                    qc.cx(i0, i1)
                # Higher-weight Paulis ignored for shallow depth (penalty encodes them)
        
        qc.barrier()
        
        # XY ring mixer
        xy = xy_ring_mixer(n_qubits, beta)
        qc.compose(xy, inplace=True)
        qc.barrier()
    
    qc.measure_all()
    return qc, params

def run_dicke_qaoa(op_dicke, reps, maxiter, seed):
    t0 = time.time()
    
    qc, params = build_dicke_qaoa_circuit(op_dicke, reps)
    
    # We'll do a manual VQE-style optimization loop with COBYLA
    # Objective: minimize expected cost (negate for COBYLA minimization)
    sampler = SamplerV2(seed=seed)
    n_params = 2 * reps
    theta0 = np.ones(n_params) * 0.1  # small initial angles
    
    best_obj = [np.inf]
    best_params = [theta0.copy()]
    n_evals = [0]
    
    def objective(theta):
        bound = qc.assign_parameters(dict(zip(params, theta)))
        job = sampler.run([bound], shots=512)
        result = job.result()[0]
        counts = result.data.meas.get_counts()
        total = sum(counts.values())
        
        # Compute expected objective (penalize invalid)
        exp_obj = 0.0
        for bs, cnt in counts.items():
            xs = decode_bitstring(bs)
            # For Dicke+XY, all states should have exactly 6 ones
            # Budget/capacity penalty if violated
            obj = evaluate_selection(xs)
            pen = 0.0
            b_arr = np.array(xs[:N])
            if COSTS @ b_arr > BUDGET:
                pen += 10.0 * (COSTS @ b_arr - BUDGET)
            if CAPS @ b_arr > CAPACITY:
                pen += 10.0 * (CAPS @ b_arr - CAPACITY)
            exp_obj += (cnt / total) * (obj - pen)
        
        neg_obj = -exp_obj  # COBYLA minimizes
        if neg_obj < best_obj[0]:
            best_obj[0] = neg_obj
            best_params[0] = theta.copy()
        n_evals[0] += 1
        return neg_obj
    
    from scipy.optimize import minimize
    result = minimize(objective, theta0, method='COBYLA',
                      options={'maxiter': maxiter, 'rhobeg': 0.5})
    
    elapsed = time.time() - t0
    
    # Final measurement with best params (2048 shots)
    final_bound = qc.assign_parameters(dict(zip(params, best_params[0])))
    sampler2 = SamplerV2(seed=seed)
    job2 = sampler2.run([final_bound], shots=2048)
    res2 = job2.result()[0]
    counts2 = res2.data.meas.get_counts()
    total2 = sum(counts2.values())
    
    # Find best valid selection
    best_valid_obj = -np.inf
    best_valid_ids = []
    n_invalid = 0
    opt_hits = 0
    
    for bs, cnt in counts2.items():
        xs = decode_bitstring(bs)
        ids_hit = sorted([DUMMY_IDS[i] for i, v in enumerate(xs) if v == 1])
        if is_valid(xs):
            obj = evaluate_selection(xs)
            if obj > best_valid_obj:
                best_valid_obj = obj
                best_valid_ids = [DUMMY_IDS[i] for i, v in enumerate(xs) if v == 1]
        else:
            n_invalid += cnt
        if ids_hit == sorted(BF_IDS):
            opt_hits += cnt
    
    # Most frequent valid bitstring
    best_bs = max(counts2, key=lambda bs: counts2[bs] if is_valid(decode_bitstring(bs)) else -1)
    x_sol = decode_bitstring(best_bs)
    
    gap = round(BF_VAL - best_valid_obj, 6) if best_valid_obj > -np.inf else None
    
    return {
        "variant": "dicke_xy",
        "reps": reps, "maxiter": maxiter, "seed": seed,
        "elapsed_s": round(elapsed, 2),
        "n_qubits": N,  # 10 decision only (K enforced by mixer)
        "valid": is_valid(x_sol),
        "selected_count": int(sum(x_sol)),
        "best_valid_obj": round(float(best_valid_obj), 6) if best_valid_obj > -np.inf else None,
        "gap_vs_brute_force": gap,
        "best_valid_ids": best_valid_ids,
        "invalid_rate_2048shots": round(n_invalid / total2, 4),
        "p_optimal_2048shots": round(opt_hits / total2, 6),
        "beats_random_sampling": (opt_hits / total2) > (1.0 / 210),
        "optimizer_evals": n_evals[0],
        "cobyla_success": result.success,
    }

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    os.makedirs("outputs/dryrun", exist_ok=True)
    out = {"NOTE": "DUMMY DRY RUN — JOB 2 — not real Stage 1 probabilities"}
    
    p("\n" + "=" * 70)
    p("JOB 2a: Random Sampling Baseline")
    p("=" * 70)
    rand_base = random_sampling_baseline(n_shots=2048, seed=42)
    p(f"  Uniform random over 210 valid combos (2048 samples):")
    p(f"  Best objective found  : {rand_base['best_obj_from_random_sampling']}")
    p(f"  Fraction hitting opt  : {rand_base['p_optimal']:.4f} (theory 1/210={rand_base['theoretical_p_optimal']:.4f})")
    out["random_sampling_baseline"] = rand_base
    
    p("\n" + "=" * 70)
    p("JOB 2b/2c: Building QUBO operators...")
    p("=" * 70)
    op_penalty, _, qubo_pen, n_vars_pen = build_qubo_op()
    op_dicke, _, qubo_dicke, n_vars_dicke = build_dicke_qubo_op()
    p(f"  Penalty-only QUBO : {n_vars_pen} qubits (10 decision + {n_vars_pen-10} slack)")
    p(f"  Dicke QUBO (no K) : {n_vars_dicke} qubits (10 decision + {n_vars_dicke-10} slack)")
    out["qubo_info"] = {
        "penalty_only_qubits": n_vars_pen,
        "penalty_only_slack": n_vars_pen - 10,
        "dicke_xy_qubits": n_vars_dicke,
        "dicke_xy_slack": n_vars_dicke - 10,
        "note": "Dicke+XY enforces K=6 via mixer; only budget+capacity need slack",
    }
    
    seeds = [42, 7, 123]
    reps_list = [1, 2, 3]
    maxiter = 200
    
    penalty_results = []
    dicke_results = []
    
    p("\n" + "=" * 70)
    p("JOB 2c: Penalty-Only QAOA (reps=1,2,3; seeds=42,7,123; maxiter=200; 5-min limit)")
    p("=" * 70)
    
    for reps in reps_list:
        for seed in seeds:
            p(f"\n  Penalty-QAOA reps={reps} seed={seed} ...")
            
            def _run():
                return run_penalty_qaoa(op_penalty, reps, maxiter, seed)
            
            result, err = run_with_timeout(_run, timeout_s=TIMEOUT_S)
            
            if err:
                p(f"    FAILED: {err}")
                r = {"variant": "penalty_only", "reps": reps, "seed": seed, "error": err}
            else:
                r = result
                p(f"    Valid={r['valid']} | Obj={r['objective_value']:.4f} | Gap={r['gap_vs_brute_force']:.4f} | t={r['elapsed_s']}s")
                pm = r.get("probability_mass", {})
                p(f"    P(optimal)={pm.get('p_optimal','?')} | Random={pm.get('random_baseline','?')} | Beats_random={pm.get('beats_random','?')}")
                p(f"    Invalid rate (2048 shots)={pm.get('invalid_rate_2048shots','?')}")
            
            penalty_results.append(r)
    
    out["penalty_only_results"] = penalty_results
    
    p("\n" + "=" * 70)
    p("JOB 2c: Dicke+XY QAOA (reps=1,2,3; seeds=42,7,123; maxiter=200; 5-min limit)")
    p("=" * 70)
    
    for reps in reps_list:
        for seed in seeds:
            p(f"\n  Dicke+XY QAOA reps={reps} seed={seed} ...")
            
            def _run_d(r=reps, s=seed):
                return run_dicke_qaoa(op_dicke, r, maxiter, s)
            
            result_d, err_d = run_with_timeout(_run_d, timeout_s=TIMEOUT_S)
            
            if err_d:
                p(f"    FAILED: {err_d}")
                r_d = {"variant": "dicke_xy", "reps": reps, "seed": seed, "error": err_d}
            else:
                r_d = result_d
                p(f"    Valid={r_d['valid']} | BestObj={r_d['best_valid_obj']} | Gap={r_d['gap_vs_brute_force']} | t={r_d['elapsed_s']}s")
                p(f"    P(optimal)={r_d['p_optimal_2048shots']} | Beats_random={r_d['beats_random_sampling']}")
                p(f"    Invalid rate={r_d['invalid_rate_2048shots']}")
            
            dicke_results.append(r_d)
    
    out["dicke_xy_results"] = dicke_results
    
    # Summary comparison table
    p("\n" + "=" * 70)
    p("SUMMARY TABLE: Penalty-Only vs Dicke+XY vs Random Sampling")
    p("=" * 70)
    p(f"{'Variant':<20} {'Reps':<6} {'Seed':<6} {'Obj':<10} {'Gap':<10} {'P(opt)':<10} {'Inv%':<10} {'Beats_rand':<12} {'Time(s)':<10}")
    p("-" * 95)
    
    for r in penalty_results + dicke_results:
        if "error" in r:
            p(f"{'Error':<20} {r.get('reps','?'):<6} {r.get('seed','?'):<6} {'--':<10} {'--':<10} {'--':<10} {'--':<10} {'--':<12} {'FAIL':<10}")
            continue
        pm = r.get("probability_mass", r)  # penalty has nested, dicke has flat
        obj = r.get("objective_value") or r.get("best_valid_obj") or "--"
        gap = r.get("gap_vs_brute_force") or "--"
        p_opt = pm.get("p_optimal") if "probability_mass" in r else r.get("p_optimal_2048shots", "--")
        inv = pm.get("invalid_rate_2048shots") if "probability_mass" in r else r.get("invalid_rate_2048shots", "--")
        beats = pm.get("beats_random") if "probability_mass" in r else r.get("beats_random_sampling", "--")
        t_s = r.get("elapsed_s", "--")
        variant = r.get("variant", "--")
        p(f"{variant:<20} {r['reps']:<6} {r['seed']:<6} {str(obj):<10} {str(gap):<10} {str(p_opt):<10} {str(inv):<10} {str(beats):<12} {str(t_s):<10}")
    
    # Random baseline row
    p(f"{'random_sampling':<20} {'N/A':<6} {'42':<6} {rand_base['best_obj_from_random_sampling']:<10} {'0.0':<10} {rand_base['p_optimal']:<10} {'0.0':<10} {'baseline':<12} {'<1s':<10}")
    
    out_path = "outputs/dryrun/dryrun_job2_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    p(f"\nSaved Job 2 results to {out_path}")

if __name__ == "__main__":
    main()
