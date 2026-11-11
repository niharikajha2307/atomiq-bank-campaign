"""
src/quantum_stage3.py
=====================
Stage 3: 6-qubit quantum success probability estimation (DRY RUN).

Each qubit represents one of the 6 customers selected by Stage 2.
Qubit i is prepared with  Ry(2 * arcsin(sqrt(p_i)))  so that
  P(measure qubit i = 1) = p_i   (probability of subscription).

Success = at least 3 of the 6 customers subscribe (≥ 3 qubits read 1).

Qiskit little-endian convention
--------------------------------
When a 6-qubit circuit is measured, the result bitstring is ordered as
  bitstring[0] = qubit 5 (most-significant, leftmost character)
  bitstring[5] = qubit 0 (least-significant, rightmost character)
So  qubit 0  ↔  rightmost bit of the bitstring.
Customer-to-qubit assignment is stated explicitly in the output.

WARNING: All 6 customer IDs and probabilities here come from the
DUMMY dryrun_stage2.json.  Final run waits for real Stage 2 output.
"""

import os
import sys
import json
import math
import time
import warnings

import numpy as np
from itertools import combinations

warnings.filterwarnings("ignore")

# Resolve workspace root (parent of src/)
WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(WORKSPACE)
sys.path.insert(0, WORKSPACE)

# Override print to always flush
import builtins
_original_print = builtins.print
def print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    _original_print(*args, **kwargs)

# ─── Qiskit ───────────────────────────────────────────────────────────────────
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel, depolarizing_error, ReadoutError
)

# ═══════════════════════════════════════════════════════════════════════════════
# LOAD SELECTED CUSTOMERS FROM dryrun_stage2.json
# If that file is missing, fall back to built-in defaults.
# ═══════════════════════════════════════════════════════════════════════════════
STAGE2_PATH = "outputs/dryrun/dryrun_stage2.json"

DEFAULT_SELECTED_IDS = ["C0005", "C0007", "C0009", "C0012", "C0019", "C0020"]
DEFAULT_PROBS = {
    "C0005": 0.3745, "C0007": 0.1950, "C0009": 0.5319,
    "C0012": 0.4247, "C0019": 0.0821, "C0020": 0.2122,
}

def load_selected(stage2_path):
    """Load the 6 selected IDs and their probabilities from Stage 2 output."""
    if not os.path.exists(stage2_path):
        print(f"[WARN] {stage2_path} not found — using DEFAULT dummy selection.")
        return DEFAULT_SELECTED_IDS, DEFAULT_PROBS

    with open(stage2_path) as f:
        s2 = json.load(f)

    # Prefer reps=2 result if valid, otherwise reps=1, otherwise classical
    sel_ids = None
    for key in ["qaoa_reps2", "qaoa_reps1"]:
        if key in s2 and s2[key].get("valid"):
            sel_ids = s2[key]["selected_ids"]
            print(f"  Using Stage 2 result from: {key}")
            break
    if sel_ids is None:
        sel_ids = s2.get("classical", {}).get("selected_ids", DEFAULT_SELECTED_IDS)
        print("  [WARN] No valid QAOA result — using classical brute-force selection.")

    # Pull probabilities from candidates list
    cand = {c["customer_id"]: c["dummy_probability"] for c in s2.get("candidates", [])}
    probs = {cid: cand.get(cid, DEFAULT_PROBS.get(cid, 0.3)) for cid in sel_ids}
    return sel_ids, probs


# ═══════════════════════════════════════════════════════════════════════════════
# EXACT P(≥ k successes) via Poisson-binomial convolution
# ═══════════════════════════════════════════════════════════════════════════════
def poisson_binomial_cdf(probs, threshold):
    """
    Exact probability that the sum of independent Bernoullis with given probs
    is >= threshold.  Uses dynamic programming (O(n^2)).
    """
    n = len(probs)
    # dp[j] = P(exactly j successes) using first i customers
    dp = np.zeros(n + 1)
    dp[0] = 1.0
    for p in probs:
        # Traverse backwards to avoid using updated values
        new_dp = np.zeros(n + 1)
        for j in range(n + 1):
            if dp[j] == 0:
                continue
            new_dp[j]     += dp[j] * (1 - p)
            new_dp[j + 1] += dp[j] * p
        dp = new_dp
    return float(np.sum(dp[threshold:]))


# ═══════════════════════════════════════════════════════════════════════════════
# BUILD THE 6-QUBIT CIRCUIT
# ═══════════════════════════════════════════════════════════════════════════════
def build_circuit(probs_list):
    """
    6-qubit circuit.  Qubit i gets Ry(2*arcsin(sqrt(p_i))).
    P(measure qubit i = 1) = p_i  exactly (ideal).
    """
    n = len(probs_list)
    qc = QuantumCircuit(n, n)
    for i, p in enumerate(probs_list):
        angle = 2 * math.asin(math.sqrt(max(0.0, min(1.0, p))))
        qc.ry(angle, i)
    qc.measure(range(n), range(n))
    return qc


def count_successes_in_bitstring(bs):
    """Count 1s in a bitstring."""
    return bs.count("1")


def p_geq_threshold_from_counts(counts, threshold=3):
    """Estimate P(≥ threshold ones) from simulator counts dict."""
    total = sum(counts.values())
    success_shots = sum(v for k, v in counts.items()
                        if count_successes_in_bitstring(k) >= threshold)
    return success_shots / total


# ═══════════════════════════════════════════════════════════════════════════════
# MONTE CARLO ESTIMATE
# ═══════════════════════════════════════════════════════════════════════════════
def monte_carlo_estimate(probs_list, n_samples=100_000, seed=42):
    rng = np.random.RandomState(seed)
    successes = 0
    for _ in range(n_samples):
        s = sum(1 for p in probs_list if rng.random() < p)
        if s >= 3:
            successes += 1
    return successes / n_samples


# ═══════════════════════════════════════════════════════════════════════════════
# NOISE MODEL
# ═══════════════════════════════════════════════════════════════════════════════
def build_noise_model(depol_1q=0.001, depol_2q=0.01, readout_err=0.02):
    """
    Simple noise model:
      - 1-qubit depolarizing error rate: depol_1q  (default 0.1%)
      - 2-qubit depolarizing error rate: depol_2q  (default 1%)
      - Symmetric readout error: readout_err        (default 2%)
    """
    nm = NoiseModel()
    # 1-qubit gate errors (ry is a 1-qubit gate)
    nm.add_all_qubit_quantum_error(depolarizing_error(depol_1q, 1), ["ry", "u", "u3"])
    # 2-qubit gate errors (not present here, but added for completeness)
    nm.add_all_qubit_quantum_error(depolarizing_error(depol_2q, 2), ["cx", "ecr"])
    # Readout errors
    ro_matrix = [[1 - readout_err, readout_err],
                 [readout_err, 1 - readout_err]]
    ro_err = ReadoutError(ro_matrix)
    for q in range(6):
        nm.add_readout_error(ro_err, [q])
    return nm


# ═══════════════════════════════════════════════════════════════════════════════
# TRANSPILE STATS
# ═══════════════════════════════════════════════════════════════════════════════
def get_circuit_stats(qc, basis_gates=None):
    """Return original and transpiled circuit stats."""
    orig_depth = qc.depth()
    orig_ops   = dict(qc.count_ops())

    if basis_gates is None:
        basis_gates = ["u3", "cx", "measure"]

    sim = AerSimulator()
    tqc = transpile(qc, sim, basis_gates=basis_gates, optimization_level=1, seed_transpiler=42)

    t_depth = tqc.depth()
    t_ops   = dict(tqc.count_ops())
    n_1q    = sum(v for k, v in t_ops.items()
                  if k not in ("cx", "ecr", "measure", "barrier", "reset"))
    n_2q    = t_ops.get("cx", 0) + t_ops.get("ecr", 0)
    n_swap  = t_ops.get("swap", 0)

    return {
        "logical_qubits":      qc.num_qubits,
        "physical_qubits":     tqc.num_qubits,
        "original_depth":      orig_depth,
        "original_ops":        orig_ops,
        "transpiled_depth":    t_depth,
        "transpiled_ops":      t_ops,
        "one_qubit_gates":     n_1q,
        "two_qubit_gates":     n_2q,
        "swap_count":          n_swap,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# ERROR PROPAGATION
# ═══════════════════════════════════════════════════════════════════════════════
def error_propagation(probs_list, shifts=(-0.05, -0.02, 0.02, 0.05)):
    """
    For each probability shift δ, compute P(≥3) when ALL probabilities
    are shifted simultaneously by δ (clamped to [0, 1]).
    """
    base = poisson_binomial_cdf(probs_list, 3)
    results = {}
    for delta in shifts:
        shifted = [min(1.0, max(0.0, p + delta)) for p in probs_list]
        new_p   = poisson_binomial_cdf(shifted, 3)
        results[f"delta_{delta:+.2f}"] = {
            "shifted_probs": [round(p, 4) for p in shifted],
            "P_geq3":        round(new_p, 6),
            "change":        round(new_p - base, 6),
            "pct_change":    round(100 * (new_p - base) / base, 2) if base > 0 else None,
        }
    return base, results


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    os.makedirs("outputs/dryrun", exist_ok=True)
    out = {"NOTE": "DUMMY DRY RUN — Stage 3 uses dummy Stage 2 selections"}

    # ── Load Stage 2 selections ──────────────────────────────────────────────
    print("=" * 60)
    print("Stage 3: 6-qubit success probability estimation")
    print("=" * 60)
    print(f"\nLoading Stage 2 selections from {STAGE2_PATH} ...")
    sel_ids, probs_map = load_selected(STAGE2_PATH)
    probs_list = [probs_map[cid] for cid in sel_ids]

    print(f"\nCustomer-to-qubit mapping (Qiskit little-endian: qubit 0 = rightmost bit):")
    print(f"  {'Qubit':<6} {'Customer ID':<12} {'p_i':>8}")
    for q, (cid, p) in enumerate(zip(sel_ids, probs_list)):
        print(f"  {q:<6} {cid:<12} {p:>8.4f}")

    out["customer_qubit_mapping"] = [
        {"qubit": i, "customer_id": cid, "probability": p,
         "note": f"Qubit {i} = bit at position {5-i} from left in 6-char bitstring"}
        for i, (cid, p) in enumerate(zip(sel_ids, probs_list))
    ]
    out["endian_note"] = (
        "Qiskit little-endian: qubit 0 is the RIGHTMOST character of "
        "the bitstring, qubit 5 is the LEFTMOST."
    )

    # ── Exact P(≥3) ──────────────────────────────────────────────────────────
    print("\n1. Exact P(≥ 3 successes) via Poisson-binomial DP ...")
    exact_p = poisson_binomial_cdf(probs_list, 3)
    print(f"   P(≥3) exact = {exact_p:.8f}")
    out["exact_P_geq3"] = round(exact_p, 8)

    # ── Monte Carlo ───────────────────────────────────────────────────────────
    print("\n2. Monte Carlo estimate (100,000 samples, seed=42) ...")
    mc_p = monte_carlo_estimate(probs_list, n_samples=100_000, seed=42)
    mc_err = abs(mc_p - exact_p)
    print(f"   Monte Carlo P(≥3) = {mc_p:.6f}  (error vs exact: {mc_err:.6f})")
    out["monte_carlo"] = {
        "n_samples": 100_000,
        "seed": 42,
        "estimate": round(mc_p, 6),
        "abs_error_vs_exact": round(mc_err, 6),
    }

    # ── Build circuit ─────────────────────────────────────────────────────────
    print("\n3. Building 6-qubit Ry circuit ...")
    qc = build_circuit(probs_list)
    print(qc.draw(output="text"))

    # ── Circuit stats ─────────────────────────────────────────────────────────
    print("\n4. Circuit statistics ...")
    stats = get_circuit_stats(qc)
    out["circuit_stats"] = stats
    print(f"   Logical qubits    : {stats['logical_qubits']}")
    print(f"   Physical qubits   : {stats['physical_qubits']}")
    print(f"   Original depth    : {stats['original_depth']}")
    print(f"   Transpiled depth  : {stats['transpiled_depth']}")
    print(f"   1-qubit gates     : {stats['one_qubit_gates']}")
    print(f"   2-qubit gates     : {stats['two_qubit_gates']}")
    print(f"   SWAP count        : {stats['swap_count']}")

    # ── Ideal AerSimulator at multiple shot counts ────────────────────────────
    print("\n5. Ideal simulation at multiple shot counts ...")
    shot_configs = [
        (512,   42),
        (1024,  7),
        (4096,  123),
        (16384, 999),
    ]
    sim_ideal = AerSimulator(method="statevector")
    shot_results = []
    for shots, seed in shot_configs:
        tqc = transpile(qc, sim_ideal, seed_transpiler=42)
        job = sim_ideal.run(tqc, shots=shots, seed_simulator=seed)
        counts = job.result().get_counts()
        est_p  = p_geq_threshold_from_counts(counts, threshold=3)
        err    = abs(est_p - exact_p)
        shot_results.append({
            "shots":          shots,
            "seed":           seed,
            "estimated_P":    round(est_p, 6),
            "abs_error":      round(err, 6),
        })
        print(f"   shots={shots:>6}, seed={seed}: P(≥3)={est_p:.4f}  "
              f"(err vs exact={err:.4f})")
    out["ideal_shot_results"] = shot_results

    # ── Noisy simulation ──────────────────────────────────────────────────────
    print("\n6. Noisy simulation (depolarizing 0.1% 1Q, 1% 2Q + 2% readout) ...")
    noise_model = build_noise_model(depol_1q=0.001, depol_2q=0.01, readout_err=0.02)
    sim_noisy   = AerSimulator(noise_model=noise_model, method="density_matrix")
    tqc_noisy   = transpile(qc, sim_noisy, seed_transpiler=42)
    job_noisy   = sim_noisy.run(tqc_noisy, shots=4096, seed_simulator=42)
    counts_noisy = job_noisy.result().get_counts()
    noisy_p  = p_geq_threshold_from_counts(counts_noisy, threshold=3)
    noisy_err = abs(noisy_p - exact_p)
    print(f"   Noisy P(≥3)   = {noisy_p:.6f}  (err vs exact: {noisy_err:.6f})")
    print(f"   Noise params  : depol_1q=0.001, depol_2q=0.01, readout_err=0.02")
    out["noisy_simulation"] = {
        "shots":          4096,
        "seed":           42,
        "noise_model": {
            "depolarizing_1q": 0.001,
            "depolarizing_2q": 0.01,
            "readout_error":   0.02,
        },
        "unmitigated_P_geq3": round(noisy_p, 6),
        "abs_error_vs_exact": round(noisy_err, 6),
        "note": "Unmitigated only. No error mitigation applied.",
    }

    # ── Error propagation ─────────────────────────────────────────────────────
    print("\n7. Error propagation (uniform prob shift of ±0.02 and ±0.05) ...")
    base_p, ep_results = error_propagation(probs_list, shifts=(-0.05, -0.02, 0.02, 0.05))
    print(f"   Base P(≥3) = {base_p:.6f}")
    for delta_key, ep in ep_results.items():
        print(f"   {delta_key}: P(≥3)={ep['P_geq3']:.6f}  "
              f"change={ep['change']:+.6f}  ({ep['pct_change']:+.2f}%)")
    out["error_propagation"] = {
        "base_P_geq3": round(base_p, 6),
        "shifts": ep_results,
        "note": (
            "All 6 probabilities shifted simultaneously by the same delta. "
            "Individual shifts would need a Jacobian/sensitivity analysis."
        ),
    }

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = "outputs/dryrun/dryrun_stage3.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
