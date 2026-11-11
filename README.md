# Atomiq Hackathon: Hybrid Quantum-Classical Term Deposit Campaign Pipeline

Final answer: **six customers to contact: C0031, C0079, C0005, C0318, C0338, C0091**, with a **19.5% probability (exact: 0.1950) that at least three of them subscribe**.

Team: PraNi Verse | Repo: https://github.com/niharikajha2307/atomiq-bank-campaign

## 1. What this project does

| Stage | Task | Main method | Quantum or classical |
|---|---|---|---|
| 1 | Predict subscription probability for 100 test customers | Logistic regression (final). Quantum kernel classifiers evaluated in CV. | Final model: classical. Quantum kernel: evaluated, not selected. |
| 2 | Choose 6 of the top 10 under budget and capacity limits | Binary optimization solved with QAOA (Dicke state + XY mixer) on a simulator | Quantum solver, checked against classical brute force |
| 3 | Estimate P(at least 3 of the 6 subscribe) | 6-qubit circuit, one qubit per customer | Quantum circuit, checked against exact and Monte Carlo |

Result files (top level): `stage1_results.csv`, `stage2_results.json` / `.csv`, `stage3_results.json` / `.csv`. All files use `customer_id` exactly as given.

## 2. Data and rules followed
- Source: organizers' corrected release `inputs/bank_campaign_participant_release (1).xlsx` (adds `customer_id` to `Stage1_Test`; all other sheets identical to the first release). Checked: 100 unique test IDs, none in `Stage1_Train`, all in `Stage2_Data`.
- Training data: 300 rows, 35 "yes" (11.67%). Heavy class imbalance, so accuracy is not used.
- `customer_id` is only a key. It never enters a model, scaler, encoder or feature selector.
- `Stage1_Test` was used once, for the final prediction, after the model was chosen from training-data cross-validation only. No test information was used for features, thresholds, calibration or hyperparameters.
- The Stage 1 top 10 was not edited by hand and is the input to Stages 2 and 3.

## 3. Stage 1: prediction

**Preprocessing (fitted on training data only, inside a scikit-learn Pipeline):** RobustScaler, OneHotEncoder, a `pdays_contacted` flag (`pdays = -1` means never contacted).

**Validation:** 5-fold stratified cross-validation on `Stage1_Train`, seed 42. Mean ± std:

| Model | ROC-AUC | PR-AUC | Precision@10 | Brier | Log loss | ECE |
|---|---|---|---|---|---|---|
| Constant prior (11.67%) | n/a | 0.117 | 0.117 | 0.1031 | 0.3602 | 0.000 |
| **LogReg unweighted, C=0.3 (final)** | 0.607 ± 0.054 | 0.235 ± 0.069 | 0.267 ± 0.133 | **0.1012 ± 0.004** | 0.3546 ± 0.017 | 0.026 ± 0.015 |
| LogReg balanced + prior shift | 0.649 ± 0.077 | 0.254 ± 0.060 | 0.200 ± 0.067 | 0.1069 | 0.3770 | 0.063 |
| HistGradientBoosting, calibrated | 0.536 ± 0.139 | 0.211 ± 0.089 | 0.180 ± 0.084 | 0.1024 | 0.3574 | 0.006 |
| Quantum kernel, 5 qubits | 0.496 ± 0.128 | 0.136 ± 0.032 | 0.100 ± 0.071 | 0.1042 | 0.3645 | 0.013 |
| Quantum kernel, 4 qubits | 0.452 ± 0.087 | 0.129 ± 0.033 | 0.060 ± 0.089 | 0.1075 | 0.3790 | 0.034 |

**Model choice (from CV only):** unweighted logistic regression, C=0.3. It has the lowest Brier score, the best Precision@10 and good calibration. The balanced models rank slightly better by ROC-AUC but their probabilities are worse than the constant baseline, and Stages 2 and 3 use the probabilities as probabilities.

**Quantum classifier:** a quantum kernel classifier (ZZ-type feature map, kernel simulated on Qiskit Aer, SVM on the kernel) on 5 and 4 features chosen by `SelectKBest(f_classif)` inside each CV fold. 5-qubit features: `day`, `month_feb`, `month_jun`, `month_may`, `poutcome_success`; 4-qubit: the same without `day`. **Both quantum kernels scored at or below chance level and did not beat the classical baselines, so they were not used for the final probabilities.** With 300 rows and 35 positives this is expected. We make no quantum-advantage claim.

**Honest limits:** all spreads are large (std 0.05 to 0.14) so differences between models are mostly within noise, and the signal in this data is weak. The final probabilities lie between 0.031 and 0.381 (mean 0.113).

**Top 10 (rank, customer, probability):** 1 C0031 0.3810; 2 C0079 0.3750; 3 C0081 0.2556; 4 C0005 0.2412; 5 C0318 0.2246; 6 C0208 0.2164; 7 C0183 0.2097; 8 C0338 0.2015; 9 C0388 0.1851; 10 C0091 0.1847. Ties are broken by lower `customer_id`.

## 4. Stage 2: constrained selection

**Problem.** For each candidate i: `expected_net_value_i = p_i * expected_deposit_value_i - contact_cost_i`. Binary variable `x_i` (1 = contact). Maximize `Z = sum(ENV_i * x_i)` subject to `sum(x_i) = 6`, `sum(cost_i * x_i) <= 25`, `sum(capacity_i * x_i) <= 22`.

**Constraint handling (two formulations, both implemented):**
1. **Penalty + slack QUBO (qiskit-optimization):** the equality is a quadratic penalty; each inequality uses binary slack variables (5 budget + 5 capacity) plus a penalty. 20 qubits. 75% to 80% of its measurement shots violate `K = 6` (a random 10-bit string has exactly six ones only 20.5% of the time), so this version needs filtering to find a valid selection.
2. **Constraint-preserving QAOA (used for the final result):** the circuit starts in the Dicke state |D(10,6)> and mixes with an XY ring mixer, so every measured bitstring has exactly six ones by construction (`K = 6` enforced by the circuit). 10 qubits.

**Important finding:** for this top 10, **all 210 combinations satisfy both limits** (the largest possible cost is 24 and capacity 18), so the budget and capacity inequalities do not bind and the slack variables do not change the answer. The optimum is simply the six customers with the highest expected net value. The formulations are kept in the code because they are needed in general.

**Solver settings:** Qiskit Aer `SamplerV2`, COBYLA (`maxiter=200`, `tol=1e-4`), 2048 shots, reps 1 to 3, seeds 42, 7, 123 (9 runs). Final selection from reps=1, seed=42; the same optimum was found across all reps and seeds.

**Result:**

| Item | Value |
|---|---|
| Selected six | C0031, C0079, C0005, C0318, C0338, C0091 |
| Best bitstring (Qiskit order, qubit 0 rightmost) | `1010011011` (six ones; 75 of 2048 shots, 3.66%) |
| Objective (expected net value) | 18.7169 |
| Total cost / budget | 14 / 25 |
| Total capacity / limit | 13 / 22 |
| Classical optimum (brute force over 210) | 18.7169 |
| Gap | 0.0 |

**Quantum vs classical, honestly.** The optimum's probability under the Dicke+XY QAOA is 3.36% (exact statevector; 3.66% from 2048 shots), against 0.48% for uniform sampling over the 210 valid combinations (theory 1/210) and 0.10% for uniform random 10-bit strings (theory 1/1024). The solver concentrates probability on the optimum, but classical brute force finds the optimum in under a millisecond. This is a demonstration on a simulator, not an advantage.

## 5. Stage 3: campaign simulation

**Circuit.** 6 qubits, one per selected customer, each prepared with `Ry(2*arcsin(sqrt(p_i)))` so that P(measure 1) = p_i. Customers respond independently, so no entanglement is needed (the circuit has no two-qubit gates). Success = at least 3 qubits read 1.

**Qubit order (qubit 0 is the rightmost character of a printed Qiskit bitstring):** q0 = C0031 (0.3810), q1 = C0079 (0.3750), q2 = C0005 (0.2412), q3 = C0318 (0.2246), q4 = C0338 (0.2015), q5 = C0091 (0.1847). The rotation angles are Ry(1.3304), Ry(1.3181), Ry(1.0267), Ry(0.9875), Ry(0.9311), Ry(0.8884). This order matches `stage3_results.json`.

| Method | P(at least 3) | Error vs exact |
|---|---|---|
| Exact (Poisson-binomial) | 0.195029 | n/a |
| Monte Carlo, 100,000 samples, seed 42 | 0.193000 | 0.002029 |
| Ideal circuit, 16,384 shots | 0.195679 | 0.000650 |
| Noisy circuit, 8192 shots (unmitigated) | 0.207764 | 0.012735 |

The ideal-circuit error shrinks with shots (512, 1024, 4096, 16384 shots are in `stage3_results.json`). **Noise model:** 0.1% single-qubit depolarizing error and 1.5% symmetric readout error. No error mitigation was applied; all noisy numbers are unmitigated.

**Circuit statistics (Aer, no coupling map, basis gates u, cx, rz, sx, x):** 6 logical and 6 physical qubits; original and transpiled depth 2 (one rotation layer plus measurement); 6 single-qubit rotation gates, 0 two-qubit gates, 0 SWAPs. `count_ops()`: original `{ry: 6, measure: 6}`, transpiled `{u: 6, measure: 6}`.

**Error propagation (Stage 1 probabilities shifted by a constant delta, all six together):**

| delta | P(at least 3) | Absolute change | Relative change |
|---|---|---|---|
| -0.05 | 0.1174 | -0.0777 | -39.8% |
| -0.02 | 0.1620 | -0.0331 | -17.0% |
| +0.02 | 0.2304 | +0.0354 | +18.1% |
| +0.05 | 0.2871 | +0.0921 | +47.2% |

The Stage 3 answer is very sensitive to Stage 1 errors: a 0.05 error in each probability moves the result by 40% to 47% in relative terms. Stage 1 probabilities are uncertain (weak signal, 300 training rows), so the 19.5% figure should be read as an estimate with a wide band.

## 6. Quantum vs classical summary
- **Classical:** all data preparation, the final Stage 1 probabilities, all baselines, the brute-force optimum, the exact and Monte Carlo Stage 3 references.
- **Quantum (simulated):** the quantum kernel classifiers (evaluated, not selected), the QAOA selection in Stage 2, the Stage 3 circuit.
- No quantum-advantage claim is made. Everything runs on a noiseless or noise-modelled simulator.

## 7. Reproducibility
- **Seeds:** CV splits and models 42; QAOA optimizer and simulator 42, 7, 123; transpiler 42; Monte Carlo 42 (`numpy.random.default_rng(42)`).
- **Versions:** Python 3.13.5, qiskit 2.6.2, qiskit-aer 0.18.2, qiskit-algorithms 0.4.0, qiskit-optimization 0.7.0, qiskit-machine-learning 0.9.1, scikit-learn 1.9.1, numpy 2.5.4, pandas 3.1.1, scipy 1.18.1, docplex 2.32.264 (see `requirements.txt`).
- **Backend:** `qiskit_aer.AerSimulator` through `qiskit_aer.primitives.SamplerV2`. Shots: Stage 2 2048; Stage 3 ideal 512 / 1024 / 4096 / 16384, noisy 8192; Monte Carlo 100,000 samples.
- **Rerun locally (Windows PowerShell, from the project root; about 3 to 4 minutes):**
```
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/export_sheets.py
.\.venv\Scripts\python.exe scripts/run_final_stage1.py
.\.venv\Scripts\python.exe scripts/run_real_stage2_and_stage3.py
```
  On macOS or Linux, use `.venv/bin/python` instead. To regenerate the Stage 1 cross-validation results:
```
.\.venv\Scripts\python.exe scripts/run_stage1_cv.py
.\.venv\Scripts\python.exe scripts/run_job1.py
```
  `run_stage1_cv.py` runs the original classical baselines and the 4- and 5-qubit quantum kernels (about 1 minute; writes `outputs/stage1_cv_metrics.csv` and `outputs/stage1_cv_curves.png`). `run_job1.py` adds the constant-prior baseline, the unweighted logistic regressions (C = 0.03, 0.1, 0.3, 1.0) and the balanced logistic regression with prior shift (about 2 seconds).
- **Smoke test:** a 2-qubit Bell-state check (`tests/`) only confirms the environment works. It is not a quantum result.

## 8. Limitations
- Weak signal and small data (300 training rows, 35 positives): CV metrics are noisy and model differences are within noise.
- Quantum kernels did not beat classical baselines. Stage 2 and Stage 3 quantum components work on simulators and do not beat their classical references.
- Stage 2 constraints do not bind for this top 10, so constraint handling was exercised mainly in the penalty-based formulation and in dry runs on dummy inputs (kept under `outputs/dryrun/`, labelled DUMMY, not used for results).
- Real hardware was not used. The noise model is simple and not calibrated to a device.

## 9. Business recommendation
Contact the six selected customers. They have the highest expected net value of the 10 candidates and fit comfortably inside the budget and capacity limits (cost 14 of 25, capacity 13 of 22), so there is room to contact more customers if the bank wants to. Plan for about a 1-in-5 chance (0.195) that the campaign reaches three conversions, and treat that figure as uncertain: it is highly sensitive to the Stage 1 probabilities. Improving the prediction model with more labelled data is the most valuable next step.
