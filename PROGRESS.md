# Atomiq Hackathon Project Progress

## Current Status: Stage 1, Stage 2, and Stage 3 Real Runs Complete (All Deliverables Generated)

### Stage 1 Final Model Prediction (OFFICIAL)
- **Selected Model:** `LogisticRegression` (unweighted, $C=0.3$, L2 penalty, solver=`lbfgs`, max_iter=1000, seed=42).
- **Selection Basis from 5-Fold Stratified CV:**
  - Ranked #1 in Precision@10 ($0.2667 \pm 0.1333$) and lowest Brier score ($0.1012 \pm 0.0040$), with near-optimal log loss ($0.3546 \pm 0.0173$) and excellent calibration (ECE $= 0.0256 \pm 0.0154$).
  - **Quantum Kernel Status:** Evaluated rigorously across 5 folds with 4 and 5 qubits (ZZ linear feature map). Neither beat the classical baselines (ROC-AUC $\approx 0.45 - 0.49$, PR-AUC $\approx 0.13 - 0.14$). In accordance with strict empirical selection criteria, quantum classifiers were NOT selected for final test predictions.
- **Preprocessing & Hygiene:** `RobustScaler`, `OneHotEncoder`, and `pdays_contacted` feature flag fit exclusively on `Stage1_Train`. `customer_id` was strictly retained as an operational identifier key and never entered any transformer or model step.
- **Prediction Execution:** Executed exactly ONCE on `Stage1_Test` (100 customers).
- **Official Top-Level Deliverable:** [stage1_results.csv](stage1_results.csv)
  - 100 rows, 4 columns: `customer_id`, `predicted_probability`, `rank`, `top10_flag`.
  - Exact rank 1..100 with ties broken by ascending `customer_id`.
  - Probability statistics: Min = `0.031462`, Max = `0.380956`, Mean = `0.113069`, Std = `0.063442`.
  - Environment details: Python 3.13.5, `scikit-learn==1.9.1`, `numpy==2.5.4`, `pandas==3.1.1`, `qiskit==2.6.2`, `qiskit-aer==0.18.2`.

---

### 1. Step 1: Verification & Problem Specification Comparison
- **Status:** Complete.
- Scanned 4-page official competition PDF (`inputs/PROBLEM_STATEMENT_THEATOMIQHACKTHON.pdf`) and compared word-for-word against `PROJECT_BRIEF.md`.
- Confirmed complete alignment on competition rules, metrics (ROC-AUC, PR-AUC, Precision@10, Recall@10, Brier score, log loss), Stage 2 constraints (K=6, Budget <= 25, Capacity <= 22), and scoring weights (35% Quantum Rigor, 30% Accuracy & Integration, 20% Code Quality, 15% Report & Presentation).
- Identified two critical data discrepancies against the PDF:
  1. `Stage1_Test` has no `customer_id` column.
  2. The PDF states `Stage2_Data` is for the 100 test customers, but the Excel sheet contains 400 rows.

### 2. Step 2: Environment Setup & Qiskit Compatibility Test
- **Status:** Complete.
- Python 3.13.5 virtual environment created (`.venv`).
- Packages installed with zero dependency conflicts and pinned in [requirements.txt](requirements.txt):
  - Data Science: `numpy==2.5.4`, `pandas==3.1.1`, `openpyxl==3.1.5`, `scikit-learn==1.9.1`, `matplotlib==3.11.2`, `scipy==1.18.1`
  - Quantum Stack: `qiskit==2.6.2`, `qiskit-aer==0.18.2`, `qiskit-algorithms==0.4.0`, `qiskit-optimization==0.7.0`, `qiskit-machine-learning==0.9.1`, `docplex==2.32.264`
- 2-qubit Aer test on Bell state passed (`{'11': 490, '00': 510}`).

### 3. Step 3 & CSV Re-Verification: Data Export & Structure Integrity
- **Status:** Complete.
- Re-verified data export with `dtype=str` and `keep_default_na=False`:
  - **Hidden Rows Check:** Exactly 0 hidden rows detected across all 5 sheets in the workbook.
  - **Fidelity Check:** 100% exact string match between raw Excel cells and exported CSVs (e.g. `"5706"` preserved strictly as `"5706"`).
  - **NaN Check:** Zero cells converted to `NaN`.
- Exported files in [data/](data/):
  - `README.csv` (20 rows, 2 cols)
  - `Stage1_Train.csv` (300 rows, 17 cols)
  - `Stage1_Test.csv` (100 rows, 15 cols)
  - `Stage2_Data.csv` (400 rows, 5 cols)
  - `Stage2_Constraints.csv` (6 rows, 4 cols)

### 4. Step 4 & 5: Data Inspection & Organizer Queries (RESOLVED)
- **Status:** **RESOLVED.**
- **Stage1_Train Analysis:** 300 rows, 16 features, target `y` has 35 "yes" (11.67%) and 265 "no" (88.33%). Heavy class imbalance. `pdays = -1` present in 254/300 rows (customer not previously contacted).
- **Stage1_Test Analysis:** 100 rows, 15 features + `customer_id`.
- **Resolution of Open Questions:**
  1. **`customer_id` Mapping Resolved:** The organizers issued a corrected release workbook (`bank_campaign_participant_release (1).xlsx`) containing the explicit `customer_id` column for `Stage1_Test`. Verified: Exactly 100 unique customer IDs, 0 overlap with `Stage1_Train`, and all 100 are present in `Stage2_Data`. All sheets were re-exported with `dtype=str` and `keep_default_na=False`.
  2. **`Stage2_Data` Scope Resolved:** The 400 rows represent the complete operational lookup table across all campaign participants. For Stage 2 optimization, we filter `Stage2_Data` to match our top-10 ranked test customers produced by Stage 1.
- **Strict Data Hygiene Commitment:** `customer_id` is treated strictly as an operational database key. It is never used as a model feature and will not enter any preprocessing, transformation, or feature selection step.

### 5. Stage 1 Model Development & 5-Fold Stratified Cross-Validation
- **Status:** Complete on `Stage1_Train` ONLY.
- **Strict Data Hygiene:** No test data was loaded, preprocessed, or touched. All preprocessing (`RobustScaler`, `OneHotEncoder`, `pdays_contacted` feature flag, and feature selection) was fit inside scikit-learn `Pipeline` objects strictly on training folds.
- **Models Evaluated:**
  1. `LogisticRegression_Balanced` (Classical Linear Baseline)
  2. `LogisticRegression_Calibrated` (Classical Linear with Sigmoid Calibration)
  3. `HistGradientBoosting_Balanced` (Classical GBDT Baseline)
  4. `HistGradientBoosting_Calibrated` (Classical GBDT with Sigmoid Calibration)
  5. `QuantumKernel_5Qubit` (Quantum Kernel with 5 Qubits, linear ZZ feature map)
  6. `QuantumKernel_4Qubit` (Quantum Kernel with 4 Qubits, linear ZZ feature map)
- **Artifacts Generated:**
  - Metrics Table: [outputs/stage1_cv_metrics.csv](outputs/stage1_cv_metrics.csv)
  - Diagnostic Plots: [outputs/stage1_cv_curves.png](outputs/stage1_cv_curves.png) (ROC, PR, and Calibration curves)

---

## 5-Fold Stratified Cross-Validation Results Summary

| Model | ROC-AUC | PR-AUC | Precision@10 | Recall@10 | Brier Score | Log Loss | ECE (Calib Err) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Constant_Prior** (Train Prevalence) | -- | 0.1167 ± 0.000 | 0.1167 ± 0.000 | 0.1000 ± 0.000 | 0.1031 ± 0.000 | 0.3602 ± 0.000 | 0.0000 ± 0.000 |
| **LogisticRegression_Balanced** | **0.6491 ± 0.086** | **0.2544 ± 0.067** | 0.1800 ± 0.045 | 0.2571 ± 0.064 | 0.2022 ± 0.033 | 0.5946 ± 0.088 | 0.2720 ± 0.054 |
| **LR_Balanced_PriorShift** | **0.6491 ± 0.077** | **0.2544 ± 0.060** | 0.2000 ± 0.067 | 0.1714 ± 0.057 | 0.1069 ± 0.006 | 0.3770 ± 0.043 | 0.0626 ± 0.006 |
| **LR_Unweighted_C1.0** | 0.6264 ± 0.069 | 0.2391 ± 0.058 | 0.2333 ± 0.082 | 0.2000 ± 0.070 | 0.1035 ± 0.006 | 0.3641 ± 0.029 | 0.0399 ± 0.009 |
| **LR_Unweighted_C0.3** | 0.6070 ± 0.054 | 0.2354 ± 0.069 | **0.2667 ± 0.133** | 0.2286 ± 0.114 | **0.1012 ± 0.004** | 0.3546 ± 0.017 | 0.0256 ± 0.015 |
| **LR_Unweighted_C0.1** | 0.5957 ± 0.051 | 0.2461 ± 0.068 | 0.2333 ± 0.082 | 0.2000 ± 0.070 | 0.1016 ± 0.002 | **0.3542 ± 0.011** | 0.0184 ± 0.005 |
| **LR_Unweighted_C0.03** | 0.5860 ± 0.068 | 0.2174 ± 0.067 | 0.2000 ± 0.125 | 0.1714 ± 0.107 | 0.1024 ± 0.002 | 0.3568 ± 0.006 | 0.0128 ± 0.007 |
| **LogisticRegression_Calibrated** | 0.5644 ± 0.101 | 0.1821 ± 0.076 | 0.1200 ± 0.110 | 0.1714 ± 0.156 | 0.1026 ± 0.001 | 0.3582 ± 0.006 | **0.0035 ± 0.003** |
| **HistGradientBoosting_Balanced** | 0.5951 ± 0.096 | 0.2467 ± 0.085 | 0.2000 ± 0.071 | **0.2857 ± 0.101** | 0.1256 ± 0.016 | 0.4580 ± 0.047 | 0.1053 ± 0.026 |
| **HistGradientBoosting_Calibrated**| 0.5358 ± 0.139 | 0.2112 ± 0.089 | 0.1800 ± 0.084 | 0.2571 ± 0.120 | 0.1024 ± 0.001 | 0.3574 ± 0.006 | 0.0064 ± 0.006 |
| **QuantumKernel_5Qubit** | 0.4962 ± 0.128 | 0.1356 ± 0.032 | 0.1000 ± 0.071 | 0.1429 ± 0.101 | 0.1042 ± 0.001 | 0.3645 ± 0.003 | 0.0129 ± 0.017 |
| **QuantumKernel_4Qubit** | 0.4515 ± 0.087 | 0.1289 ± 0.033 | 0.0600 ± 0.089 | 0.0857 ± 0.128 | 0.1075 ± 0.005 | 0.3790 ± 0.021 | 0.0338 ± 0.036 |

*(Note: Random baseline PR-AUC is equal to the positive class rate: 0.1167 / 11.67%)*

---

## 6. Stage 2 & 3 Dummy Dry Run Verification (Completed)
- **Status:** Complete on **DUMMY data ONLY** (10 test candidate IDs from `Stage2_Data` with seed=42 dummy probabilities).
- **Strict Boundary:** No test data was evaluated, and top-level final results files were NOT created. All dry-run artifacts are isolated in [outputs/dryrun/](outputs/dryrun/).

### Stage 2 QUBO & Quantum Optimization
- **QUBO Formulation & Qubit Count:**
  - 10 decision variables ($x_0 \dots x_9$) for customer selection ($K=6$).
  - Budget ($\le 25$) encoded via 5 binary slack bits.
  - Capacity ($\le 22$) encoded via 5 binary slack bits.
  - **Total QUBO size:** Exactly **20 qubits** (185 Pauli terms in Ising Hamiltonian).
- **Constraint Enforcement & Violation Analysis:**
  - **Constraints Enforced Inside QUBO:** All three constraints ($C_1: \sum x_i = 6$, $C_2: \sum c_i x_i \le 25$, $C_3: \sum u_i x_i \le 22$) are mathematically encoded inside the QUBO Hamiltonian via quadratic penalties and binary slack expansions.
  - **Why Full-Slack QUBO Was Chosen:** It allows the quantum state evolution itself to penalize infeasible allocations in Hilbert space rather than relying on unguided post-selection. The full-slack formulation is retained in the codebase as the rigorous baseline.
  - **Sample Violation Rates (across 2,048 measurement shots):**
    - **QAOA `reps=1`:** Overall Infeasible = **80.0%** ($C_1$ size violation: 80.0%, $C_2$ budget violation: 0.0%, $C_3$ capacity violation: 6.1%).
    - **QAOA `reps=2`:** Overall Infeasible = **75.2%** ($C_1$ size violation: 75.2%, $C_2$ budget violation: 0.0%, $C_3$ capacity violation: 8.3%).
    - *Observation:* The penalty terms strongly suppress budget and capacity overruns, while the cardinality constraint ($K=6$) is the primary source of sample infeasibility at shallow depths ($p=1, 2$).
- **Solver Performance Comparison:**
  - **Classical Brute Force** ($C(10,6) = 210$ combinations): Optimal objective = **41.9494** (Selected: `C0005`, `C0007`, `C0009`, `C0025`, `C0028`, `C0031`; Budget used: 14.0/25; Capacity used: 16.0/22; 210/210 valid combinations).
  - **NumPyMinimumEigensolver** (Exact QUBO diagonalization): Objective = **41.9494** in 1.44s (100% match with classical brute force).
  - **QAOA reps=1** (`SamplerV2` on Aer, COBYLA 50 iters, seed=42): Solved in **15.5s**. Best valid selection has objective **41.9494** (Gap vs Classical: **0.0**, exact match!).
  - **QAOA reps=2** (`SamplerV2` on Aer, COBYLA 50 iters, seed=42): Solved in **47.0s**. Best valid selection has objective **39.4252** (Gap vs Classical: **2.5242**; selected rank #4 combination).
  - *Diagnosis on prior timeouts:* Qiskit's `StatevectorSampler` hangs on matrix exponential decomposition for 20-qubit Hamiltonians in Python 3.13; decomposing the `QAOAAnsatz` down to native gates and executing on `qiskit_aer.primitives.SamplerV2` executes within seconds.
- **Dry-run Artifacts:**
  - [outputs/dryrun/dryrun_stage2.json](outputs/dryrun/dryrun_stage2.json)
  - [outputs/dryrun/dryrun_classical_benchmark.json](outputs/dryrun/dryrun_classical_benchmark.json)
  - [outputs/dryrun/dryrun_job2_results.json](outputs/dryrun/dryrun_job2_results.json)
  - [outputs/dryrun/stage2_diagnosis.json](outputs/dryrun/stage2_diagnosis.json)

### Stage 2 Diagnosis & Benchmark: Pure 10-Qubit Dicke(6)+XY vs Baselines
- **Diagnosis of Prior ~99.9% Infeasibility in Dicke+XY (Job 2):**
  - **Root Cause Identified:** In the previous script, slack variables for inequality constraints expanded the QuadraticProgram to 20 qubits. The approximate Dicke excitation ladder was inadvertently applied over 20 qubits rather than confining 6 ones strictly to the 10 decision qubits, dispersing Hamming weight into slack bits ($0.0\%$ shots had $K=6$ on decision bits).
- **Pure 10-Qubit Formulation (Exact $|D(10,6)\rangle$ + XY Ring Mixer):**
  - When implemented strictly on the 10 decision qubits with an XY ring mixer and exact Dicke state initialization, Hamming weight $K=6$ is preserved **100.0%** across all shots.
  - Since all combinations with $K=6$ on this dummy dataset satisfy Budget ($\le 25$) and Capacity ($\le 22$), the infeasibility rate drops from $100\%$ to **0.0%**.
- **Comparative Performance across Solvers (2,048 shots):**
  - **Baseline 1 (Uniform over $C(10,6)=210$ combinations):** Feasibility = 100%, Best Obj = 41.9494, $P(\text{opt}) = 0.004883 \approx 0.49\%$ (theory $1/210 = 0.476\%$).
  - **Baseline 2 (Uniform random 10-bit strings from $\{0,1\}^{10}$):** Feasibility = 19.58% (theory 20.51%), Best Obj = 41.9494, $P(\text{opt}) = 0.001465 \approx 0.15\%$ (theory $1/1024 = 0.098\%$).
  - **Pure Dicke(6) + XY Mixer QAOA (10 qubits, seed 42):**
    - `reps=1`: Infeasible = **0.0%**, Best Obj = **41.9494** (Gap = 0.0), $P(\text{opt}) = \mathbf{0.008301} \approx \mathbf{0.83\%}$ (Elapsed: 9.5s).
    - `reps=2`: Infeasible = **0.0%**, Best Obj = **41.9494** (Gap = 0.0), $P(\text{opt}) = \mathbf{0.007324} \approx \mathbf{0.73\%}$ (Elapsed: 12.5s).
    - `reps=3`: Infeasible = **0.0%**, Best Obj = **41.9494** (Gap = 0.0), $P(\text{opt}) = \mathbf{0.014160} \approx \mathbf{1.42\%}$ (Elapsed: 13.2s).
- **Honest Finding on Quantum Concentration:**
  - With a strictly constraint-preserving mixer on 10 qubits, QAOA achieves **100% feasibility** and concentrates ground state probability up to **1.42%** at $p=3$ — **beating uniform random combination sampling ($0.48\%$) by 3x** and uniform bitstring sampling ($0.10\%$) by 14x. However, classical brute force still finds the exact optimum in < 1 millisecond.

### Stage 2 Quantum Solver Benchmark: Penalty-Only vs Dicke+XY vs Random Sampling
- **Evaluation Setup (Seed 42, 7, 123; reps=1, 2, 3; COBYLA maxiter=200; 5-min timeout per run):**
  - **Random Sampling Baseline (2,048 samples over 210 feasible combinations):**
    - Finds the exact optimum (41.9494) in < 1 second.
    - Hit rate on optimal bitstring: $0.49\%$ (matches theoretical $1/210 = 0.476\%$).
    - Infeasibility rate: $0.0\%$.
  - **Penalty-Only QAOA (20 qubits: 10 decision + 10 slack):**
    - Infeasibility rate: $74.6\% - 79.7\%$ of measurement shots violate constraints.
    - Hit rate on optimal bitstring: $0.05\% - 0.24\%$ across seeds (below uniform random baseline of $0.48\%$).
    - Does find the optimal selection ($41.9494$) among valid measured samples at reps=2 (seed 42), but does not beat random sampling in probability concentration.
  - **Dicke(6) + XY Ring Mixer QAOA (20 qubits, K=6 enforced via mixer structure):**
    - High sample invalidity ($99.7\% - 100\%$) due to heavy budget/capacity penalty terms distorting the state on shallow depths ($p=1, 2, 3$).
    - Best valid objective reached: $40.6422$ (Gap vs classic: $1.3072$).
    - Hit rate on optimal bitstring: $0.0\%$. Does not beat random sampling.
  - **Honest Finding on Quantum Advantage:** On a 10-candidate binary selection problem, classical random sampling and brute-force evaluate all 210 states in milliseconds with 100% feasibility. Shallow QAOA ($p \le 3$) produces high invalid sample rates ($>74\%$) and achieves lower probability on the ground state than uniform random guessing ($0.1\% \text{ vs } 0.48\%$). This provides clear, honest empirical documentation of current NISQ solver limitations on constrained combinatorial problems.

### Stage 3 Success Probability Circuit Simulation
- **Setup:** 6 qubits prepared with $R_y(2 \arcsin(\sqrt{p_i}))$ rotation gates representing subscription probabilities of Stage 2 selected customers.
- **Endianness Convention:** Explicitly handled Qiskit little-endian format (Qubit 0 is rightmost bit).
- **Exact vs Simulation Results (Success = $\ge 3$ subscribers):**
  - **Exact Poisson-Binomial DP:** $P(\ge 3) = 0.496398$
  - **Monte Carlo (100,000 samples):** $P(\ge 3) = 0.494680$ (Absolute error: 0.001718)
  - **Ideal Circuit Execution:**
    - Shots = 512: $P = 0.4785$ (Absolute error: 0.0179)
    - Shots = 1024: $P = 0.4883$ (Absolute error: 0.0081)
    - Shots = 4096: $P = 0.5034$ (Absolute error: 0.0070)
    - Shots = 16384: $P = 0.4924$ (Absolute error: 0.0040 — error shrinks monotonically with $1/\sqrt{N}$)
  - **Noisy Simulation (Depolarizing + 2% Readout Error):** Unmitigated $P(\ge 3) = 0.511230$ (Absolute error: 0.014833).
  - **Circuit Characteristics:** 6 logical/physical qubits, circuit depth = 2, 6 1-qubit gates, 0 2-qubit gates, 0 SWAPs.
  - **Sensitivity Analysis (Relative vs Absolute Labels):**
    - Base probability: $P(\ge 3) = 0.496398$
    - Shift $\delta = -0.05$: Absolute change = $-0.1075$ | **Relative change = -21.66%**
    - Shift $\delta = -0.02$: Absolute change = $-0.0430$ | **Relative change = -8.67%**
    - Shift $\delta = +0.02$: Absolute change = $+0.0426$ | **Relative change = +8.59%**
    - Shift $\delta = +0.05$: Absolute change = $+0.1051$ | **Relative change = +21.16%**
    - *(Note: The $\pm 8.6\%$ and $\pm 21.4\%$ values represent relative changes in $P(\ge 3)$ relative to base probability $0.4964$, corresponding to absolute shifts of roughly $\pm 0.043$ and $\pm 0.106$.)*
- **Dry-run Artifact:**
  - [outputs/dryrun/dryrun_stage3.json](outputs/dryrun/dryrun_stage3.json)

---

## 7. Real Stage 2 Optimization Results (Official)

- **Deliverables:** [stage2_results.json](stage2_results.json), [stage2_results.csv](stage2_results.csv) at top level.
- **Top 10 Selection Pool:** Merged from [stage1_results.csv](stage1_results.csv) (`top10_flag == 1`) and `data/Stage2_Data.csv`.
- **Constraint Binding Analysis:**
  - All $C(10,6) = 210$ combinations strictly satisfy Budget ($\sum c_i \le 25$) and Capacity ($\sum u_i \le 22$).
  - Therefore, the two inequality constraints **do not bind** for this top 10 pool, meaning slack variables do not alter the feasibility boundary or optimal solution. The full formulation (equations, penalty weights, and binary slack expansion) is retained in the codebase and write-up for methodological completeness.
- **Classical Brute Force Optimum:**
  - Objective ($ENV$): **18.716898**
  - Selected 6 Customers: `['C0031', 'C0079', 'C0005', 'C0318', 'C0338', 'C0091']`
  - Budget Used: 14.00 / 25.0 | Capacity Used: 13.00 / 22.0
- **Classical Random Baselines (2048 shots):**
  - Baseline 1 (Uniform over 210 combinations): $P(\text{opt}) = 0.007324$ ($15/2048$), Best Obj = 18.7169.
  - Baseline 2 (Uniform $\{0,1\}^{10}$ bitstrings): Feasible = $19.58\%$, $P(\text{opt}) = 0.000977$ ($2/2048$), Best Obj = 18.7169.
- **Penalty-Only QAOA (reps=2, seed=42):**
  - Feasible fraction: $21.73\%$
  - Best valid objective: **18.7169** (Gap: 0.0000), $P(\text{opt}) = 0.000977$, Runtime: 53.3s.
- **Constraint-Preserving Dicke(6) + XY-Ring Mixer QAOA (10 decision qubits, 2048 shots):**
  - All 9 configurations achieved **0.0% invalid rate** (100% feasibility), **Gap = 0.0000**, and found the exact global optimum bitstring:
    - `reps=1, seed=42`: Obj = 18.716898, $P(\text{opt})_{\text{shots}} = 0.0366$, $P(\text{opt})_{\text{SV}} = 0.0336$, Runtime = 6.5s
    - `reps=1, seed=7`: Obj = 18.716898, $P(\text{opt})_{\text{shots}} = 0.0303$, $P(\text{opt})_{\text{SV}} = 0.0323$, Runtime = 7.1s
    - `reps=1, seed=123`: Obj = 18.716898, $P(\text{opt})_{\text{shots}} = 0.0278$, $P(\text{opt})_{\text{SV}} = 0.0334$, Runtime = 6.6s
    - `reps=2, seed=42`: Obj = 18.716898, $P(\text{opt})_{\text{shots}} = 0.0337$, $P(\text{opt})_{\text{SV}} = 0.0313$, Runtime = 8.0s
    - `reps=2, seed=7`: Obj = 18.716898, $P(\text{opt})_{\text{shots}} = 0.0303$, $P(\text{opt})_{\text{SV}} = 0.0293$, Runtime = 8.1s
    - `reps=2, seed=123`: Obj = 18.716898, $P(\text{opt})_{\text{shots}} = 0.0278$, $P(\text{opt})_{\text{SV}} = 0.0314$, Runtime = 9.8s
    - `reps=3, seed=42`: Obj = 18.716898, $P(\text{opt})_{\text{shots}} = 0.0366$, $P(\text{opt})_{\text{SV}} = 0.0354$, Runtime = 11.4s
    - `reps=3, seed=7`: Obj = 18.716898, $P(\text{opt})_{\text{shots}} = 0.0317$, $P(\text{opt})_{\text{SV}} = 0.0311$, Runtime = 11.8s
    - `reps=3, seed=123`: Obj = 18.716898, $P(\text{opt})_{\text{shots}} = 0.0278$, $P(\text{opt})_{\text{SV}} = 0.0319$, Runtime = 27.0s
- **Final Quantum Selection:** `['C0031', 'C0079', 'C0005', 'C0318', 'C0338', 'C0091']` (exact match to classical brute force).

---

## 8. Real Stage 3 Quantum Simulation Results (Official)

- **Deliverables:** [stage3_results.json](stage3_results.json), [stage3_results.csv](stage3_results.csv) at top level.
- **Design Justification:** No entanglement is needed because the 6 customers' subscription decisions are independent events (confirmed by competition design); each qubit independently models one customer's outcome, which the organizers confirmed is acceptable when the design choice is explained.
- **Qubit Mapping & Input Probabilities (Rank Order $\to$ Qubit Index):**
  - Qubit 0: `C0031` ($p = 0.380956$) $\to R_y(\theta_0 = 1.3308)$
  - Qubit 1: `C0079` ($p = 0.374971$) $\to R_y(\theta_1 = 1.3178)$
  - Qubit 2: `C0005` ($p = 0.241190$) $\to R_y(\theta_2 = 1.0264)$
  - Qubit 3: `C0318` ($p = 0.224620$) $\to R_y(\theta_3 = 0.9870)$
  - Qubit 4: `C0338` ($p = 0.201533$) $\to R_y(\theta_4 = 0.9298)$
  - Qubit 5: `C0091` ($p = 0.184674$) $\to R_y(\theta_5 = 0.8856)$
- **Circuit Architecture:**
  - 6 logical / physical qubits, depth 2, 7 single-qubit gates, 0 two-qubit gates, 0 SWAPs.
- **Benchmark Estimations of $P(\ge 3)$:**
  - **Exact Poisson-Binomial DP:** **0.195029**
  - **Monte Carlo (100,000 samples):** **0.193000** (Abs Error: 0.002029)
  - **Ideal Quantum Simulation across Shots (Seed 42):**
    - Shots = 512: $P(\ge 3) = 0.197266$ (Abs Error: 0.002237)
    - Shots = 1024: $P(\ge 3) = 0.202148$ (Abs Error: 0.007120)
    - Shots = 4096: $P(\ge 3) = 0.198486$ (Abs Error: 0.003458)
    - Shots = 16384: $P(\ge 3) = \mathbf{0.195679}$ (Abs Error: **0.000650**)
- **Noise Analysis (Depolarizing + 1.5% Readout Error, 8192 shots):**
  - Unmitigated $P(\ge 3) = 0.207764$ (Abs Error vs Exact: 0.012735)
- **Sensitivity & Error Propagation:**
  - $\delta = -0.05 \implies P(\ge 3) = 0.117362$ (Abs: $-0.077667$, **Relative: -39.82%**)
  - $\delta = -0.02 \implies P(\ge 3) = 0.161966$ (Abs: $-0.033063$, **Relative: -16.95%**)
  - $\delta = +0.02 \implies P(\ge 3) = 0.230414$ (Abs: $+0.035385$, **Relative: +18.14%**)
  - $\delta = +0.05 \implies P(\ge 3) = 0.287132$ (Abs: $+0.092103$, **Relative: +47.23%**)

---

## 9. Deliverables Summary (All Stages Complete)

| Stage | Deliverable Files (Top-Level) | Status | Description |
|---|---|---|---|
| **Stage 1** | `stage1_results.csv` | **Complete** | 100 customer subscription probabilities, ranks, top-10 flag |
| **Stage 2** | `stage2_results.json`, `stage2_results.csv` | **Complete** | QUBO / QAOA selection of 6 customers, comparison tables, metrics |
| **Stage 3** | `stage3_results.json`, `stage3_results.csv` | **Complete** | 6-qubit circuit simulation for $P(\ge 3)$, noise, error propagation |
