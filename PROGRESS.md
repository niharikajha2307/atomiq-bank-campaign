# Atomiq Hackathon Project Progress

## Current Status: Stage 1 Training & 5-Fold Stratified Cross-Validation Complete

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
| **LogisticRegression_Balanced** | **0.6491 ± 0.086** | **0.2544 ± 0.067** | 0.1800 ± 0.045 | 0.2571 ± 0.064 | 0.2022 ± 0.033 | 0.5946 ± 0.088 | 0.2720 ± 0.054 |
| **HistGradientBoosting_Balanced** | 0.5951 ± 0.096 | 0.2467 ± 0.085 | **0.2000 ± 0.071** | **0.2857 ± 0.101** | 0.1256 ± 0.016 | 0.4580 ± 0.047 | 0.1053 ± 0.026 |
| **HistGradientBoosting_Calibrated**| 0.5358 ± 0.139 | 0.2112 ± 0.089 | 0.1800 ± 0.084 | 0.2571 ± 0.120 | **0.1024 ± 0.001** | **0.3574 ± 0.006** | **0.0064 ± 0.006** |
| **LogisticRegression_Calibrated** | 0.5644 ± 0.101 | 0.1821 ± 0.076 | 0.1200 ± 0.110 | 0.1714 ± 0.156 | 0.1026 ± 0.001 | 0.3582 ± 0.006 | **0.0035 ± 0.003** |
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

## What Is Next
- Awaiting organizer response on the two open questions (`customer_id` mapping and `Stage2_Data` row count scope).
- Once confirmed:
  1. Fit selected model (`HistGradientBoosting_Calibrated`) on full `Stage1_Train`.
  2. Run single prediction on `Stage1_Test`.
  3. Produce `stage1_results.csv` with continuous probabilities, ranks, and top-10 flag.
  4. Run Stage 2 QUBO on the true top-10 customers $\to$ `outputs/stage2_results.json`.
  5. Run Stage 3 circuit on the 6 chosen customers $\to$ `outputs/stage3_results.json`.
  6. Final README and presentation preparation.
