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

### 4. Step 4 & 5: Data Inspection & Open Questions for Organizers
- **Status:** Analysis complete; awaiting organizer response.
- **Stage1_Train Analysis:** 300 rows, 16 features, target `y` has 35 "yes" (11.67%) and 265 "no" (88.33%). Heavy class imbalance. `pdays = -1` present in 254/300 rows (customer not previously contacted).
- **Stage1_Test Analysis:** 100 rows, 15 features, no `y` and no `customer_id`.
- **Open Questions for Hackathon Organizers:**
  1. **Missing `customer_id` Mapping:** `Stage1_Test` has 100 feature rows but no `customer_id`. `Stage2_Data` contains 400 IDs, of which exactly 100 IDs do not appear in `Stage1_Train`. Does the row order of `Stage1_Test` match these 100 unmatched IDs in ascending alphanumeric order, or will an updated file with IDs be provided?
  2. **`Stage2_Data` Customer Count Mismatch:** PDF Section 3 describes `Stage2_Data` as *"operational inputs for the 100 test customers"*, but the sheet contains **400 rows** (spanning all training and test customer IDs). Are we expected to filter `Stage2_Data` to only the 100 test customer IDs, or does the full 400-row table remain the operational lookup table?

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

## What Is Next
- Awaiting organizer response on the two open questions (`customer_id` mapping and `Stage2_Data` row count scope).
- Once confirmed:
  1. Fit selected model (`HistGradientBoosting_Calibrated`) on full `Stage1_Train`.
  2. Run single prediction on `Stage1_Test`.
  3. Produce `stage1_results.csv` with continuous probabilities, ranks, and top-10 flag.
  4. Proceed to Stage 2 QUBO formulation and quantum solver.
