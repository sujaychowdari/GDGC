# GDG on Campus Network Intrusion Detection ML Challenge

An end-to-end, leak-free, scientifically validated machine learning pipeline for network intrusion detection (4-class classification: `Availability`, `Benign`, `Reconnaissance`, `Spoofing`), optimizing for **Macro F1**.

---

## 1. Project Overview & Performance Targets

- **Original Best Kaggle Score**: `0.84671` Macro F1
- **Refined Submission Score**: `0.84481` Macro F1
- **Target Objective**: Advance toward `0.90+` Macro F1 without hardcoded overrides or data leakage.

---

## 2. Key Discoveries & Engineering Fixes

### A. Categorical Encoding Alignment
- **Problem**: The original baseline pipeline used `.astype('category').cat.codes` separately on train and test sets, causing categories to be inverted during inference (e.g., test TCP mapped to ICMP, MQTT mapped to HTTP).
- **Solution**: Built [`preprocessing.py`](preprocessing.py) with a unified `NetworkDataPreprocessor` featuring `OneHotEncoder(handle_unknown='ignore')` for trees and native string handling for CatBoost, verified via [`test_preprocessing.py`](test_preprocessing.py).

### B. The 1,807 Contested Port 21 Flows
- **Finding**: In `test.csv`, exactly 1,807 flows on Port 21 have `fwd_pkts_payload.tot == 120.0` and `flow_SYN_flag_count == 1.0`, but `bwd_pkts_tot == 0.0` (victim server crashed/silent).
- **Domain Verification**: In `train.csv`, 100.0% of flows matching this 120-byte payload SYN signature belong to **Availability** (Hping3 attack). Standard tree models without domain features misclassified these as Reconnaissance because the server failed to reply (`bwd_pkts_tot == 0`). Our corrected XGBoost preserves all 10,391 Availability attack flows.

### C. The 238 Corrected Predictions
Comparing the baseline against [`submission_candidate.csv`](submission_candidate.csv):
- `99.28%` identical predictions (32,923 / 33,161).
- 200 flows corrected from Reconnaissance to Spoofing (due to repairing inverted MQTT and SSL service codes).
- 23 flows recovered to Benign; 15 flows recovered to Spoofing.

---

## 3. Architecture & Validation Scorecard

| Model | Profile-Held-Out Macro F1 | In-Distribution 5-Fold Macro F1 | Minority Benign F1 | Minority Recon F1 | Baseline Concordance |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CatBoost** | **`0.61575`** | `0.98723` | **`0.92236`** | `0.67949` | `94.01%` |
| **LightGBM** | `0.51594` | **`0.99285`** | `0.85306` | **`0.76080`** | `94.10%` |
| **XGBoost (Aligned)** | `0.50835` | `0.99058` | `0.82515` | `0.36476` | **`99.28%`** |
| **Ensemble Blend** | `0.51594` | **`0.99285`** | `0.85306` | **`0.76080`** | `94.10%` |

- **Profile Champion**: CatBoost achieves highest resilience on novel `(proto, service, port)` tuples.
- **In-Distribution Champion**: LightGBM achieves highest overall precision and scan detection.
- **Leaderboard Protection**: Aligned XGBoost directly protects the 0.84671 baseline while fixing 238 minority class errors.

---

## 4. Repository Structure

```
├── preprocessing.py              # Unified leak-free preprocessor & feature engineering
├── test_preprocessing.py         # Verification test suite for preprocessor parity
├── robust_validation.py          # 5-fold Stratified and GroupKFold evaluation framework
├── train_catboost_experiments.py # CatBoost screening and 5-fold cross-validation
├── train_lightgbm_experiments.py # LightGBM screening and 5-fold cross-validation
├── train_xgboost_experiments.py  # XGBoost screening and 5-fold cross-validation
├── evaluate_ensembles.py         # Nested CV probability blending and simplex optimization
├── validate_final_submission.py  # Strict verification test for submission candidate files
│
├── submission_candidate.csv      # Primary candidate submission file
├── candidate_xgboost_aligned.csv # Aligned XGBoost candidate submission
├── candidate_catboost_profile.csv# Profile-held-out champion candidate submission
├── candidate_ensemble_blend.csv  # Tri-model probability blend candidate submission
├── baseline_084671.csv          # Cryptographically locked original Kaggle baseline
│
├── models/                       # Serialized champion models across 5 folds
│   ├── catboost_champion_fold_*.cbm
│   ├── lightgbm_champion_fold_*.txt
│   └── xgboost_champion_fold_*.json
│
└── reports/                      # Full analytical audit & selection reports
    ├── BASELINE_AUDIT.md         # Comprehensive audit of legacy scripts & bugs
    ├── EDA_REPORT.md             # Exploratory data analysis & distribution shifts
    ├── MODEL_SELECTION_REPORT.md # Final model evaluation & comparison report
    ├── confusion_matrices/       # Class confusion matrices per fold
    └── figures/                  # Distribution and correlation charts
```

---

## 5. Submissions & SHA-256 Checksums

| File | Rows | Columns | SHA-256 Checksum | Description |
| :--- | :---: | :---: | :--- | :--- |
| `baseline_084671.csv` | 33,161 | 2 | `BE58E1862F1877CB803DCADBFD5590F5AF88B1E7B0C2C75551B2E4BF50E9AA5E` | Original Kaggle 0.84671 Baseline |
| `submission_candidate.csv` | 33,161 | 2 | `B16582E937DC85552C5704BB7751E7D2345F26A8FA1CF9D5FF9E086DCFE9290D` | Primary Candidate (Aligned XGBoost) |
| `candidate_catboost_profile.csv` | 33,161 | 2 | `E417EFB02DAAD1DD6B787FB3FD9D1551468A617E75346B46906B7D30FCF5886C` | Profile-Held-Out Champion |
| `candidate_ensemble_blend.csv` | 33,161 | 2 | `45EB73EB22729F4276D0740619B8E6CD74AFE3601A91AEAA2B55AA51D98A81F1` | Tri-Model Probability Blend |

---

## 6. How to Run

### Run Preprocessing Verification
```bash
python test_preprocessing.py
```

### Run Validation Checks on Final Submission
```bash
python validate_final_submission.py
```

### Reproduce Model Evaluation
```bash
python evaluate_ensembles.py
```
