# GDG on Campus Network Intrusion Detection — Baseline Audit Report

**Date & Time:** 2026-10-30 22:30 IST  
**Workspace:** `D:\Sujay\GDGC_AIML`  
**Dataset Directory:** `D:\Sujay\GDGC_AIML\network-intrusion-detection2-ml-challenge`  
**Current Best Kaggle Score:** **0.84671** Macro F1  
**Refined Submission Score:** **0.84481** Macro F1  
**Target:** **0.90+** Macro F1  

---

## 1. Inventory of All Existing Python Scripts

Below is the complete inventory of all Python scripts currently residing in `D:\Sujay\GDGC_AIML`:

| Script Name | Size (Bytes) | Category | Description & Purpose |
| :--- | :---: | :---: | :--- |
| `train_ensemble.py` | 9,452 | **Training Pipeline** | **The Original Pipeline:** 5-Fold Stratified CV ensemble of CatBoost + XGBoost with feature engineering and Port 21 SYN flood server-dropout augmentation. Produced `submission_ensemble.py.csv` (**0.84671**). |
| `baseline_cv.py` | 1,730 | Baseline / Validation | Initial 5-Fold StratifiedKFold test using LightGBM on raw features. |
| `check_test_preds.py` | 2,836 | Model Evaluation | 5-Fold CatBoost without data augmentation; demonstrated that raw models misclassify ~9,000 Port 21 test rows as Benign/Reconnaissance due to server response drops. |
| `test_augmentation.py` | 3,215 | Experimentation | Single-model CatBoost script testing partial backward-feature zeroing on Port 21 Availability samples. |
| `test_precise_aug.py` | 2,498 | Experimentation | Refined augmentation script isolating the 60 exact zero-valued columns in unresponsive test flows. |
| `test_p21_preds.py` | 365 | Diagnostics | Short snippet checking Port 21 prediction cross-tabulations. |
| `eda.py` | 2,064 | Exploratory Analysis | Initial dataset shapes, target distribution, missing values, and column data types. |
| `deep_eda.py` | 2,143 | Exploratory Analysis | Distribution statistics of packet rates, flow duration, SYN/FIN/ACK flag counts. |
| `explore_features.py` | 1,906 | Exploratory Analysis | Port frequencies, protocol breakdowns, and target-by-port cross-tabulations. |
| `cluster_traffic.py` | 2,093 | Exploratory Analysis | Target traffic profile characteristics (medians, IQRs) and Port 80 HTTP attack behaviors. |
| `deep_port_inspection.py`| 2,616 | Exploratory Analysis | Statistical comparisons between train and test across top ports (53, 80, 443, 5353). |
| `inspect_port21.py` | 1,060 | Exploratory Analysis | Feature comparison between Port 21 in train vs. test; revealed server response drop. |
| `compare_p21.py` | 1,400 | Exploratory Analysis | Feature-by-feature divergence between train and test Port 21 records. |
| `check_p21_details.py` | 1,404 | Exploratory Analysis | Proved payload (120 bytes) and window size (64) match identical `hping3` parameters across train and test. |
| `diagnose_p21.py` | 1,317 | Diagnostics | Proved train contains 0 Benign records on Port 21, yet unaugmented models predicted Benign on test Port 21. |
| `inspect_other_ports.py`| 1,188 | Exploratory Analysis | Analyzed the 6,881 non-standard port flows in test (identifying NMAP TCP/FIN/XMAS probe scans). |
| `investigate_features.py`| 1,534 | Validation Research | Adversarial validation classifier distinguishing train vs. test (achieved AUC 0.9999). |
| `audit_issue_a.py` | 1,165 | Audit Utility | Verifies the categorical code desynchronization between train and test in XGBoost. |
| `audit_issue_b.py` | 2,533 | Audit Utility | Quantifies destination port overlap and identifies that 17.65% of test flows operate on unseen ports. |
| `audit_issue_c.py` | 2,420 | Audit Utility | Pinpoints the 1,773 exact prediction differences between the 0.84671 baseline and 0.84481 refined submission. |
| `audit_issue_e.py` | 5,645 | Audit Utility | Evaluates mathematical consistency of engineered features on augmented rows. |
| `audit_issue_f.py` | 2,975 | Audit Utility | Analyzes ID structure, duplicate feature vectors, and cross-dataset feature leakage. |
| `compute_hashes.py` | 1,605 | Audit Utility | Computes SHA256 hashes and schema validations for all CSV submission files. |

---

## 2. Identification of the Original Training Pipeline (Score: 0.84671)

The pipeline responsible for the **0.84671** Kaggle Macro F1 score is implemented in **`train_ensemble.py`**:

- **Execution Output:** Generated `submission_ensemble.py.csv` (1,390,121 bytes, created at 19:57:19 IST).
- **Core Pipeline Components:**
  1. **Feature Engineering:** Computes 21 domain features including forward/backward payload-per-packet ratios, header-per-packet ratios, binary payload presence, TCP flag combination signatures (`is_syn_only`, `is_fin_only`, `is_xmas_scan`), port indicators, and log-transformed rates.
  2. **Data Augmentation:** Samples 35% of `Availability` instances on Port 21 (27,894 records) and sets responder-dependent features (backward packet counts, backward payloads, backward IAT, duration, rate, and ACK flags) to 0.0 to simulate an overwhelmed server dropping responses during SYN floods.
  3. **CatBoost Component:** 5-Fold StratifiedKFold, 450 iterations, learning rate 0.07, max depth 6, `auto_class_weights='Balanced'`, categorical features `['proto', 'service']`. Out-of-fold Macro F1: `0.97878`.
  4. **XGBoost Component:** 5-Fold StratifiedKFold, 350 trees, learning rate 0.07, max depth 6, `tree_method='hist'`, balanced sample weights. Out-of-fold Macro F1: `0.99077`.
  5. **Ensemble Blending:** Blends predicted class probabilities (50% CatBoost + 50% XGBoost) and selects the argmax class.

---

## 3. Identification of the Refined Pipeline (Score: 0.84481)

The refined pipeline generated **`submission_v2.csv`** (1,400,871 bytes, created at 21:12:25 IST) and previously overwrote `network-intrusion-detection2-ml-challenge/submission.csv`.

- **Nature of the Refinement:** Post-processing rule overrides applied on top of the original model predictions.
- **Specific Overrides Applied:**
  1. **Port 5353 Destination Override:** Forced **1,765 test records** where `id.resp_p == 5353.0` to be classified as `Reconnaissance` (overriding 1,692 baseline `Spoofing` predictions and 73 baseline `Benign` predictions).
  2. **Port 21 Destination Override:** Forced **5 test records** where `id.resp_p == 21.0` to be classified as `Availability` (overriding 5 baseline `Reconnaissance` predictions).
  3. **Port 5353 Source Override:** Forced **80 test records** where `id.orig_p == 5353.0` to be classified as `Reconnaissance` (overriding 76 baseline `Benign` predictions).
- **Why the Score Dropped (0.84671 $\rightarrow$ 0.84481):**
  - In `train.csv`, Port 5353 contains **0** Reconnaissance records (only Spoofing and Benign mDNS). Overriding all test mDNS flows to Reconnaissance created a massive spike of false positives, degrading both Reconnaissance precision and Spoofing recall.
  - On Port 21, the 5 overridden records had `fwd_pkts_payload.tot == 0.0` and originated from scanner source ports (`36242`, `59247`, `51036`, `46600`). The baseline model had correctly predicted them as `Reconnaissance` scans; forcing them to `Availability` corrupted valid predictions.

---

## 4. Comprehensive Submission Files Inventory & SHA256 Hashes

Every submission file in the project was hashed using SHA256 and verified for schema compliance:

| Submission File Path | File Size | SHA256 Hash | Status / Version |
| :--- | :---: | :---: | :--- |
| `D:\Sujay\GDGC_AIML\baseline_084671.csv` | 1,390,121 B | `be58e1862f1877cb803dcadbfd5590f5af88b1e7b0c2c75551b2e4bf50e9aa5e` | **Original Baseline (0.84671)** |
| `D:\Sujay\GDGC_AIML\submission_ensemble.py.csv` | 1,390,121 B | `be58e1862f1877cb803dcadbfd5590f5af88b1e7b0c2c75551b2e4bf50e9aa5e` | **Original Baseline (0.84671)** |
| `D:\Sujay\GDGC_AIML\submission.csv` | 1,390,121 B | `be58e1862f1877cb803dcadbfd5590f5af88b1e7b0c2c75551b2e4bf50e9aa5e` | **Restored Baseline (0.84671)** |
| `D:\Sujay\GDGC_AIML\submission_v2.csv` | 1,400,871 B | `f1d6f00946fc712e4b88179f531b1422a45a386503f21dc790d90612f80acd06` | Refined Version (0.84481) |
| `D:\Sujay\GDGC_AIML\network-intrusion-detection2-ml-challenge\submission.csv` | 1,400,871 B | `f1d6f00946fc712e4b88179f531b1422a45a386503f21dc790d90612f80acd06` | Refined Version (0.84481) |
| `D:\Sujay\GDGC_AIML\network-intrusion-detection2-ml-challenge\sample_submission.csv` | 1,260,129 B | `f5b124659f85a63beb700a4302f1a1f49f7a0a3fc1b6e6f31b2f8880620515df` | Kaggle Benchmark Template |

---

## 5. Schema, Row Count & Label Verification

All submission files were verified against `test.csv` (33,161 rows, 83 columns):

- **Row Count Verification:** Exactly **33,161 prediction rows** (+ 1 header row = 33,162 total lines) across all files. Pass.
- **ID Alignment Verification:** Every single ID matches `test.csv` in exact sequential order (`(df['id'] == test_df['id']).all() == True`). Pass.
- **Missing / Null Values:** **0 nulls** across all submission files. Pass.
- **Target Label Validity:** All labels are strictly members of the 4 permissible classes: `{'Availability', 'Benign', 'Reconnaissance', 'Spoofing'}`. Pass.

---

## 6. Class Distribution Breakdown

| Target Label | Baseline (0.84671) Count | Baseline Share (%) | Refined (0.84481) Count | Refined Share (%) | Net Change |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Benign** | 10,427 | 31.44% | 10,351 | 31.21% | -76 |
| **Availability** | 10,391 | 31.34% | 10,396 | 31.35% | +5 |
| **Reconnaissance** | 7,160 | 21.59% | 8,923 | 26.91% | **+1,763** |
| **Spoofing** | 5,183 | 15.63% | 3,491 | 10.53% | **-1,692** |
| **Total** | **33,161** | **100.00%** | **33,161** | **100.00%** | 0 |

---

## 7. Prediction Discrepancy Analysis (Original vs. Refined)

Comparing `baseline_084671.csv` against `submission_v2.csv` reveals:
- **Identical Predictions:** **31,388 rows (94.65%)**
- **Differing Predictions:** **1,773 rows (5.35%)**

### Cross-Tabulation Matrix of Differing Rows:

```
                          Refined Prediction (0.84481)
Baseline (0.84671)      Availability    Reconnaissance    Total Changed
-----------------------------------------------------------------------
Benign                        0                76               76
Reconnaissance                5                 0                5
Spoofing                      0             1,692            1,692
-----------------------------------------------------------------------
Total Changed                 5             1,768            1,773
```

### Breakdown of Ports Where Discrepancies Occur:
1. **Destination Port 5353.0 (mDNS):** 1,765 changed records.
   - 1,692 records flipped from `Spoofing` $\rightarrow$ `Reconnaissance`.
   - 73 records flipped from `Benign` $\rightarrow$ `Reconnaissance`.
2. **Destination Port 21.0 (FTP):** 5 changed records.
   - 5 records flipped from `Reconnaissance` $\rightarrow$ `Availability`.
3. **Other Destination Ports (143, 53, 8084):** 3 changed records.

---

## 8. Summary of Confirmed Technical Bugs in Existing Pipeline

1. **XGBoost Categorical Encoding Bug:**
   - In `train_ensemble.py`, `augmented_train[col].astype('category').cat.codes` and `test_fe[col].astype('category').cat.codes` were executed separately.
   - Because `icmp` was present in train but absent in test, `proto` codes desynchronized: test TCP became code 0 (ICMP in train), test UDP became code 1 (TCP in train).
   - Because `service` had 9 categories in train and 5 in test, DNS, HTTP, MQTT, and SSL mapped to completely wrong services in test.
   - **XGBoost's test predictions were corrupted by invalid categorical mapping, degrading the ensemble.**
2. **Test-Set Snooping in Augmentation:**
   - In `train_ensemble.py`, the list of zeroed columns was derived by filtering `test_df`. Preprocessing must be independent of test data inspection.
3. **Mathematical Inconsistency on Augmented Features:**
   - `flow_pkts_payload.min` was zeroed out during augmentation, but for a 1-packet forward flow of 120 bytes, the minimum payload is mathematically 120.0, which created an artificial mismatch with test data.
4. **Validation Leakage & Over-optimism:**
   - Standard StratifiedKFold cross-validation scores ~0.987 F1 due to port memorization, masking the fact that 17.65% of test records run on unseen ports.

---

## 9. Baseline Protection Confirmation

- **Baseline File Created:** `D:\Sujay\GDGC_AIML\baseline_084671.csv` (SHA256: `be58e1862f1877cb803dcadbfd5590f5af88b1e7b0c2c75551b2e4bf50e9aa5e`).
- **Submission Restored:** `D:\Sujay\GDGC_AIML\submission.csv` matches the 0.84671 baseline hash.
- **Lock State:** No files have been deleted or overwritten. All future experimental models will write to new output files without touching `baseline_084671.csv` or `submission.csv`.
