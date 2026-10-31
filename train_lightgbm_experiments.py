"""
LightGBM Multiclass Intrusion Detection Experimentation Framework
GDG on Campus Network Intrusion Detection Challenge

Key Objectives:
1. Systematic comparison across:
   - Number of leaves (num_leaves: 15, 31, 63, 127)
   - Maximum depth (max_depth: 4, 6, 8, -1)
   - Learning rate (learning_rate: 0.04, 0.08, 0.12)
   - Minimum child samples (min_child_samples: 10, 20, 50)
   - Class weights (None, Balanced, SqrtBalanced)
   - Feature subsampling (colsample_bytree: 0.70, 0.85, 1.0)
2. Uses exact same validation folds as CatBoost:
   - Stratified 5-Fold CV (random_state=42)
   - StratifiedGroupKFold on (proto, service, id.resp_p) (random_state=42)
3. Corrected leak-free preprocessing via NetworkDataPreprocessor(model_type='tree', categorical_encoding='onehot').
4. Full out-of-fold probability generation and per-class F1 reporting.
5. Strict preservation of baseline and previous experiments.
"""

import os
import sys
import time
import json
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from preprocessing import NetworkDataPreprocessor

warnings.filterwarnings('ignore', category=FutureWarning)

# Configure Paths
DATA_DIR = r"D:\Sujay\GDGC_AIML\network-intrusion-detection2-ml-challenge"
REPORTS_DIR = r"D:\Sujay\GDGC_AIML\reports"
MODELS_DIR = r"D:\Sujay\GDGC_AIML\models"
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Target Mapping
LABEL_MAPPING = {'Availability': 0, 'Benign': 1, 'Reconnaissance': 2, 'Spoofing': 3}
INV_MAPPING = {v: k for k, v in LABEL_MAPPING.items()}
CLASS_NAMES = ['Availability', 'Benign', 'Reconnaissance', 'Spoofing']


def load_dataset():
    print(f"Loading datasets from {DATA_DIR}...")
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    print(f"Loaded train: {train_df.shape}, test: {test_df.shape}")
    y = train_df['Target'].map(LABEL_MAPPING).values
    return train_df, test_df, y


def macro_f1_metric(y_true, y_pred):
    preds = y_pred.reshape(-1, 4)
    pred_labels = np.argmax(preds, axis=1)
    f1 = f1_score(y_true, pred_labels, average='macro', zero_division=0)
    return 'macro_f1', f1, True


def compute_sqrt_balanced_weights(y):
    """
    Computes sample weights proportional to square root of inverse class frequency.
    Prevents over-penalizing the dominant Availability class while boosting minority classes.
    """
    classes, counts = np.unique(y, return_counts=True)
    inv_sqrt = np.sqrt(len(y) / counts)
    inv_sqrt_normalized = inv_sqrt / np.mean(inv_sqrt)
    weight_map = dict(zip(classes, inv_sqrt_normalized))
    return np.array([weight_map[label] for label in y], dtype=np.float32)


def run_cv_experiment(name, params, train_df, y, splits, n_splits=5, is_profile_held_out=False):
    """
    Runs cross-validation with LightGBM using clean fold preprocessing.
    """
    print(f"\n=======================================================")
    print(f"Running Experiment: {name}")
    print(f"Parameters: {params}")
    print(f"Splits: {'Profile-Held-Out (StratifiedGroupKFold)' if is_profile_held_out else 'StratifiedKFold'}")
    print(f"=======================================================")
    
    oof_probs = np.zeros((len(train_df), 4), dtype=np.float32)
    fold_results = []
    
    use_sqrt_weights = params.get('use_sqrt_weights', False)
    clean_params = {k: v for k, v in params.items() if k not in ['use_sqrt_weights', 'early_stopping_rounds']}
    early_rounds = params.get('early_stopping_rounds', 35)
    
    t_start = time.time()
    for fold, (train_idx, val_idx) in enumerate(splits):
        f_start = time.time()
        
        train_slice = train_df.iloc[train_idx].copy()
        val_slice = train_df.iloc[val_idx].copy()
        y_train = y[train_idx]
        y_val = y[val_idx]
        
        # Fit preprocessor strictly on train slice (One-Hot for tree models)
        preprocessor = NetworkDataPreprocessor(model_type='tree', categorical_encoding='onehot')
        X_train = preprocessor.fit_transform(train_slice)
        X_val = preprocessor.transform(val_slice)
        
        # Determine sample weights if custom
        sample_weights = None
        if use_sqrt_weights:
            sample_weights = compute_sqrt_balanced_weights(y_train)
            
        clf = lgb.LGBMClassifier(
            objective='multiclass',
            num_class=4,
            random_state=42 + fold,
            verbose=-1,
            n_jobs=8,
            **clean_params
        )
        
        clf.fit(
            X_train, y_train,
            sample_weight=sample_weights,
            eval_set=[(X_val, y_val)],
            eval_metric=macro_f1_metric,
            callbacks=[early_stopping(stopping_rounds=early_rounds, verbose=False), log_evaluation(0)]
        )
        
        val_probs = clf.predict_proba(X_val)
        oof_probs[val_idx] = val_probs
        val_preds = np.argmax(val_probs, axis=1)
        
        # Metrics
        macro_f1 = f1_score(y_val, val_preds, average='macro', zero_division=0)
        class_f1 = f1_score(y_val, val_preds, average=None, labels=[0, 1, 2, 3], zero_division=0)
        
        f_dur = time.time() - f_start
        best_iter = clf.best_iteration_
        print(f"  Fold {fold+1}/{n_splits} | Macro F1: {macro_f1:.5f} | "
              f"Avail: {class_f1[0]:.4f}, Ben: {class_f1[1]:.4f}, Rec: {class_f1[2]:.4f}, Spf: {class_f1[3]:.4f} | "
              f"BestIter: {best_iter} | Time: {f_dur:.1f}s")
        
        fold_results.append({
            'fold': fold + 1,
            'macro_f1': macro_f1,
            'f1_availability': class_f1[0],
            'f1_benign': class_f1[1],
            'f1_reconnaissance': class_f1[2],
            'f1_spoofing': class_f1[3],
            'best_iteration': best_iter,
            'duration': f_dur
        })
    
    total_dur = time.time() - t_start
    
    # Calculate overall metrics
    overall_preds = np.argmax(oof_probs, axis=1)
    overall_macro_f1 = f1_score(y, overall_preds, average='macro', zero_division=0)
    overall_class_f1 = f1_score(y, overall_preds, average=None, labels=[0, 1, 2, 3], zero_division=0)
    overall_precision = precision_score(y, overall_preds, average='macro', zero_division=0)
    overall_recall = recall_score(y, overall_preds, average='macro', zero_division=0)
    
    mean_fold_f1 = np.mean([r['macro_f1'] for r in fold_results])
    std_fold_f1 = np.std([r['macro_f1'] for r in fold_results])
    
    print(f"\n>> {name} Overall Results:")
    print(f"   Overall OOF Macro F1: {overall_macro_f1:.5f} (Fold Mean: {mean_fold_f1:.5f} ± {std_fold_f1:.5f})")
    print(f"   Availability F1:     {overall_class_f1[0]:.5f}")
    print(f"   Benign F1:           {overall_class_f1[1]:.5f}")
    print(f"   Reconnaissance F1:   {overall_class_f1[2]:.5f}")
    print(f"   Spoofing F1:         {overall_class_f1[3]:.5f}")
    print(f"   Macro Precision:     {overall_precision:.5f} | Macro Recall: {overall_recall:.5f}")
    print(f"   Total Duration:      {total_dur:.1f}s")
    
    summary = {
        'experiment': name,
        'overall_macro_f1': overall_macro_f1,
        'mean_fold_f1': mean_fold_f1,
        'std_fold_f1': std_fold_f1,
        'f1_availability': overall_class_f1[0],
        'f1_benign': overall_class_f1[1],
        'f1_reconnaissance': overall_class_f1[2],
        'f1_spoofing': overall_class_f1[3],
        'macro_precision': overall_precision,
        'macro_recall': overall_recall,
        'duration_seconds': total_dur,
        'params': json.dumps(params)
    }
    
    return summary, fold_results, oof_probs


def main():
    train_df, test_df, y = load_dataset()
    
    # 1. Stratified 5-Fold Splits (Exact same folds as CatBoost: random_state=42)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    stratified_splits = list(skf.split(train_df, y))
    
    # 2. StratifiedGroupKFold on (proto, service, id.resp_p) (Exact same folds: random_state=42)
    groups = (train_df['proto'].astype(str) + "_" + 
              train_df['service'].astype(str) + "_" + 
              train_df['id.resp_p'].astype(str))
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    profile_splits = list(sgkf.split(train_df, y, groups=groups))
    
    # 3-Fold splits for systematic screening
    skf_3 = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    splits_screen = list(skf_3.split(train_df, y))
    
    experiments = [
        # --- Experiment 1: Baseline ---
        ("1. Baseline (Leaves=31, Depth=-1, Balanced)", {
            'n_estimators': 300, 'num_leaves': 31, 'max_depth': -1, 'learning_rate': 0.08,
            'min_child_samples': 20, 'class_weight': 'balanced', 'colsample_bytree': 0.85
        }),
        # --- Experiment 2 & 3: Class Weights ---
        ("2. Weights: None (Uniform)", {
            'n_estimators': 300, 'num_leaves': 31, 'max_depth': -1, 'learning_rate': 0.08,
            'min_child_samples': 20, 'class_weight': None, 'colsample_bytree': 0.85
        }),
        ("3. Weights: SqrtBalanced (Custom)", {
            'n_estimators': 300, 'num_leaves': 31, 'max_depth': -1, 'learning_rate': 0.08,
            'min_child_samples': 20, 'class_weight': None, 'use_sqrt_weights': True, 'colsample_bytree': 0.85
        }),
        # --- Experiment 4 & 5: Number of Leaves ---
        ("4. Leaves: 15 (Small / Constrained)", {
            'n_estimators': 300, 'num_leaves': 15, 'max_depth': -1, 'learning_rate': 0.08,
            'min_child_samples': 20, 'use_sqrt_weights': True, 'colsample_bytree': 0.85
        }),
        ("5. Leaves: 63 (Large / Complex)", {
            'n_estimators': 300, 'num_leaves': 63, 'max_depth': -1, 'learning_rate': 0.08,
            'min_child_samples': 20, 'use_sqrt_weights': True, 'colsample_bytree': 0.85
        }),
        ("6. Leaves: 127 (Very Large)", {
            'n_estimators': 300, 'num_leaves': 127, 'max_depth': -1, 'learning_rate': 0.08,
            'min_child_samples': 20, 'use_sqrt_weights': True, 'colsample_bytree': 0.85
        }),
        # --- Experiment 7 & 8: Maximum Depth ---
        ("7. Max Depth: 4 (Shallow)", {
            'n_estimators': 300, 'num_leaves': 15, 'max_depth': 4, 'learning_rate': 0.08,
            'min_child_samples': 20, 'use_sqrt_weights': True, 'colsample_bytree': 0.85
        }),
        ("8. Max Depth: 8 (Controlled)", {
            'n_estimators': 300, 'num_leaves': 63, 'max_depth': 8, 'learning_rate': 0.08,
            'min_child_samples': 20, 'use_sqrt_weights': True, 'colsample_bytree': 0.85
        }),
        # --- Experiment 9 & 10: Learning Rate ---
        ("9. LR: 0.04 (Low, 500 trees)", {
            'n_estimators': 500, 'num_leaves': 31, 'max_depth': -1, 'learning_rate': 0.04,
            'min_child_samples': 20, 'use_sqrt_weights': True, 'colsample_bytree': 0.85
        }),
        ("10. LR: 0.12 (High, 250 trees)", {
            'n_estimators': 250, 'num_leaves': 31, 'max_depth': -1, 'learning_rate': 0.12,
            'min_child_samples': 20, 'use_sqrt_weights': True, 'colsample_bytree': 0.85
        }),
        # --- Experiment 11 & 12: Minimum Child Samples ---
        ("11. Min Child Samples: 10 (Fine Granularity)", {
            'n_estimators': 300, 'num_leaves': 31, 'max_depth': -1, 'learning_rate': 0.08,
            'min_child_samples': 10, 'use_sqrt_weights': True, 'colsample_bytree': 0.85
        }),
        ("12. Min Child Samples: 50 (Strong Regularization)", {
            'n_estimators': 300, 'num_leaves': 31, 'max_depth': -1, 'learning_rate': 0.08,
            'min_child_samples': 50, 'use_sqrt_weights': True, 'colsample_bytree': 0.85
        }),
        # --- Experiment 13 & 14: Feature Subsampling ---
        ("13. Subsampling: 0.70 (High Diversity)", {
            'n_estimators': 300, 'num_leaves': 31, 'max_depth': -1, 'learning_rate': 0.08,
            'min_child_samples': 20, 'use_sqrt_weights': True, 'colsample_bytree': 0.70
        }),
        ("14. Subsampling: 1.0 (Full Features)", {
            'n_estimators': 300, 'num_leaves': 31, 'max_depth': -1, 'learning_rate': 0.08,
            'min_child_samples': 20, 'use_sqrt_weights': True, 'colsample_bytree': 1.0
        }),
    ]
    
    screen_summaries = []
    print("\n=======================================================")
    print("STAGE 1: SYSTEMATIC HYPERPARAMETER SCREENING (3-Fold CV)")
    print("=======================================================")
    
    for name, params in experiments:
        summary, _, _ = run_cv_experiment(name, params, train_df, y, splits_screen, n_splits=3)
        screen_summaries.append(summary)
    
    screen_df = pd.DataFrame(screen_summaries)
    screen_csv = os.path.join(REPORTS_DIR, "lightgbm_screen_results.csv")
    screen_df.to_csv(screen_csv, index=False)
    print(f"\nScreening results saved to {screen_csv}")
    
    # Sort and identify top configuration
    top_configs = screen_df.sort_values(by='overall_macro_f1', ascending=False)
    print("\nTop LightGBM Screening Configurations:")
    print(top_configs[['experiment', 'overall_macro_f1', 'f1_benign', 'f1_reconnaissance', 'f1_spoofing', 'duration_seconds']])
    
    best_config_row = top_configs.iloc[0]
    best_params = json.loads(best_config_row['params'])
    best_name = f"Champion: {best_config_row['experiment']}"
    
    print("\n=======================================================")
    print(f"STAGE 2: FULL 5-FOLD CV ON CHAMPION MODEL")
    print(f"Champion: {best_name}")
    print("=======================================================")
    
    # 1. Full 5-Fold Stratified CV
    champ_summary, champ_folds, champ_oof_probs = run_cv_experiment(
        f"{best_name} (Stratified 5-Fold)", best_params, train_df, y, stratified_splits, n_splits=5
    )
    
    # Save OOF Probabilities
    oof_df = pd.DataFrame(champ_oof_probs, columns=[f"prob_{c}" for c in CLASS_NAMES])
    oof_df['id'] = train_df['id']
    oof_df['true_target'] = train_df['Target']
    oof_df['pred_target'] = [INV_MAPPING[p] for p in np.argmax(champ_oof_probs, axis=1)]
    oof_csv = os.path.join(REPORTS_DIR, "oof_probabilities_lightgbm.csv")
    oof_df.to_csv(oof_csv, index=False)
    print(f"OOF probabilities saved to {oof_csv}")
    
    # Confusion matrix
    cm = confusion_matrix(y, np.argmax(champ_oof_probs, axis=1))
    cm_df = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
    cm_csv = os.path.join(REPORTS_DIR, "confusion_matrices", "cm_lightgbm_champion_oof.csv")
    cm_df.to_csv(cm_csv)
    print(f"Champion confusion matrix:\n{cm_df}")
    
    # 2. Profile-Held-Out Evaluation on Champion Model
    print("\n=======================================================")
    print("STAGE 3: PROFILE-HELD-OUT EVALUATION (StratifiedGroupKFold)")
    print("=======================================================")
    profile_summary, profile_folds, profile_oof_probs = run_cv_experiment(
        f"{best_name} (Profile-Held-Out SGKF)", best_params, train_df, y, profile_splits, n_splits=5, is_profile_held_out=True
    )
    profile_csv = os.path.join(REPORTS_DIR, "lightgbm_profile_held_out.csv")
    pd.DataFrame(profile_folds).to_csv(profile_csv, index=False)
    print(f"Profile-held-out fold metrics saved to {profile_csv}")
    
    # 3. Train on full 5 folds and predict on test.csv (generate test probabilities)
    print("\n=======================================================")
    print("STAGE 4: TEST SET INFERENCE VIA OUT-OF-FOLD ENSEMBLE")
    print("=======================================================")
    test_probs = np.zeros((len(test_df), 4), dtype=np.float32)
    use_sqrt_weights = best_params.get('use_sqrt_weights', False)
    clean_params = {k: v for k, v in best_params.items() if k not in ['use_sqrt_weights', 'early_stopping_rounds']}
    early_rounds = best_params.get('early_stopping_rounds', 35)
    
    for fold, (train_idx, val_idx) in enumerate(stratified_splits):
        print(f"Training fold {fold+1} for test inference...")
        train_slice = train_df.iloc[train_idx].copy()
        val_slice = train_df.iloc[val_idx].copy()
        y_train = y[train_idx]
        y_val = y[val_idx]
        
        preprocessor = NetworkDataPreprocessor(model_type='tree', categorical_encoding='onehot')
        X_train = preprocessor.fit_transform(train_slice)
        X_val = preprocessor.transform(val_slice)
        X_test = preprocessor.transform(test_df)
        
        sample_weights = None
        if use_sqrt_weights:
            sample_weights = compute_sqrt_balanced_weights(y_train)
            
        clf = lgb.LGBMClassifier(
            objective='multiclass',
            num_class=4,
            random_state=42 + fold,
            verbose=-1,
            n_jobs=8,
            **clean_params
        )
        clf.fit(
            X_train, y_train,
            sample_weight=sample_weights,
            eval_set=[(X_val, y_val)],
            eval_metric=macro_f1_metric,
            callbacks=[early_stopping(stopping_rounds=early_rounds, verbose=False), log_evaluation(0)]
        )
        
        # Save model
        model_path = os.path.join(MODELS_DIR, f"lightgbm_champion_fold_{fold+1}.txt")
        clf.booster_.save_model(model_path)
        
        test_probs += clf.predict_proba(X_test) / 5.0
    
    # Save test predictions independently
    test_pred_indices = np.argmax(test_probs, axis=1)
    test_preds_df = pd.DataFrame({
        'id': test_df['id'],
        'Target': [INV_MAPPING[idx] for idx in test_pred_indices]
    })
    for i, c in enumerate(CLASS_NAMES):
        test_preds_df[f'prob_{c}'] = test_probs[:, i]
        
    test_pred_csv = os.path.join(REPORTS_DIR, "test_predictions_lightgbm.csv")
    test_preds_df.to_csv(test_pred_csv, index=False)
    print(f"Test predictions and probabilities saved to {test_pred_csv}")
    print("Predicted class distribution on test set:")
    print(test_preds_df['Target'].value_counts())
    
    # Save master summary of all experiments
    all_summaries = screen_summaries + [champ_summary, profile_summary]
    master_df = pd.DataFrame(all_summaries)
    master_csv = os.path.join(REPORTS_DIR, "lightgbm_experiments.csv")
    master_df.to_csv(master_csv, index=False)
    print(f"\nMaster experimental results saved to {master_csv}")
    print("ALL LIGHTGBM EXPERIMENTS COMPLETED SUCCESSFULLY.")


if __name__ == '__main__':
    main()
