"""
CatBoost Multiclass Intrusion Detection Experimentation Framework
GDG on Campus Network Intrusion Detection Challenge

Key Objectives:
1. Systematic comparison across:
   - Class weights (None, Balanced, SqrtBalanced, Custom Macro)
   - Tree depth (4, 6, 8)
   - Learning rates (0.04, 0.08, 0.12)
   - Iterations & Early Stopping on Macro F1
   - L2 Regularization (1.0, 5.0, 12.0)
2. Direct optimization for Macro F1 (eval_metric='TotalF1:average=Macro').
3. Evaluation across both:
   - Stratified 5-Fold CV (in-distribution stability & OOF probabilities)
   - Profile-Held-Out CV (StratifiedGroupKFold on proto, service, id.resp_p)
4. Full out-of-fold probability generation and per-class F1 reporting.
5. Zero leakage: NetworkDataPreprocessor fit strictly on training fold.
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import catboost as cb
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, classification_report
from preprocessing import NetworkDataPreprocessor

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


def run_cv_experiment(name, params, train_df, y, splits, n_splits=5, is_profile_held_out=False):
    """
    Runs cross-validation with CatBoost using clean fold preprocessing.
    """
    print(f"\n=======================================================")
    print(f"Running Experiment: {name}")
    print(f"Parameters: {params}")
    print(f"Splits: {'Profile-Held-Out (StratifiedGroupKFold)' if is_profile_held_out else 'StratifiedKFold'}")
    print(f"=======================================================")
    
    oof_probs = np.zeros((len(train_df), 4), dtype=np.float32)
    fold_results = []
    
    t_start = time.time()
    for fold, (train_idx, val_idx) in enumerate(splits):
        f_start = time.time()
        
        train_slice = train_df.iloc[train_idx].copy()
        val_slice = train_df.iloc[val_idx].copy()
        y_train = y[train_idx]
        y_val = y[val_idx]
        
        # Check for missing classes in validation fold
        val_classes_present = set(y_val)
        all_classes = set(range(4))
        missing_classes = all_classes - val_classes_present
        
        # Fit preprocessor strictly on train slice
        preprocessor = NetworkDataPreprocessor(model_type='catboost', categorical_encoding='passthrough')
        X_train = preprocessor.fit_transform(train_slice)
        X_val = preprocessor.transform(val_slice)
        
        cat_features = ['proto', 'service']
        
        # Build CatBoost parameters
        cb_params = {
            'loss_function': 'MultiClass',
            'eval_metric': 'TotalF1:average=Macro',
            'random_seed': 42 + fold,
            'verbose': False,
            'thread_count': 8,
            'cat_features': cat_features,
            **params
        }
        
        early_stopping = params.get('early_stopping_rounds', 40)
        
        clf = cb.CatBoostClassifier(**cb_params)
        clf.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=early_stopping, verbose=False)
        
        # Predict probabilities
        val_probs = clf.predict_proba(X_val)
        oof_probs[val_idx] = val_probs
        val_preds = np.argmax(val_probs, axis=1)
        
        # Metrics
        macro_f1 = f1_score(y_val, val_preds, average='macro', zero_division=0)
        class_f1 = f1_score(y_val, val_preds, average=None, labels=[0, 1, 2, 3], zero_division=0)
        
        f_dur = time.time() - f_start
        best_iter = clf.get_best_iteration()
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
            'duration': f_dur,
            'missing_classes': list(missing_classes)
        })
    
    total_dur = time.time() - t_start
    
    # Calculate overall metrics
    # In profile-held-out, some samples might not be scored if custom split, but here all N samples are covered
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
    
    # 1. Stratified 5-Fold Splits
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    stratified_splits = list(skf.split(train_df, y))
    
    # 2. StratifiedGroupKFold (Profile-Held-Out on proto, service, id.resp_p)
    groups = (train_df['proto'].astype(str) + "_" + 
              train_df['service'].astype(str) + "_" + 
              train_df['id.resp_p'].astype(str))
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    profile_splits = list(sgkf.split(train_df, y, groups=groups))
    
    # Define Parameter Grid for Systematic Comparison
    # Using 3 folds for hyperparameter screen to rapidly explore grid, then 5 folds for top configs
    skf_3 = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    splits_screen = list(skf_3.split(train_df, y))
    
    experiments = [
        # --- Experiment Set 1: Class Weights ---
        ("1. Weights: None (Uniform)", {
            'iterations': 400, 'depth': 6, 'learning_rate': 0.08, 'l2_leaf_reg': 3.0,
            'auto_class_weights': None
        }),
        ("2. Weights: Balanced (Inverse Freq)", {
            'iterations': 400, 'depth': 6, 'learning_rate': 0.08, 'l2_leaf_reg': 3.0,
            'auto_class_weights': 'Balanced'
        }),
        ("3. Weights: SqrtBalanced", {
            'iterations': 400, 'depth': 6, 'learning_rate': 0.08, 'l2_leaf_reg': 3.0,
            'auto_class_weights': 'SqrtBalanced'
        }),
        ("4. Weights: Custom Macro-Tuned", {
            'iterations': 400, 'depth': 6, 'learning_rate': 0.08, 'l2_leaf_reg': 3.0,
            'class_weights': [0.15, 1.0, 1.2, 1.5]
        }),
        
        # --- Experiment Set 2: Tree Depth ---
        ("5. Depth: 4 (Shallow)", {
            'iterations': 400, 'depth': 4, 'learning_rate': 0.08, 'l2_leaf_reg': 3.0,
            'auto_class_weights': 'SqrtBalanced'
        }),
        ("6. Depth: 8 (Deep)", {
            'iterations': 400, 'depth': 8, 'learning_rate': 0.08, 'l2_leaf_reg': 3.0,
            'auto_class_weights': 'SqrtBalanced'
        }),
        
        # --- Experiment Set 3: Learning Rate & Iterations ---
        ("7. LR: 0.04 (Low, 600 iters)", {
            'iterations': 600, 'depth': 6, 'learning_rate': 0.04, 'l2_leaf_reg': 3.0,
            'auto_class_weights': 'SqrtBalanced'
        }),
        ("8. LR: 0.12 (High, 300 iters)", {
            'iterations': 300, 'depth': 6, 'learning_rate': 0.12, 'l2_leaf_reg': 3.0,
            'auto_class_weights': 'SqrtBalanced'
        }),
        
        # --- Experiment Set 4: L2 Regularization ---
        ("9. L2 Reg: 1.0 (Low)", {
            'iterations': 400, 'depth': 6, 'learning_rate': 0.08, 'l2_leaf_reg': 1.0,
            'auto_class_weights': 'SqrtBalanced'
        }),
        ("10. L2 Reg: 12.0 (High)", {
            'iterations': 400, 'depth': 6, 'learning_rate': 0.08, 'l2_leaf_reg': 12.0,
            'auto_class_weights': 'SqrtBalanced'
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
    screen_csv = os.path.join(REPORTS_DIR, "catboost_screen_results.csv")
    screen_df.to_csv(screen_csv, index=False)
    print(f"\nScreening results saved to {screen_csv}")
    
    # Sort and identify top 2 configurations
    top_configs = screen_df.sort_values(by='overall_macro_f1', ascending=False)
    print("\nTop Screening Configurations:")
    print(top_configs[['experiment', 'overall_macro_f1', 'f1_benign', 'f1_reconnaissance', 'f1_spoofing']])
    
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
    oof_csv = os.path.join(REPORTS_DIR, "oof_probabilities_catboost.csv")
    oof_df.to_csv(oof_csv, index=False)
    print(f"OOF probabilities saved to {oof_csv}")
    
    # Confusion matrix
    cm = confusion_matrix(y, np.argmax(champ_oof_probs, axis=1))
    cm_df = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
    cm_csv = os.path.join(REPORTS_DIR, "confusion_matrices", "cm_catboost_champion_oof.csv")
    cm_df.to_csv(cm_csv)
    print(f"Champion confusion matrix:\n{cm_df}")
    
    # 2. Profile-Held-Out Evaluation on Champion Model
    print("\n=======================================================")
    print("STAGE 3: PROFILE-HELD-OUT EVALUATION (StratifiedGroupKFold)")
    print("=======================================================")
    profile_summary, profile_folds, profile_oof_probs = run_cv_experiment(
        f"{best_name} (Profile-Held-Out SGKF)", best_params, train_df, y, profile_splits, n_splits=5, is_profile_held_out=True
    )
    profile_csv = os.path.join(REPORTS_DIR, "catboost_profile_held_out.csv")
    pd.DataFrame(profile_folds).to_csv(profile_csv, index=False)
    print(f"Profile-held-out fold metrics saved to {profile_csv}")
    
    # 3. Train on full 5 folds and predict on test.csv (generate test probabilities)
    print("\n=======================================================")
    print("STAGE 4: TEST SET INFERENCE VIA OUT-OF-FOLD ENSEMBLE")
    print("=======================================================")
    test_probs = np.zeros((len(test_df), 4), dtype=np.float32)
    
    for fold, (train_idx, val_idx) in enumerate(stratified_splits):
        print(f"Training fold {fold+1} for test inference...")
        train_slice = train_df.iloc[train_idx].copy()
        val_slice = train_df.iloc[val_idx].copy()
        y_train = y[train_idx]
        y_val = y[val_idx]
        
        preprocessor = NetworkDataPreprocessor(model_type='catboost', categorical_encoding='passthrough')
        X_train = preprocessor.fit_transform(train_slice)
        X_val = preprocessor.transform(val_slice)
        X_test = preprocessor.transform(test_df)
        
        cb_params = {
            'loss_function': 'MultiClass',
            'eval_metric': 'TotalF1:average=Macro',
            'random_seed': 42 + fold,
            'verbose': False,
            'thread_count': 8,
            'cat_features': ['proto', 'service'],
            **best_params
        }
        clf = cb.CatBoostClassifier(**cb_params)
        clf.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=best_params.get('early_stopping_rounds', 40), verbose=False)
        
        # Save model
        model_path = os.path.join(MODELS_DIR, f"catboost_champion_fold_{fold+1}.cbm")
        clf.save_model(model_path)
        
        test_probs += clf.predict_proba(X_test) / 5.0
    
    # Save test predictions independently
    test_pred_indices = np.argmax(test_probs, axis=1)
    test_preds_df = pd.DataFrame({
        'id': test_df['id'],
        'Target': [INV_MAPPING[idx] for idx in test_pred_indices]
    })
    for i, c in enumerate(CLASS_NAMES):
        test_preds_df[f'prob_{c}'] = test_probs[:, i]
        
    test_pred_csv = os.path.join(REPORTS_DIR, "test_predictions_catboost.csv")
    test_preds_df.to_csv(test_pred_csv, index=False)
    print(f"Test predictions and probabilities saved to {test_pred_csv}")
    print("Predicted class distribution on test set:")
    print(test_preds_df['Target'].value_counts())
    
    # Save master summary of all experiments
    all_summaries = screen_summaries + [champ_summary, profile_summary]
    master_df = pd.DataFrame(all_summaries)
    master_csv = os.path.join(REPORTS_DIR, "catboost_experiments.csv")
    master_df.to_csv(master_csv, index=False)
    print(f"\nMaster experimental results saved to {master_csv}")
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY.")


if __name__ == '__main__':
    main()
