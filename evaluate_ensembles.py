"""
Rigorous Model Ensemble & Probability Blending Framework
GDG on Campus Network Intrusion Detection Challenge

Models Compared:
1. CatBoost alone
2. LightGBM alone
3. XGBoost alone
4. CatBoost + LightGBM (Equal & Nested-Tuned)
5. CatBoost + XGBoost (Equal & Nested-Tuned)
6. CatBoost + LightGBM + XGBoost (Equal & Nested-Tuned)

Validation Architecture:
- Uses the exact same 5 Stratified Folds (random_state=42).
- Blending weights tuned strictly on inner training folds via Nested Cross-Validation
  to prevent overfitting ensemble weights to validation data.
- Evaluates Macro F1, Availability F1, Benign F1, Reconnaissance F1, Spoofing F1.
- Evaluates whether ensemble provides a genuine, repeatable improvement over the strongest single model.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

# Paths
REPORTS_DIR = r"D:\Sujay\GDGC_AIML\reports"
CLASS_NAMES = ['Availability', 'Benign', 'Reconnaissance', 'Spoofing']
LABEL_MAPPING = {'Availability': 0, 'Benign': 1, 'Reconnaissance': 2, 'Spoofing': 3}
INV_MAPPING = {v: k for k, v in LABEL_MAPPING.items()}


def load_oof_and_test_probabilities():
    print("Loading Out-of-Fold (OOF) and Test probability predictions...")
    
    cb_oof_df = pd.read_csv(os.path.join(REPORTS_DIR, "oof_probabilities_catboost.csv"))
    lgb_oof_df = pd.read_csv(os.path.join(REPORTS_DIR, "oof_probabilities_lightgbm.csv"))
    xgb_oof_df = pd.read_csv(os.path.join(REPORTS_DIR, "oof_probabilities_xgboost.csv"))
    
    # Ground truth
    y_true = cb_oof_df['true_target'].map(LABEL_MAPPING).values
    
    prob_cols = [f"prob_{c}" for c in CLASS_NAMES]
    
    cb_oof = cb_oof_df[prob_cols].values.astype(np.float32)
    lgb_oof = lgb_oof_df[prob_cols].values.astype(np.float32)
    xgb_oof = xgb_oof_df[prob_cols].values.astype(np.float32)
    
    # Test predictions
    cb_test_df = pd.read_csv(os.path.join(REPORTS_DIR, "test_predictions_catboost.csv"))
    lgb_test_df = pd.read_csv(os.path.join(REPORTS_DIR, "test_predictions_lightgbm.csv"))
    xgb_test_df = pd.read_csv(os.path.join(REPORTS_DIR, "test_predictions_xgboost.csv"))
    
    cb_test = cb_test_df[prob_cols].values.astype(np.float32)
    lgb_test = lgb_test_df[prob_cols].values.astype(np.float32)
    xgb_test = xgb_test_df[prob_cols].values.astype(np.float32)
    
    test_ids = cb_test_df['id'].values
    
    return (cb_oof, lgb_oof, xgb_oof, y_true, 
            cb_test, lgb_test, xgb_test, test_ids)


def compute_metrics(y_true, y_pred):
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    class_f1 = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2, 3], zero_division=0)
    macro_p = precision_score(y_true, y_pred, average='macro', zero_division=0)
    macro_r = recall_score(y_true, y_pred, average='macro', zero_division=0)
    return {
        'macro_f1': macro_f1,
        'f1_availability': class_f1[0],
        'f1_benign': class_f1[1],
        'f1_reconnaissance': class_f1[2],
        'f1_spoofing': class_f1[3],
        'macro_precision': macro_p,
        'macro_recall': macro_r
    }


def optimize_2way_weights(p1, p2, y):
    """
    Finds scalar weight w in [0, 1] maximizing Macro F1 for w*p1 + (1-w)*p2.
    Grid search with 101 points.
    """
    best_w = 0.5
    best_f1 = -1.0
    for w in np.linspace(0.0, 1.0, 101):
        blend = w * p1 + (1.0 - w) * p2
        pred = np.argmax(blend, axis=1)
        f1 = f1_score(y, pred, average='macro', zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_w = w
    return best_w, best_f1


def optimize_3way_weights(p1, p2, p3, y, step=0.02):
    """
    Finds weights (w1, w2, w3) on 2D simplex maximizing Macro F1.
    """
    best_weights = (1/3, 1/3, 1/3)
    best_f1 = -1.0
    
    steps = int(round(1.0 / step))
    for i in range(steps + 1):
        w1 = i * step
        for j in range(steps + 1 - i):
            w2 = j * step
            w3 = 1.0 - w1 - w2
            if w3 < -1e-6:
                continue
            w3 = max(0.0, w3)
            
            blend = w1 * p1 + w2 * p2 + w3 * p3
            pred = np.argmax(blend, axis=1)
            f1 = f1_score(y, pred, average='macro', zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_weights = (round(w1, 3), round(w2, 3), round(w3, 3))
                
    return best_weights, best_f1


def evaluate_single_model(name, oof_probs, y_true, splits):
    fold_f1s = []
    for fold, (tr_idx, val_idx) in enumerate(splits):
        val_pred = np.argmax(oof_probs[val_idx], axis=1)
        f1 = f1_score(y_true[val_idx], val_pred, average='macro', zero_division=0)
        fold_f1s.append(f1)
        
    overall_preds = np.argmax(oof_probs, axis=1)
    metrics = compute_metrics(y_true, overall_preds)
    metrics['experiment'] = name
    metrics['mean_fold_f1'] = np.mean(fold_f1s)
    metrics['std_fold_f1'] = np.std(fold_f1s)
    metrics['fold_f1s'] = fold_f1s
    metrics['weights_used'] = 'Single Model'
    return metrics, overall_preds


def evaluate_fixed_blend(name, prob_list, weights, y_true, splits):
    norm_w = np.array(weights) / np.sum(weights)
    blended_oof = sum(w * p for w, p in zip(norm_w, prob_list))
    
    fold_f1s = []
    for fold, (tr_idx, val_idx) in enumerate(splits):
        val_pred = np.argmax(blended_oof[val_idx], axis=1)
        f1 = f1_score(y_true[val_idx], val_pred, average='macro', zero_division=0)
        fold_f1s.append(f1)
        
    overall_preds = np.argmax(blended_oof, axis=1)
    metrics = compute_metrics(y_true, overall_preds)
    metrics['experiment'] = name
    metrics['mean_fold_f1'] = np.mean(fold_f1s)
    metrics['std_fold_f1'] = np.std(fold_f1s)
    metrics['fold_f1s'] = fold_f1s
    metrics['weights_used'] = str([round(w, 3) for w in norm_w])
    return metrics, overall_preds, blended_oof


def evaluate_nested_2way_blend(name, p1, p2, y_true, splits):
    """
    Nested cross-validation: weights tuned on inner folds, evaluated on outer fold.
    """
    nested_oof = np.zeros_like(p1)
    fold_f1s = []
    fold_weights = []
    
    for fold, (tr_idx, val_idx) in enumerate(splits):
        # Learn optimal w on inner training folds
        best_w, inner_f1 = optimize_2way_weights(p1[tr_idx], p2[tr_idx], y_true[tr_idx])
        fold_weights.append(best_w)
        
        # Apply to outer held-out validation fold
        val_blend = best_w * p1[val_idx] + (1.0 - best_w) * p2[val_idx]
        nested_oof[val_idx] = val_blend
        
        val_pred = np.argmax(val_blend, axis=1)
        fold_f1 = f1_score(y_true[val_idx], val_pred, average='macro', zero_division=0)
        fold_f1s.append(fold_f1)
        
    overall_preds = np.argmax(nested_oof, axis=1)
    metrics = compute_metrics(y_true, overall_preds)
    metrics['experiment'] = name
    metrics['mean_fold_f1'] = np.mean(fold_f1s)
    metrics['std_fold_f1'] = np.std(fold_f1s)
    metrics['fold_f1s'] = fold_f1s
    metrics['weights_used'] = f"Nested mean w1={np.mean(fold_weights):.3f} (folds: {[round(w, 3) for w in fold_weights]})"
    return metrics, overall_preds, nested_oof, fold_weights


def evaluate_nested_3way_blend(name, p1, p2, p3, y_true, splits):
    """
    Nested cross-validation for 3-way blend.
    """
    nested_oof = np.zeros_like(p1)
    fold_f1s = []
    fold_weights = []
    
    for fold, (tr_idx, val_idx) in enumerate(splits):
        # Learn optimal (w1, w2, w3) on inner training folds
        best_w, inner_f1 = optimize_3way_weights(p1[tr_idx], p2[tr_idx], p3[tr_idx], y_true[tr_idx], step=0.02)
        fold_weights.append(best_w)
        
        # Apply to outer held-out validation fold
        w1, w2, w3 = best_w
        val_blend = w1 * p1[val_idx] + w2 * p2[val_idx] + w3 * p3[val_idx]
        nested_oof[val_idx] = val_blend
        
        val_pred = np.argmax(val_blend, axis=1)
        fold_f1 = f1_score(y_true[val_idx], val_pred, average='macro', zero_division=0)
        fold_f1s.append(fold_f1)
        
    overall_preds = np.argmax(nested_oof, axis=1)
    metrics = compute_metrics(y_true, overall_preds)
    metrics['experiment'] = name
    metrics['mean_fold_f1'] = np.mean(fold_f1s)
    metrics['std_fold_f1'] = np.std(fold_f1s)
    metrics['fold_f1s'] = fold_f1s
    mean_w = np.mean(fold_weights, axis=0)
    metrics['weights_used'] = f"Nested mean: CB={mean_w[0]:.3f}, LGB={mean_w[1]:.3f}, XGB={mean_w[2]:.3f}"
    return metrics, overall_preds, nested_oof, fold_weights


def main():
    (cb_oof, lgb_oof, xgb_oof, y_true, 
     cb_test, lgb_test, xgb_test, test_ids) = load_oof_and_test_probabilities()
    
    # Stratified 5-Fold splits (exact same splits as original models)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(skf.split(cb_oof, y_true))
    
    results = []
    oof_predictions_dict = {}
    
    print("\n" + "="*70)
    print("STEP 1: EVALUATING INDIVIDUAL MODELS (OOF Ground Truth)")
    print("="*70)
    
    # 1. CatBoost alone
    m_cb, pred_cb = evaluate_single_model("1. CatBoost Alone", cb_oof, y_true, splits)
    results.append(m_cb)
    oof_predictions_dict["CatBoost"] = pred_cb
    print(f">> CatBoost Alone:       Macro F1 = {m_cb['macro_f1']:.5f} (Fold Mean: {m_cb['mean_fold_f1']:.5f} ± {m_cb['std_fold_f1']:.5f})")
    
    # 2. LightGBM alone
    m_lgb, pred_lgb = evaluate_single_model("2. LightGBM Alone", lgb_oof, y_true, splits)
    results.append(m_lgb)
    oof_predictions_dict["LightGBM"] = pred_lgb
    print(f">> LightGBM Alone:       Macro F1 = {m_lgb['macro_f1']:.5f} (Fold Mean: {m_lgb['mean_fold_f1']:.5f} ± {m_lgb['std_fold_f1']:.5f})")
    
    # 3. XGBoost alone
    m_xgb, pred_xgb = evaluate_single_model("3. XGBoost Alone", xgb_oof, y_true, splits)
    results.append(m_xgb)
    oof_predictions_dict["XGBoost"] = pred_xgb
    print(f">> XGBoost Alone:        Macro F1 = {m_xgb['macro_f1']:.5f} (Fold Mean: {m_xgb['mean_fold_f1']:.5f} ± {m_xgb['std_fold_f1']:.5f})")
    
    print("\n" + "="*70)
    print("STEP 2: EVALUATING TWO-MODEL ENSEMBLES (Equal vs Nested-Tuned)")
    print("="*70)
    
    # 4a. CatBoost + LightGBM (Equal)
    m_cb_lgb_eq, pred_cb_lgb_eq, _ = evaluate_fixed_blend("4a. CatBoost + LightGBM (Equal 50/50)", [cb_oof, lgb_oof], [0.5, 0.5], y_true, splits)
    results.append(m_cb_lgb_eq)
    print(f">> CB + LGB (Equal):     Macro F1 = {m_cb_lgb_eq['macro_f1']:.5f} (Fold Mean: {m_cb_lgb_eq['mean_fold_f1']:.5f} ± {m_cb_lgb_eq['std_fold_f1']:.5f})")
    
    # 4b. CatBoost + LightGBM (Nested-Tuned)
    m_cb_lgb_opt, pred_cb_lgb_opt, oof_cb_lgb, w_cb_lgb = evaluate_nested_2way_blend("4b. CatBoost + LightGBM (Nested-Tuned)", cb_oof, lgb_oof, y_true, splits)
    results.append(m_cb_lgb_opt)
    oof_predictions_dict["CB_LGB_Nested"] = pred_cb_lgb_opt
    print(f">> CB + LGB (Nested):    Macro F1 = {m_cb_lgb_opt['macro_f1']:.5f} (Fold Mean: {m_cb_lgb_opt['mean_fold_f1']:.5f} ± {m_cb_lgb_opt['std_fold_f1']:.5f}) | {m_cb_lgb_opt['weights_used']}")
    
    # 5a. CatBoost + XGBoost (Equal)
    m_cb_xgb_eq, pred_cb_xgb_eq, _ = evaluate_fixed_blend("5a. CatBoost + XGBoost (Equal 50/50)", [cb_oof, xgb_oof], [0.5, 0.5], y_true, splits)
    results.append(m_cb_xgb_eq)
    print(f">> CB + XGB (Equal):     Macro F1 = {m_cb_xgb_eq['macro_f1']:.5f} (Fold Mean: {m_cb_xgb_eq['mean_fold_f1']:.5f} ± {m_cb_xgb_eq['std_fold_f1']:.5f})")
    
    # 5b. CatBoost + XGBoost (Nested-Tuned)
    m_cb_xgb_opt, pred_cb_xgb_opt, oof_cb_xgb, w_cb_xgb = evaluate_nested_2way_blend("5b. CatBoost + XGBoost (Nested-Tuned)", cb_oof, xgb_oof, y_true, splits)
    results.append(m_cb_xgb_opt)
    oof_predictions_dict["CB_XGB_Nested"] = pred_cb_xgb_opt
    print(f">> CB + XGB (Nested):    Macro F1 = {m_cb_xgb_opt['macro_f1']:.5f} (Fold Mean: {m_cb_xgb_opt['mean_fold_f1']:.5f} ± {m_cb_xgb_opt['std_fold_f1']:.5f}) | {m_cb_xgb_opt['weights_used']}")
    
    # 5c. LightGBM + XGBoost (Equal & Nested-Tuned)
    m_lgb_xgb_eq, pred_lgb_xgb_eq, _ = evaluate_fixed_blend("5c. LightGBM + XGBoost (Equal 50/50)", [lgb_oof, xgb_oof], [0.5, 0.5], y_true, splits)
    results.append(m_lgb_xgb_eq)
    m_lgb_xgb_opt, pred_lgb_xgb_opt, oof_lgb_xgb, w_lgb_xgb = evaluate_nested_2way_blend("5d. LightGBM + XGBoost (Nested-Tuned)", lgb_oof, xgb_oof, y_true, splits)
    results.append(m_lgb_xgb_opt)
    print(f">> LGB + XGB (Nested):   Macro F1 = {m_lgb_xgb_opt['macro_f1']:.5f} (Fold Mean: {m_lgb_xgb_opt['mean_fold_f1']:.5f} ± {m_lgb_xgb_opt['std_fold_f1']:.5f}) | {m_lgb_xgb_opt['weights_used']}")
    
    print("\n" + "="*70)
    print("STEP 3: EVALUATING THREE-MODEL ENSEMBLE (CatBoost + LightGBM + XGBoost)")
    print("="*70)
    
    # 6a. Three-way Equal Blend
    m_tri_eq, pred_tri_eq, _ = evaluate_fixed_blend("6a. CB + LGB + XGB (Equal 33/33/33)", [cb_oof, lgb_oof, xgb_oof], [1/3, 1/3, 1/3], y_true, splits)
    results.append(m_tri_eq)
    print(f">> Tri-Model (Equal):    Macro F1 = {m_tri_eq['macro_f1']:.5f} (Fold Mean: {m_tri_eq['mean_fold_f1']:.5f} ± {m_tri_eq['std_fold_f1']:.5f})")
    
    # 6b. Three-way Nested-Tuned Blend
    m_tri_opt, pred_tri_opt, oof_tri, w_tri = evaluate_nested_3way_blend("6b. CB + LGB + XGB (Nested-Tuned)", cb_oof, lgb_oof, xgb_oof, y_true, splits)
    results.append(m_tri_opt)
    oof_predictions_dict["Tri_Model_Nested"] = pred_tri_opt
    print(f">> Tri-Model (Nested):   Macro F1 = {m_tri_opt['macro_f1']:.5f} (Fold Mean: {m_tri_opt['mean_fold_f1']:.5f} ± {m_tri_opt['std_fold_f1']:.5f}) | {m_tri_opt['weights_used']}")
    
    # Full dataset global weight optimization for final test predictions
    print("\n" + "="*70)
    print("STEP 4: FINAL TEST ENSEMBLE GENERATION (Strictly Unlabeled)")
    print("="*70)
    
    # Optimize globally across full OOF to get final ensemble weights for test inference
    full_opt_w_tri, full_opt_f1 = optimize_3way_weights(cb_oof, lgb_oof, xgb_oof, y_true, step=0.01)
    w_cb_final, w_lgb_final, w_xgb_final = full_opt_w_tri
    print(f"Optimal 3-Way Weights on Full OOF: CB={w_cb_final:.2f}, LGB={w_lgb_final:.2f}, XGB={w_xgb_final:.2f} (Full OOF F1: {full_opt_f1:.5f})")
    
    full_opt_w_cb_lgb, full_opt_f1_cb_lgb = optimize_2way_weights(cb_oof, lgb_oof, y_true)
    print(f"Optimal CB+LGB Weights on Full OOF: CB={full_opt_w_cb_lgb:.2f}, LGB={1.0 - full_opt_w_cb_lgb:.2f} (Full OOF F1: {full_opt_f1_cb_lgb:.5f})")
    
    full_opt_w_lgb_xgb, full_opt_f1_lgb_xgb = optimize_2way_weights(lgb_oof, xgb_oof, y_true)
    print(f"Optimal LGB+XGB Weights on Full OOF: LGB={full_opt_w_lgb_xgb:.2f}, XGB={1.0 - full_opt_w_lgb_xgb:.2f} (Full OOF F1: {full_opt_f1_lgb_xgb:.5f})")
    
    # Generate final ensemble test predictions using the optimal tri-model blend
    final_test_blend = w_cb_final * cb_test + w_lgb_final * lgb_test + w_xgb_final * xgb_test
    final_test_preds = np.argmax(final_test_blend, axis=1)
    
    ens_test_df = pd.DataFrame({
        'id': test_ids,
        'Target': [INV_MAPPING[idx] for idx in final_test_preds]
    })
    for i, c in enumerate(CLASS_NAMES):
        ens_test_df[f'prob_{c}'] = final_test_blend[:, i]
        
    ens_test_csv = os.path.join(REPORTS_DIR, "ensemble_test_predictions.csv")
    ens_test_df.to_csv(ens_test_csv, index=False)
    print(f"\nFinal ensemble test predictions saved to: {ens_test_csv}")
    print("Class distribution on test set:")
    print(ens_test_df['Target'].value_counts())
    
    # Save master ensemble comparison CSV
    summary_df = pd.DataFrame(results)
    # Reorder columns
    cols_order = ['experiment', 'macro_f1', 'mean_fold_f1', 'std_fold_f1',
                  'f1_availability', 'f1_benign', 'f1_reconnaissance', 'f1_spoofing',
                  'macro_precision', 'macro_recall', 'weights_used']
    summary_df = summary_df[cols_order]
    
    comparison_csv = os.path.join(REPORTS_DIR, "ensemble_comparison_results.csv")
    summary_df.to_csv(comparison_csv, index=False)
    print(f"\nMaster ensemble comparison results saved to: {comparison_csv}")
    
    # Generate Confusion Matrix for Champion Tri-Model Ensemble
    tri_cm = confusion_matrix(y_true, np.argmax(oof_tri, axis=1))
    tri_cm_df = pd.DataFrame(tri_cm, index=CLASS_NAMES, columns=CLASS_NAMES)
    cm_csv = os.path.join(REPORTS_DIR, "confusion_matrices", "cm_tri_model_nested_ensemble_oof.csv")
    tri_cm_df.to_csv(cm_csv)
    print(f"\nChampion Tri-Model Confusion Matrix:\n{tri_cm_df}")


if __name__ == '__main__':
    main()
