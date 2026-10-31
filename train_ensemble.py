import pandas as pd
import numpy as np
import os
import catboost as cb
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report

data_dir = r"d:\Sujay\GDGC_AIML\network-intrusion-detection2-ml-challenge"
train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
sample_sub = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"))

label_mapping = {'Availability': 0, 'Benign': 1, 'Reconnaissance': 2, 'Spoofing': 3}
inv_mapping = {v: k for k, v in label_mapping.items()}

print(f"Loaded train: {train_df.shape}, test: {test_df.shape}")

# 1. Feature Engineering Function
def engineer_features(df):
    data = df.copy()
    
    # Ratios and Per-Packet statistics
    fwd_pkts = np.maximum(1.0, data['fwd_pkts_tot'])
    bwd_pkts = np.maximum(1.0, data['bwd_pkts_tot'])
    
    data['fwd_payload_per_pkt'] = data['fwd_pkts_payload.tot'] / fwd_pkts
    data['bwd_payload_per_pkt'] = data['bwd_pkts_payload.tot'] / bwd_pkts
    data['fwd_header_per_pkt'] = data['fwd_header_size_tot'] / fwd_pkts
    data['bwd_header_per_pkt'] = data['bwd_header_size_tot'] / bwd_pkts
    
    # Payload flags
    data['has_fwd_payload'] = (data['fwd_pkts_payload.tot'] > 0).astype(float)
    data['has_bwd_payload'] = (data['bwd_pkts_payload.tot'] > 0).astype(float)
    
    # TCP Flag Combinations
    data['is_syn_only'] = ((data['flow_SYN_flag_count'] > 0) & (data['flow_ACK_flag_count'] == 0)).astype(float)
    data['is_fin_only'] = ((data['flow_FIN_flag_count'] > 0) & (data['flow_ACK_flag_count'] == 0)).astype(float)
    data['is_xmas_scan'] = ((data['flow_FIN_flag_count'] > 0) & (data['fwd_PSH_flag_count'] > 0) & (data['fwd_URG_flag_count'] > 0)).astype(float)
    
    # Signatures
    data['is_syn_flood_sig'] = ((data['proto'] == 'tcp') & (data['flow_SYN_flag_count'] >= 1) & (data['fwd_pkts_payload.tot'] == 120.0)).astype(float)
    data['is_port_21'] = (data['id.resp_p'] == 21.0).astype(float)
    data['is_port_1883'] = (data['id.resp_p'] == 1883.0).astype(float)
    data['is_port_53'] = (data['id.resp_p'] == 53.0).astype(float)
    data['is_port_80'] = (data['id.resp_p'] == 80.0).astype(float)
    data['is_port_443'] = (data['id.resp_p'] == 443.0).astype(float)
    data['is_port_5353'] = (data['id.resp_p'] == 5353.0).astype(float)
    
    # Port ranges
    data['is_well_known_resp'] = (data['id.resp_p'] <= 1024).astype(float)
    data['is_ephemeral_resp'] = (data['id.resp_p'] >= 49152).astype(float)
    
    # Log transforms for skewed rate/duration features
    data['log_duration'] = np.log1p(np.maximum(0, data['flow_duration']))
    data['log_flow_pkts_per_sec'] = np.log1p(np.maximum(0, data['flow_pkts_per_sec']))
    data['log_payload_bytes_per_sec'] = np.log1p(np.maximum(0, data['payload_bytes_per_second']))
    
    return data

print("Engineering features for train and test...")
train_fe = engineer_features(train_df)
test_fe = engineer_features(test_df)

# Zero out columns that were zero in unresponsive server SYN floods
t21_0 = test_df[(test_df['id.resp_p'] == 21.0) & (test_df['bwd_pkts_tot'] == 0.0)]
zero_cols = [c for c in t21_0.columns if c not in ['id', 'proto', 'service'] and t21_0[c].max() == 0 and t21_0[c].min() == 0]

# Perform domain-consistent data augmentation on training set
augmented_train = train_fe.copy()
syn_flood_mask = (augmented_train['Target'] == 'Availability') & (augmented_train['id.resp_p'] == 21.0)
drop_indices = augmented_train[syn_flood_mask].sample(frac=0.35, random_state=42).index

for c in zero_cols:
    if c in augmented_train.columns:
        augmented_train.loc[drop_indices, c] = 0.0

# Also update the engineered features for the augmented rows
augmented_train.loc[drop_indices, 'bwd_payload_per_pkt'] = 0.0
augmented_train.loc[drop_indices, 'bwd_header_per_pkt'] = 0.0
augmented_train.loc[drop_indices, 'has_bwd_payload'] = 0.0
augmented_train.loc[drop_indices, 'is_syn_only'] = 1.0
augmented_train.loc[drop_indices, 'log_duration'] = 0.0
augmented_train.loc[drop_indices, 'log_flow_pkts_per_sec'] = 0.0
augmented_train.loc[drop_indices, 'log_payload_bytes_per_sec'] = 0.0

print(f"Augmented {len(drop_indices)} samples of Availability in training set.")

feature_cols = [c for c in train_fe.columns if c not in ['id', 'Target']]
print(f"Total feature columns: {len(feature_cols)}")

# Encode categoricals
for col in ['proto', 'service']:
    augmented_train[col] = augmented_train[col].astype('category')
    test_fe[col] = test_fe[col].astype('category')

X = augmented_train[feature_cols].copy()
y = augmented_train['Target'].map(label_mapping).values
X_test = test_fe[feature_cols].copy()

# For XGBoost, convert category dtypes properly
X_xgb = X.copy()
X_test_xgb = X_test.copy()
for col in ['proto', 'service']:
    X_xgb[col] = X_xgb[col].cat.codes
    X_test_xgb[col] = X_test_xgb[col].cat.codes

# 5-Fold Stratified Cross Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cb_oof_probs = np.zeros((len(augmented_train), 4))
cb_test_probs = np.zeros((len(test_df), 4))

xgb_oof_probs = np.zeros((len(augmented_train), 4))
xgb_test_probs = np.zeros((len(test_df), 4))

cat_features = ['proto', 'service']
X_cb = X.copy()
X_test_cb = X_test.copy()
for col in cat_features:
    X_cb[col] = X_cb[col].astype(str)
    X_test_cb[col] = X_test_cb[col].astype(str)

print("\n--- Training CatBoost Models ---")
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\nFold {fold+1}/5:")
    X_tr, y_tr = X_cb.iloc[train_idx], y[train_idx]
    X_va, y_va = X_cb.iloc[val_idx], y[val_idx]
    
    cb_model = cb.CatBoostClassifier(
        iterations=450,
        learning_rate=0.07,
        depth=6,
        cat_features=cat_features,
        auto_class_weights='Balanced',
        random_seed=42 + fold,
        verbose=150,
        thread_count=-1
    )
    cb_model.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=50, verbose=150)
    
    val_pred = cb_model.predict_proba(X_va)
    cb_oof_probs[val_idx] = val_pred
    fold_f1 = f1_score(y_va, np.argmax(val_pred, axis=1), average='macro')
    print(f"Fold {fold+1} CatBoost Macro F1: {fold_f1:.5f}")
    
    cb_test_probs += cb_model.predict_proba(X_test_cb) / 5.0

cb_total_f1 = f1_score(y, np.argmax(cb_oof_probs, axis=1), average='macro')
print(f"\nOverall CatBoost OOF Macro F1: {cb_total_f1:.5f}")

print("\n--- Training XGBoost Models ---")
# Compute class weights for XGBoost
from sklearn.utils.class_weight import compute_sample_weight
sample_weights = compute_sample_weight('balanced', y)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\nFold {fold+1}/5:")
    X_tr, y_tr = X_xgb.iloc[train_idx], y[train_idx]
    X_va, y_va = X_xgb.iloc[val_idx], y[val_idx]
    sw_tr = sample_weights[train_idx]
    
    xgb_model = xgb.XGBClassifier(
        n_estimators=350,
        learning_rate=0.07,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42 + fold,
        n_jobs=-1,
        tree_method='hist'
    )
    xgb_model.fit(X_tr, y_tr, sample_weight=sw_tr, eval_set=[(X_va, y_va)], verbose=150)
    
    val_pred = xgb_model.predict_proba(X_va)
    xgb_oof_probs[val_idx] = val_pred
    fold_f1 = f1_score(y_va, np.argmax(val_pred, axis=1), average='macro')
    print(f"Fold {fold+1} XGBoost Macro F1: {fold_f1:.5f}")
    
    xgb_test_probs += xgb_model.predict_proba(X_test_xgb) / 5.0

xgb_total_f1 = f1_score(y, np.argmax(xgb_oof_probs, axis=1), average='macro')
print(f"\nOverall XGBoost OOF Macro F1: {xgb_total_f1:.5f}")

# Ensemble Blending (50% CatBoost + 50% XGBoost)
ensemble_oof_probs = 0.5 * cb_oof_probs + 0.5 * xgb_oof_probs
ensemble_f1 = f1_score(y, np.argmax(ensemble_oof_probs, axis=1), average='macro')
print(f"\n==========================================")
print(f"Ensemble (CatBoost + XGBoost) OOF Macro F1: {ensemble_f1:.5f}")
print(f"==========================================")
print("\nEnsemble Classification Report:")
print(classification_report(y, np.argmax(ensemble_oof_probs, axis=1), target_names=['Availability', 'Benign', 'Reconnaissance', 'Spoofing'], digits=5))

# Final Test Predictions
ensemble_test_probs = 0.5 * cb_test_probs + 0.5 * xgb_test_probs
final_preds = np.argmax(ensemble_test_probs, axis=1)
final_labels = [inv_mapping[i] for i in final_preds]

sub_df = pd.DataFrame({
    'id': test_df['id'],
    'Target': final_labels
})

sub_path = os.path.join(r"d:\Sujay\GDGC_AIML", "submission_ensemble.py.csv")
sub_df.to_csv(sub_path, index=False)
print(f"\nSaved final submission to {sub_path}")

# Check test predictions distribution
print("\n=== Final Test Predictions Distribution ===")
print(sub_df['Target'].value_counts())
print("\nNormalized:")
print(sub_df['Target'].value_counts(normalize=True))

# Check submission integrity
print("\n=== Integrity Check ===")
print(f"Shape: {sub_df.shape} (Expected: (33161, 2))")
print(f"Null count: {sub_df.isnull().sum().to_dict()}")
print(f"Exact column names: {list(sub_df.columns)}")
print("First 10 rows:")
print(sub_df.head(10))

# Save also as submission.csv in current dir and challenge dir for easy access
sub_df.to_csv(os.path.join(r"d:\Sujay\GDGC_AIML", "submission.csv"), index=False)
sub_df.to_csv(os.path.join(data_dir, "submission.csv"), index=False)
print("Saved also as submission.csv!")
