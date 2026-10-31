import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight

# Output directories
base_dir = r"D:\Sujay\GDGC_AIML"
reports_dir = os.path.join(base_dir, "reports")
cm_dir = os.path.join(reports_dir, "confusion_matrices")
os.makedirs(cm_dir, exist_ok=True)

data_dir = os.path.join(base_dir, "network-intrusion-detection2-ml-challenge")
train_path = os.path.join(data_dir, "train.csv")

print("================================================================================")
print("ROBUST VALIDATION FRAMEWORK: NETWORK INTRUSION DETECTION")
print("================================================================================")
print(f"Loading data from: {train_path}")
train_df = pd.read_csv(train_path)
print(f"Train dataset loaded: {train_df.shape[0]:,} rows, {train_df.shape[1]} columns")

# Target mapping
label_names = ['Availability', 'Benign', 'Reconnaissance', 'Spoofing']
label_to_idx = {name: i for i, name in enumerate(label_names)}
idx_to_label = {i: name for i, name in enumerate(label_names)}
y = train_df['Target'].map(label_to_idx).values

# Raw feature columns (Strictly excluding 'id' and 'Target')
feature_cols = [c for c in train_df.columns if c not in ['id', 'Target']]
print(f"Total predictive features: {len(feature_cols)} (ID and Target excluded)")

# Categorical columns
cat_cols = ['proto', 'service']

# Helper to plot and save confusion matrix
def save_confusion_matrix_plot(cm, title, filepath):
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(cm, cmap='Blues', alpha=0.85)
    fig.colorbar(cax)
    
    for i in range(len(label_names)):
        for j in range(len(label_names)):
            val = cm[i, j]
            ax.text(j, i, f"{val:,}", ha='center', va='center', color='black' if val < cm.max()/2 else 'white', fontweight='bold')
            
    ax.set_xticks(range(len(label_names)))
    ax.set_yticks(range(len(label_names)))
    ax.set_xticklabels(label_names, rotation=30, ha='left', fontsize=9)
    ax.set_yticklabels(label_names, fontsize=9)
    ax.set_xlabel('Predicted Label', fontweight='bold')
    ax.set_ylabel('True Label', fontweight='bold')
    ax.set_title(title, fontweight='bold', pad=20, fontsize=11)
    plt.tight_layout()
    plt.savefig(filepath, dpi=180)
    plt.close()

# Evaluation runner for a specific split strategy
def run_validation_experiment(exp_name, splits):
    print("\n" + "="*80)
    print(f"RUNNING EXPERIMENT: {exp_name}")
    print("="*80)
    
    fold_metrics_list = []
    oof_preds = np.full(len(train_df), -1)
    oof_cm = np.zeros((4, 4), dtype=int)
    
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        print(f"\n--- Fold {fold_idx + 1} ---")
        
        # Split data
        X_train_raw = train_df.iloc[train_idx][feature_cols].copy()
        y_train = y[train_idx]
        X_val_raw = train_df.iloc[val_idx][feature_cols].copy()
        y_val = y[val_idx]
        
        # Class distribution check
        train_class_counts = {label_names[i]: int((y_train == i).sum()) for i in range(4)}
        val_class_counts = {label_names[i]: int((y_val == i).sum()) for i in range(4)}
        
        train_missing = [name for name, cnt in train_class_counts.items() if cnt == 0]
        val_missing = [name for name, cnt in val_class_counts.items() if cnt == 0]
        
        print(f"Train size: {len(X_train_raw):,} | Val size: {len(X_val_raw):,}")
        print(f"Train Class Counts: {train_class_counts}")
        print(f"Val Class Counts:   {val_class_counts}")
        if val_missing:
            print(f"  [ALERT] Missing classes in validation fold: {val_missing}")
        if train_missing:
            print(f"  [ALERT] Missing classes in training fold: {train_missing}")
            
        # Fit categorical encoders strictly on X_train_raw
        X_train = X_train_raw.copy()
        X_val = X_val_raw.copy()
        for col in cat_cols:
            cats = sorted(X_train_raw[col].dropna().unique())
            cat_map = {cat: idx for idx, cat in enumerate(cats)}
            X_train[col] = X_train_raw[col].map(cat_map).fillna(-1).astype(int)
            X_val[col] = X_val_raw[col].map(cat_map).fillna(-1).astype(int)
            
        # Compute balanced sample weights on train
        sample_weights = compute_sample_weight('balanced', y_train)
        
        # Train XGBoost model
        clf = xgb.XGBClassifier(
            n_estimators=300,
            learning_rate=0.08,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            tree_method='hist',
            random_state=42 + fold_idx,
            n_jobs=-1
        )
        clf.fit(X_train, y_train, sample_weight=sample_weights, eval_set=[(X_val, y_val)], verbose=False)
        
        val_pred = clf.predict(X_val)
        oof_preds[val_idx] = val_pred
        
        # Compute metrics
        labels_all = [0, 1, 2, 3]
        f1_per_class = f1_score(y_val, val_pred, labels=labels_all, average=None, zero_division=0)
        p_per_class = precision_score(y_val, val_pred, labels=labels_all, average=None, zero_division=0)
        r_per_class = recall_score(y_val, val_pred, labels=labels_all, average=None, zero_division=0)
        
        macro_f1 = f1_score(y_val, val_pred, labels=labels_all, average='macro', zero_division=0)
        macro_p = precision_score(y_val, val_pred, labels=labels_all, average='macro', zero_division=0)
        macro_r = recall_score(y_val, val_pred, labels=labels_all, average='macro', zero_division=0)
        
        # Fold Confusion Matrix
        fold_cm = confusion_matrix(y_val, val_pred, labels=labels_all)
        oof_cm += fold_cm
        
        # Save fold confusion matrix CSV
        clean_exp = exp_name.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "").replace(".", "_")
        cm_fold_csv = os.path.join(cm_dir, f"cm_{clean_exp}_fold{fold_idx+1}.csv")
        pd.DataFrame(fold_cm, index=label_names, columns=label_names).to_csv(cm_fold_csv)
        
        print(f"Fold {fold_idx + 1} Metrics -> Macro F1: {macro_f1:.5f} | Precision: {macro_p:.5f} | Recall: {macro_r:.5f}")
        print(f"  Availability F1: {f1_per_class[0]:.5f} | Benign F1: {f1_per_class[1]:.5f} | Reconnaissance F1: {f1_per_class[2]:.5f} | Spoofing F1: {f1_per_class[3]:.5f}")
        
        fold_metrics_list.append({
            'experiment': exp_name,
            'fold': fold_idx + 1,
            'train_size': len(train_idx),
            'val_size': len(val_idx),
            'missing_val_classes': ";".join(val_missing) if val_missing else "None",
            'missing_train_classes': ";".join(train_missing) if train_missing else "None",
            'macro_f1': macro_f1,
            'macro_precision': macro_p,
            'macro_recall': macro_r,
            'f1_availability': f1_per_class[0],
            'f1_benign': f1_per_class[1],
            'f1_reconnaissance': f1_per_class[2],
            'f1_spoofing': f1_per_class[3],
            'prec_availability': p_per_class[0],
            'prec_benign': p_per_class[1],
            'prec_reconnaissance': p_per_class[2],
            'prec_spoofing': p_per_class[3],
            'rec_availability': r_per_class[0],
            'rec_benign': r_per_class[1],
            'rec_reconnaissance': r_per_class[2],
            'rec_spoofing': r_per_class[3],
            'train_counts': str(train_class_counts),
            'val_counts': str(val_class_counts)
        })
        
    # Aggregate OOF Metrics
    valid_oof_mask = (oof_preds != -1)
    oof_f1_overall = f1_score(y[valid_oof_mask], oof_preds[valid_oof_mask], labels=labels_all, average='macro', zero_division=0)
    print(f"\n=== {exp_name} Overall OOF Macro F1: {oof_f1_overall:.5f} ===")
    
    # Save overall OOF Confusion Matrix
    oof_cm_df = pd.DataFrame(oof_cm, index=label_names, columns=label_names)
    cm_oof_csv = os.path.join(cm_dir, f"cm_{clean_exp}_overall_oof.csv")
    oof_cm_df.to_csv(cm_oof_csv)
    
    cm_oof_png = os.path.join(cm_dir, f"cm_{clean_exp}_overall_oof.png")
    save_confusion_matrix_plot(oof_cm, f"{exp_name} (Overall OOF Confusion Matrix)", cm_oof_png)
    
    return fold_metrics_list, oof_f1_overall

# ----------------------------------------------------------------------------------
# Define Validation Split Strategies
# ----------------------------------------------------------------------------------

all_fold_metrics = []
summary_results = []

# 1. Stratified 5-Fold Random Validation
print("\nPreparing Strategy 1: Stratified 5-Fold Random Validation...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
stratified_splits = list(skf.split(train_df, y))
f_metrics1, oof1 = run_validation_experiment("1. Stratified 5-Fold Random", stratified_splits)
all_fold_metrics.extend(f_metrics1)

# 2. StratifiedGroupKFold on (proto, service, id.resp_p)
print("\nPreparing Strategy 2: StratifiedGroupKFold on (proto, service, id.resp_p)...")
groups_combo = train_df['proto'].astype(str) + "_" + train_df['service'].astype(str) + "_" + train_df['id.resp_p'].astype(str)
sgkf_combo = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
combo_splits = list(sgkf_combo.split(train_df, y, groups=groups_combo))
f_metrics2, oof2 = run_validation_experiment("2. StratifiedGroupKFold (proto, service, resp_p)", combo_splits)
all_fold_metrics.extend(f_metrics2)

# 3. Destination-Port-Held-Out Validation
print("\nPreparing Strategy 3: Destination-Port-Held-Out Validation...")
groups_port = train_df['id.resp_p'].astype(str)
sgkf_port = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
port_splits = list(sgkf_port.split(train_df, y, groups=groups_port))
f_metrics3, oof3 = run_validation_experiment("3. Destination-Port-Held-Out (resp_p)", port_splits)
all_fold_metrics.extend(f_metrics3)

# 4. Traffic-Profile-Cluster-Held-Out Validation
print("\nPreparing Strategy 4: Traffic-Profile-Cluster-Held-Out Validation...")
# Cluster traffic based on physical protocol behavior:
# Cluster 0: IoT Telemetry (MQTT, NTP, DHCP)
# Cluster 1: Domain Resolution (DNS ports 53, 5353)
# Cluster 2: Web & Secure HTTP (HTTP port 80, SSL port 443)
# Cluster 3: Scanning & Ephemeral Probing (Service '-' on non-21 ports)
# Cluster 4: Volumetric FTP Flood (Port 21 FTP)
def assign_traffic_cluster(row):
    p = row['id.resp_p']
    s = row['service']
    if s == 'mqtt' or p in [1883.0, 123.0, 67.0, 68.0]:
        return 0 # IoT Telemetry
    elif s == 'dns' or p in [53.0, 5353.0]:
        return 1 # Domain Name Services
    elif s in ['http', 'ssl'] or p in [80.0, 443.0]:
        return 2 # Web & Encrypted Traffic
    elif p == 21.0:
        return 4 # Volumetric Flood
    else:
        return 3 # Scanning & Probing Sweeps

profile_clusters = train_df.apply(assign_traffic_cluster, axis=1).values
cluster_names = {
    0: "IoT Telemetry (MQTT/NTP)",
    1: "DNS Resolution (Port 53/5353)",
    2: "Web Applications (Port 80/443)",
    3: "Scanning Probes (Non-21 '-' Traffic)",
    4: "Volumetric Flood (Port 21 FTP)"
}

profile_splits = []
for c_idx in range(5):
    val_mask = (profile_clusters == c_idx)
    tr_idx = np.where(~val_mask)[0]
    va_idx = np.where(val_mask)[0]
    profile_splits.append((tr_idx, va_idx))

f_metrics4, oof4 = run_validation_experiment("4. Traffic-Profile-Cluster-Held-Out", profile_splits)
all_fold_metrics.extend(f_metrics4)

# ----------------------------------------------------------------------------------
# Compile and Save Reports
# ----------------------------------------------------------------------------------

fold_metrics_df = pd.DataFrame(all_fold_metrics)
fold_metrics_csv_path = os.path.join(reports_dir, "fold_metrics.csv")
fold_metrics_df.to_csv(fold_metrics_csv_path, index=False)
print(f"\nSuccessfully saved detailed fold metrics to: {fold_metrics_csv_path}")

# Calculate summary results (mean and std across folds)
summary_rows = []
for exp_name, group in fold_metrics_df.groupby('experiment', sort=False):
    summary_rows.append({
        'experiment': exp_name,
        'mean_macro_f1': group['macro_f1'].mean(),
        'std_macro_f1': group['macro_f1'].std(),
        'mean_macro_precision': group['macro_precision'].mean(),
        'std_macro_precision': group['macro_precision'].std(),
        'mean_macro_recall': group['macro_recall'].mean(),
        'std_macro_recall': group['macro_recall'].std(),
        'mean_f1_availability': group['f1_availability'].mean(),
        'std_f1_availability': group['f1_availability'].std(),
        'mean_f1_benign': group['f1_benign'].mean(),
        'std_f1_benign': group['f1_benign'].std(),
        'mean_f1_reconnaissance': group['f1_reconnaissance'].mean(),
        'std_f1_reconnaissance': group['f1_reconnaissance'].std(),
        'mean_f1_spoofing': group['f1_spoofing'].mean(),
        'std_f1_spoofing': group['f1_spoofing'].std()
    })

summary_df = pd.DataFrame(summary_rows)
summary_csv_path = os.path.join(reports_dir, "validation_results.csv")
summary_df.to_csv(summary_csv_path, index=False)
print(f"Successfully saved summary validation results to: {summary_csv_path}")

print("\n" + "="*80)
print("FINAL VALIDATION SUMMARY ACROSS EXPERIMENTS")
print("="*80)
print(summary_df[['experiment', 'mean_macro_f1', 'std_macro_f1', 'mean_f1_benign', 'mean_f1_reconnaissance', 'mean_f1_spoofing', 'mean_f1_availability']].to_string(index=False))
