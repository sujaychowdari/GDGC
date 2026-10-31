import pandas as pd
import numpy as np

train_df = pd.read_csv(r'D:\Sujay\GDGC_AIML\network-intrusion-detection2-ml-challenge\train.csv')
test_df = pd.read_csv(r'D:\Sujay\GDGC_AIML\network-intrusion-detection2-ml-challenge\test.csv')

print("="*60)
print("AUDIT ISSUE B: VALIDATION SCHEME & LEAKAGE")
print("="*60)

# Check port overlap between train and test
train_resp_ports = set(train_df['id.resp_p'].unique())
test_resp_ports = set(test_df['id.resp_p'].unique())
print(f"Total unique resp_p: Train={len(train_resp_ports)}, Test={len(test_resp_ports)}")
print(f"Overlap: {len(train_resp_ports & test_resp_ports)}")
print(f"Test ports NOT in train: {len(test_resp_ports - train_resp_ports)} / {len(test_resp_ports)} ({len(test_resp_ports - train_resp_ports)/len(test_resp_ports)*100:.1f}%)")

# How many test rows have a port never seen in train?
unseen_port_rows = test_df[test_df['id.resp_p'].isin(test_resp_ports - train_resp_ports)]
print(f"Test rows with unseen resp_p: {len(unseen_port_rows)} / {len(test_df)} ({len(unseen_port_rows)/len(test_df)*100:.2f}%)")

# Check orig_p overlap
train_orig_ports = set(train_df['id.orig_p'].unique())
test_orig_ports = set(test_df['id.orig_p'].unique())
print(f"\nTotal unique orig_p: Train={len(train_orig_ports)}, Test={len(test_orig_ports)}")
print(f"Test orig_p NOT in train: {len(test_orig_ports - train_orig_ports)} / {len(test_orig_ports)} ({len(test_orig_ports - train_orig_ports)/len(test_orig_ports)*100:.1f}%)")

# Check StratifiedKFold vs Grouping on traffic profile / port
print("\nIn standard StratifiedKFold, what fraction of validation rows share EXACT duplicate feature vectors with training rows?")
feature_cols = [c for c in train_df.columns if c not in ['id', 'Target']]
duplicates_in_train = train_df.duplicated(subset=feature_cols, keep=False).sum()
print(f"Exact feature duplicate rows in train: {duplicates_in_train} / {len(train_df)} ({duplicates_in_train/len(train_df)*100:.2f}%)")

# Duplicate feature rows with DIFFERENT targets? (Label noise / ambiguity)
dups = train_df[train_df.duplicated(subset=feature_cols, keep=False)]
dups_grouped = dups.groupby(feature_cols)['Target'].nunique()
conflicting_dups = dups_grouped[dups_grouped > 1]
print(f"Feature patterns with conflicting target labels: {len(conflicting_dups)}")
if len(conflicting_dups) > 0:
    total_conflicting_rows = dups[dups[feature_cols].apply(tuple, axis=1).isin(conflicting_dups.index)].shape[0]
    print(f"Total rows involved in conflicting duplicate patterns: {total_conflicting_rows}")
