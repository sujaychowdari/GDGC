import pandas as pd
import numpy as np

train_df = pd.read_csv(r'D:\Sujay\GDGC_AIML\network-intrusion-detection2-ml-challenge\train.csv')
test_df = pd.read_csv(r'D:\Sujay\GDGC_AIML\network-intrusion-detection2-ml-challenge\test.csv')

print("="*60)
print("AUDIT ISSUE F: ID ANALYSIS, DUPLICATES, AND POTENTIAL LEAKAGE")
print("="*60)

# Check ID column structure
print("Train ID sample:")
print(train_df['id'].head(5))
print("Test ID sample:")
print(test_df['id'].head(5))

# Check length of ID strings
print("Train ID lengths:", train_df['id'].str.len().unique())
print("Test ID lengths: ", test_df['id'].str.len().unique())

# Check hex characters in ID
train_hex = train_df['id'].str.replace('train_', '')
test_hex = test_df['id'].str.replace('test_', '')
print("Train hex length:", train_hex.str.len().unique())
print("Test hex length: ", test_hex.str.len().unique())

# Check if hex strings have any ordering or timestamp structure (e.g. first 8 chars = timestamp?)
first_8 = train_hex.str[:8].apply(lambda x: int(x, 16))
print("First 8 hex chars as int (train): min=", first_8.min(), "max=", first_8.max(), "std=", first_8.std())
first_8_test = test_hex.str[:8].apply(lambda x: int(x, 16))
print("First 8 hex chars as int (test): min=", first_8_test.min(), "max=", first_8_test.max(), "std=", first_8_test.std())

# Check duplicates within train
features = [c for c in train_df.columns if c not in ['id', 'Target']]
train_dup_count = train_df.duplicated(subset=features, keep=False).sum()
print(f"\nDuplicate feature rows within train: {train_dup_count} / {len(train_df)}")

# Check duplicates within test
test_dup_count = test_df.duplicated(subset=features, keep=False).sum()
print(f"Duplicate feature rows within test: {test_dup_count} / {len(test_df)}")

# Check overlap between train and test features (Exact matches)
train_tuples = set(map(tuple, train_df[features].values))
test_tuples = set(map(tuple, test_df[features].values))
overlap = train_tuples & test_tuples
print(f"\nUnique feature vectors in train: {len(train_tuples)}")
print(f"Unique feature vectors in test:  {len(test_tuples)}")
print(f"Exact matching feature vectors between train and test: {len(overlap)}")

# How many rows in test EXACTLY match a row in train?
test_matches = test_df[features].apply(tuple, axis=1).isin(overlap)
print(f"Total test rows that have an exact feature match in train: {test_matches.sum()} / {len(test_df)} ({test_matches.sum()/len(test_df)*100:.2f}%)")

if len(overlap) > 0:
    # Check target labels of the overlapping rows in train
    overlapping_train = train_df[train_df[features].apply(tuple, axis=1).isin(overlap)]
    print("\nTarget labels in train for the overlapping feature vectors:")
    print(overlapping_train['Target'].value_counts())
    print("\nCheck if any overlapping feature vector has conflicting labels in train:")
    grp = overlapping_train.groupby(features)['Target'].nunique()
    print("Conflicting vectors in overlap:", (grp > 1).sum())
