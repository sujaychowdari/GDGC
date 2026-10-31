import pandas as pd
import numpy as np
import os

data_dir = r"d:\Sujay\GDGC_AIML\network-intrusion-detection2-ml-challenge"
train_path = os.path.join(data_dir, "train.csv")
test_path = os.path.join(data_dir, "test.csv")
sample_sub_path = os.path.join(data_dir, "sample_submission.csv")

print("Loading train...")
train_df = pd.read_csv(train_path)
print("Train shape:", train_df.shape)

print("Loading test...")
test_df = pd.read_csv(test_path)
print("Test shape:", test_df.shape)

print("\n--- Target Distribution ---")
if 'Target' in train_df.columns:
    print(train_df['Target'].value_counts())
    print("\nNormalized:")
    print(train_df['Target'].value_counts(normalize=True))

print("\n--- Missing values ---")
train_missing = train_df.isnull().sum()
test_missing = test_df.isnull().sum()
print("Train cols with missing:", train_missing[train_missing > 0])
print("Test cols with missing:", test_missing[test_missing > 0])

print("\n--- Column Data Types ---")
dtypes_counts = train_df.dtypes.value_counts()
print(dtypes_counts)

print("\n--- Categorical / Object columns ---")
obj_cols = train_df.select_dtypes(include=['object', 'category']).columns.tolist()
print("Object columns:", obj_cols)
for c in obj_cols:
    if c in ['id', 'Target']:
        continue
    train_u = set(train_df[c].dropna().unique())
    test_u = set(test_df[c].dropna().unique())
    print(f"\nCol '{c}': {len(train_u)} unique in train, {len(test_u)} unique in test")
    print(f"Overlap: {len(train_u.intersection(test_u))}, Only in train: {len(train_u - test_u)}, Only in test: {len(test_u - train_u)}")
    if len(train_u) < 20:
        print("Train values:", train_df[c].value_counts(dropna=False).to_dict())
        print("Test values:", test_df[c].value_counts(dropna=False).to_dict())

print("\n--- First few column names ---")
print(list(train_df.columns))

print("\n--- Check ID columns ---")
if 'id' in train_df.columns:
    print("Train id head:", train_df['id'].head().tolist())
if 'id' in test_df.columns:
    print("Test id head:", test_df['id'].head().tolist())
