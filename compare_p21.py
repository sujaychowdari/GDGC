import pandas as pd
import numpy as np
import os

data_dir = r"d:\Sujay\GDGC_AIML\network-intrusion-detection2-ml-challenge"
train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))

train_p21 = train_df[train_df['id.resp_p'] == 21.0]
test_p21 = test_df[test_df['id.resp_p'] == 21.0]

print("Train Port 21 count:", len(train_p21))
print("Test Port 21 count:", len(test_p21))

# Compare feature values between train p21 and test p21
diff_cols = []
for c in train_p21.columns:
    if c in ['id', 'Target', 'proto', 'service']:
        continue
    train_mean = train_p21[c].mean()
    test_mean = test_p21[c].mean()
    if abs(train_mean - test_mean) > 1e-4:
        diff_cols.append((c, train_mean, test_mean))

print(f"Features with differences between train and test Port 21 ({len(diff_cols)}):")
for c, tm, sm in diff_cols[:25]:
    print(f"  {c:30s}: train_mean={tm:.4f}, test_mean={sm:.4f}")

# What features are IDENTICAL or near-identical?
same_cols = []
for c in train_p21.columns:
    if c in ['id', 'Target', 'proto', 'service']:
        continue
    train_mean = train_p21[c].mean()
    test_mean = test_p21[c].mean()
    if abs(train_mean - test_mean) <= 1e-4 and train_p21[c].std() < 1e-4:
        same_cols.append(c)

print(f"\nFeatures that are IDENTICAL in both train and test Port 21 ({len(same_cols)}):")
print(same_cols)
