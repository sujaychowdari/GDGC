import pandas as pd
import numpy as np
import os

data_dir = r"d:\Sujay\GDGC_AIML\network-intrusion-detection2-ml-challenge"
train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))

print("=== Target Breakdown by Protocol ===")
print(pd.crosstab(train_df['proto'], train_df['Target'], margins=True))

print("\n=== Target Breakdown by Service ===")
print(pd.crosstab(train_df['service'], train_df['Target'], margins=True))

print("\n=== Test Proto breakdown ===")
print(test_df['proto'].value_counts())

print("\n=== Test Service breakdown ===")
print(test_df['service'].value_counts())

# Check constant features in train or test
numeric_cols = [c for c in train_df.columns if c not in ['id', 'Target', 'proto', 'service']]
zero_var_train = [c for c in numeric_cols if train_df[c].nunique() <= 1]
zero_var_test = [c for c in numeric_cols if test_df[c].nunique() <= 1]
print(f"\nZero variance cols in train ({len(zero_var_train)}):", zero_var_train)
print(f"Zero variance cols in test ({len(zero_var_test)}):", zero_var_test)

# Let's inspect ports: id.orig_p and id.resp_p
print("\nTrain top 10 resp_p:")
print(train_df['id.resp_p'].value_counts().head(10))
print("\nTest top 10 resp_p:")
print(test_df['id.resp_p'].value_counts().head(10))

print("\nTrain top 10 orig_p:")
print(train_df['id.orig_p'].value_counts().head(10))
print("\nTest top 10 orig_p:")
print(test_df['id.orig_p'].value_counts().head(10))

# Target vs resp_p
print("\nTarget by top resp_p in train:")
top_resp_p = train_df['id.resp_p'].value_counts().head(8).index
print(pd.crosstab(train_df['id.resp_p'].isin(top_resp_p).map({True: 'Top', False: 'Other'}), train_df['Target']))

for p in [80, 21, 22, 53, 1883, 443]:
    sub = train_df[train_df['id.resp_p'] == p]
    if len(sub) > 0:
        print(f"Port {p} (count={len(sub)}):", sub['Target'].value_counts().to_dict())
