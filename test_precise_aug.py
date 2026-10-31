import pandas as pd
import numpy as np
import os
import catboost as cb

data_dir = r"d:\Sujay\GDGC_AIML\network-intrusion-detection2-ml-challenge"
train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))

label_mapping = {'Availability': 0, 'Benign': 1, 'Reconnaissance': 2, 'Spoofing': 3}
inv_mapping = {v: k for k, v in label_mapping.items()}

# Let's find exactly all columns that are 0 in test t21_0:
t21_0 = test_df[(test_df['id.resp_p'] == 21.0) & (test_df['bwd_pkts_tot'] == 0.0)]
zero_cols = [c for c in t21_0.columns if c not in ['id', 'proto', 'service'] and t21_0[c].max() == 0 and t21_0[c].min() == 0]
print(f"Number of columns that are 0 in test t21_0: {len(zero_cols)}")
print("Zero cols:", zero_cols)

# Non-zero cols in test t21_0:
non_zero_cols = [c for c in t21_0.columns if c not in ['id', 'proto', 'service'] and c not in zero_cols]
print(f"\nNon-zero cols in test t21_0 ({len(non_zero_cols)}):")
for c in non_zero_cols:
    print(f"  {c:30s}: unique={t21_0[c].unique()}")

# Create augmented training set
augmented_train = train_df.copy()
syn_flood_mask = (augmented_train['Target'] == 'Availability') & (augmented_train['id.resp_p'] == 21.0)
drop_indices = augmented_train[syn_flood_mask].sample(frac=0.3, random_state=42).index

for c in zero_cols:
    augmented_train.loc[drop_indices, c] = 0.0

features = [c for c in train_df.columns if c not in ['id', 'Target']]
X_aug = augmented_train[features].copy()
y_aug = augmented_train['Target'].map(label_mapping).values
X_test = test_df[features].copy()

cat_features = ['proto', 'service']
for col in cat_features:
    X_aug[col] = X_aug[col].astype(str)
    X_test[col] = X_test[col].astype(str)

clf = cb.CatBoostClassifier(
    iterations=300,
    learning_rate=0.08,
    depth=6,
    cat_features=cat_features,
    auto_class_weights='Balanced',
    random_seed=42,
    verbose=0,
    thread_count=-1
)
clf.fit(X_aug, y_aug)

test_preds = clf.predict(X_test).ravel()
pred_series = pd.Series([inv_mapping[i] for i in test_preds])
print("\n=== Test Predictions Distribution after Precise Augmentation ===")
print(pred_series.value_counts())
print("\nPercentage:")
print(pred_series.value_counts(normalize=True))

test_eval = pd.DataFrame({'resp_p': test_df['id.resp_p'], 'service': test_df['service'], 'proto': test_df['proto'], 'pred': pred_series})
print("\nPredictions on Port 21 in Test:")
print(test_eval[test_eval['resp_p'] == 21.0]['pred'].value_counts())
