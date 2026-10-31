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

label_mapping = {'Availability': 0, 'Benign': 1, 'Reconnaissance': 2, 'Spoofing': 3}
inv_mapping = {v: k for k, v in label_mapping.items()}

# Let's inspect what happens if we augment train with unresponsive server SYN floods:
# In train, there are 79,698 Availability rows on Port 21 where bwd_pkts_tot=1, ACK=1, RST=1.
# What if 20% of them have bwd features set to 0 (server dropped response)?
augmented_train = train_df.copy()
syn_flood_mask = (augmented_train['Target'] == 'Availability') & (augmented_train['id.resp_p'] == 21.0)
drop_indices = augmented_train[syn_flood_mask].sample(frac=0.3, random_state=42).index

# Zero out backward/server response features for these dropped samples
bwd_cols = [c for c in augmented_train.columns if 'bwd' in c or 'ACK' in c or 'RST' in c]
for c in bwd_cols:
    augmented_train.loc[drop_indices, c] = 0.0
augmented_train.loc[drop_indices, 'flow_duration'] = 0.0
augmented_train.loc[drop_indices, 'flow_pkts_per_sec'] = 0.0
augmented_train.loc[drop_indices, 'payload_bytes_per_second'] = 0.0

print(f"Augmented {len(drop_indices)} samples of Availability to have unresponsive server profiles.")

# Train CatBoost and check test predictions
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
print("\n=== Test Predictions Distribution after Server Unresponsive Augmentation ===")
print(pred_series.value_counts())
print("\nPercentage:")
print(pred_series.value_counts(normalize=True))

test_eval = pd.DataFrame({'resp_p': test_df['id.resp_p'], 'service': test_df['service'], 'proto': test_df['proto'], 'pred': pred_series})
print("\nPredictions on Port 21 in Test:")
print(test_eval[test_eval['resp_p'] == 21.0]['pred'].value_counts())

print("\nPredictions on Port 1883 in Test:")
print(test_eval[test_eval['resp_p'] == 1883.0]['pred'].value_counts())

print("\nPredictions on Port 53 in Test:")
print(test_eval[test_eval['resp_p'] == 53.0]['pred'].value_counts())

print("\nPredictions on Port 80 in Test:")
print(test_eval[test_eval['resp_p'] == 80.0]['pred'].value_counts())

print("\nPredictions on Other Ports in Test:")
print(test_eval[~test_eval['resp_p'].isin([21.0, 1883.0, 53.0, 80.0, 443.0, 5353.0])]['pred'].value_counts())
