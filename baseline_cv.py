import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
import lightgbm as lgb

data_dir = r"d:\Sujay\GDGC_AIML\network-intrusion-detection2-ml-challenge"
train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))

label_mapping = {'Availability': 0, 'Benign': 1, 'Reconnaissance': 2, 'Spoofing': 3}
inv_mapping = {v: k for k, v in label_mapping.items()}
y = train_df['Target'].map(label_mapping).values

features = [c for c in train_df.columns if c not in ['id', 'Target']]
X = train_df[features].copy()

# Categorical encoding
for col in ['proto', 'service']:
    X[col] = X[col].astype('category')

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros((len(train_df), 4))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_val, y_val = X.iloc[val_idx], y[val_idx]
    
    clf = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1,
        verbose=-1
    )
    clf.fit(X_train, y_train)
    oof_preds[val_idx] = clf.predict_proba(X_val)
    fold_pred = np.argmax(oof_preds[val_idx], axis=1)
    fold_f1 = f1_score(y_val, fold_pred, average='macro')
    print(f"Fold {fold} Macro F1: {fold_f1:.5f}")

oof_pred_labels = np.argmax(oof_preds, axis=1)
total_f1 = f1_score(y, oof_pred_labels, average='macro')
print(f"\nOverall OOF Macro F1: {total_f1:.5f}")
print("\nClassification Report:")
print(classification_report(y, oof_pred_labels, target_names=['Availability', 'Benign', 'Reconnaissance', 'Spoofing'], digits=5))
