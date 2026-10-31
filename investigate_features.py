import pandas as pd
import numpy as np
import os
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

data_dir = r"d:\Sujay\GDGC_AIML\network-intrusion-detection2-ml-challenge"
train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))

features = [c for c in train_df.columns if c not in ['id', 'Target', 'proto', 'service']]

# Subsample train to avoid 80k port 21 dominating adversarial validation
train_sub = train_df.sample(n=min(len(train_df), 35000), random_state=42)
test_sub = test_df.sample(n=min(len(test_df), 33000), random_state=42)

X_adv = pd.concat([train_sub[features], test_sub[features]], ignore_index=True)
y_adv = np.array([0] * len(train_sub) + [1] * len(test_sub))

clf = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42, n_jobs=-1)
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X_adv))
for train_idx, val_idx in skf.split(X_adv, y_adv):
    clf.fit(X_adv.iloc[train_idx], y_adv[train_idx])
    oof_preds[val_idx] = clf.predict_proba(X_adv.iloc[val_idx])[:, 1]

auc = roc_auc_score(y_adv, oof_preds)
print(f"Adversarial Validation AUC: {auc:.4f}")

# Train on full adv set to get feature importances
clf.fit(X_adv, y_adv)
fi = pd.Series(clf.feature_importances_, index=features).sort_values(ascending=False)
print("\nTop 20 features driving train vs test shift (Adversarial Importance):")
print(fi.head(20))
