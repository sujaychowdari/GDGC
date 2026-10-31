import pandas as pd
import numpy as np
import os
import catboost as cb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report

data_dir = r"d:\Sujay\GDGC_AIML\network-intrusion-detection2-ml-challenge"
train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))

label_mapping = {'Availability': 0, 'Benign': 1, 'Reconnaissance': 2, 'Spoofing': 3}
inv_mapping = {v: k for k, v in label_mapping.items()}
y = train_df['Target'].map(label_mapping).values

features = [c for c in train_df.columns if c not in ['id', 'Target']]
X = train_df[features].copy()
X_test = test_df[features].copy()

# Categoricals for CatBoost
cat_features = ['proto', 'service']
for col in cat_features:
    X[col] = X[col].astype(str)
    X_test[col] = X_test[col].astype(str)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_probs = np.zeros((len(train_df), 4))
test_probs = np.zeros((len(test_df), 4))

print("Starting CatBoost 5-fold CV...")
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_val, y_val = X.iloc[val_idx], y[val_idx]
    
    clf = cb.CatBoostClassifier(
        iterations=400,
        learning_rate=0.08,
        depth=6,
        cat_features=cat_features,
        auto_class_weights='Balanced',
        random_seed=42 + fold,
        verbose=100,
        thread_count=-1
    )
    clf.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=40, verbose=False)
    
    val_pred_p = clf.predict_proba(X_val)
    oof_probs[val_idx] = val_pred_p
    fold_f1 = f1_score(y_val, np.argmax(val_pred_p, axis=1), average='macro')
    print(f"Fold {fold} Macro F1: {fold_f1:.5f}")
    
    test_probs += clf.predict_proba(X_test) / 5.0

oof_pred_labels = np.argmax(oof_probs, axis=1)
total_f1 = f1_score(y, oof_pred_labels, average='macro')
print(f"\nOverall CatBoost OOF Macro F1: {total_f1:.5f}")
print("\nClassification Report:")
print(classification_report(y, oof_pred_labels, target_names=['Availability', 'Benign', 'Reconnaissance', 'Spoofing'], digits=5))

test_preds = np.argmax(test_probs, axis=1)
pred_series = pd.Series([inv_mapping[i] for i in test_preds])
print("\n=== CatBoost Test Predictions Distribution ===")
print(pred_series.value_counts())
print("\nPercentage:")
print(pred_series.value_counts(normalize=True))

test_eval_df = pd.DataFrame({'proto': test_df['proto'], 'service': test_df['service'], 'resp_p': test_df['id.resp_p'], 'pred': pred_series})
print("\n=== Test Predictions by Protocol ===")
print(pd.crosstab(test_eval_df['proto'], test_eval_df['pred'], margins=True))

print("\n=== Test Predictions by Service ===")
print(pd.crosstab(test_eval_df['service'], test_eval_df['pred'], margins=True))
