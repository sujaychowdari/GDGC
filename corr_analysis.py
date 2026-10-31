import pandas as pd
import numpy as np

train_df = pd.read_csv(r'D:\Sujay\GDGC_AIML\network-intrusion-detection2-ml-challenge\train.csv')
test_df = pd.read_csv(r'D:\Sujay\GDGC_AIML\network-intrusion-detection2-ml-challenge\test.csv')

print("=== Collinear Feature Pairs in Train (Correlation > 0.98) ===")
num_cols = [c for c in train_df.columns if c not in ['id', 'Target', 'proto', 'service']]
corr_matrix = train_df[num_cols].corr().abs()

# Find upper triangle pairs
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
high_corr = upper_tri.stack().reset_index()
high_corr.columns = ['Feature 1', 'Feature 2', 'Correlation']
high_corr = high_corr[high_corr['Correlation'] > 0.98].sort_values(by='Correlation', ascending=False)
print(f"Total collinear pairs with r > 0.98: {len(high_corr)}")
print(high_corr.head(20).to_string(index=False))

print("\n=== Feature Discriminability (ANOVA F-statistic / Mutual Info proxy) ===")
from sklearn.feature_selection import f_classif
label_mapping = {'Availability': 0, 'Benign': 1, 'Reconnaissance': 2, 'Spoofing': 3}
y = train_df['Target'].map(label_mapping).values

f_scores, p_vals = f_classif(train_df[num_cols], y)
f_df = pd.DataFrame({'feature': num_cols, 'f_statistic': f_scores, 'p_value': p_vals}).sort_values(by='f_statistic', ascending=False)
print("Top 15 Most Discriminative Features by F-Statistic:")
print(f_df.head(15).to_string(index=False))

print("\nBottom 10 Least Discriminative Features by F-Statistic:")
print(f_df.tail(10).to_string(index=False))
