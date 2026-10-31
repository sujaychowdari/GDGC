import hashlib
import os
import pandas as pd

files = [
    r'D:\Sujay\GDGC_AIML\baseline_084671.csv',
    r'D:\Sujay\GDGC_AIML\submission.csv',
    r'D:\Sujay\GDGC_AIML\submission_ensemble.py.csv',
    r'D:\Sujay\GDGC_AIML\submission_v2.csv',
    r'D:\Sujay\GDGC_AIML\network-intrusion-detection2-ml-challenge\submission.csv',
    r'D:\Sujay\GDGC_AIML\network-intrusion-detection2-ml-challenge\sample_submission.csv'
]

test_df = pd.read_csv(r'D:\Sujay\GDGC_AIML\network-intrusion-detection2-ml-challenge\test.csv')
valid_labels = {'Availability', 'Benign', 'Reconnaissance', 'Spoofing'}

print(f"Test shape: {test_df.shape}")

results = []

for f in files:
    with open(f, 'rb') as fp:
        sha256 = hashlib.sha256(fp.read()).hexdigest()
    df = pd.read_csv(f)
    rows = len(df)
    cols = list(df.columns)
    id_match = (df['id'] == test_df['id']).all() if 'id' in df.columns else False
    unique_labels = set(df['Target'].unique()) if 'Target' in df.columns else set()
    labels_valid = unique_labels.issubset(valid_labels)
    null_count = int(df.isnull().sum().sum())
    val_counts = df['Target'].value_counts().to_dict() if 'Target' in df.columns else {}
    
    results.append({
        'path': f,
        'sha256': sha256,
        'rows': rows,
        'cols': cols,
        'nulls': null_count,
        'id_match': id_match,
        'val_counts': val_counts,
        'labels_valid': labels_valid
    })

for r in results:
    print(f"\nFile: {os.path.basename(r['path'])} ({r['path']})")
    print(f"  SHA256: {r['sha256']}")
    print(f"  Rows: {r['rows']} (33161: {r['rows'] == 33161})")
    print(f"  Cols: {r['cols']}")
    print(f"  Null count: {r['nulls']}")
    print(f"  IDs exact match: {r['id_match']}")
    print(f"  Labels valid: {r['labels_valid']}")
    print(f"  Class counts: {r['val_counts']}")
