import pandas as pd

train_df = pd.read_csv(r'D:\Sujay\GDGC_AIML\network-intrusion-detection2-ml-challenge\train.csv')
test_df = pd.read_csv(r'D:\Sujay\GDGC_AIML\network-intrusion-detection2-ml-challenge\test.csv')

print("="*60)
print("AUDIT ISSUE A: CATEGORICAL ENCODING IN XGBOOST")
print("="*60)

for col in ['proto', 'service']:
    s_tr = train_df[col].astype('category')
    s_te = test_df[col].astype('category')
    print(f"\n--- Column: {col} ---")
    print("Train categories:", dict(enumerate(s_tr.cat.categories)))
    print("Test categories: ", dict(enumerate(s_te.cat.categories)))
    
    train_map = dict(enumerate(s_tr.cat.categories))
    test_map = dict(enumerate(s_te.cat.categories))
    
    mismatches = []
    for code, cat in test_map.items():
        tr_cat = train_map.get(code, None)
        if tr_cat != cat:
            mismatches.append((code, tr_cat, cat))
            
    if mismatches:
        print(f"CRITICAL BUG CONFIRMED for '{col}'!")
        for code, tr_cat, te_cat in mismatches:
            print(f"  Code {code}: In Train = '{tr_cat}' | In Test = '{te_cat}'")
    else:
        print(f"No code mismatch for '{col}'.")
