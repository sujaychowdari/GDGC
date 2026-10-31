import pandas as pd

base = pd.read_csv(r'D:\Sujay\GDGC_AIML\baseline_084671.csv')
cb = pd.read_csv(r'D:\Sujay\GDGC_AIML\reports\test_predictions_catboost.csv')
lgb = pd.read_csv(r'D:\Sujay\GDGC_AIML\reports\test_predictions_lightgbm.csv')
xgb = pd.read_csv(r'D:\Sujay\GDGC_AIML\reports\test_predictions_xgboost.csv')
ens = pd.read_csv(r'D:\Sujay\GDGC_AIML\reports\ensemble_test_predictions.csv')

print('--- TEST PREDICTION CLASS DISTRIBUTIONS (33,161 samples) ---')
dist_df = pd.DataFrame({
    'Baseline (0.84671)': base['Target'].value_counts(),
    'CatBoost': cb['Target'].value_counts(),
    'LightGBM': lgb['Target'].value_counts(),
    'XGBoost': xgb['Target'].value_counts(),
    'Ensemble (Tri-Model)': ens['Target'].value_counts(),
})
print(dist_df)
print('\nPercentage Distribution:')
print((dist_df / 33161 * 100).round(2))

print('\nAgreement with Original Baseline (0.84671):')
print(f"  CatBoost vs Baseline:  {(cb['Target'] == base['Target']).mean() * 100:.2f}%")
print(f"  LightGBM vs Baseline:  {(lgb['Target'] == base['Target']).mean() * 100:.2f}%")
print(f"  XGBoost vs Baseline:   {(xgb['Target'] == base['Target']).mean() * 100:.2f}%")
print(f"  Ensemble vs Baseline:  {(ens['Target'] == base['Target']).mean() * 100:.2f}%")

# Disagreements breakdown
diff_mask = cb['Target'] != base['Target']
print(f'\nTotal disagreements between CatBoost and Baseline: {diff_mask.sum()}')
ct_cb_base = pd.crosstab(base['Target'], cb['Target'], rownames=['Baseline'], colnames=['CatBoost'])
print('\nBaseline vs CatBoost Contingency:')
print(ct_cb_base)

ct_lgb_base = pd.crosstab(base['Target'], lgb['Target'], rownames=['Baseline'], colnames=['LightGBM'])
print('\nBaseline vs LightGBM Contingency:')
print(ct_lgb_base)
