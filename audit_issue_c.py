import pandas as pd
import numpy as np

sub_ens = pd.read_csv(r'D:\Sujay\GDGC_AIML\submission_ensemble.py.csv')
sub_v2 = pd.read_csv(r'D:\Sujay\GDGC_AIML\submission_v2.csv')
test_df = pd.read_csv(r'D:\Sujay\GDGC_AIML\network-intrusion-detection2-ml-challenge\test.csv')

print("="*60)
print("AUDIT ISSUE C: HARDCODED RULES & REFINED SUBMISSION ANALYSIS")
print("="*60)

diff_mask = (sub_ens['Target'] != sub_v2['Target'])
print(f"Total differences between Baseline (0.84671) and Refined (0.84481): {diff_mask.sum()}")

diff_df = test_df[diff_mask].copy()
diff_df['baseline_pred'] = sub_ens.loc[diff_mask, 'Target']
diff_df['refined_pred'] = sub_v2.loc[diff_mask, 'Target']

print("\n--- Breakdown of All Changed Predictions ---")
print(pd.crosstab(diff_df['baseline_pred'], diff_df['refined_pred']))

print("\n--- Changed rows by destination port (id.resp_p) ---")
print(diff_df['id.resp_p'].value_counts())

print("\n--- Changed rows by originating port (id.orig_p) ---")
print(diff_df['id.orig_p'].value_counts().head(10))

print("\n--- Changed rows by Protocol & Service ---")
print(pd.crosstab(diff_df['proto'], diff_df['service']))

# Specifically inspect Port 5353 changed rows:
p5353_diff = diff_df[diff_df['id.resp_p'] == 5353.0]
print(f"\nPort 5353 changed rows count: {len(p5353_diff)}")
print("Baseline predictions on these rows:", p5353_diff['baseline_pred'].value_counts().to_dict())
print("Refined predictions on these rows: ", p5353_diff['refined_pred'].value_counts().to_dict())
print("Duration on these rows: median=", p5353_diff['flow_duration'].median(), "mean=", p5353_diff['flow_duration'].mean())
print("Payload on these rows: median=", p5353_diff['fwd_pkts_payload.tot'].median(), "mean=", p5353_diff['fwd_pkts_payload.tot'].mean())
print("bwd_pkts on these rows: median=", p5353_diff['bwd_pkts_tot'].median(), "max=", p5353_diff['bwd_pkts_tot'].max())

# Specifically inspect Port 21 changed rows:
p21_diff = diff_df[diff_df['id.resp_p'] == 21.0]
print(f"\nPort 21 changed rows count: {len(p21_diff)}")
print("Baseline predictions:", p21_diff['baseline_pred'].value_counts().to_dict())
print("Refined predictions: ", p21_diff['refined_pred'].value_counts().to_dict())
print("Features of these 5 rows:")
for idx, r in p21_diff.iterrows():
    print(f"  id: {r['id']}, orig_p: {r['id.orig_p']}, payload: {r['fwd_pkts_payload.tot']}, syn: {r['flow_SYN_flag_count']}, bwd_pkts: {r['bwd_pkts_tot']}")
