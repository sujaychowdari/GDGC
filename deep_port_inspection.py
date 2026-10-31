import pandas as pd
import numpy as np
import os

data_dir = r"d:\Sujay\GDGC_AIML\network-intrusion-detection2-ml-challenge"
train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))

print("=== PORT 53 (DNS) ===")
p53_train = train_df[train_df['id.resp_p'] == 53.0]
for target, group in p53_train.groupby('Target'):
    print(f"\nTarget {target} on Port 53 (n={len(group)}):")
    for f in ['flow_duration', 'fwd_pkts_tot', 'bwd_pkts_tot', 'fwd_pkts_payload.tot', 'bwd_pkts_payload.tot', 
              'fwd_pkts_payload.avg', 'bwd_pkts_payload.avg', 'flow_SYN_flag_count', 'flow_ACK_flag_count']:
        print(f"  {f:25s}: median={group[f].median():.2f}, mean={group[f].mean():.2f}, std={group[f].std():.2f}")

print("\n=== PORT 53 in Test ===")
p53_test = test_df[test_df['id.resp_p'] == 53.0]
print(f"Total on Port 53 in Test: {len(p53_test)}")
for f in ['flow_duration', 'fwd_pkts_tot', 'bwd_pkts_tot', 'fwd_pkts_payload.tot', 'bwd_pkts_payload.tot', 
          'fwd_pkts_payload.avg', 'bwd_pkts_payload.avg', 'flow_SYN_flag_count', 'flow_ACK_flag_count']:
    print(f"  {f:25s}: median={p53_test[f].median():.2f}, mean={p53_test[f].mean():.2f}, std={p53_test[f].std():.2f}")

print("\n=== PORT 5353 (mDNS) in Train vs Test ===")
p5353_train = train_df[train_df['id.resp_p'] == 5353.0]
p5353_test = test_df[test_df['id.resp_p'] == 5353.0]
print("Train targets:", p5353_train['Target'].value_counts().to_dict())
print("Train fwd_payload.tot:", p5353_train.groupby('Target')['fwd_pkts_payload.tot'].describe())
print("Test fwd_payload.tot:", p5353_test['fwd_pkts_payload.tot'].describe())
print("Test bwd_pkts_tot:", p5353_test['bwd_pkts_tot'].value_counts().to_dict())
print("Test duration:", p5353_test['flow_duration'].describe())

print("\n=== PORT 443 (SSL/HTTPS) ===")
p443_train = train_df[train_df['id.resp_p'] == 443.0]
for target, group in p443_train.groupby('Target'):
    print(f"\nTarget {target} on Port 443 (n={len(group)}):")
    for f in ['flow_duration', 'fwd_pkts_tot', 'bwd_pkts_tot', 'fwd_pkts_payload.tot', 'bwd_pkts_payload.tot', 
              'fwd_pkts_payload.avg', 'bwd_pkts_payload.avg']:
        print(f"  {f:25s}: median={group[f].median():.2f}, mean={group[f].mean():.2f}")

print("\n=== PORT 443 in Test ===")
p443_test = test_df[test_df['id.resp_p'] == 443.0]
for f in ['flow_duration', 'fwd_pkts_tot', 'bwd_pkts_tot', 'fwd_pkts_payload.tot', 'bwd_pkts_payload.tot', 
          'fwd_pkts_payload.avg', 'bwd_pkts_payload.avg']:
    print(f"  {f:25s}: median={p443_test[f].median():.2f}, mean={p443_test[f].mean():.2f}")
