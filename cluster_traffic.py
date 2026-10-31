import pandas as pd
import numpy as np
import os

data_dir = r"d:\Sujay\GDGC_AIML\network-intrusion-detection2-ml-challenge"
train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))

print("=== Train Target Profiles ===")
for target in ['Availability', 'Benign', 'Reconnaissance', 'Spoofing']:
    sub = train_df[train_df['Target'] == target]
    print(f"\n--- Target: {target} (Total: {len(sub)}) ---")
    print(f"Top 5 resp_p: {sub['id.resp_p'].value_counts().head(5).to_dict()}")
    print(f"Protocols: {sub['proto'].value_counts().to_dict()}")
    print(f"Services: {sub['service'].value_counts().head(5).to_dict()}")
    print(f"Payload Tot median: {sub['fwd_pkts_payload.tot'].median()}, IQR: [{sub['fwd_pkts_payload.tot'].quantile(0.25)}, {sub['fwd_pkts_payload.tot'].quantile(0.75)}]")
    print(f"SYN count: {sub['flow_SYN_flag_count'].value_counts(normalize=True).head(3).to_dict()}")
    print(f"FIN count: {sub['flow_FIN_flag_count'].value_counts(normalize=True).head(3).to_dict()}")
    print(f"ACK count: {sub['flow_ACK_flag_count'].value_counts(normalize=True).head(3).to_dict()}")
    print(f"RST count: {sub['flow_RST_flag_count'].value_counts(normalize=True).head(3).to_dict()}")

# Now inspect Port 80 in train
print("\n=== Port 80 in Train ===")
p80_train = train_df[train_df['id.resp_p'] == 80.0]
print(p80_train['Target'].value_counts())
for target in p80_train['Target'].unique():
    t_sub = p80_train[p80_train['Target'] == target]
    print(f"  Target {target} (n={len(t_sub)}): duration_median={t_sub['flow_duration'].median():.4f}, fwd_pkts_median={t_sub['fwd_pkts_tot'].median()}, bwd_pkts_median={t_sub['bwd_pkts_tot'].median()}, fwd_payload_median={t_sub['fwd_pkts_payload.tot'].median()}")

# Inspect Port 80 in test
print("\n=== Port 80 in Test ===")
p80_test = test_df[test_df['id.resp_p'] == 80.0]
print(f"Total port 80 in test: {len(p80_test)}")
print(f"Services: {p80_test['service'].value_counts().to_dict()}")
print(f"fwd_payload distribution: {p80_test['fwd_pkts_payload.tot'].describe()}")
