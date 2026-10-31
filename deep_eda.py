import pandas as pd
import numpy as np
import os

data_dir = r"d:\Sujay\GDGC_AIML\network-intrusion-detection2-ml-challenge"
train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))

print("=== Train vs Test Shapes ===")
print("Train:", train_df.shape)
print("Test:", test_df.shape)

print("\n=== Check Feature Names ===")
train_features = [c for c in train_df.columns if c not in ['id', 'Target']]
test_features = [c for c in test_df.columns if c not in ['id']]
print("Features match:", set(train_features) == set(test_features))

print("\n=== Summary Stats for Key Features in Train vs Test ===")
cols_to_check = [
    'flow_duration', 'fwd_pkts_tot', 'bwd_pkts_tot', 'flow_pkts_per_sec',
    'payload_bytes_per_second', 'flow_SYN_flag_count', 'flow_FIN_flag_count',
    'flow_ACK_flag_count', 'fwd_pkts_payload.tot', 'bwd_pkts_payload.tot'
]

stats_train = train_df[cols_to_check].describe().T[['mean', 'std', '50%', 'max']]
stats_test = test_df[cols_to_check].describe().T[['mean', 'std', '50%', 'max']]
print("--- Train Stats ---")
print(stats_train)
print("\n--- Test Stats ---")
print(stats_test)

# Let's inspect Target characteristics in train
print("\n=== Target Stats in Train ===")
for target, group in train_df.groupby('Target'):
    print(f"\nTarget: {target} (count={len(group)})")
    print(f"  proto: {group['proto'].value_counts().to_dict()}")
    print(f"  top 5 service: {group['service'].value_counts().head(5).to_dict()}")
    print(f"  top 5 resp_p: {group['id.resp_p'].value_counts().head(5).to_dict()}")
    print(f"  median flow_duration: {group['flow_duration'].median()}")
    print(f"  median flow_pkts_per_sec: {group['flow_pkts_per_sec'].median()}")
    print(f"  median flow_SYN_flag_count: {group['flow_SYN_flag_count'].median()}")
    print(f"  median flow_ACK_flag_count: {group['flow_ACK_flag_count'].median()}")

# Let's check test set port distribution
print("\n=== Test set top 15 resp_p ===")
print(test_df['id.resp_p'].value_counts().head(15))

print("\n=== Train set top 15 resp_p ===")
print(train_df['id.resp_p'].value_counts().head(15))
