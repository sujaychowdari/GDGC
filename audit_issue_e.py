import pandas as pd
import numpy as np

train_df = pd.read_csv(r'D:\Sujay\GDGC_AIML\network-intrusion-detection2-ml-challenge\train.csv')
test_df = pd.read_csv(r'D:\Sujay\GDGC_AIML\network-intrusion-detection2-ml-challenge\test.csv')

print("="*60)
print("AUDIT ISSUE E: MATHEMATICAL CONSISTENCY OF ENGINEERED FEATURES")
print("="*60)

# Simulate the exact steps of train_ensemble.py
def engineer_features(df):
    data = df.copy()
    fwd_pkts = np.maximum(1.0, data['fwd_pkts_tot'])
    bwd_pkts = np.maximum(1.0, data['bwd_pkts_tot'])
    
    data['fwd_payload_per_pkt'] = data['fwd_pkts_payload.tot'] / fwd_pkts
    data['bwd_payload_per_pkt'] = data['bwd_pkts_payload.tot'] / bwd_pkts
    data['fwd_header_per_pkt'] = data['fwd_header_size_tot'] / fwd_pkts
    data['bwd_header_per_pkt'] = data['bwd_header_size_tot'] / bwd_pkts
    
    data['has_fwd_payload'] = (data['fwd_pkts_payload.tot'] > 0).astype(float)
    data['has_bwd_payload'] = (data['bwd_pkts_payload.tot'] > 0).astype(float)
    
    data['is_syn_only'] = ((data['flow_SYN_flag_count'] > 0) & (data['flow_ACK_flag_count'] == 0)).astype(float)
    data['is_fin_only'] = ((data['flow_FIN_flag_count'] > 0) & (data['flow_ACK_flag_count'] == 0)).astype(float)
    data['is_xmas_scan'] = ((data['flow_FIN_flag_count'] > 0) & (data['fwd_PSH_flag_count'] > 0) & (data['fwd_URG_flag_count'] > 0)).astype(float)
    
    data['is_syn_flood_sig'] = ((data['proto'] == 'tcp') & (data['flow_SYN_flag_count'] >= 1) & (data['fwd_pkts_payload.tot'] == 120.0)).astype(float)
    data['is_port_21'] = (data['id.resp_p'] == 21.0).astype(float)
    data['is_port_1883'] = (data['id.resp_p'] == 1883.0).astype(float)
    data['is_port_53'] = (data['id.resp_p'] == 53.0).astype(float)
    data['is_port_80'] = (data['id.resp_p'] == 80.0).astype(float)
    data['is_port_443'] = (data['id.resp_p'] == 443.0).astype(float)
    data['is_port_5353'] = (data['id.resp_p'] == 5353.0).astype(float)
    
    data['is_well_known_resp'] = (data['id.resp_p'] <= 1024).astype(float)
    data['is_ephemeral_resp'] = (data['id.resp_p'] >= 49152).astype(float)
    
    data['log_duration'] = np.log1p(np.maximum(0, data['flow_duration']))
    data['log_flow_pkts_per_sec'] = np.log1p(np.maximum(0, data['flow_pkts_per_sec']))
    data['log_payload_bytes_per_sec'] = np.log1p(np.maximum(0, data['payload_bytes_per_second']))
    
    return data

train_fe = engineer_features(train_df)

t21_0 = test_df[(test_df['id.resp_p'] == 21.0) & (test_df['bwd_pkts_tot'] == 0.0)]
zero_cols = [c for c in t21_0.columns if c not in ['id', 'proto', 'service'] and t21_0[c].max() == 0 and t21_0[c].min() == 0]

augmented_train = train_fe.copy()
syn_flood_mask = (augmented_train['Target'] == 'Availability') & (augmented_train['id.resp_p'] == 21.0)
drop_indices = augmented_train[syn_flood_mask].sample(frac=0.35, random_state=42).index

for c in zero_cols:
    if c in augmented_train.columns:
        augmented_train.loc[drop_indices, c] = 0.0

augmented_train.loc[drop_indices, 'bwd_payload_per_pkt'] = 0.0
augmented_train.loc[drop_indices, 'bwd_header_per_pkt'] = 0.0
augmented_train.loc[drop_indices, 'has_bwd_payload'] = 0.0
augmented_train.loc[drop_indices, 'is_syn_only'] = 1.0
augmented_train.loc[drop_indices, 'log_duration'] = 0.0
augmented_train.loc[drop_indices, 'log_flow_pkts_per_sec'] = 0.0
augmented_train.loc[drop_indices, 'log_payload_bytes_per_sec'] = 0.0

# Now verify consistency on drop_indices:
print("Checking inconsistencies on augmented rows:")
inconsistencies = []

# Check 1: down_up_ratio vs bwd_pkts_tot / fwd_pkts_tot
# In raw train, down_up_ratio was 1.0. Was it zeroed?
print(f"  down_up_ratio on augmented rows: unique = {augmented_train.loc[drop_indices, 'down_up_ratio'].unique()}")

# Check 2: flow_SYN_flag_count vs flow_ACK_flag_count
print(f"  flow_SYN_flag_count: {augmented_train.loc[drop_indices, 'flow_SYN_flag_count'].unique()}")
print(f"  flow_ACK_flag_count: {augmented_train.loc[drop_indices, 'flow_ACK_flag_count'].unique()}")
print(f"  is_syn_only: {augmented_train.loc[drop_indices, 'is_syn_only'].unique()}")

# Check 3: What about other ratio features or raw features not in zero_cols?
for c in augmented_train.columns:
    if 'bwd' in c:
        u = augmented_train.loc[drop_indices, c].unique()
        if len(u) > 1 or (len(u) == 1 and u[0] != 0.0):
            print(f"  Inconsistent backward column '{c}' is NOT zero! Unique: {u}")

# Check 4: What about flow_duration and derived fields?
print(f"  flow_duration: {augmented_train.loc[drop_indices, 'flow_duration'].unique()}")
print(f"  flow_pkts_per_sec: {augmented_train.loc[drop_indices, 'flow_pkts_per_sec'].unique()}")
print(f"  payload_bytes_per_second: {augmented_train.loc[drop_indices, 'payload_bytes_per_second'].unique()}")
print(f"  log_duration: {augmented_train.loc[drop_indices, 'log_duration'].unique()}")
print(f"  log_flow_pkts_per_sec: {augmented_train.loc[drop_indices, 'log_flow_pkts_per_sec'].unique()}")
print(f"  log_payload_bytes_per_sec: {augmented_train.loc[drop_indices, 'log_payload_bytes_per_sec'].unique()}")

# Check 5: flow_pkts_payload vs fwd_pkts_payload
print(f"  fwd_pkts_payload.tot: {augmented_train.loc[drop_indices, 'fwd_pkts_payload.tot'].unique()}")
print(f"  bwd_pkts_payload.tot: {augmented_train.loc[drop_indices, 'bwd_pkts_payload.tot'].unique()}")
print(f"  flow_pkts_payload.tot: {augmented_train.loc[drop_indices, 'flow_pkts_payload.tot'].unique()}")
print(f"  flow_pkts_payload.min: {augmented_train.loc[drop_indices, 'flow_pkts_payload.min'].unique()}")
print(f"  flow_pkts_payload.max: {augmented_train.loc[drop_indices, 'flow_pkts_payload.max'].unique()}")
