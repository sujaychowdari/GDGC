import pandas as pd
import numpy as np
import os

data_dir = r"d:\Sujay\GDGC_AIML\network-intrusion-detection2-ml-challenge"
train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))

train_p21 = train_df[train_df['id.resp_p'] == 21.0]
test_p21 = test_df[test_df['id.resp_p'] == 21.0]

print("Train Port 21 count:", len(train_p21))
print("Test Port 21 count:", len(test_p21))

cols_compare = ['flow_duration', 'fwd_pkts_tot', 'bwd_pkts_tot', 'flow_pkts_per_sec', 
                'flow_SYN_flag_count', 'flow_ACK_flag_count', 'fwd_header_size_tot', 'proto', 'service']

print("\n--- Train Port 21 Stats ---")
print(train_p21[cols_compare].describe().T[['mean', 'std', '50%', 'max']])

print("\n--- Test Port 21 Stats ---")
print(test_p21[cols_compare].describe().T[['mean', 'std', '50%', 'max']])

# Also check non-port-21 in test
print("\n--- Test Other Ports Stats ---")
test_other = test_df[test_df['id.resp_p'] != 21.0]
print(test_other[cols_compare].describe().T[['mean', 'std', '50%', 'max']])
