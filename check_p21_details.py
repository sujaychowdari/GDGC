import pandas as pd
import numpy as np
import os
import catboost as cb

data_dir = r"d:\Sujay\GDGC_AIML\network-intrusion-detection2-ml-challenge"
train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))

label_mapping = {'Availability': 0, 'Benign': 1, 'Reconnaissance': 2, 'Spoofing': 3}
inv_mapping = {v: k for k, v in label_mapping.items()}

# Let's inspect test_df port 21 rows where bwd_pkts_tot == 0
t21_0 = test_df[(test_df['id.resp_p'] == 21.0) & (test_df['bwd_pkts_tot'] == 0.0)]
print("Total test Port 21 rows with bwd_pkts == 0:", len(t21_0))

# Check ALL features of t21_0
print("\nUnique values in key features of t21_0:")
for c in ['proto', 'service', 'fwd_pkts_tot', 'bwd_pkts_tot', 'flow_SYN_flag_count', 'flow_FIN_flag_count', 
          'flow_ACK_flag_count', 'fwd_pkts_payload.tot', 'fwd_header_size_tot', 'fwd_init_window_size']:
    print(f"  {c}: {t21_0[c].value_counts().head(3).to_dict()}")

# Now check train Availability on Port 21
tr21 = train_df[train_df['id.resp_p'] == 21.0]
print("\nUnique values in key features of train Port 21:")
for c in ['proto', 'service', 'fwd_pkts_tot', 'bwd_pkts_tot', 'flow_SYN_flag_count', 'flow_FIN_flag_count', 
          'flow_ACK_flag_count', 'fwd_pkts_payload.tot', 'fwd_header_size_tot', 'fwd_init_window_size']:
    print(f"  {c}: {tr21[c].value_counts().head(3).to_dict()}")
