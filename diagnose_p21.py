import pandas as pd
import numpy as np
import os
import catboost as cb

data_dir = r"d:\Sujay\GDGC_AIML\network-intrusion-detection2-ml-challenge"
train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))

# Inspect Benign rows in train:
benign = train_df[train_df['Target'] == 'Benign']
print("Benign rows in train:", len(benign))
print("Benign resp_p:")
print(benign['id.resp_p'].value_counts().head(10))

# What features of test port 21 are matching Benign?
test_p21 = test_df[test_df['id.resp_p'] == 21.0].head(5)
print("\nTest Port 21 sample features:")
for col in ['id.resp_p', 'flow_duration', 'fwd_pkts_tot', 'bwd_pkts_tot', 'flow_SYN_flag_count', 'flow_ACK_flag_count', 'fwd_pkts_payload.tot']:
    print(f"  {col}: {test_p21[col].tolist()}")

# Does train have ANY Benign on Port 21?
print("Train Benign on Port 21:", len(train_df[(train_df['Target'] == 'Benign') & (train_df['id.resp_p'] == 21.0)]))
# Does train have ANY Reconnaissance on Port 21?
print("Train Reconnaissance on Port 21:", len(train_df[(train_df['Target'] == 'Reconnaissance') & (train_df['id.resp_p'] == 21.0)]))
# Does train have ANY Spoofing on Port 21?
print("Train Spoofing on Port 21:", len(train_df[(train_df['Target'] == 'Spoofing') & (train_df['id.resp_p'] == 21.0)]))
