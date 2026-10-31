import pandas as pd
import numpy as np
import os

data_dir = r"d:\Sujay\GDGC_AIML\network-intrusion-detection2-ml-challenge"
test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))

ports = [21.0, 53.0, 80.0, 443.0, 1883.0, 5353.0]
te_other = test_df[~test_df['id.resp_p'].isin(ports)]
print(f"Total other rows in test: {len(te_other)}")
print("Proto breakdown:")
print(te_other['proto'].value_counts())
print("\nService breakdown:")
print(te_other['service'].value_counts())
print("\nTop 20 resp_p in other:")
print(te_other['id.resp_p'].value_counts().head(20))

print("\nDuration distribution in other:")
print(te_other['flow_duration'].describe())
print("\nbwd_pkts distribution in other:")
print(te_other['bwd_pkts_tot'].describe())
print("\nfwd_payload distribution in other:")
print(te_other['fwd_pkts_payload.tot'].describe())
print("\nSYN flag distribution in other:")
print(te_other['flow_SYN_flag_count'].value_counts())
print("\nFIN flag distribution in other:")
print(te_other['flow_FIN_flag_count'].value_counts())
print("\nACK flag distribution in other:")
print(te_other['flow_ACK_flag_count'].value_counts())
