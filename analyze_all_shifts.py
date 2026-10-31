import pandas as pd
import numpy as np
import os

data_dir = r"d:\Sujay\GDGC_AIML\network-intrusion-detection2-ml-challenge"
train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))

ports = [21.0, 53.0, 80.0, 443.0, 1883.0, 5353.0]

print("==================================================")
print("PORT ANALYSIS IN TRAIN VS TEST")
print("==================================================")

for p in ports:
    tr = train_df[train_df['id.resp_p'] == p]
    te = test_df[test_df['id.resp_p'] == p]
    print(f"\n>>> Port {p}: Train count = {len(tr)}, Test count = {len(te)}")
    if len(tr) > 0:
        print("  Train Target distribution:")
        print("   ", tr['Target'].value_counts().to_dict())
        print(f"  Train protocols: {tr['proto'].value_counts().to_dict()}")
        print(f"  Train services: {tr['service'].value_counts().to_dict()}")
        print(f"  Train fwd_payload medians by target:")
        for t, g in tr.groupby('Target'):
            print(f"     {t}: fwd_payload={g['fwd_pkts_payload.tot'].median()}, bwd_pkts={g['bwd_pkts_tot'].median()}, duration={g['flow_duration'].median():.4f}")
    if len(te) > 0:
        print(f"  Test protocols: {te['proto'].value_counts().to_dict()}")
        print(f"  Test services: {te['service'].value_counts().to_dict()}")
        print(f"  Test stats: fwd_payload_median={te['fwd_pkts_payload.tot'].median()}, bwd_pkts_median={te['bwd_pkts_tot'].median()}, duration_median={te['flow_duration'].median():.4f}")

# Check remaining test rows not in these top ports
te_other = test_df[~test_df['id.resp_p'].isin(ports)]
tr_other = train_df[~train_df['id.resp_p'].isin(ports)]
print(f"\n>>> Other Ports: Train count = {len(tr_other)}, Test count = {len(te_other)}")
print("  Train Target distribution on other ports:")
print("   ", tr_other['Target'].value_counts().to_dict())
print(f"  Train proto: {tr_other['proto'].value_counts().to_dict()}")
print(f"  Test proto: {te_other['proto'].value_counts().to_dict()}")
print(f"  Test top 10 other ports: {te_other['id.resp_p'].value_counts().head(10).to_dict()}")
