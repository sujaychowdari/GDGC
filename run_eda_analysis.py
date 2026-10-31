import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

# Create output directories
os.makedirs(r'D:\Sujay\GDGC_AIML\reports\figures', exist_ok=True)

data_dir = r'D:\Sujay\GDGC_AIML\network-intrusion-detection2-ml-challenge'
train_path = os.path.join(data_dir, 'train.csv')
test_path = os.path.join(data_dir, 'test.csv')

print("Loading train and test datasets...")
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

# 1. Missing Values
train_na = train_df.isna().sum()
test_na = test_df.isna().sum()
print(f"Total missing in train: {train_na.sum()}, in test: {test_na.sum()}")

# 2. Duplicate Records
raw_features = [c for c in train_df.columns if c not in ['id', 'Target']]
train_dups = train_df.duplicated(subset=raw_features).sum()
test_dups = test_df.duplicated(subset=raw_features).sum()
print(f"Feature duplicates in train: {train_dups}, in test: {test_dups}")

# 3. Target Distribution
target_counts = train_df['Target'].value_counts()
target_props = train_df['Target'].value_counts(normalize=True) * 100
print("\nTarget Class Distribution:")
for k, v in target_counts.items():
    print(f"  {k:15s}: {v:6d} ({target_props[k]:6.2f}%)")

# 4. Train-Test Comparison & KS-Test
numeric_cols = [c for c in raw_features if c not in ['proto', 'service']]
comparison_rows = []

for col in numeric_cols:
    tr_s = train_df[col]
    te_s = test_df[col]
    
    # KS-test for distribution difference
    ks_stat, ks_pval = ks_2samp(tr_s, te_s)
    
    comparison_rows.append({
        'feature': col,
        'train_mean': tr_s.mean(),
        'test_mean': te_s.mean(),
        'train_std': tr_s.std(),
        'test_std': te_s.std(),
        'train_median': tr_s.median(),
        'test_median': te_s.median(),
        'train_min': tr_s.min(),
        'test_min': te_s.min(),
        'train_max': tr_s.max(),
        'test_max': te_s.max(),
        'ks_statistic': ks_stat,
        'ks_pvalue': ks_pval,
        'mean_ratio_test_to_train': (te_s.mean() + 1e-6) / (tr_s.mean() + 1e-6)
    })

comp_df = pd.DataFrame(comparison_rows).sort_values(by='ks_statistic', ascending=False)
comp_csv_path = r'D:\Sujay\GDGC_AIML\reports\train_test_comparison.csv'
comp_df.to_csv(comp_csv_path, index=False)
print(f"\nSaved train_test_comparison.csv with {len(comp_df)} numerical features.")

# 5. Top Distribution Shifts (highest KS statistic)
print("\nTop 15 Features with Largest Train-Test Shift (by KS Statistic):")
for idx, r in comp_df.head(15).iterrows():
    print(f"  {r['feature']:30s}: KS={r['ks_statistic']:.4f} (Tr Median={r['train_median']:.2f}, Te Median={r['test_median']:.2f})")

# 6. Unseen Combinations of (proto, service, id.resp_p)
train_combos = set(zip(train_df['proto'], train_df['service'], train_df['id.resp_p']))
test_combos = set(zip(test_df['proto'], test_df['service'], test_df['id.resp_p']))

unseen_combos = test_combos - train_combos
seen_combos = test_combos & train_combos

print(f"\nTotal (proto, service, id.resp_p) combinations in train: {len(train_combos)}")
print(f"Total combinations in test: {len(test_combos)}")
print(f"Overlapping combinations: {len(seen_combos)}")
print(f"Previously unseen combinations in test: {len(unseen_combos)} ({len(unseen_combos)/len(test_combos)*100:.2f}%)")

test_combo_series = list(zip(test_df['proto'], test_df['service'], test_df['id.resp_p']))
test_unseen_mask = [c in unseen_combos for c in test_combo_series]
unseen_test_rows = sum(test_unseen_mask)
print(f"Total test records belonging to UNSEEN combinations: {unseen_test_rows} / {len(test_df)} ({unseen_test_rows/len(test_df)*100:.2f}%)")

# Inspect top unseen combinations
test_df['combo'] = test_combo_series
top_unseen_df = test_df[test_unseen_mask]['combo'].value_counts().head(15)
print("\nTop 15 Unseen Combinations in Test:")
for c, cnt in top_unseen_df.items():
    print(f"  proto={c[0]}, service={c[1]}, resp_p={c[2]}: {cnt} rows")

# 7. Class-wise feature profiles
key_features = [
    'flow_duration', 'fwd_pkts_tot', 'bwd_pkts_tot', 'fwd_pkts_payload.tot',
    'bwd_pkts_payload.tot', 'flow_SYN_flag_count', 'flow_FIN_flag_count',
    'flow_ACK_flag_count', 'flow_pkts_per_sec', 'payload_bytes_per_second'
]

print("\nClass-wise Medians for Key Features:")
class_medians = train_df.groupby('Target')[key_features].median().T
print(class_medians)

# 8. Visualizations
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# Figure 1: Target Class Imbalance
fig, ax = plt.subplots(figsize=(8, 5))
classes = target_counts.index
counts = target_counts.values
colors = ['#d9534f', '#f0ad4e', '#5bc0de', '#5cb85c']
bars = ax.bar(classes, counts, color=colors, edgecolor='black', alpha=0.85)
ax.set_title('Training Set Target Class Distribution (Extreme Class Imbalance)', fontsize=13, fontweight='bold')
ax.set_ylabel('Number of Samples (Log Scale)', fontsize=11)
ax.set_yscale('log')
for bar, count, prop in zip(bars, counts, target_props.values):
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval * 1.15, f"{count:,}\n({prop:.2f}%)", ha='center', va='bottom', fontsize=10, fontweight='bold')
plt.tight_layout()
fig1_path = r'D:\Sujay\GDGC_AIML\reports\figures\fig1_target_distribution.png'
plt.savefig(fig1_path, dpi=200)
plt.close()
print("Saved fig1_target_distribution.png")

# Figure 2: Proto & Service Shift between Train and Test
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Proto
proto_df = pd.DataFrame({
    'Train (%)': train_df['proto'].value_counts(normalize=True) * 100,
    'Test (%)': test_df['proto'].value_counts(normalize=True) * 100
}).fillna(0)
proto_df.plot(kind='bar', ax=axes[0], color=['#337ab7', '#e74c3c'], edgecolor='black', alpha=0.85)
axes[0].set_title('Protocol Distribution Shift (Train vs. Test)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Percentage (%)', fontsize=11)
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)

# Service
service_df = pd.DataFrame({
    'Train (%)': train_df['service'].value_counts(normalize=True) * 100,
    'Test (%)': test_df['service'].value_counts(normalize=True) * 100
}).fillna(0)
service_df.plot(kind='bar', ax=axes[1], color=['#337ab7', '#e74c3c'], edgecolor='black', alpha=0.85)
axes[1].set_title('Service Distribution Shift (Train vs. Test)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Percentage (%)', fontsize=11)
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right')

plt.tight_layout()
fig2_path = r'D:\Sujay\GDGC_AIML\reports\figures\fig2_proto_service_shift.png'
plt.savefig(fig2_path, dpi=200)
plt.close()
print("Saved fig2_proto_service_shift.png")

# Figure 3: Payload vs Packets by Class
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
train_df['log_fwd_payload'] = np.log1p(train_df['fwd_pkts_payload.tot'])
train_df['log_fwd_pkts'] = np.log1p(train_df['fwd_pkts_tot'])

class_order = ['Availability', 'Benign', 'Reconnaissance', 'Spoofing']
train_df.boxplot(column='log_fwd_payload', by='Target', ax=axes[0], grid=True)
axes[0].set_title('Forward Payload (log1p bytes) by Class', fontsize=12, fontweight='bold')
axes[0].set_ylabel('log1p(fwd_pkts_payload.tot)', fontsize=11)
axes[0].set_xlabel('')

train_df.boxplot(column='log_fwd_pkts', by='Target', ax=axes[1], grid=True)
axes[1].set_title('Forward Packets (log1p count) by Class', fontsize=12, fontweight='bold')
axes[1].set_ylabel('log1p(fwd_pkts_tot)', fontsize=11)
axes[1].set_xlabel('')

fig.suptitle('')
plt.tight_layout()
fig3_path = r'D:\Sujay\GDGC_AIML\reports\figures\fig3_payload_vs_packets_by_class.png'
plt.savefig(fig3_path, dpi=200)
plt.close()
print("Saved fig3_payload_vs_packets_by_class.png")

# Figure 4: TCP Flags Prevalence Across Target Classes
flag_cols = ['flow_SYN_flag_count', 'flow_FIN_flag_count', 'flow_RST_flag_count', 'flow_ACK_flag_count', 'fwd_PSH_flag_count']
flag_means = train_df.groupby('Target')[flag_cols].mean()

fig, ax = plt.subplots(figsize=(10, 5))
flag_means.T.plot(kind='bar', ax=ax, colormap='tab10', edgecolor='black', alpha=0.85)
ax.set_title('Average TCP Flag Counts by Target Class', fontsize=13, fontweight='bold')
ax.set_ylabel('Mean Flag Count per Flow', fontsize=11)
ax.set_xticklabels([c.replace('flow_', '').replace('_flag_count', '') for c in flag_cols], rotation=0)
plt.tight_layout()
fig4_path = r'D:\Sujay\GDGC_AIML\reports\figures\fig4_tcp_flags_by_class.png'
plt.savefig(fig4_path, dpi=200)
plt.close()
print("Saved fig4_tcp_flags_by_class.png")

# Figure 5: Flow Duration and Rates by Class
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
train_df['log_duration'] = np.log1p(np.maximum(0, train_df['flow_duration']))
train_df['log_flow_pkts_per_sec'] = np.log1p(np.maximum(0, train_df['flow_pkts_per_sec']))

train_df.boxplot(column='log_duration', by='Target', ax=axes[0], grid=True)
axes[0].set_title('Flow Duration (log1p seconds) by Class', fontsize=12, fontweight='bold')
axes[0].set_ylabel('log1p(flow_duration)', fontsize=11)
axes[0].set_xlabel('')

train_df.boxplot(column='log_flow_pkts_per_sec', by='Target', ax=axes[1], grid=True)
axes[1].set_title('Flow Packet Rate (log1p pkts/sec) by Class', fontsize=12, fontweight='bold')
axes[1].set_ylabel('log1p(flow_pkts_per_sec)', fontsize=11)
axes[1].set_xlabel('')

fig.suptitle('')
plt.tight_layout()
fig5_path = r'D:\Sujay\GDGC_AIML\reports\figures\fig5_flow_duration_rates.png'
plt.savefig(fig5_path, dpi=200)
plt.close()
print("Saved fig5_flow_duration_rates.png")

# Figure 6: Correlation Matrix of Key Flow Features
corr_features = [
    'flow_duration', 'fwd_pkts_tot', 'bwd_pkts_tot', 'fwd_header_size_tot',
    'fwd_pkts_payload.tot', 'bwd_pkts_payload.tot', 'flow_SYN_flag_count',
    'flow_ACK_flag_count', 'flow_FIN_flag_count', 'flow_pkts_per_sec',
    'payload_bytes_per_second'
]
corr_matrix = train_df[corr_features].corr()

fig, ax = plt.subplots(figsize=(10, 8))
cax = ax.matshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(len(corr_features))
ax.set_xticks(ticks)
ax.set_yticks(ticks)
labels = [c.replace('_pkts', '').replace('_tot', '') for c in corr_features]
ax.set_xticklabels(labels, rotation=90, fontsize=9)
ax.set_yticklabels(labels, fontsize=9)
ax.set_title('Feature Correlation Matrix (Key Flow Metrics)', fontsize=12, fontweight='bold', pad=20)
plt.tight_layout()
fig6_path = r'D:\Sujay\GDGC_AIML\reports\figures\fig6_feature_correlations.png'
plt.savefig(fig6_path, dpi=200)
plt.close()
print("Saved fig6_feature_correlations.png")

# Figure 7: Unseen Traffic Volume in Test
fig, ax = plt.subplots(figsize=(7, 5))
categories = ['Seen in Train\n(Combinations)', 'Novel in Test\n(Unseen Combinations)']
shares = [len(test_df) - unseen_test_rows, unseen_test_rows]
colors = ['#5cb85c', '#d9534f']
bars = ax.bar(categories, shares, color=colors, edgecolor='black', alpha=0.85, width=0.5)
ax.set_title('Test Set Traffic: Seen vs. Unseen (proto, service, resp_p)', fontsize=12, fontweight='bold')
ax.set_ylabel('Number of Test Instances', fontsize=11)
for bar, share in zip(bars, shares):
    yval = bar.get_height()
    pct = (share / len(test_df)) * 100
    ax.text(bar.get_x() + bar.get_width()/2.0, yval * 1.02, f"{share:,}\n({pct:.2f}%)", ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_ylim(0, len(test_df) * 1.15)
plt.tight_layout()
fig7_path = r'D:\Sujay\GDGC_AIML\reports\figures\fig7_unseen_traffic_breakdown.png'
plt.savefig(fig7_path, dpi=200)
plt.close()
print("Saved fig7_unseen_traffic_breakdown.png")
