"""
Validation script for submission_candidate.csv
Performs strict schema, integrity, and comparative validation against test.csv and baseline_084671.csv.
"""

import sys
import pandas as pd
import numpy as np

def validate_submission(candidate_path, test_path, baseline_path):
    print("=" * 70)
    print("SUBMISSION CANDIDATE VALIDATION REPORT")
    print("=" * 70)
    
    # 1. Load Files
    print(f"\n[1] Loading files...")
    candidate_df = pd.read_csv(candidate_path)
    test_df = pd.read_csv(test_path, usecols=['id'])
    baseline_df = pd.read_csv(baseline_path)
    
    print(f"  Candidate path: {candidate_path}")
    print(f"  Test path:      {test_path}")
    print(f"  Baseline path:  {baseline_path}")
    
    status_checks = []
    
    # Check 1: Exactly 33,161 rows
    row_count = len(candidate_df)
    expected_rows = 33161
    c1 = (row_count == expected_rows)
    status_checks.append(("Row Count (Exactly 33,161)", c1, f"{row_count} rows"))
    
    # Check 2: Exactly two columns: id and Target
    expected_cols = ['id', 'Target']
    actual_cols = list(candidate_df.columns)
    c2 = (actual_cols == expected_cols)
    status_checks.append(("Column Names (['id', 'Target'])", c2, f"{actual_cols}"))
    
    # Check 3: Every ID matching test.csv
    test_ids = set(test_df['id'])
    cand_ids = set(candidate_df['id'])
    c3 = (test_ids == cand_ids)
    status_checks.append(("All IDs Match test.csv", c3, f"{len(cand_ids.intersection(test_ids))} / {len(test_ids)} matching"))
    
    # Check 4: Original test ID ordering preserved
    c4 = (candidate_df['id'] == test_df['id']).all()
    status_checks.append(("Original ID Ordering Preserved", c4, "Exact element-wise sequence match"))
    
    # Check 5: No missing predictions
    missing_count = candidate_df['Target'].isna().sum() + candidate_df['id'].isna().sum()
    c5 = (missing_count == 0)
    status_checks.append(("No Missing / Null Predictions", c5, f"{missing_count} nulls found"))
    
    # Check 6: No duplicate IDs
    dup_count = candidate_df['id'].duplicated().sum()
    c6 = (dup_count == 0)
    status_checks.append(("No Duplicate IDs", c6, f"{dup_count} duplicates found"))
    
    # Check 7: Only valid labels
    valid_labels = {'Benign', 'Availability', 'Reconnaissance', 'Spoofing'}
    actual_labels = set(candidate_df['Target'].unique())
    c7 = actual_labels.issubset(valid_labels) and len(actual_labels) == 4
    status_checks.append(("Valid Target Classes Only", c7, f"Classes: {sorted(list(actual_labels))}"))

    # Print Item 1 & 2: Prediction distribution and percentages
    print("\n" + "=" * 70)
    print("1. PREDICTION DISTRIBUTION & 2. PERCENTAGES")
    print("=" * 70)
    counts = candidate_df['Target'].value_counts()
    percentages = (counts / row_count) * 100
    
    dist_df = pd.DataFrame({
        'Predicted Count': counts,
        'Percentage (%)': percentages.round(4)
    })
    dist_df['Formatted'] = dist_df.apply(lambda r: f"{int(r['Predicted Count']):,d} ({r['Percentage (%)']:.2f}%)", axis=1)
    
    for idx, row in dist_df.iterrows():
        print(f"  - {idx:<15}: {int(row['Predicted Count']):>6,d} flows  ({row['Percentage (%)']:>6.2f}%)")
    print(f"  {'Total':<15}: {row_count:>6,d} flows  (100.00%)")
    
    # Compare with Baseline
    print("\n" + "=" * 70)
    print("3. DIFFERENCE FROM ORIGINAL BASELINE PREDICTIONS")
    print("=" * 70)
    baseline_counts = baseline_df['Target'].value_counts()
    comp_df = pd.DataFrame({
        'Candidate': counts,
        'Baseline (0.84671)': baseline_counts,
        'Net Delta': counts - baseline_counts
    }).fillna(0)
    comp_df['Candidate %'] = (comp_df['Candidate'] / row_count * 100).round(2)
    comp_df['Baseline %'] = (comp_df['Baseline (0.84671)'] / row_count * 100).round(2)
    
    print(comp_df[['Baseline (0.84671)', 'Candidate', 'Net Delta', 'Baseline %', 'Candidate %']].to_string())
    
    print("\nCross-Tabulation Matrix (Row: Baseline 0.84671 -> Column: Candidate):")
    ct = pd.crosstab(baseline_df['Target'], candidate_df['Target'], margins=True)
    print(ct.to_string())
    
    # Print Item 4: Number of changed predictions
    print("\n" + "=" * 70)
    print("4. NUMBER OF CHANGED PREDICTIONS")
    print("=" * 70)
    changed_mask = (baseline_df['Target'] != candidate_df['Target'])
    num_changed = changed_mask.sum()
    pct_changed = (num_changed / row_count) * 100
    agreement = ((row_count - num_changed) / row_count) * 100
    
    print(f"  Identical Predictions: {row_count - num_changed:,d} / {row_count:,d} ({agreement:.2f}%)")
    print(f"  Changed Predictions:   {num_changed:,d} / {row_count:,d} ({pct_changed:.2f}%)")
    
    print("\nBreakdown of Changes by Transition:")
    changes_series = baseline_df.loc[changed_mask, 'Target'] + " -> " + candidate_df.loc[changed_mask, 'Target']
    for transition, count in changes_series.value_counts().items():
        print(f"  - {transition:<35}: {count:>4d} flows ({count/num_changed*100:>5.1f}% of changes)")
        
    # Print Item 5: CSV Validation Status
    print("\n" + "=" * 70)
    print("5. CSV VALIDATION STATUS")
    print("=" * 70)
    all_passed = True
    for name, passed, details in status_checks:
        status_tag = "[PASS]" if passed else "[FAIL]"
        if not passed:
            all_passed = False
        print(f"  {status_tag} {name:<35}: {details}")
        
    print("-" * 70)
    if all_passed:
        print("OVERALL STATUS: [PASSED] ALL SUBMISSION REQUIREMENTS FULLY SATISFIED")
    else:
        print("OVERALL STATUS: [FAILED] ONE OR MORE CRITICAL CHECKS FAILED")
    print("=" * 70)
    
    return all_passed

if __name__ == "__main__":
    cand_path = r"D:\Sujay\GDGC_AIML\submission_candidate.csv"
    test_path = r"D:\Sujay\GDGC_AIML\network-intrusion-detection2-ml-challenge\test.csv"
    base_path = r"D:\Sujay\GDGC_AIML\baseline_084671.csv"
    
    success = validate_submission(cand_path, test_path, base_path)
    if not success:
        sys.exit(1)
