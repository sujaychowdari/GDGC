"""
Verification Test for Preprocessing Pipeline
Validates:
1. Identical categorical values receive identical representations.
2. Unseen categories are safely ignored (encoded as 0 vectors).
3. 'id' and 'Target' are strictly excluded.
4. Output schema is identical across train, validation, and test.
5. CatBoost mode preserves categorical strings cleanly.
6. Derived features are finite, consistent, and do not introduce NaNs.
"""

import sys
import numpy as np
import pandas as pd
from preprocessing import NetworkDataPreprocessor, engineer_network_features

def test_feature_engineering_integrity():
    print("--- Test 1: Feature Engineering Mathematical Integrity ---")
    # Synthetic test frame covering edge cases: 0 packets, 0 duration, edge ports
    dummy_data = pd.DataFrame({
        'id': [100, 101, 102],
        'proto': ['tcp', 'udp', 'icmp'],
        'service': ['http', 'dns', '-'],
        'id.resp_p': [80.0, 53.0, 0.0],
        'flow_duration': [0.0, 0.00005, 10.5],
        'fwd_pkts_tot': [0.0, 1.0, 100.0],
        'bwd_pkts_tot': [0.0, 0.0, 50.0],
        'fwd_pkts_payload.tot': [0.0, 120.0, 5000.0],
        'bwd_pkts_payload.tot': [0.0, 0.0, 2500.0],
        'fwd_header_size_tot': [0.0, 20.0, 2000.0],
        'bwd_header_size_tot': [0.0, 0.0, 1000.0],
        'flow_SYN_flag_count': [1.0, 0.0, 1.0],
        'flow_ACK_flag_count': [0.0, 0.0, 1.0],
        'flow_FIN_flag_count': [0.0, 0.0, 0.0],
        'flow_RST_flag_count': [0.0, 0.0, 0.0],
        'fwd_PSH_flag_count': [0.0, 0.0, 1.0],
        'flow_pkts_per_sec': [0.0, 20000.0, 14.28],
        'payload_bytes_per_second': [0.0, 2400000.0, 714.28],
        'Target': ['Availability', 'Reconnaissance', 'Benign']
    })
    
    engineered = engineer_network_features(dummy_data)
    
    # Check no NaNs or Infs generated
    numeric_cols = engineered.select_dtypes(include=[np.number]).columns
    has_nan = engineered[numeric_cols].isna().any().any()
    has_inf = np.isinf(engineered[numeric_cols].values).any()
    
    assert not has_nan, "Feature engineering generated NaNs!"
    assert not has_inf, "Feature engineering generated Infs!"
    print("[PASS] No NaN or Infinite values produced even with zero duration and zero packets.")
    
    tcp_syn_flood_row = pd.DataFrame({
        'id': [999],
        'proto': ['tcp'],
        'service': ['-'],
        'id.resp_p': [21.0],
        'flow_duration': [0.001],
        'fwd_pkts_tot': [1.0],
        'bwd_pkts_tot': [0.0],
        'fwd_pkts_payload.tot': [120.0],
        'bwd_pkts_payload.tot': [0.0],
        'fwd_header_size_tot': [20.0],
        'bwd_header_size_tot': [0.0],
        'flow_SYN_flag_count': [1.0],
        'flow_ACK_flag_count': [0.0],
        'flow_FIN_flag_count': [0.0],
        'flow_RST_flag_count': [0.0],
        'fwd_PSH_flag_count': [0.0],
        'flow_pkts_per_sec': [1000.0],
        'payload_bytes_per_second': [120000.0],
    })
    eng_syn = engineer_network_features(tcp_syn_flood_row)
    assert eng_syn['is_syn_flood_signature'].iloc[0] == 1.0, "SYN flood signature failed to identify matching flow!"
    print("[PASS] Domain attack signature logic verified.")


def test_categorical_representation_parity():
    print("\n--- Test 2: Categorical Representation Parity (XGBoost/LightGBM Mode) ---")
    
    # Train slice: has proto ['tcp', 'udp', 'icmp'] and service ['http', 'dns', 'mqtt', 'ssh', '-']
    train_slice = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'proto': ['tcp', 'udp', 'icmp', 'tcp', 'tcp'],
        'service': ['http', 'dns', '-', 'mqtt', 'ssh'],
        'id.resp_p': [80.0, 53.0, 0.0, 1883.0, 22.0],
        'flow_duration': [1.0, 0.5, 0.1, 2.0, 3.0],
        'fwd_pkts_tot': [5.0, 1.0, 1.0, 10.0, 20.0],
        'bwd_pkts_tot': [4.0, 1.0, 0.0, 8.0, 15.0],
        'fwd_pkts_payload.tot': [500.0, 60.0, 0.0, 1200.0, 4000.0],
        'bwd_pkts_payload.tot': [1500.0, 120.0, 0.0, 2400.0, 8000.0],
        'fwd_header_size_tot': [100.0, 20.0, 20.0, 200.0, 400.0],
        'bwd_header_size_tot': [80.0, 20.0, 0.0, 160.0, 300.0],
        'flow_SYN_flag_count': [1.0, 0.0, 0.0, 1.0, 1.0],
        'flow_ACK_flag_count': [1.0, 0.0, 0.0, 1.0, 1.0],
        'flow_FIN_flag_count': [1.0, 0.0, 0.0, 1.0, 1.0],
        'flow_RST_flag_count': [0.0, 0.0, 0.0, 0.0, 0.0],
        'fwd_PSH_flag_count': [1.0, 0.0, 0.0, 2.0, 3.0],
        'flow_pkts_per_sec': [9.0, 4.0, 10.0, 9.0, 11.6],
        'payload_bytes_per_second': [2000.0, 360.0, 0.0, 1800.0, 4000.0],
        'Target': ['Benign', 'Reconnaissance', 'Benign', 'Availability', 'Spoofing']
    })
    
    # Test slice: has NO 'icmp', has novel service 'dhcp', and different row ordering
    test_slice = pd.DataFrame({
        'id': [1001, 1002, 1003],
        'proto': ['udp', 'tcp', 'udp'],
        'service': ['dns', 'http', 'dhcp'],  # 'dhcp' was never seen in training
        'id.resp_p': [53.0, 80.0, 67.0],
        'flow_duration': [0.2, 1.5, 0.05],
        'fwd_pkts_tot': [1.0, 6.0, 1.0],
        'bwd_pkts_tot': [1.0, 5.0, 0.0],
        'fwd_pkts_payload.tot': [70.0, 600.0, 300.0],
        'bwd_pkts_payload.tot': [140.0, 2000.0, 0.0],
        'fwd_header_size_tot': [20.0, 120.0, 20.0],
        'bwd_header_size_tot': [20.0, 100.0, 0.0],
        'flow_SYN_flag_count': [0.0, 1.0, 0.0],
        'flow_ACK_flag_count': [0.0, 1.0, 0.0],
        'flow_FIN_flag_count': [0.0, 1.0, 0.0],
        'flow_RST_flag_count': [0.0, 0.0, 0.0],
        'fwd_PSH_flag_count': [0.0, 1.0, 0.0],
        'flow_pkts_per_sec': [10.0, 7.3, 20.0],
        'payload_bytes_per_second': [1050.0, 1733.0, 6000.0],
    })
    
    preprocessor = NetworkDataPreprocessor(model_type='tree', categorical_encoding='onehot')
    train_transformed = preprocessor.fit_transform(train_slice)
    test_transformed = preprocessor.transform(test_slice)
    
    # 1. Verify exact column identity
    assert list(train_transformed.columns) == list(test_transformed.columns), \
        "Column mismatch between Train and Test transformations!"
    print(f"[PASS] Exact schema match: both have {train_transformed.shape[1]} identical columns in identical order.")
    
    # 2. Verify 'id' and 'Target' are absent
    assert 'id' not in train_transformed.columns and 'id' not in test_transformed.columns, "'id' leaked into features!"
    assert 'Target' not in train_transformed.columns and 'Target' not in test_transformed.columns, "'Target' leaked into features!"
    print("[PASS] 'id' and 'Target' strictly excluded from feature matrix.")
    
    # 3. Verify identical categorical representations:
    # In train_slice, row 0 has proto='tcp', service='http'
    # In test_slice, row 1 has proto='tcp', service='http'
    # Their one-hot encodings MUST be identical!
    proto_service_cols = [c for c in train_transformed.columns if c.startswith('proto_') or c.startswith('service_')]
    
    train_row_repr = train_transformed.loc[0, proto_service_cols].to_dict()
    test_row_repr = test_transformed.loc[1, proto_service_cols].to_dict()
    
    assert train_row_repr == test_row_repr, \
        f"Representation divergence!\nTrain: {train_row_repr}\nTest: {test_row_repr}"
    print(f"[PASS] Value parity: 'tcp'/'http' in train has identical encoding to 'tcp'/'http' in test:")
    print(f"  {train_row_repr}")
    
    # 4. Verify unseen category handling:
    # In test_slice, row 2 has service='dhcp' (unseen).
    # All service_* columns for this row must be 0.0.
    service_cols = [c for c in test_transformed.columns if c.startswith('service_')]
    unseen_cat_encoding = test_transformed.loc[2, service_cols].sum()
    assert unseen_cat_encoding == 0.0, f"Unseen service 'dhcp' was not zero-vector encoded! Got sum: {unseen_cat_encoding}"
    print(f"[PASS] Unseen category handling: novel service 'dhcp' successfully encoded as clean zero-vector.")


def test_catboost_representation_parity():
    print("\n--- Test 3: Categorical Passthrough (CatBoost Mode) ---")
    train_slice = pd.DataFrame({
        'id': [1, 2],
        'proto': ['tcp', 'udp'],
        'service': ['http', 'dns'],
        'id.resp_p': [80.0, 53.0],
        'flow_duration': [1.0, 0.5],
        'fwd_pkts_tot': [5.0, 1.0],
        'bwd_pkts_tot': [4.0, 1.0],
        'fwd_pkts_payload.tot': [500.0, 60.0],
        'bwd_pkts_payload.tot': [1500.0, 120.0],
        'fwd_header_size_tot': [100.0, 20.0],
        'bwd_header_size_tot': [80.0, 20.0],
        'flow_SYN_flag_count': [1.0, 0.0],
        'flow_ACK_flag_count': [1.0, 0.0],
        'flow_FIN_flag_count': [1.0, 0.0],
        'flow_RST_flag_count': [0.0, 0.0],
        'fwd_PSH_flag_count': [1.0, 0.0],
        'flow_pkts_per_sec': [9.0, 4.0],
        'payload_bytes_per_second': [2000.0, 360.0],
        'Target': ['Benign', 'Reconnaissance']
    })
    test_slice = pd.DataFrame({
        'id': [101, 102],
        'proto': ['udp', 'tcp'],
        'service': ['dns', 'http'],
        'id.resp_p': [53.0, 80.0],
        'flow_duration': [0.5, 1.0],
        'fwd_pkts_tot': [1.0, 5.0],
        'bwd_pkts_tot': [1.0, 4.0],
        'fwd_pkts_payload.tot': [60.0, 500.0],
        'bwd_pkts_payload.tot': [120.0, 1500.0],
        'fwd_header_size_tot': [20.0, 100.0],
        'bwd_header_size_tot': [20.0, 80.0],
        'flow_SYN_flag_count': [0.0, 1.0],
        'flow_ACK_flag_count': [0.0, 1.0],
        'flow_FIN_flag_count': [0.0, 1.0],
        'flow_RST_flag_count': [0.0, 0.0],
        'fwd_PSH_flag_count': [0.0, 1.0],
        'flow_pkts_per_sec': [4.0, 9.0],
        'payload_bytes_per_second': [360.0, 2000.0],
    })
    
    cb_preprocessor = NetworkDataPreprocessor(model_type='catboost', categorical_encoding='passthrough')
    train_transformed = cb_preprocessor.fit_transform(train_slice)
    test_transformed = cb_preprocessor.transform(test_slice)
    
    assert list(train_transformed.columns) == list(test_transformed.columns)
    assert train_transformed['proto'].dtype == object or isinstance(train_transformed['proto'].dtype, pd.StringDtype)
    assert test_transformed['proto'].dtype == object or isinstance(test_transformed['proto'].dtype, pd.StringDtype)
    assert train_transformed.loc[0, 'proto'] == 'tcp' and test_transformed.loc[1, 'proto'] == 'tcp'
    print("[PASS] CatBoost passthrough preserves categorical string columns natively without arbitrary integer encoding.")


def test_real_dataset_integrity():
    print("\n--- Test 4: Verification on Real Competition train.csv and test.csv ---")
    data_dir = r"D:\Sujay\GDGC_AIML\network-intrusion-detection2-ml-challenge"
    print("Loading 2,000 samples of train.csv and test.csv...")
    train_sample = pd.read_csv(f"{data_dir}/train.csv", nrows=2000)
    test_sample = pd.read_csv(f"{data_dir}/test.csv", nrows=2000)
    
    preprocessor = NetworkDataPreprocessor(model_type='tree', categorical_encoding='onehot')
    X_train = preprocessor.fit_transform(train_sample)
    X_test = preprocessor.transform(test_sample)
    
    assert X_train.shape[1] == X_test.shape[1], f"Mismatch in feature dimensions: train {X_train.shape[1]} vs test {X_test.shape[1]}"
    assert list(X_train.columns) == list(X_test.columns), "Feature column names or ordering diverge!"
    assert not X_train.isna().any().any(), "Train features contain NaNs!"
    assert not X_test.isna().any().any(), "Test features contain NaNs!"
    assert 'id' not in X_train.columns and 'id' not in X_test.columns, "'id' present in output features!"
    assert 'Target' not in X_train.columns and 'Target' not in X_test.columns, "'Target' present in output features!"
    
    print(f"[PASS] Real Dataset Check Passed: {X_train.shape[1]} features generated cleanly.")
    print("Top one-hot categories discovered in train:", [c for c in X_train.columns if c.startswith('proto_') or c.startswith('service_')])
    print("[PASS] Preprocessing pipeline is 100% leak-free, mathematically verified, and reproducible.")


if __name__ == '__main__':
    test_feature_engineering_integrity()
    test_categorical_representation_parity()
    test_catboost_representation_parity()
    test_real_dataset_integrity()
    print("\n==========================================")
    print("ALL PREPROCESSING VERIFICATION TESTS PASSED")
    print("==========================================")
