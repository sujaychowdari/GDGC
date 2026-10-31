"""
Unified, Leak-Free Preprocessing and Feature Engineering Pipeline
for Network Intrusion Detection (GDG on Campus ML Challenge).

Key Design Principles:
1. Consistent categorical encoding across Train, Validation, and Test.
   - For XGBoost / LightGBM: OneHotEncoder(handle_unknown='ignore', sparse_output=False).
   - For CatBoost: Passthrough as string objects.
2. Fit strictly on training data inside each fold; transform validation and test without refitting.
3. Strict exclusion of 'id' from predictive features.
4. Strict exclusion of 'Target' from feature transformations to prevent leakage.
5. Mathematically validated derived features (no redundant duplicate clones or dead flag checks).
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder


def engineer_network_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes domain-grounded network flow features.
    Operates on a copy of the input DataFrame and preserves all raw features.
    """
    data = df.copy()

    fwd_pkts = np.maximum(1.0, data['fwd_pkts_tot'])
    bwd_pkts = np.maximum(1.0, data['bwd_pkts_tot'])
    fwd_payload = np.maximum(0.0, data['fwd_pkts_payload.tot'])
    bwd_payload = np.maximum(0.0, data['bwd_pkts_payload.tot'])

    # 1. Flow Directionality & Symmetry
    # Raw dataset has down_up_ratio, but explicit packet/byte ratios add stability
    data['bwd_to_fwd_pkt_ratio'] = data['bwd_pkts_tot'] / fwd_pkts
    data['bwd_to_fwd_byte_ratio'] = bwd_payload / np.maximum(1.0, fwd_payload)
    data['is_unidirectional'] = (data['bwd_pkts_tot'] == 0.0).astype(float)

    # 2. Header Overhead vs. Data Ratios
    # Automated probes carry high header relative to payload; real transfers carry more payload
    data['fwd_header_per_pkt'] = data['fwd_header_size_tot'] / fwd_pkts
    data['bwd_header_per_pkt'] = data['bwd_header_size_tot'] / bwd_pkts
    data['fwd_header_to_payload_ratio'] = data['fwd_header_size_tot'] / np.maximum(1.0, fwd_payload)

    # 3. TCP Handshake and Flag States
    # (FIN, SYN, RST, ACK, PSH combinations)
    syn_cnt = data['flow_SYN_flag_count']
    ack_cnt = data['flow_ACK_flag_count']
    fin_cnt = data['flow_FIN_flag_count']
    rst_cnt = data['flow_RST_flag_count']
    psh_cnt = data['fwd_PSH_flag_count']

    data['is_syn_only'] = ((syn_cnt > 0) & (ack_cnt == 0)).astype(float)
    data['is_fin_only'] = ((fin_cnt > 0) & (ack_cnt == 0)).astype(float)
    data['is_syn_ack'] = ((syn_cnt > 0) & (ack_cnt > 0)).astype(float)
    data['has_rst_flag'] = (rst_cnt > 0).astype(float)
    data['has_psh_flag'] = (psh_cnt > 0).astype(float)

    # 4. Invariant Attack Signatures (Physical Network Flow Level)
    # Hping3 SYN flood in RT-IoT2022: TCP + SYN flag + exactly 120 bytes payload
    proto_str = data['proto'].astype(str).str.lower()
    data['is_syn_flood_signature'] = (
        (proto_str == 'tcp') & (syn_cnt >= 1) & (fwd_payload == 120.0)
    ).astype(float)

    # Probe scan signature: 0 bytes payload on forward packet + single packet flow
    data['is_probe_scan_signature'] = (
        (fwd_payload == 0.0) & (data['fwd_pkts_tot'] == 1.0) & (data['flow_duration'] <= 1e-4)
    ).astype(float)

    # 5. Semantic Port Categories
    # Abstract port numbers into protocol roles to allow generalization across unseen ports
    resp_p = data['id.resp_p']
    data['is_port_ftp'] = (resp_p == 21.0).astype(float)
    data['is_port_ssh'] = (resp_p == 22.0).astype(float)
    data['is_port_dns'] = ((resp_p == 53.0) | (resp_p == 5353.0)).astype(float)
    data['is_port_http'] = (resp_p == 80.0).astype(float)
    data['is_port_https'] = (resp_p == 443.0).astype(float)
    data['is_port_mqtt'] = (resp_p == 1883.0).astype(float)
    data['is_port_ntp'] = (resp_p == 123.0).astype(float)
    data['is_well_known_resp'] = (resp_p <= 1024.0).astype(float)
    data['is_registered_resp'] = ((resp_p > 1024.0) & (resp_p < 49152.0)).astype(float)
    data['is_ephemeral_resp'] = (resp_p >= 49152.0).astype(float)

    # 6. Log-Scale Transforms for Extreme Range Rate Features
    # Protects tree models from microsecond rate spikes
    data['log_duration'] = np.log1p(np.maximum(0.0, data['flow_duration']))
    data['log_flow_pkts_per_sec'] = np.log1p(np.maximum(0.0, data['flow_pkts_per_sec']))
    data['log_payload_bytes_per_sec'] = np.log1p(np.maximum(0.0, data['payload_bytes_per_second']))

    return data


class NetworkDataPreprocessor(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compliant preprocessor ensuring identical, leak-free
    feature transformation for training, validation, and test datasets.
    """

    def __init__(self, model_type='tree', categorical_encoding='onehot'):
        """
        Args:
            model_type (str): 'tree' (XGBoost, LightGBM, HistGB) or 'catboost'.
            categorical_encoding (str): 'onehot' (recommended for XGBoost) or 'passthrough'.
        """
        self.model_type = model_type
        self.categorical_encoding = categorical_encoding
        self.cat_cols = ['proto', 'service']
        self.ohe_ = None
        self.numeric_cols_ = None
        self.feature_names_out_ = None
        self.median_imputer_ = {}

    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit transformers strictly on training data.
        X must be the feature DataFrame (without 'id' and 'Target').
        """
        # Exclude 'id' and 'Target' if present
        clean_X = X.copy()
        drop_cols = [c for c in ['id', 'Target'] if c in clean_X.columns]
        if drop_cols:
            clean_X = clean_X.drop(columns=drop_cols)

        # Apply feature engineering
        fe_X = engineer_network_features(clean_X)

        # Identify numeric vs categorical columns
        self.numeric_cols_ = [c for c in fe_X.columns if c not in self.cat_cols]

        # Learn median values on training set for numerical columns (defensive imputation)
        for col in self.numeric_cols_:
            self.median_imputer_[col] = float(fe_X[col].median())

        # Fit OneHotEncoder on training categoricals with handle_unknown='ignore'
        if self.categorical_encoding == 'onehot' and self.model_type != 'catboost':
            self.ohe_ = OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=np.float32)
            self.ohe_.fit(fe_X[self.cat_cols].astype(str))
            ohe_cols = list(self.ohe_.get_feature_names_out(self.cat_cols))
            self.feature_names_out_ = self.numeric_cols_ + ohe_cols
        else:
            self.feature_names_out_ = self.numeric_cols_ + self.cat_cols

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform validation or test data using the fitted preprocessor.
        Guarantees 100% identical columns, encodings, and order as training.
        """
        if self.feature_names_out_ is None:
            raise RuntimeError("Preprocessor has not been fitted yet! Call .fit() first.")

        clean_X = X.copy()
        drop_cols = [c for c in ['id', 'Target'] if c in clean_X.columns]
        if drop_cols:
            clean_X = clean_X.drop(columns=drop_cols)

        # Feature engineering
        fe_X = engineer_network_features(clean_X)

        # Numeric transformations and defensive fillna
        res_numeric = fe_X[self.numeric_cols_].copy()
        for col in self.numeric_cols_:
            if col in res_numeric.columns:
                res_numeric[col] = res_numeric[col].fillna(self.median_imputer_[col])
            else:
                res_numeric[col] = self.median_imputer_[col]

        # Categorical transformations
        if self.categorical_encoding == 'onehot' and self.model_type != 'catboost':
            cat_array = self.ohe_.transform(fe_X[self.cat_cols].astype(str))
            ohe_cols = list(self.ohe_.get_feature_names_out(self.cat_cols))
            res_cat = pd.DataFrame(cat_array, columns=ohe_cols, index=fe_X.index)
            output_df = pd.concat([res_numeric, res_cat], axis=1)
        else:
            # For CatBoost: preserve raw strings
            res_cat = fe_X[self.cat_cols].astype(str).copy()
            output_df = pd.concat([res_numeric, res_cat], axis=1)

        # Enforce exact column order
        return output_df[self.feature_names_out_]
