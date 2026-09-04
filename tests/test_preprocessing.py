"""
Unit tests for leakage-safe preprocessing pipeline, data cleaning, feature engineering, encoding, scaling, and artifact serialization.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.preprocessing.cleaning import DataFrameCleaner
from src.preprocessing.encoding import CategoricalEncoder
from src.preprocessing.feature_engineering import CybersecurityFeatureEngineer
from src.preprocessing.pipeline import CyberthreatPreprocessingPipeline
from src.preprocessing.scaling import FeatureScaler


@pytest.fixture
def mock_cyber_dataset():
    """Generates a synthetic network traffic DataFrame for testing preprocessing."""
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        " Total Fwd Packets": np.random.randint(1, 50, n),
        " Total Backward Packets": np.random.randint(0, 50, n),
        " Total Length of Fwd Packets": np.random.uniform(10, 5000, n),
        " Total Length of Bwd Packets": np.random.uniform(0, 5000, n),
        " Flow Duration": np.random.exponential(1000, n),
        " Flow Bytes/s": np.random.uniform(0, 1e6, n),
        " Constant Feature": [1.0] * n,
        " Inf Feature": [np.nan if i % 10 == 0 else (np.inf if i % 5 == 0 else float(i)) for i in range(n)],
        " Proto": np.random.choice(["tcp", "udp", "icmp"], n),
        " Label": np.random.choice(["BENIGN", "DDoS"], n, p=[0.7, 0.3]),
    })
    y_binary = (df[" Label"] != "BENIGN").astype(int)
    y_multi = df[" Label"]
    X = df.drop(columns=[" Label"])
    return X, y_binary, y_multi


def test_cleaner_leakage_safety(mock_cyber_dataset):
    X, _, _ = mock_cyber_dataset
    X_train = X.iloc[:150]
    X_test = X.iloc[150:]

    cleaner = DataFrameCleaner(remove_constant=True)
    cleaner.fit(X_train)

    assert " Constant Feature" in cleaner.constant_cols_
    assert " Constant Feature" not in cleaner.get_feature_names_out()

    X_train_clean = cleaner.transform(X_train)
    X_test_clean = cleaner.transform(X_test)

    assert not X_train_clean.isnull().any().any()
    assert not X_test_clean.isnull().any().any()
    assert not np.isinf(X_train_clean.select_dtypes(include=[np.number])).any().any()
    assert not np.isinf(X_test_clean.select_dtypes(include=[np.number])).any().any()


def test_feature_engineer(mock_cyber_dataset):
    X, _, _ = mock_cyber_dataset
    fe = CybersecurityFeatureEngineer()
    X_fe = fe.fit_transform(X)

    assert "fe_packet_ratio" in X_fe.columns
    assert "fe_bytes_per_packet" in X_fe.columns
    assert "fe_byte_ratio" in X_fe.columns
    assert "fe_flow_intensity" in X_fe.columns
    assert not np.isinf(X_fe[["fe_packet_ratio", "fe_bytes_per_packet"]]).any().any()


def test_categorical_encoder():
    df_train = pd.DataFrame({"num": [1, 2, 3], "cat": ["tcp", "udp", "tcp"]})
    df_test = pd.DataFrame({"num": [4, 5], "cat": ["icmp", "tcp"]})  # 'icmp' is unknown in train

    encoder = CategoricalEncoder(encoding_type="onehot")
    encoder.fit(df_train)
    df_test_enc = encoder.transform(df_test)

    assert "cat_tcp" in df_test_enc.columns
    assert "cat_udp" in df_test_enc.columns
    assert len(df_test_enc) == 2


def test_feature_scaler():
    df = pd.DataFrame({"a": [1.0, 100.0, 50.0], "b": [0.1, 0.2, 0.3]})
    scaler = FeatureScaler(scaling_method="robust")
    scaler.fit(df)
    df_scaled = scaler.transform(df)

    assert list(df_scaled.columns) == ["a", "b"]
    assert isinstance(df_scaled, pd.DataFrame)


def test_end_to_end_preprocessing_pipeline(mock_cyber_dataset, tmp_path):
    X, y_bin, y_multi = mock_cyber_dataset
    pipeline = CyberthreatPreprocessingPipeline(
        impute_strategy="median", scaling_method="robust", encoding_type="onehot"
    )

    processed = pipeline.fit_transform_splits(X, y_bin, y_multi)

    assert "X_train" in processed
    assert "X_val" in processed
    assert "X_test" in processed
    assert not processed["X_train"].isnull().any().any()
    assert not processed["X_val"].isnull().any().any()
    assert not processed["X_test"].isnull().any().any()

    # Save & Load Artifact Test
    artifact_path = tmp_path / "preprocessing_pipeline.joblib"
    pipeline.save_pipeline(artifact_path)
    reloaded_pipeline = CyberthreatPreprocessingPipeline.load_pipeline(artifact_path)

    X_new_proc = reloaded_pipeline.transform_new_data(X.iloc[:10])
    assert X_new_proc.shape[1] == processed["X_train"].shape[1]
