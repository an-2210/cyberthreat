"""
Unit tests for data validation and dataset adapter logic using synthetic mock DataFrames.
"""

import numpy as np
import pandas as pd
import pytest
from src.data.dataset_adapters import CICIDS2017Adapter, UNSWNB15Adapter
from src.data.validators import DataValidator


@pytest.fixture
def mock_cicids_df():
    """Creates a mock DataFrame mimicking CIC-IDS2017 raw columns with spaces and anomalies."""
    data = {
        " Destination Port": [80, 443, 80, 22, 80],
        " Flow Duration": [1000, 2000, np.inf, 1500, 1000],
        " Total Fwd Packets": [10, 20, 15, np.nan, 10],
        " Constant Feature": [1, 1, 1, 1, 1],
        " Label": ["BENIGN", "DDoS", "BENIGN", "PortScan", "BENIGN"],
    }
    return pd.DataFrame(data)


@pytest.fixture
def mock_unsw_df():
    """Creates a mock DataFrame mimicking UNSW-NB15 raw columns."""
    data = {
        "id": [1, 2, 3, 4],
        "dur": [0.1, 0.5, 1.2, 0.05],
        "proto": ["tcp", "udp", "tcp", "icmp"],
        "attack_cat": ["Normal", "Fuzzers", "Normal", "Exploits"],
        "label": [0, 1, 0, 1],
    }
    return pd.DataFrame(data)


def test_data_validator(mock_cicids_df):
    validator = DataValidator(mock_cicids_df, dataset_name="MockCICIDS", target_col=" Label")
    report = validator.run_full_validation()

    assert report["n_rows"] == 5
    assert report["n_columns"] == 5
    assert report["missing_values"]["total_missing_cells"] == 1
    assert report["infinite_values"]["total_infinite_cells"] == 1
    assert " Constant Feature" in report["constant_features"]
    assert report["label_distribution"]["counts"]["BENIGN"] == 3


def test_cicids_adapter(mock_cicids_df):
    adapter = CICIDS2017Adapter(target_column="Label")
    X, y_binary, y_multi = adapter.extract_labels(mock_cicids_df)

    assert "Flow Duration" in X.columns
    assert "Destination Port" in X.columns
    assert "Label" not in X.columns
    assert list(y_binary) == [0, 1, 0, 1, 0]
    assert list(y_multi) == ["BENIGN", "DDoS", "BENIGN", "PortScan", "BENIGN"]


def test_unsw_adapter(mock_unsw_df):
    adapter = UNSWNB15Adapter(target_column="label", cat_target_column="attack_cat")
    X, y_binary, y_multi = adapter.extract_labels(mock_unsw_df)

    assert "dur" in X.columns
    assert "proto" in X.columns
    assert "id" not in X.columns
    assert "label" not in X.columns
    assert "attack_cat" not in X.columns
    assert list(y_binary) == [0, 1, 0, 1]
    assert list(y_multi) == ["Normal", "Fuzzers", "Normal", "Exploits"]
