"""
Unit tests for Dataset Loaders and downcasting functions.
"""

import numpy as np
import pandas as pd
import pytest
from src.data.loaders import CICIDS2017Loader, DatasetNotFoundError, UNSWNB15Loader, downcast_numeric_dtypes


def test_downcast_numeric_dtypes():
    df = pd.DataFrame({
        "a": np.array([1.0, 2.0, 3.0], dtype=np.float64),
        "b": np.array([10, 20, 30], dtype=np.int64),
        "c": ["foo", "bar", "baz"],
    })
    downcasted = downcast_numeric_dtypes(df)

    assert downcasted["a"].dtype == np.float32
    assert downcasted["b"].dtype == np.int32
    assert str(downcasted["c"].dtype) in ["object", "string", "str"] or pd.api.types.is_string_dtype(downcasted["c"])


def test_cicids_loader_missing_files(tmp_path):
    loader = CICIDS2017Loader(raw_dir=tmp_path)
    assert loader.list_available_files() == []
    with pytest.raises(DatasetNotFoundError):
        loader.load_merged_dataset()


def test_unsw_loader_missing_files(tmp_path):
    loader = UNSWNB15Loader(raw_dir=tmp_path)
    assert loader.list_available_files() == []
    with pytest.raises(DatasetNotFoundError):
        loader.load_dataset()
