"""
Tests for the experiment tracker. Values are compared with the saved metric files they come from.
"""

import json
from pathlib import Path

import pytest

from src.evaluation.tracker import COLUMNS, build_table
from src.utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]
XGB_BINARY = ROOT / "results" / "metrics" / "cicids2017_binary_xgboost.json"


@pytest.mark.skipif(not XGB_BINARY.exists(), reason="CIC-IDS2017 binary XGBoost results not present")
def test_xgboost_binary_row_matches_its_metric_file():
    cfg = load_config("configs/config.yaml")
    df = build_table(cfg)
    assert list(df.columns) == COLUMNS
    row = df[(df["model"] == "xgboost") & (df["task"] == "binary") & (df["dataset"] == "CIC-IDS2017")].iloc[0]
    saved = json.loads(XGB_BINARY.read_text(encoding="utf-8"))
    assert row["recall"] == pytest.approx(round(saved["recall"], 6))
    assert row["fnr"] == pytest.approx(round(saved["fnr"], 6))
    assert row["training_time_s"] == pytest.approx(round(saved["train_fit_seconds"], 6))


@pytest.mark.skipif(not XGB_BINARY.exists(), reason="CIC-IDS2017 binary XGBoost results not present")
def test_unmeasured_fields_stay_empty_not_estimated():
    cfg = load_config("configs/config.yaml")
    df = build_table(cfg)
    binary_rows = df[df["task"] == "binary"]
    # Binary metric files have no macro or weighted averages, so those cells must stay empty.
    assert binary_rows["macro_f1"].isna().all()
    assert binary_rows["weighted_f1"].isna().all()
