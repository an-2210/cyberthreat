"""
Tests for the ablation table, the UNSW-NB15 cross-dataset column mapping, and the latency helpers.

Checks against saved artifacts use the real files when they exist and skip otherwise. The UNSW mapping test
uses literal input values so the unit conversion can be checked by hand; no UNSW data is needed.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation.ablation import ablation_rows, unseen_attack_rows
from src.evaluation.cross_dataset import CIC_COLUMNS, SHARED_FEATURES, cic_shared_frame, unsw_to_shared

ROOT = Path(__file__).resolve().parents[1]
HYBRID = ROOT / "results" / "metrics" / "cicids2017_hybrid_detection.json"
LOAO = ROOT / "results" / "metrics" / "cicids2017_leave_one_attack_out.json"
RUN_SUMMARY = ROOT / "results" / "reports" / "cicids2017_run_summary.json"


def test_unsw_mapping_converts_units_and_renames():
    unsw = pd.DataFrame([{"dur": 0.5, "spkts": 3, "dpkts": 2, "sbytes": 180, "dbytes": 90,
                          "smean": 60, "dmean": 45, "swin": 255, "dwin": 0}])
    shared = unsw_to_shared(unsw)
    assert list(shared.columns) == CIC_COLUMNS
    assert shared.loc[0, "Flow Duration"] == pytest.approx(500_000.0)  # seconds -> microseconds
    assert shared.loc[0, "Total Fwd Packets"] == 3
    assert shared.loc[0, "Init_Win_bytes_backward"] == 0


def test_unsw_mapping_turns_infinite_values_into_missing():
    unsw = pd.DataFrame([{k: 1.0 for k in SHARED_FEATURES}])
    unsw.loc[0, "sbytes"] = np.inf
    assert np.isnan(unsw_to_shared(unsw).loc[0, "Total Length of Fwd Packets"])


def test_unsw_mapping_rejects_missing_columns():
    with pytest.raises(ValueError, match="missing shared feature columns"):
        unsw_to_shared(pd.DataFrame({"dur": [1.0]}))


@pytest.mark.skipif(not RUN_SUMMARY.exists(), reason="CIC run summary not present")
def test_every_shared_cic_column_exists_in_the_trained_feature_list():
    names = set(json.loads(RUN_SUMMARY.read_text())["feature_names"])
    base = {n.replace(".1", "") for n in names}
    for col in CIC_COLUMNS:
        assert col in names or col in base, f"{col} is not a feature the CIC pipeline uses"


def test_cic_shared_frame_rejects_missing_columns():
    with pytest.raises(ValueError, match="missing shared feature columns"):
        cic_shared_frame(pd.DataFrame({"Flow Duration": [1.0]}))


@pytest.mark.skipif(not HYBRID.exists(), reason="hybrid detection results not present")
def test_ablation_rows_reproduce_saved_hybrid_test_metrics():
    hybrid = json.loads(HYBRID.read_text())
    rows = {r["variant"]: r for r in ablation_rows(hybrid)}
    fused = hybrid["test_results"]["hybrid_fusion"]
    row = rows["Hybrid: XGBoost + Autoencoder fusion"]
    assert row["f1"] == round(fused["f1"], 4)
    assert row["fnr"] == round(fused["fnr"], 4)
    assert rows["XGBoost only"]["delta_f1_vs_xgboost"] == 0.0
    assert row["delta_f1_vs_xgboost"] == round(fused["f1"] - hybrid["test_results"]["supervised_xgboost"]["f1"], 4)


@pytest.mark.skipif(not LOAO.exists(), reason="leave-one-attack-out results not present")
def test_unseen_attack_rows_match_saved_summary():
    loao = json.loads(LOAO.read_text())
    rows = {r["variant"]: r for r in unseen_attack_rows(loao)}
    s = loao["summary"]["mean_detection_over_held_out_attacks"]
    assert rows["XGBoost only"]["mean_detection_unseen"] == round(s["supervised"], 4)
    assert rows["Autoencoder only"]["delta_vs_xgboost"] == round(s["autoencoder"] - s["supervised"], 4)
