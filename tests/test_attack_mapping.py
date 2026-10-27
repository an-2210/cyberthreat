"""
ATT&CK mapping tests. The first group is SYNTHETIC (tiny in-memory catalog, no downloads). The last
test checks the real official bundle and is skipped when data/external/enterprise-attack.json is absent.
"""

import json
from pathlib import Path

import pytest

from src.threat_intelligence.attack_mapping import (
    CIC_IDS2017_LABELS,
    LABEL_TO_TECHNIQUES,
    load_attack_catalog,
    map_predictions,
    techniques_for_label,
    validate_mapping,
)

REAL_BUNDLE = Path(__file__).resolve().parents[1] / "data" / "external" / "enterprise-attack.json"


def _synthetic_catalog():
    return {
        "version": "test",
        "techniques": {
            "T1046": {"name": "Network Service Discovery", "tactics": ["discovery"], "is_subtechnique": False, "url": None},
            "T1110.001": {"name": "Password Guessing", "tactics": ["credential-access"], "is_subtechnique": True, "url": None},
        },
    }


def test_labels_cover_all_cic_ids2017_classes():
    assert len(CIC_IDS2017_LABELS) == 15
    assert "BENIGN" in CIC_IDS2017_LABELS and LABEL_TO_TECHNIQUES["BENIGN"] == []


def test_validation_accepts_known_ids_and_rejects_unknown_ones():
    catalog = _synthetic_catalog()
    validate_mapping(catalog, {"PortScan": [("T1046", "high", "x")]})
    with pytest.raises(ValueError, match="unknown technique T9999"):
        validate_mapping(catalog, {"PortScan": [("T9999", "high", "x")]})


def test_validation_rejects_invalid_confidence():
    with pytest.raises(ValueError, match="invalid confidence"):
        validate_mapping(_synthetic_catalog(), {"PortScan": [("T1046", "certain", "x")]})


def test_techniques_for_label_enriches_with_catalog_fields():
    catalog = _synthetic_catalog()
    out = techniques_for_label("PortScan", catalog, {"PortScan": [("T1046", "high", "probing")]})
    assert out == [{"technique_id": "T1046", "technique_name": "Network Service Discovery", "tactics": ["discovery"],
                    "is_subtechnique": False, "url": None, "confidence": "high", "rationale": "probing"}]
    assert techniques_for_label("BENIGN", catalog) == []


def test_map_predictions_handles_repeated_and_unknown_labels():
    series = map_predictions(["PortScan", "BENIGN", "PortScan", "Unknown label"], _synthetic_catalog() | {})
    assert len(series) == 4
    assert series.iloc[1] == []
    assert series.iloc[3] == []


@pytest.mark.skipif(not REAL_BUNDLE.exists(), reason="official ATT&CK bundle not downloaded")
def test_every_mapped_technique_exists_in_official_bundle():
    catalog = load_attack_catalog(REAL_BUNDLE)
    validate_mapping(catalog)
    assert catalog["version"] is not None
    for label in CIC_IDS2017_LABELS:
        for tech in techniques_for_label(label, catalog):
            assert tech["tactics"], f"{tech['technique_id']} has no tactic in the bundle"


def test_unmapped_label_is_reported_and_has_reason():
    from src.threat_intelligence.attack_mapping import UNMAPPED_LABELS, label_status

    assert label_status("Infiltration") == "unmapped"
    assert "Infiltration" in UNMAPPED_LABELS
    assert label_status("BENIGN") == "benign"
    assert techniques_for_label("Infiltration", _synthetic_catalog()) == []


def test_summarize_detections_counts_technique_level_outcomes():
    from src.threat_intelligence.attack_mapping import summarize_detections

    catalog = _synthetic_catalog()
    catalog["techniques"]["T1110.001"]["name"] = "Password Guessing"
    true = ["PortScan", "FTP-Patator", "BENIGN", "Infiltration"]
    pred = ["PortScan", "SSH-Patator", "BENIGN", "BENIGN"]
    s = summarize_detections(true, pred, catalog)
    per = {r["technique_id"]: r for r in s["per_technique"]}
    assert s["label_accuracy"] == 0.5
    assert per["T1046"]["correct_rows"] == 1 and per["T1046"]["true_rows"] == 1
    assert per["T1110.001"]["true_rows"] == 1 and per["T1110.001"]["predicted_rows"] == 1
    assert per["T1110.001"]["correct_rows"] == 1  # FTP and SSH Patator share T1110.001
    assert s["unmapped_true_rows"] == 1
    assert s["technique_set_agreement"] == 1.0  # all four rows have identical technique sets (FTP/SSH share T1110.001; empty vs empty)


def test_summarize_detections_rejects_length_mismatch():
    import pytest as _pytest
    from src.threat_intelligence.attack_mapping import summarize_detections

    with _pytest.raises(ValueError):
        summarize_detections(["BENIGN"], [], _synthetic_catalog())
