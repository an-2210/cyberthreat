"""
Tests for retrieval-grounded report generation.

No fake clients or indexes are used. Pure-function tests check parsing, citation and severity logic on
their real inputs. The integration test uses the real FAISS index, the real saved multiclass detector on
rows from a real CIC-IDS2017 CSV, and the real Ollama server; it is skipped if any of these is missing.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.threat_intelligence.report_generator import (
    OllamaClient,
    OllamaError,
    build_prompt,
    parse_llm_json,
    severity_for,
    validate_citations,
)

THRESHOLDS = {"critical": 0.85, "high": 0.70, "medium": 0.45, "low": 0.00}
ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = ROOT / "models" / "artifacts" / "vector_store"
CSV = ROOT / "data" / "raw" / "CICIDS2017" / "MachineLearningCVE" / "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"


def test_severity_uses_configured_thresholds_and_none_for_benign():
    assert severity_for("PortScan", 0.9, THRESHOLDS) == "critical"
    assert severity_for("PortScan", 0.72, THRESHOLDS) == "high"
    assert severity_for("PortScan", 0.5, THRESHOLDS) == "medium"
    assert severity_for("PortScan", 0.1, THRESHOLDS) == "low"
    assert severity_for("BENIGN", 0.99, THRESHOLDS) == "none"


def test_parse_llm_json_accepts_wrapped_objects_and_rejects_missing_fields():
    wrapped = ('Here you go: {"attack_description": "a", "potential_impact": "b", '
               '"recommended_mitigation": [], "citations": []} done')
    assert parse_llm_json(wrapped)["attack_description"] == "a"
    with pytest.raises(ValueError, match="missing fields"):
        parse_llm_json('{"attack_description": "a"}')
    with pytest.raises(ValueError, match="no JSON"):
        parse_llm_json("no json here")


def test_validate_citations_splits_valid_from_invented_ids():
    out = validate_citations(["PROFILE:PortScan", "ATTACK:T9999", "PROFILE:PortScan"],
                             ["PROFILE:PortScan", "ATTACK:T1046"])
    assert out == {"valid": ["PROFILE:PortScan"], "dropped": ["ATTACK:T9999"]}


def test_prompt_contains_detection_mapping_and_only_the_given_context():
    hits = [{"id": "ATTACK:T1046", "title": "T1046 Network Service Discovery", "text": "scan text"}]
    techs = [{"technique_id": "T1046", "technique_name": "Network Service Discovery",
              "tactics": ["discovery"], "confidence": "high"}]
    detection = {"predicted_label": "PortScan", "confidence": 0.9,
                 "top_class_probabilities": [("PortScan", 0.9), ("BENIGN", 0.05)]}
    prompt = build_prompt(detection, hits, techs)
    assert "predicted label: PortScan" in prompt
    assert "[ATTACK:T1046]" in prompt
    assert "T1046 Network Service Discovery" in prompt


def test_ollama_unreachable_gives_clear_error():
    client = OllamaClient("any-model", host="http://127.0.0.1:9", timeout=2)
    with pytest.raises(OllamaError, match="not reachable"):
        client.available_models()


def _live_ollama_model(name: str = "qwen2.5:14b"):
    client = OllamaClient(name, timeout=600)
    try:
        return client if name in client.available_models() else None
    except OllamaError:
        return None


@pytest.mark.skipif(not INDEX_DIR.exists() or not CSV.exists(), reason="index or real CSV not present")
def test_end_to_end_report_from_real_detection_and_real_llm():
    client = _live_ollama_model()
    if client is None:
        pytest.skip("Ollama server or model qwen2.5:14b not available")

    from src.data.dataset_adapters import CICIDS2017Adapter
    from src.data.loaders import read_cicids2017_csv
    from src.models.inference import CICIDS2017Detector
    from src.threat_intelligence.attack_mapping import load_attack_catalog
    from src.threat_intelligence.rag import ThreatIntelIndex
    from src.threat_intelligence.report_generator import generate_report
    from src.utils.config import load_config

    cfg = load_config(str(ROOT / "configs" / "config.yaml"))
    detector = CICIDS2017Detector.load("multiclass", "xgboost", artifacts_dir=ROOT / "models" / "artifacts")
    X, _, y = CICIDS2017Adapter("Label").extract_labels(read_cicids2017_csv(CSV).sample(400, random_state=0))
    proba = detector.predict_proba(X)
    row = int(np.argmax(proba[:, detector.labels.index("PortScan")]))
    label = detector.labels[int(proba[row].argmax())]
    detection = {
        "predicted_label": label,
        "true_label": str(y.iloc[row]),
        "confidence": float(proba[row].max()),
        "top_class_probabilities": [(detector.labels[j], float(proba[row, j]))
                                    for j in np.argsort(proba[row])[::-1][:3]],
    }
    index = ThreatIntelIndex.load(INDEX_DIR)
    catalog = load_attack_catalog(ROOT / "data" / "external" / "enterprise-attack.json")
    rep = generate_report(detection, index, catalog, client, cfg.fusion.severity_thresholds, top_k=5)

    retrieved = {c["id"] for c in rep["evidence"]["retrieved_context"]}
    assert rep["citations"] and set(rep["citations"]) <= retrieved
    assert rep["predicted_label"] == label
    assert rep["severity"] == severity_for(label, detection["confidence"], cfg.fusion.severity_thresholds)
    assert rep["attack_description"] and rep["potential_impact"] and rep["recommended_mitigation"]
