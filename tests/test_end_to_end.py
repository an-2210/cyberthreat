"""
End-to-end test on real CIC-IDS2017 rows: CSV -> detection -> SHAP explanation -> ATT&CK mapping -> risk -> CTI report.

No component is replaced by a stub. The test skips (with the reason) when the raw CSV, the saved artifacts, the SHAP
background or a running Ollama server are not available. The report step needs the FAISS index built by
src.threat_intelligence.rag.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
DDOS_CSV = ROOT / "data" / "raw" / "CICIDS2017" / "MachineLearningCVE" / "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
ARTIFACTS = ROOT / "models" / "artifacts"
BACKGROUND = ROOT / "results" / "explainability" / "shap_background.npy"
STIX = ROOT / "data" / "external" / "enterprise-attack.json"
INDEX_DIR = ROOT / "models" / "artifacts" / "vector_store"
RISK_LEVELS = {"CRITICAL", "HIGH", "MEDIUM", "LOW", "NONE", "UNRATED"}


def _real_rows() -> pd.DataFrame:
    """10 real DDoS flows and 10 real benign flows from the same CSV (labels are dropped before detection)."""
    raw = pd.read_csv(DDOS_CSV, low_memory=False, encoding="utf-8", encoding_errors="replace")
    raw.columns = raw.columns.str.strip()
    label = raw["Label"].astype(str).str.strip()
    attacks = raw[label == "DDoS"].head(10)
    benign = raw[label == "BENIGN"].head(10)
    return pd.concat([attacks, benign]).reset_index(drop=True)


@pytest.mark.skipif(not DDOS_CSV.exists(), reason="CIC-IDS2017 CSV not present in data/raw")
@pytest.mark.skipif(not (ARTIFACTS / "cicids2017_binary_xgboost.joblib").exists(), reason="saved models not present")
def test_csv_to_detection_mapping_and_risk_on_real_flows():
    from src.models.inference import CICIDS2017Detector
    from src.threat_intelligence.attack_mapping import LABEL_TO_TECHNIQUES, UNMAPPED_LABELS
    from src.threat_intelligence.risk_engine import risk_for_detection
    from src.utils.config import load_config

    cfg = load_config("configs/config.yaml")
    rows = _real_rows()
    features = rows.drop(columns=["Label"])

    binary = CICIDS2017Detector.load("binary", "xgboost", artifacts_dir=ARTIFACTS)
    multi = CICIDS2017Detector.load("multiclass", "xgboost", artifacts_dir=ARTIFACTS)
    flagged = np.asarray(binary.predict(features)) == "ATTACK"
    p_attack = binary.predict_proba(features)[:, list(binary.labels).index("ATTACK")]
    types = np.asarray(multi.predict(features))
    type_conf = multi.predict_proba(features).max(axis=1)

    assert flagged[:10].mean() >= 0.5, "most real DDoS flows should be flagged"
    assert flagged[10:].mean() <= 0.2, "most real benign flows should not be flagged"

    for name in set(types[flagged & (types != "BENIGN")]):
        assert name in LABEL_TO_TECHNIQUES or name in UNMAPPED_LABELS, f"{name} has no ATT&CK mapping decision"
    for i in np.flatnonzero(flagged & (types != "BENIGN")):
        risk = risk_for_detection(str(types[i]), float(type_conf[i]), cfg.risk)
        assert risk["level"] in RISK_LEVELS
        assert risk["not_available"]

    if not BACKGROUND.exists():
        pytest.skip("SHAP background not present; run python -m src.explainability.shap_explainer")
    from src.explainability.shap_explainer import build_explainer, explain_single

    explainer = build_explainer(binary.model.model.get_booster(), np.load(BACKGROUND))
    names = list(binary.pipeline.feature_names_out_)
    row = int(np.flatnonzero(flagged)[0])
    x = binary.pipeline.transform_new_data(features.iloc[[row]][binary.input_columns]).to_numpy(dtype=np.float32)[0]
    explanation = explain_single(explainer, x, names, top_k=5)
    margin = float(binary.model.model.get_booster().predict(xgb.DMatrix(x.reshape(1, -1)), output_margin=True)[0])
    assert explanation["expected_value_log_odds"] + explanation["margin_total_log_odds"] == pytest.approx(margin, abs=1e-3)
    assert len(explanation["top_features"]) == 5


@pytest.mark.skipif(not DDOS_CSV.exists(), reason="CIC-IDS2017 CSV not present in data/raw")
@pytest.mark.skipif(not STIX.exists(), reason="ATT&CK STIX bundle not present")
@pytest.mark.skipif(not INDEX_DIR.exists(), reason="FAISS index not built")
def test_end_to_end_report_from_real_flow_with_ollama():
    from src.models.inference import CICIDS2017Detector
    from src.threat_intelligence.attack_mapping import load_attack_catalog
    from src.threat_intelligence.rag import ThreatIntelIndex
    from src.threat_intelligence.report_generator import OllamaClient, OllamaError, generate_report
    from src.utils.config import load_config

    cfg = load_config("configs/config.yaml")
    client = OllamaClient(cfg.llm.ollama_model, host=cfg.llm.ollama_host, timeout=float(cfg.llm.timeout_seconds))
    try:
        if cfg.llm.ollama_model not in client.available_models():
            pytest.skip(f"Ollama model {cfg.llm.ollama_model} is not pulled")
    except OllamaError as exc:
        pytest.skip(str(exc))

    rows = _real_rows().head(10)  # the DDoS flows
    features = rows.drop(columns=["Label"])
    multi = CICIDS2017Detector.load("multiclass", "xgboost", artifacts_dir=ARTIFACTS)
    proba = multi.predict_proba(features)
    i = int(np.argmax(np.max(proba, axis=1)))
    label = str(multi.labels[int(np.argmax(proba[i]))])
    detection = {
        "predicted_label": label,
        "true_label": "DDoS",
        "confidence": float(proba[i].max()),
        "top_class_probabilities": [(str(multi.labels[j]), float(proba[i, j])) for j in np.argsort(proba[i])[::-1][:3]],
    }
    index = ThreatIntelIndex.load(INDEX_DIR)
    catalog = load_attack_catalog(STIX)
    report = generate_report(detection, index, catalog, client, cfg.risk, top_k=5)

    assert report["incident_id"].startswith("INC-") and report["created_utc"]
    assert report["severity"] in RISK_LEVELS
    assert report["statement_basis"]["retrieved_knowledge"], "the report must cite retrieved knowledge"
    assert report["limitations"], "every report states its limitations"
    assert set(report["citations"]) <= {h["id"] for h in report["statement_basis"]["retrieved_knowledge"]}
