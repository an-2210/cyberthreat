"""
Grounded threat-intelligence reports: retrieval (FAISS) + local LLM (Ollama).

Division of labour (kept deliberately narrow):
  * Code decides: predicted label, confidence, severity (config thresholds), the ATT&CK mapping, the
    retrieved evidence list, and which citations are valid.
  * The LLM writes only: attack description, potential impact, recommended mitigations and the
    citations it relies on. It receives only the retrieved context and must cite document IDs from it.

Citations that name a document that was not retrieved are dropped and recorded in the report.

Requires a running Ollama server (default http://localhost:11434) and a pulled model
(default from configs/config.yaml: llm.ollama_model).

Run from the repository root (after building the index in src/threat_intelligence/rag.py):
    python -m src.threat_intelligence.report_generator
"""

import argparse
import datetime as _dt
import json
import re
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from src.threat_intelligence.attack_mapping import label_status, load_attack_catalog, techniques_for_label
from src.threat_intelligence.rag import ThreatIntelIndex
from src.threat_intelligence.risk_engine import risk_for_detection
from src.utils.config import get_absolute_path, load_config
from src.utils.io_utils import save_json
from src.utils.logging import setup_logger

logger = setup_logger("report_generator")

REQUIRED_LLM_FIELDS = ("attack_description", "potential_impact", "recommended_mitigation", "citations")


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    """Minimal client for Ollama's /api/chat endpoint (standard library only)."""

    def __init__(self, model: str, host: str = "http://localhost:11434", timeout: float = 600.0, seed: int = 42):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.seed = seed

    def available_models(self) -> List[str]:
        try:
            with urllib.request.urlopen(f"{self.host}/api/tags", timeout=10) as r:
                return [m["name"] for m in json.loads(r.read())["models"]]
        except (urllib.error.URLError, OSError) as exc:
            raise OllamaError(f"Ollama is not reachable at {self.host}. Start it with `ollama serve`. ({exc})") from exc

    def chat_json(self, system: str, user: str) -> str:
        body = json.dumps({
            "model": self.model,
            "stream": False,
            "format": "json",
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "options": {"temperature": 0, "seed": self.seed, "num_ctx": 8192},
        }).encode("utf-8")
        req = urllib.request.Request(f"{self.host}/api/chat", data=body, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                payload = json.loads(r.read())
        except (urllib.error.URLError, OSError) as exc:
            raise OllamaError(f"Ollama request failed: {exc}") from exc
        return payload["message"]["content"]


SYSTEM_PROMPT = (
    "You are a SOC analyst writing a threat-intelligence note for a detected network event. "
    "Use ONLY the context documents provided. Do not invent CVEs, IP addresses, tools, or facts that are not in "
    "the context. Cite the document IDs you relied on in the 'citations' list, using IDs exactly as given (for "
    "example ATTACK:T1046 or PROFILE:PortScan). If the context is insufficient for a field, say so explicitly. "
    "Respond with a single JSON object with keys: attack_description (string), potential_impact (string), "
    "recommended_mitigation (list of strings), citations (list of strings)."
)


def build_prompt(detection: Dict[str, Any], hits: Sequence[Dict[str, Any]], techniques: List[Dict[str, Any]]) -> str:
    context = "\n\n".join(f"[{h['id']}] {h['title']}\n{h['text']}" for h in hits)
    tech_lines = "\n".join(f"- {t['technique_id']} {t['technique_name']} (tactics: {', '.join(t['tactics'])}; confidence {t['confidence']})"
                           for t in techniques) or "- none assigned"
    top = ", ".join(f"{name} {p:.3f}" for name, p in detection["top_class_probabilities"])
    return (
        f"DETECTION\n"
        f"- predicted label: {detection['predicted_label']}\n"
        f"- model confidence (probability of predicted label): {detection['confidence']:.3f}\n"
        f"- top class probabilities: {top}\n\n"
        f"ATT&CK MAPPING ASSIGNED BY THE PROJECT (do not change it)\n{tech_lines}\n\n"
        f"CONTEXT DOCUMENTS\n{context}\n\n"
        "Write the JSON object now."
    )


def parse_llm_json(raw: str) -> Dict[str, Any]:
    """Parses the model's JSON. Extracts the first object if the model added surrounding text."""
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.S)
        if not match:
            raise ValueError("LLM response contained no JSON object")
        obj = json.loads(match.group(0))
    missing = [k for k in REQUIRED_LLM_FIELDS if k not in obj]
    if missing:
        raise ValueError(f"LLM response is missing fields: {missing}")
    if not isinstance(obj["recommended_mitigation"], list) or not isinstance(obj["citations"], list):
        raise ValueError("recommended_mitigation and citations must be lists")
    return obj


def validate_citations(citations: List[str], retrieved_ids: Sequence[str]) -> Dict[str, List[str]]:
    allowed = set(retrieved_ids)
    valid = [c for c in dict.fromkeys(citations) if c in allowed]
    dropped = [c for c in dict.fromkeys(citations) if c not in allowed]
    return {"valid": valid, "dropped": dropped}


def limitations_for(status: str, has_explanation: bool) -> List[str]:
    """Limitations stated in every report. Written by code, not by the LLM."""
    items = [
        "Flow-level records: the CIC-IDS2017 flows used here carry no timestamps, host names or IP addresses, so the "
        "affected asset, frequency and persistence of this activity are not assessed.",
        "Confidence is the classifier's probability for this flow. It is not the probability that an attack occurred.",
        "No attribution is made: threat actor, malware family and IP reputation are not assessed from this data.",
        "The attack-to-technique mapping is a project mapping of dataset labels to ATT&CK techniques, with confidence "
        "levels assigned by the project (see src/threat_intelligence/attack_mapping.py).",
    ]
    if status != "mapped":
        items.append("Insufficient evidence to assign a single ATT&CK technique to this label.")
    if not has_explanation:
        items.append("No local SHAP explanation was computed for this flow.")
    return items


def generate_report(
    detection: Dict[str, Any],
    index: ThreatIntelIndex,
    catalog: Dict[str, Any],
    client: OllamaClient,
    risk_cfg: Any,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Builds one grounded report for a detection. Raises if the LLM output cannot be grounded.

    The risk level comes from src/threat_intelligence/risk_engine.py (every term is stored in the report). The LLM
    writes only the description, impact and mitigations, from the retrieved text it is given.
    """
    label = detection["predicted_label"]
    confidence = float(detection["confidence"])
    techniques = techniques_for_label(label, catalog)
    query = f"{label} network attack. " + " ".join(t["technique_name"] for t in techniques)
    hits = index.search(query, k=top_k)
    retrieved_ids = [h["id"] for h in hits]

    raw = client.chat_json(SYSTEM_PROMPT, build_prompt(detection, hits, techniques))
    llm = parse_llm_json(raw)
    cites = validate_citations([str(c) for c in llm["citations"]], retrieved_ids)
    if not cites["valid"]:
        raise ValueError(f"No valid citations in LLM output; cited {llm['citations']}, retrieved {retrieved_ids}")
    if cites["dropped"]:
        logger.warning(f"Dropped citations not in retrieved context: {cites['dropped']}")

    risk = risk_for_detection(label, confidence, risk_cfg)
    report_id = str(uuid.uuid4())
    status = label_status(label)
    explanation = detection.get("explanation")
    return {
        "report_id": report_id,
        "incident_id": f"INC-{report_id[:8].upper()}",
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "predicted_label": label,
        "true_label": detection.get("true_label"),
        "confidence": round(confidence, 4),
        "severity": risk["level"],
        "risk": risk,
        "label_status": status,
        "attack_description": str(llm["attack_description"]).strip(),
        "potential_impact": str(llm["potential_impact"]).strip(),
        "recommended_mitigation": [str(m).strip() for m in llm["recommended_mitigation"]],
        "mitre_mapping": [
            {"technique_id": t["technique_id"], "technique_name": t["technique_name"], "tactics": t["tactics"],
             "confidence": t["confidence"], "url": t["url"]} for t in techniques
        ],
        "statement_basis": {
            "observed_evidence": {
                "source": "features of one test flow (CIC-IDS2017 MachineLearningCVE, flow level)",
                "top_class_probabilities": detection["top_class_probabilities"],
                "local_explanation": explanation,
            },
            "model_inference": {
                "classifier": "XGBoost multiclass: predicted label and class probabilities",
                "explanation_model": "XGBoost binary with SHAP: contributions toward ATTACK, in log-odds",
                "predicted_label": label,
                "confidence": round(confidence, 4),
            },
            "retrieved_knowledge": [{"id": h["id"], "title": h["title"], "score": h["score"]} for h in hits],
            "generated_from_retrieved_text": ["attack_description", "potential_impact", "recommended_mitigation"],
        },
        "citations": cites["valid"],
        "limitations": limitations_for(status, explanation is not None),
        "validation": {"dropped_citations": cites["dropped"], "llm_fields_ok": True},
        "generator": {"llm": f"ollama:{client.model}", "retrieval": "faiss IndexFlatIP, all-MiniLM-L6-v2",
                      "temperature": 0, "seed": client.seed},
        "disclaimer": "Generated from retrieved project and ATT&CK text. Analyst review required before action.",
    }


def run(cfg=None, examples_per_label: int = 1, labels: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    """Selects real test-set detections for the chosen labels, then writes one report per detection."""
    import numpy as np
    import pandas as pd
    from src.models.inference import CICIDS2017Detector
    from src.run_rare_class_experiment import prepare_splits

    cfg = cfg or load_config("configs/config.yaml")
    rag_cfg = cfg.rag_cti
    llm_model = cfg.llm.ollama_model
    host = cfg.llm.ollama_host
    catalog = load_attack_catalog(get_absolute_path(cfg.mitre_attack.stix_data_path))
    client = OllamaClient(llm_model, host=host, timeout=float(cfg.llm.timeout_seconds), seed=int(cfg.system.seed))
    if llm_model not in client.available_models():
        raise OllamaError(f"Model '{llm_model}' is not pulled. Run `ollama pull {llm_model}`.")

    # Detector and data first, FAISS index last (see _faiss() in rag.py for why the order matters).
    detector = CICIDS2017Detector.load("multiclass", "xgboost", artifacts_dir=cfg.paths.artifacts_dir)
    splits = prepare_splits(cfg)["splits"]
    index = ThreatIntelIndex.load(rag_cfg.vector_store_dir)
    X_test = splits["X_test"].to_numpy(dtype=np.float32)
    true_names = pd.Series(splits["y_test_multi"]).to_numpy(dtype=object)
    proba = detector.model.predict_proba(X_test)
    classes = np.asarray(detector.model.model.classes_).astype(int)
    names = np.asarray(detector.labels, dtype=object)
    pred_ids = classes[proba.argmax(axis=1)]
    pred_names = names[pred_ids]
    conf = proba.max(axis=1)

    wanted = list(labels) if labels else ["PortScan", "DDoS", "Web Attack - Sql Injection", "Web Attack - XSS", "Bot"]
    selected = []
    for lab in wanted:
        rows = np.flatnonzero(pred_names == lab)[:examples_per_label]
        if len(rows) == 0:
            logger.warning(f"No test rows were predicted as '{lab}', so no report was generated for it.")
        for r in rows:
            top = np.argsort(proba[r])[::-1][:3]
            selected.append({
                "predicted_label": str(pred_names[r]),
                "true_label": str(true_names[r]),
                "confidence": float(conf[r]),
                "top_class_probabilities": [(str(names[classes[j]]), float(proba[r, j])) for j in top],
                "test_row": int(r),
            })
    # Local SHAP explanation of the binary decision for each selected flow (when the SHAP background was saved).
    from src.explainability.shap_explainer import BACKGROUND_FILE, build_explainer, explain_single
    bg_path = get_absolute_path(cfg.paths.results_dir) / "explainability" / BACKGROUND_FILE
    explainer, feature_names = None, None
    if bg_path.exists() and selected:
        binary = CICIDS2017Detector.load("binary", "xgboost", artifacts_dir=cfg.paths.artifacts_dir)
        explainer = build_explainer(binary.model.model.get_booster(), np.load(bg_path))
        feature_names = list(binary.pipeline.feature_names_out_)
    else:
        logger.warning(f"No SHAP background at {bg_path}; reports will say that no local explanation was computed.")
    for det in selected:
        if explainer is not None:
            det["explanation"] = explain_single(explainer, X_test[det["test_row"]], feature_names, top_k=5)

    out_dir = get_absolute_path(cfg.paths.results_dir) / "threat_intelligence" / "reports"
    reports = []
    for det in selected:
        rep = generate_report(det, index, catalog, client, cfg.risk, top_k=int(rag_cfg.top_k_retrieval))
        save_json(rep, out_dir / f"report_{det['predicted_label'].replace(' ', '_').replace('-', '')}_{det['test_row']}.json")
        logger.info(f"Wrote report for {det['predicted_label']} (true: {det['true_label']}, risk {rep['severity']}, score {rep['risk']['score']})")
        reports.append(rep)
    return reports


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Generate grounded threat-intelligence reports.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--per-label", type=int, default=1)
    parser.add_argument("--labels", nargs="*", default=None)
    args = parser.parse_args(argv)
    reports = run(load_config(args.config), examples_per_label=args.per_label, labels=args.labels)
    for r in reports:
        print(f"{r['predicted_label']:28s} true={str(r['true_label']):28s} severity={r['severity']:8s} "
              f"citations={r['citations']}")


if __name__ == "__main__":
    main()
