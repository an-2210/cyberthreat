"""
Backend for the CyberThreat dashboard (app/frontend).

It only reads what the CIC-IDS2017 pipeline already wrote, plus one prediction endpoint:

    python -m src.run_cicids2017                   -> results/ and models/artifacts/
    python -m src.run_rare_class_experiment        -> results/rare_class_experiment/
    uvicorn app.backend.main:app --port 8000       -> this file

Endpoints (all JSON, all under /api):
    GET  /api/health         is the backend up
    GET  /api/overview       numbers from the last pipeline run (split sizes, sampling, cleaning)
    GET  /api/data           data-quality audit and class distribution
    GET  /api/models         test-set results for each saved model
    GET  /api/rare-class     results of the rare-class experiment
    GET  /api/figures        list of saved plots; the images themselves are served under /figures/
    POST /api/predict        classify an uploaded CSV with one saved model
"""

import io
import json
import re
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.inference import CICIDS2017Detector, MODEL_NAMES, TASKS  # noqa: E402
from src.threat_intelligence.report_generator import OllamaClient, OllamaError  # noqa: E402
from src.threat_intelligence.risk_engine import risk_for_detection  # noqa: E402
from src.threat_intelligence.attack_mapping import LABEL_TO_TECHNIQUES  # noqa: E402
from src.explainability.shap_explainer import BACKGROUND_FILE, build_explainer, explain_single  # noqa: E402
from src.utils.config import load_config  # noqa: E402

RESULTS = REPO_ROOT / "results"
RUN_SUMMARY = RESULTS / "reports" / "cicids2017_run_summary.json"
QUALITY_REPORT = RESULTS / "reports" / "cicids2017_data_quality_report.json"
CLASS_DISTRIBUTION = RESULTS / "tables" / "cicids2017_class_distribution.csv"
RARE_CLASS_SUMMARY = RESULTS / "rare_class_experiment" / "comparison_summary.csv"
FIGURES = RESULTS / "figures"
METRICS = RESULTS / "metrics"
ARTIFACTS = REPO_ROOT / "models" / "artifacts"

BENIGN = "BENIGN"
ATTACK = "ATTACK"
MAX_PREDICTION_ROWS = 1000  # per-row results returned to the browser; counts always cover the whole file

app = FastAPI(title="CyberThreat dashboard API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/figures", StaticFiles(directory=FIGURES, check_dir=False), name="figures")


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{path.relative_to(REPO_ROOT)} is missing. Run the pipeline first.")
    return json.loads(path.read_text(encoding="utf-8"))


def _model_key(task: str, model: str) -> str:
    return f"{task}_{model}"


def _parse_model_key(key: str) -> tuple:
    for task in TASKS:
        for model in MODEL_NAMES:
            if key == _model_key(task, model):
                return task, model
    raise HTTPException(status_code=400, detail=f"Unknown model '{key}'. Use one of: {[_model_key(t, m) for t in TASKS for m in MODEL_NAMES]}")


@lru_cache(maxsize=None)
def _load_detector(task: str, model: str) -> CICIDS2017Detector:
    return CICIDS2017Detector.load(task, model, artifacts_dir=ARTIFACTS)


@app.get("/api/health")
def health() -> Dict[str, Any]:
    return {"ok": True, "pipelineRunFound": RUN_SUMMARY.exists()}


@app.get("/api/overview")
def overview() -> Dict[str, Any]:
    run = _read_json(RUN_SUMMARY)
    return {
        "dataSource": run["data_source"],
        "seed": run["seed"],
        "elapsedSeconds": run["elapsed_seconds"],
        "splitSizes": run["split_sizes"],
        "duplicateRowsRemoved": run["duplicate_rows_removed_before_split"],
        "identifierColumnsDropped": run["identifier_columns_dropped"],
        "leakageFlaggedFeatures": run["single_feature_leakage_flagged"],
        "classNames": run["class_names"],
        "featureCount": run["n_features"],
        "sampling": run["sampling"],
    }


@app.get("/api/data")
def data() -> Dict[str, Any]:
    quality = _read_json(QUALITY_REPORT)
    if not CLASS_DISTRIBUTION.exists():
        raise HTTPException(status_code=404, detail="Class distribution table is missing. Run the pipeline first.")
    classes = pd.read_csv(CLASS_DISTRIBUTION)
    return {
        "rows": quality["n_rows"],
        "featureColumns": quality["n_feature_columns"],
        "missingCells": quality["missing_values"]["total_missing_cells"],
        "missingByFeature": quality["missing_values"]["features_with_missing"],
        "infiniteCells": quality["infinite_values"]["total_infinite_cells"],
        "infiniteByFeature": quality["infinite_values"]["features_with_infinite"],
        "duplicateRows": quality["duplicate_rows_exact_including_label"]["count"],
        "conflictingFeatureRows": quality["duplicate_feature_rows_with_conflicting_labels"]["rows_involved"],
        "constantFeatures": quality["constant_features"],
        "classes": [
            {"label": str(r["label"]), "count": int(r["count"]), "percent": float(r["percent"])}
            for _, r in classes.iterrows()
        ],
    }


@app.get("/api/models")
def models() -> List[Dict[str, Any]]:
    out = []
    for task in TASKS:
        for model in MODEL_NAMES:
            path = METRICS / f"cicids2017_{task}_{model}.json"
            if not path.exists():
                continue
            m = json.loads(path.read_text(encoding="utf-8"))
            out.append({
                "key": _model_key(task, model),
                "task": task,
                "model": model,
                "accuracy": m["accuracy"],
                "mcc": m["mcc"],
                # binary metrics are for the ATTACK class; multiclass metrics are macro-averaged over the 15 labels
                "precision": m["precision"] if task == "binary" else m["precision_macro"],
                "recall": m["recall"] if task == "binary" else m["recall_macro"],
                "f1": m["f1"] if task == "binary" else m["f1_macro"],
                "rocAuc": m["roc_auc"] if task == "binary" else m["roc_auc_ovr_macro"],
                "fpr": m.get("fpr"),
                "trainSeconds": m["train_fit_seconds"],
                "trainRows": m["n_train_rows"],
                "testRows": m["n_test_rows"],
                "confusionMatrix": m["confusion_matrix"],
                "perClass": m.get("per_class"),
                "topFeatures": list(m["top_feature_importances"].items())[:15],
            })
    return out


@app.get("/api/rare-class")
def rare_class() -> List[Dict[str, Any]]:
    if not RARE_CLASS_SUMMARY.exists():
        raise HTTPException(status_code=404, detail="Rare-class results are missing. Run src.run_rare_class_experiment first.")
    df = pd.read_csv(RARE_CLASS_SUMMARY)
    return df.where(pd.notna(df), None).to_dict(orient="records")


@app.get("/api/figures")
def figures() -> List[str]:
    if not FIGURES.exists():
        return []
    return sorted(p.name for p in FIGURES.glob("*.png"))


@app.post("/api/predict")
async def predict(file: UploadFile = File(...), modelKey: str = Form(...)) -> Dict[str, Any]:
    task, model = _parse_model_key(modelKey)
    try:
        detector = _load_detector(task, model)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    content = await file.read()
    try:
        raw = pd.read_csv(io.BytesIO(content))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read the file as CSV: {exc}")
    raw.columns = raw.columns.str.strip()
    if raw.empty:
        raise HTTPException(status_code=400, detail="The CSV has no rows.")

    started = time.perf_counter()
    try:
        labels = detector.predict(raw)
        proba = detector.predict_proba(raw)
    except ValueError as exc:
        # raised by the detector when required feature columns are missing
        raise HTTPException(status_code=400, detail=str(exc))
    elapsed_ms = (time.perf_counter() - started) * 1000

    counts = pd.Series(labels).value_counts().to_dict()
    rows = [
        {"row": i + 1, "prediction": str(labels[i]), "confidence": round(float(proba[i].max()), 4)}
        for i in range(min(len(labels), MAX_PREDICTION_ROWS))
    ]
    return {
        "modelKey": modelKey,
        "rowsClassified": int(len(labels)),
        "counts": {str(k): int(v) for k, v in counts.items()},
        "benignRows": int(counts.get(BENIGN, 0)),
        "rows": rows,
        "milliseconds": round(elapsed_ms, 1),
    }


# ---------------------------------------------------------------------------
# anomaly detection, hybrid fusion and unseen attacks
# ---------------------------------------------------------------------------

HYBRID = METRICS / "cicids2017_hybrid_detection.json"
HYBRID_TABLE = RESULTS / "tables" / "cicids2017_hybrid_per_class_detection.csv"
LEAVE_ONE_OUT = METRICS / "cicids2017_leave_one_attack_out.json"

DETECTOR_LABELS = {
    "supervised_xgboost": "XGBoost (supervised)",
    "isolation_forest_raw": "Isolation Forest (raw features)",
    "isolation_forest_log": "Isolation Forest (log features)",
    "autoencoder": "Autoencoder (benign only)",
    "hybrid_fusion": "Fusion (XGBoost + autoencoder)",
}


@app.get("/api/detectors")
def detectors() -> Dict[str, Any]:
    """Test-set comparison of the supervised model, the anomaly detectors and their fusion (all at 1% benign FPR)."""
    hybrid = _read_json(HYBRID)
    table = pd.read_csv(HYBRID_TABLE) if HYBRID_TABLE.exists() else None
    rows = []
    for key, label in DETECTOR_LABELS.items():
        m = hybrid["test_results"].get(key)
        if m is None:
            continue
        rows.append({
            "key": key,
            "label": label,
            "recall": m["recall"],
            "precision": m["precision"],
            "f1": m["f1"],
            "fpr": m["fpr"],
            "rocAuc": m["roc_auc"],
        })
    per_class = []
    if table is not None:
        for _, r in table.iterrows():
            per_class.append({
                "class": str(r["class"]),
                "testRows": int(r["test_rows"]),
                "isAttack": bool(r["is_attack"]),
                "supervised": float(r["supervised_xgboost"]),
                "autoencoder": float(r["autoencoder"]),
                "fusion": float(r["hybrid_fusion"]),
            })
    return {
        "targetFpr": hybrid["target_fpr"],
        "detectors": rows,
        "anomalyAucOnValidation": hybrid["anomaly_candidate_validation_roc_auc"],
        "chosenAnomalyDetector": hybrid["anomaly_candidate_chosen"],
        "fusion": {
            "alpha": hybrid["fusion"]["alpha"],
            "threshold": hybrid["fusion"]["threshold"],
            "validationF1": hybrid["fusion"]["val_f1"],
        },
        "perClass": per_class,
    }


@app.get("/api/unseen-attacks")
def unseen_attacks() -> Dict[str, Any]:
    """Leave-one-attack-out results: each attack type is removed from the supervised training and then tested."""
    data = _read_json(LEAVE_ONE_OUT)
    s = data["summary"]
    folds = [
        {
            "heldOut": f["held_out"],
            "testRows": f["test_rows_held_out"],
            "supervised": f["supervised_detection"],
            "autoencoder": f["autoencoder_detection"],
            "fusion": f["fusion_detection"],
            "fusionAlpha": f["fusion_alpha"],
            "falsePositiveRate": f["supervised_fpr"],
        }
        for f in data["folds"]
    ]
    return {
        "targetFpr": s["target_fpr"],
        "baseline": next(f for f in folds if f["heldOut"] == "none"),
        "folds": [f for f in folds if f["heldOut"] != "none"],
        "meanDetection": s["mean_detection_over_held_out_attacks"],
        "detectedAtLeastHalf": s["held_out_attacks_detected_at_least_50pct"],
        "nHeldOut": s["n_held_out_attacks"],
        "nLowSample": s["n_low_sample_held_out"],
    }


FAMILY_HOLDOUT = METRICS / "cicids2017_family_holdout.json"


@app.get("/api/family-holdout")
def family_holdout() -> Dict[str, Any]:
    """Whole attack families held out of supervised training together (harder test than one type at a time)."""
    data = _read_json(FAMILY_HOLDOUT)
    s = data["summary"]
    folds = [
        {
            "heldOut": f["held_out"],
            "heldOutLabels": f["held_out_labels"],
            "testRows": f["test_rows_held_out"],
            "supervised": f["supervised_detection"],
            "autoencoder": f["autoencoder_detection"],
            "fusion": f["fusion_detection"],
            "fusionAlpha": f["fusion_alpha"],
            "falsePositiveRate": f["supervised_fpr"],
        }
        for f in data["folds"]
    ]
    return {
        "targetFpr": s["target_fpr"],
        "baseline": next(f for f in folds if f["heldOut"] == "none"),
        "folds": [f for f in folds if f["heldOut"] != "none"],
        "meanDetection": s["mean_detection_over_held_out_groups"],
        "detectedAtLeastHalf": s["held_out_groups_detected_at_least_50pct"],
        "nHeldOut": s["n_held_out_groups"],
        "nLowSample": s["n_low_sample_held_out"],
    }


# ---------------------------------------------------------------------------
# UNSW-NB15: native model, TTL ablation, Fuzzers, and CIC-to-UNSW transfer
# ---------------------------------------------------------------------------

UNSW_NATIVE = METRICS / "unsw_native_all_features.json"
UNSW_TTL = METRICS / "unsw_native_ttl_ablation.json"
UNSW_FUZZERS = METRICS / "unsw_native_fuzzers_fix.json"
UNSW_DERIVED = METRICS / "unsw_native_derived_features.json"
UNSW_IN_DOMAIN = METRICS / "unsw_in_domain_shared_features.json"
CIC_TO_UNSW = METRICS / "cross_dataset_cic_to_unsw.json"

TTL_VARIANT_LABELS = {
    "all_features": "All features",
    "without_ttl": "Without TTL fields",
    "only_ttl": "Only TTL fields",
}


def _binary_summary(m: Dict[str, Any]) -> Dict[str, Any]:
    return {k: m[k] for k in ("accuracy", "precision", "recall", "f1", "fpr", "roc_auc")}


@app.get("/api/unsw")
def unsw() -> Dict[str, Any]:
    """UNSW-NB15 results: the UNSW-native model, what it relies on, Fuzzers, and the CIC-to-UNSW transfer."""
    native = _read_json(UNSW_NATIVE)
    ttl = _read_json(UNSW_TTL)
    fuzz = _read_json(UNSW_FUZZERS)
    derived = _read_json(UNSW_DERIVED)
    cross = _read_json(CIC_TO_UNSW)
    in_domain = _read_json(UNSW_IN_DOMAIN)

    variants = ttl["variants"]
    categories = sorted(variants["all_features"]["per_attack_category"].items(), key=lambda kv: -kv[1]["support"])
    per_category = [
        {
            "category": cat,
            "support": stats["support"],
            **{key: variants[key]["per_attack_category"][cat]["recall"] for key in TTL_VARIANT_LABELS},
        }
        for cat, stats in categories
    ]

    seed_runs = derived["summary"]
    fuzzers_variants = [
        {"key": key, "label": label, "fuzzersRecall": fuzz["variants"][key]["per_attack_category"]["Fuzzers"]["recall"],
         "f1": fuzz["variants"][key]["test"]["f1"]}
        for key, label in (("binary_xgboost", "Binary XGBoost (headline model)"),
                           ("multiclass_xgboost", "Multiclass XGBoost"))
    ]
    return {
        "native": {
            "featureCount": len(native["features_used"]),
            "threshold": native["threshold"]["value"],
            "test": _binary_summary(native["unsw_test"]),
            "confusionMatrix": native["unsw_test"]["confusion_matrix"],
            "topFeatures": native["top_features_by_importance"],
            "rows": native["shape"],
            "leakageFlagged": native["leakage_scan_flagged"],
            "leakageTopFeature": native["leakage_scan_top"][0],
        },
        "ttl": {
            "variants": [
                {"key": key, "label": label, "featureCount": len(variants[key]["features"]),
                 "test": variants[key]["test"], "fuzzersRecall": variants[key]["per_attack_category"]["Fuzzers"]["recall"]}
                for key, label in TTL_VARIANT_LABELS.items()
            ],
            "perCategory": per_category,
        },
        "fuzzers": {
            "chosenOnValidation": fuzz["chosen_on_validation"],
            "variants": fuzzers_variants,
            "seedRuns": [
                {"key": "baseline", "label": "Binary XGBoost, 3 seeds",
                 "fuzzersRecall": seed_runs["baseline"]["test_fuzzers_recall"],
                 "f1": seed_runs["baseline"]["test_f1"]},
                {"key": "with_derived", "label": "Binary + ratio features, 3 seeds",
                 "fuzzersRecall": seed_runs["with_derived"]["test_fuzzers_recall"],
                 "f1": seed_runs["with_derived"]["test_f1"]},
            ],
        },
        "transfer": {
            "cicInDomain": _binary_summary(cross["in_domain_cic_test"]),
            "transferAtHalf": _binary_summary(cross["transfer_unsw_nb15_test_threshold_0_5"]),
            "calibrated": {"threshold": cross["calibrated_threshold"]["value"],
                           **_binary_summary(cross["transfer_unsw_nb15_test_calibrated"])},
            "unswInDomain": {"threshold": in_domain["threshold"]["value"], **_binary_summary(in_domain["unsw_test"])},
            "featureShift": [
                {"feature": r["feature"], "aucCic": r["auc_cic"], "aucUnsw": r["auc_unsw"]}
                for r in cross["feature_shift"]
            ],
        },
    }


# ---------------------------------------------------------------------------
# threat-intelligence reports and the local LLM (Ollama)
# ---------------------------------------------------------------------------

REPORTS = RESULTS / "threat_intelligence" / "reports"
CFG = load_config(REPO_ROOT / "configs" / "config.yaml")
LLM = CFG.llm


def _ollama() -> OllamaClient:
    return OllamaClient(model=LLM.ollama_model, host=LLM.ollama_host, timeout=float(LLM.timeout_seconds))


@app.get("/api/ollama")
def ollama_status() -> Dict[str, Any]:
    """Is the local Ollama server up, and is the configured model installed?"""
    try:
        installed = _ollama().available_models()
    except OllamaError as exc:
        return {"reachable": False, "model": LLM.ollama_model, "host": LLM.ollama_host, "modelInstalled": False, "message": str(exc)}
    return {"reachable": True, "model": LLM.ollama_model, "host": LLM.ollama_host,
            "modelInstalled": LLM.ollama_model in installed, "message": ""}


def _saved_reports_for(labels: List[str]) -> List[Dict[str, Any]]:
    """Saved threat-intelligence reports (written earlier by the report generator) for the labels that were detected."""
    if not REPORTS.exists():
        return []
    wanted = set(labels)
    return [r for r in (json.loads(p.read_text(encoding="utf-8")) for p in sorted(REPORTS.glob("report_*.json")))
            if r.get("predicted_label") in wanted]


REPORT_SYSTEM = (
    "You write a detection report for a security operations analyst. You are given facts computed from one batch of network "
    "flows: how many were flagged as attacks, which attack types a second model assigned to them, the model's confidence, "
    "and reference text for those attack types. Explain what the facts mean and what the analyst should do next. Be specific: "
    "name the attack types and counts, say how confident the model was, and tie each next step to an attack type or to the "
    "borderline flows. Use only the facts and reference text given. Do not invent IP addresses, ports, timings or specific "
    "flows. The model output has no ground-truth labels, so never say whether predictions are correct or mention accuracy, "
    "precision, recall or false positives. Do not repeat the same number in more than one finding. "
    "Reply with JSON: {\"headline\": one sentence, \"summary\": two or three sentences, "
    "\"findings\": 2 to 3 short strings, \"next_steps\": 2 to 3 short strings}."
)
HIGH_CONFIDENCE = 0.9
BORDERLINE = 0.7


def _report_facts(raw: pd.DataFrame) -> Dict[str, Any]:
    """Computed facts about one batch: the binary detector finds attacks, the multiclass detector names their types."""
    binary = _load_detector("binary", "xgboost")
    multi = _load_detector("multiclass", "xgboost")
    labels = np.asarray(binary.predict(raw))
    attack_idx = list(binary.labels).index(ATTACK)
    attack_conf = binary.predict_proba(raw)[:, attack_idx]
    flagged = labels == ATTACK
    types = np.asarray(multi.predict(raw))

    breakdown = []
    for name in sorted(set(types[flagged]) - {BENIGN}):
        m = flagged & (types == name)
        mean_conf = round(float(attack_conf[m].mean()), 3)
        breakdown.append({"label": str(name), "count": int(m.sum()), "meanConfidence": mean_conf,
                          "risk": risk_for_detection(str(name), mean_conf, CFG.risk)["level"]})
    breakdown.sort(key=lambda d: -d["count"])
    return {
        "rows": int(len(labels)),
        "flaggedAttack": int(flagged.sum()),
        "benign": int((~flagged).sum()),
        "highConfidenceAttack": int((flagged & (attack_conf >= HIGH_CONFIDENCE)).sum()),
        "borderlineAttack": int((flagged & (attack_conf < BORDERLINE)).sum()),
        "meanAttackConfidence": round(float(attack_conf[flagged].mean()), 3) if flagged.any() else None,
        "attackTypes": breakdown,
        "flaggedWithoutType": int((flagged & (types == BENIGN)).sum()),
    }


@app.post("/api/summarize")
async def summarize(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Detection report for one uploaded file. Facts are computed from the file; the local LLM only writes the text."""
    content = await file.read()
    try:
        raw = pd.read_csv(io.BytesIO(content))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read the file as CSV: {exc}")
    raw.columns = raw.columns.str.strip()
    if raw.empty:
        raise HTTPException(status_code=400, detail="The CSV has no rows.")
    try:
        facts = _report_facts(raw)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    references = [
        {"label": r["predicted_label"], "description": r["attack_description"], "impact": r["potential_impact"],
         "mitigations": r["recommended_mitigation"][:3], "techniques": [t["technique_id"] for t in r["mitre_mapping"]]}
        for r in _saved_reports_for([t["label"] for t in facts["attackTypes"]])
    ]
    user = json.dumps({"facts": facts, "reference_descriptions": references}, indent=1)
    try:
        raw_reply = _ollama().chat_json(REPORT_SYSTEM, user)
    except OllamaError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    try:
        parsed = json.loads(raw_reply)
    except json.JSONDecodeError:
        raise HTTPException(status_code=502, detail="The model did not return valid JSON. Try again.")

    # The counts carry no ground truth, so any line that claims correctness is dropped rather than shown.
    unsupported = re.compile(r"actual|accura|false positive|true positive|correct|discrepan|precision|recall", re.IGNORECASE)

    def keep(lines: Any) -> List[str]:
        return [s for s in (str(x) for x in lines or []) if not unsupported.search(s)][:3]

    summary = str(parsed.get("summary", ""))
    return {
        "model": LLM.ollama_model,
        "headline": str(parsed.get("headline", "")),
        "summary": "" if unsupported.search(summary) else summary,
        "findings": keep(parsed.get("findings")),
        "nextSteps": keep(parsed.get("next_steps")),
        "referencesUsed": [r["label"] for r in references],
        "facts": facts,
    }


# ---------------------------------------------------------------------------
# per-flow analysis and single-flow explanations for the dashboard
# ---------------------------------------------------------------------------

MAX_ANALYSIS_ROWS = 20000


async def _read_upload(file: UploadFile) -> pd.DataFrame:
    content = await file.read()
    try:
        raw = pd.read_csv(io.BytesIO(content))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read the file as CSV: {exc}")
    raw.columns = raw.columns.str.strip()
    if raw.empty:
        raise HTTPException(status_code=400, detail="The CSV has no rows.")
    return raw


@lru_cache(maxsize=1)
def _explainer():
    """SHAP explainer for the binary XGBoost detector, built once from the saved background sample."""
    background_path = RESULTS / "explainability" / BACKGROUND_FILE
    if not background_path.exists():
        raise HTTPException(status_code=503, detail="The SHAP background is missing. Run python -m src.explainability.shap_explainer first.")
    det = _load_detector("binary", "xgboost")
    names = list(det.pipeline.feature_names_out_)
    return build_explainer(det.model.model.get_booster(), np.load(background_path)), names


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Every flow in the file: attack or benign, the attack type for flagged flows, and a transparent risk level."""
    raw = await _read_upload(file)
    if len(raw) > MAX_ANALYSIS_ROWS:
        raise HTTPException(status_code=400, detail=f"Upload at most {MAX_ANALYSIS_ROWS} rows at a time.")
    try:
        binary = _load_detector("binary", "xgboost")
        multi = _load_detector("multiclass", "xgboost")
        flagged = np.asarray(binary.predict(raw)) == ATTACK
        attack_conf = binary.predict_proba(raw)[:, list(binary.labels).index(ATTACK)]
        types = np.asarray(multi.predict(raw))
        type_conf = multi.predict_proba(raw).max(axis=1)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    rows: List[Dict[str, Any]] = []
    for i in range(len(raw)):
        if not flagged[i]:
            rows.append({"row": i + 1, "prediction": BENIGN, "attackType": None, "confidence": round(float(attack_conf[i]), 4),
                         "risk": "NONE", "riskScore": 0.0})
            continue
        attack_type = str(types[i]) if types[i] != BENIGN else None
        if attack_type is None:
            # flagged by the binary model, but the multiclass model did not name an attack type: not rated
            rows.append({"row": i + 1, "prediction": ATTACK, "attackType": None, "confidence": round(float(attack_conf[i]), 4),
                         "risk": "UNRATED", "riskScore": None})
            continue
        risk = risk_for_detection(attack_type, float(type_conf[i]), CFG.risk)
        rows.append({"row": i + 1, "prediction": ATTACK, "attackType": attack_type,
                     "confidence": round(float(attack_conf[i]), 4), "risk": risk["level"], "riskScore": risk["score"]})

    levels = pd.Series([r["risk"] for r in rows]).value_counts().to_dict()
    types_count = pd.Series([r["attackType"] for r in rows if r["attackType"]]).value_counts().to_dict()
    return {
        "rowsAnalyzed": len(rows),
        "attackRows": int(flagged.sum()),
        "benignRows": int((~flagged).sum()),
        "riskCounts": {str(k): int(v) for k, v in levels.items()},
        "attackTypeCounts": {str(k): int(v) for k, v in types_count.items()},
        "mitre": {
            name: [{"technique": t, "confidence": c} for t, c, _ in LABEL_TO_TECHNIQUES.get(name, [])]
            for name in types_count
        },
        "rows": rows,
    }


@app.post("/api/explain")
async def explain(file: UploadFile = File(...), row: int = Form(...)) -> Dict[str, Any]:
    """Why one flow (1-based row number in the upload) was or was not flagged: SHAP contributions of the binary model."""
    raw = await _read_upload(file)
    if row < 1 or row > len(raw):
        raise HTTPException(status_code=400, detail=f"Row must be between 1 and {len(raw)}.")
    explainer, names = _explainer()
    det = _load_detector("binary", "xgboost")
    one = raw.iloc[[row - 1]]
    try:
        x = det.pipeline.transform_new_data(one[det.input_columns]).to_numpy(dtype=np.float32)[0]
        p_attack = float(det.predict_proba(one)[0, list(det.labels).index(ATTACK)])
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    out = explain_single(explainer, x, names, top_k=10)
    return {"row": row, "prediction": ATTACK if p_attack >= 0.5 else BENIGN, "attackProbability": round(p_attack, 4), **out}


@app.get("/api/experiments")
def experiments() -> List[Dict[str, Any]]:
    """Every saved experiment, one row per model and split (written by src.evaluation.tracker)."""
    path = METRICS / "experiments.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="experiments.csv is missing. Run python -m src.evaluation.tracker first.")
    df = pd.read_csv(path)
    return df.where(pd.notna(df), None).to_dict(orient="records")

