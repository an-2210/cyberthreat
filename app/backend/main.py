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
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.inference import CICIDS2017Detector, MODEL_NAMES, TASKS  # noqa: E402

RESULTS = REPO_ROOT / "results"
RUN_SUMMARY = RESULTS / "reports" / "cicids2017_run_summary.json"
QUALITY_REPORT = RESULTS / "reports" / "cicids2017_data_quality_report.json"
CLASS_DISTRIBUTION = RESULTS / "tables" / "cicids2017_class_distribution.csv"
RARE_CLASS_SUMMARY = RESULTS / "rare_class_experiment" / "comparison_summary.csv"
FIGURES = RESULTS / "figures"
METRICS = RESULTS / "metrics"
ARTIFACTS = REPO_ROOT / "models" / "artifacts"

BENIGN = "BENIGN"
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
