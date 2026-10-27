"""
Per-request detection latency for the CIC-IDS2017 detectors, measured on real rows.

Two processes, because XGBoost and PyTorch crash when they share a process (see src/run_hybrid_detection.py):

  stage "xgboost"  (no torch)  : raw row -> preprocessing -> XGBoost probability, one request at a time.
                                 Also caches the transformed test split, the XGBoost test probabilities and
                                 the benign validation rows that the fusion normalizer needs.
  stage "anomaly"  (torch)     : rebuilds the autoencoder and Isolation Forest from saved artifacts, recomputes the
                                 fusion reference, checks that the hybrid test metrics are reproduced, and times
                                 autoencoder scoring and fusion one request at a time.
  stage "summary"              : combines the cached timings into one table.

Requests are real CIC-IDS2017 rows (a random sample of one raw CSV), processed one at a time after a warm-up.
Timings are wall-clock, single-threaded per request. Hardware is recorded in the output.

Run from the repository root, in this order:
    python -m src.evaluation.latency xgboost
    python -m src.evaluation.latency anomaly
    python -m src.evaluation.latency summary
"""

import argparse
import json
import os
import platform
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from src.utils.config import get_absolute_path, load_config
from src.utils.io_utils import save_json
from src.utils.logging import setup_logger

logger = setup_logger("latency")

CACHE = Path("data/interim/latency_cache")
N_REQUESTS = 500
WARMUP = 20
SEED = 42
SAMPLE_CSV = "Wednesday-workingHours.pcap_ISCX.csv"


def _hardware() -> Dict[str, Any]:
    return {"machine": platform.machine(), "system": platform.platform(), "python": platform.python_version(),
            "cpu_count": os.cpu_count()}


def _percentiles(ms: np.ndarray) -> Dict[str, float]:
    return {"p50_ms": round(float(np.percentile(ms, 50)), 4), "p95_ms": round(float(np.percentile(ms, 95)), 4),
            "p99_ms": round(float(np.percentile(ms, 99)), 4), "mean_ms": round(float(ms.mean()), 4), "n": int(len(ms))}


def stage_xgboost(cfg) -> None:
    from src.data.dataset_adapters import CICIDS2017Adapter
    from src.data.loaders import read_cicids2017_csv
    from src.data.splits import prepare_splits
    from src.models.inference import CICIDS2017Detector

    cache = get_absolute_path(CACHE)
    cache.mkdir(parents=True, exist_ok=True)
    det = CICIDS2017Detector.load("binary", "xgboost", artifacts_dir=get_absolute_path(cfg.paths.artifacts_dir))

    # Real raw requests: a random sample of one real CSV, with the label dropped.
    raw_csv = get_absolute_path(cfg.paths.cicids2017_dir) / "MachineLearningCVE" / SAMPLE_CSV
    raw = read_cicids2017_csv(raw_csv)
    X_raw, _, _ = CICIDS2017Adapter(cfg.data.target_column_cicids2017).extract_labels(raw)
    sample = X_raw.sample(N_REQUESTS + WARMUP, random_state=SEED).reset_index(drop=True)
    rows = [sample.iloc[[i]] for i in range(len(sample))]  # one-row DataFrames, built before timing
    del raw, X_raw

    for r in rows[:WARMUP]:
        det.predict_proba(r)

    e2e, pipe_ms, xgb_ms = [], [], []
    transformed = []
    p_sup = []
    for r in rows[WARMUP:]:
        t0 = time.perf_counter()
        p = det.predict_proba(r)[0, 1]
        e2e.append((time.perf_counter() - t0) * 1e3)
        p_sup.append(p)

        t0 = time.perf_counter()
        xt = det.pipeline.transform_new_data(r[det.input_columns]).to_numpy(dtype=np.float32)
        pipe_ms.append((time.perf_counter() - t0) * 1e3)
        transformed.append(xt[0])

        t0 = time.perf_counter()
        det.model.predict_proba(xt)
        xgb_ms.append((time.perf_counter() - t0) * 1e3)

    # Verification material: the full transformed test split and benign validation rows (same pipeline refit).
    splits = prepare_splits(cfg)["splits"]
    X_test = splits["X_test"].to_numpy(dtype=np.float32)
    p_test = det.model.predict_proba(X_test)[:, 1]
    np.savez_compressed(
        cache / "xgboost_stage.npz",
        X_sample=np.stack(transformed), p_sample=np.asarray(p_sup),
        X_test=X_test, p_test=p_test, y_test=splits["y_test_binary"].to_numpy(dtype=int),
        X_val_benign=splits["X_val"].to_numpy(dtype=np.float32)[splits["y_val_binary"].to_numpy(dtype=int) == 0],
    )
    save_json({
        "hardware": _hardware(),
        "n_requests": N_REQUESTS, "warmup": WARMUP, "seed": SEED,
        "raw_sample_source": SAMPLE_CSV,
        "pipeline_transform": _percentiles(np.asarray(pipe_ms)),
        "xgboost_predict_transformed": _percentiles(np.asarray(xgb_ms)),
        "xgboost_end_to_end_raw": _percentiles(np.asarray(e2e)),
        "e2e_ms_per_request": e2e,
    }, cache / "xgboost_timings.json")
    logger.info(f"XGBoost stage done: end-to-end p50 {np.percentile(e2e, 50):.3f} ms")


def stage_anomaly(cfg) -> None:
    import torch
    from src.anomaly_detection.autoencoder import AutoencoderDetector
    from src.anomaly_detection.common import signed_log1p
    from src.anomaly_detection.isolation_forest import IsolationForestDetector
    from src.evaluation.detection_metrics import binary_detection_metrics
    from src.fusion.decision_fusion import fuse, rank_normalizer

    cache = get_absolute_path(CACHE)
    stage = np.load(cache / "xgboost_stage.npz")
    timings = json.loads((cache / "xgboost_timings.json").read_text())
    artifacts = get_absolute_path(cfg.paths.artifacts_dir)
    hybrid = json.loads(get_absolute_path(cfg.paths.results_dir).joinpath("metrics", "cicids2017_hybrid_detection.json").read_text())
    alpha, threshold = hybrid["fusion"]["alpha"], hybrid["fusion"]["threshold"]

    ck = torch.load(artifacts / "cicids2017_anomaly_autoencoder.pt", map_location="cpu", weights_only=False)
    ae = AutoencoderDetector(hidden_dims=tuple(ck["hidden_dims"]))
    ae.network = ae._build(int(ck["n_features"]))
    ae.network.load_state_dict(ck["state_dict"])
    ae.network.eval()
    ae.mean_, ae.std_ = np.asarray(ck["mean"]), np.asarray(ck["std"])
    if_log = IsolationForestDetector.load(artifacts / "cicids2017_anomaly_isolation_forest_log.joblib")

    # Fusion reference: autoencoder scores on benign validation rows (not saved by the hybrid run; recomputed).
    normalize = rank_normalizer(ae.score(stage["X_val_benign"]))

    # Verification: reproduce the hybrid test metrics from the cached test split.
    y_test = stage["y_test"]
    fused_test = fuse(stage["p_test"], normalize(ae.score(stage["X_test"])), alpha)
    ours = binary_detection_metrics(y_test, (fused_test > threshold).astype(int), fused_test)
    theirs = hybrid["test_results"]["hybrid_fusion"]
    checks = {k: {"recomputed": round(ours[k], 6), "saved": round(theirs[k], 6),
                  "abs_diff": float(abs(ours[k] - theirs[k]))} for k in ("precision", "recall", "f1", "fpr", "fnr")}
    logger.info(f"Hybrid reproduction check: {checks}")

    # Per-request timings on the same sampled requests as the XGBoost stage.
    X = stage["X_sample"]
    p = stage["p_sample"]
    ae_ms, fuse_ms, if_ms, hyb_decision = [], [], [], []
    for i in range(min(WARMUP, len(X))):  # warm-up on the first requests; they are timed again below
        ae.score(X[i:i + 1])
        if_log.score(signed_log1p(X[i:i + 1]))
    for i in range(len(X)):
        row = X[i:i + 1]
        t0 = time.perf_counter()
        a = ae.score(row)
        ae_ms.append((time.perf_counter() - t0) * 1e3)

        t0 = time.perf_counter()
        decision = int(fuse(p[i:i + 1], normalize(a), alpha)[0] > threshold)
        fuse_ms.append((time.perf_counter() - t0) * 1e3)
        hyb_decision.append(decision)

        t0 = time.perf_counter()
        if_log.score(signed_log1p(row))
        if_ms.append((time.perf_counter() - t0) * 1e3)

    e2e_xgb = np.asarray(timings["e2e_ms_per_request"])  # same N requests, same order as X
    assert len(e2e_xgb) == len(ae_ms), "XGBoost and anomaly stages must time the same requests"
    e2e_hybrid = e2e_xgb + np.asarray(ae_ms) + np.asarray(fuse_ms)

    out = {
        "hardware": _hardware(),
        "torch_threads": int(torch.get_num_threads()),
        "reproduction_check_vs_saved_hybrid_metrics": checks,
        "fusion": {"alpha": alpha, "threshold": threshold},
        "autoencoder_score_single_request": _percentiles(np.asarray(ae_ms)),
        "fusion_single_request": _percentiles(np.asarray(fuse_ms)),
        "isolation_forest_log_score_single_request": _percentiles(np.asarray(if_ms)),
        "hybrid_end_to_end_raw": _percentiles(e2e_hybrid),
        "hybrid_decisions_on_sample": {"flagged": int(sum(hyb_decision)), "requests": len(hyb_decision)},
        "e2e_hybrid_ms_per_request": e2e_hybrid.tolist(),
    }
    save_json(out, cache / "anomaly_timings.json")
    logger.info(f"Anomaly stage done: AE p50 {np.percentile(ae_ms, 50):.3f} ms")


def stage_summary(cfg) -> None:
    cache = get_absolute_path(CACHE)
    x = json.loads((cache / "xgboost_timings.json").read_text())
    a = json.loads((cache / "anomaly_timings.json").read_text())
    rows = [
        ("Preprocessing (pipeline transform, one row)", x["pipeline_transform"]),
        ("XGBoost predict (transformed row)", x["xgboost_predict_transformed"]),
        ("XGBoost end-to-end (raw row to probability)", x["xgboost_end_to_end_raw"]),
        ("Autoencoder score (transformed row)", a["autoencoder_score_single_request"]),
        ("Fusion (normalize + fuse + threshold)", a["fusion_single_request"]),
        ("Isolation Forest (log) score (not in hybrid)", a["isolation_forest_log_score_single_request"]),
        ("Hybrid end-to-end (raw row to decision)", a["hybrid_end_to_end_raw"]),
    ]
    df = pd.DataFrame([{"stage": name, **stats} for name, stats in rows])
    tables = get_absolute_path(cfg.paths.results_dir) / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    df.to_csv(tables / "cicids2017_latency_per_request.csv", index=False)
    summary = {
        "data_source": "real CIC-IDS2017 CSVs",
        "measurement": "one request at a time, after warm-up; stages measured in separate processes and summed per request",
        "hardware": x["hardware"],
        "n_requests": x["n_requests"],
        "warmup": x["warmup"],
        "table": df.to_dict(orient="records"),
        "reproduction_check_vs_saved_hybrid_metrics": a["reproduction_check_vs_saved_hybrid_metrics"],
    }
    save_json(summary, get_absolute_path(cfg.paths.results_dir) / "reports" / "cicids2017_latency_benchmark.json")
    print(df[["stage", "p50_ms", "p95_ms", "p99_ms", "mean_ms"]].to_string(index=False))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Per-request latency benchmark for the CIC-IDS2017 detectors.")
    parser.add_argument("stage", choices=["xgboost", "anomaly", "summary"])
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    {"xgboost": stage_xgboost, "anomaly": stage_anomaly, "summary": stage_summary}[args.stage](cfg)


if __name__ == "__main__":
    main()
