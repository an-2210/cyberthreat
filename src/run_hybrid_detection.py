"""
Hybrid known + unknown threat detection for CIC-IDS2017.

Compares, on the same split as src.run_cicids2017:
  - supervised: the saved binary XGBoost model (knows the attack classes it was trained on)
  - anomaly candidates trained on benign flows only: Isolation Forest on raw features, Isolation Forest on
    signed-log features, and an autoencoder (reconstruction error)
  - fusion: a weighted combination of the supervised probability and the chosen anomaly score

Every choice is made on the validation split: which anomaly candidate to use (by ROC-AUC), the fusion weight, and
all thresholds (set so that 1% of benign validation flows are flagged). The test split is only used for the
final numbers.

Outputs:
    models/artifacts/cicids2017_anomaly_autoencoder.pt
    models/artifacts/cicids2017_anomaly_isolation_forest_log.joblib
    results/metrics/cicids2017_hybrid_detection.json
    results/tables/cicids2017_hybrid_per_class_detection.csv

Run from the repository root (after src.run_cicids2017 has produced the supervised model):
    python -m src.run_hybrid_detection
"""

import argparse
import subprocess
import sys
import time
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.anomaly_detection.common import signed_log1p, threshold_for_fpr
from src.anomaly_detection.isolation_forest import IsolationForestDetector
from src.evaluation.detection_metrics import binary_detection_metrics
from src.fusion.decision_fusion import fuse, rank_normalizer, select_fusion
from src.data.splits import prepare_splits
from src.utils.config import get_absolute_path, load_config
from src.utils.io_utils import save_json
from src.utils.logging import setup_logger
from src.utils.reproducibility import set_seed

logger = setup_logger("run_hybrid_detection")

BENIGN = "BENIGN"
ARTIFACTS = "models/artifacts"
SUPERVISED_ARTIFACT = f"{ARTIFACTS}/cicids2017_binary_xgboost.joblib"
SUPERVISED_CACHE = "results/hybrid_cache/xgboost_binary_probabilities.npz"
AE_ARTIFACT = f"{ARTIFACTS}/cicids2017_anomaly_autoencoder.pt"
IF_LOG_ARTIFACT = f"{ARTIFACTS}/cicids2017_anomaly_isolation_forest_log.joblib"
METRICS_PATH = "results/metrics/cicids2017_hybrid_detection.json"
PER_CLASS_PATH = "results/tables/cicids2017_hybrid_per_class_detection.csv"
ALPHA_GRID = np.round(np.linspace(0.0, 1.0, 21), 2)

# Why there are two processes: PyTorch and XGBoost each ship their own OpenMP runtime. Running XGBoost predictions in
# the same process as PyTorch crashes (segfault). So XGBoost scores are computed in a child process and cached to a
# file, and everything that uses torch runs in this process.


def report(name: str, y_true: np.ndarray, score: np.ndarray, threshold: float) -> Dict[str, Any]:
    """Binary metrics for one scoring rule at a fixed threshold, plus the flagged vector for per-class reporting."""
    flagged = (score > threshold).astype(int)
    metrics = binary_detection_metrics(y_true, flagged, score)
    metrics["detector"] = name
    metrics["threshold"] = float(threshold)
    metrics["_flagged"] = flagged
    return metrics


def supervised_probabilities(config: str) -> Dict[str, np.ndarray]:
    """Runs the XGBoost predictions in a child process (see the note above) and loads the cached result."""
    subprocess.run([sys.executable, "-m", "src.run_hybrid_detection", "--xgboost-child", "--config", config],
                   check=True)
    cache = np.load(get_absolute_path(SUPERVISED_CACHE))
    return {"val": cache["val"], "test": cache["test"]}


def xgboost_child(config: str) -> None:
    """Child-process entry point: computes XGBoost attack probabilities for validation and test rows."""
    import joblib
    from src.models.supervised import predict_full_proba

    splits = prepare_splits(load_config(config))["splits"]
    xgb = joblib.load(get_absolute_path(SUPERVISED_ARTIFACT))
    out = get_absolute_path(SUPERVISED_CACHE)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **{
        name: predict_full_proba(xgb, splits[f"X_{name}"].to_numpy(dtype=np.float32), 2)[:, 1]
        for name in ("val", "test")
    })
    logger.info(f"Cached XGBoost probabilities to {out}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Hybrid supervised + anomaly detection on CIC-IDS2017.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--target-fpr", type=float, default=0.01)
    parser.add_argument("--xgboost-child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.xgboost_child:
        xgboost_child(args.config)
        return

    cfg = load_config(args.config)
    seed = int(cfg.system.seed)
    set_seed(seed)
    started = time.perf_counter()

    splits = prepare_splits(cfg)["splits"]
    X = {s: splits[f"X_{s}"].to_numpy(dtype=np.float32) for s in ("train", "val", "test")}
    y = {s: splits[f"y_{s}_binary"].to_numpy(dtype=int) for s in ("train", "val", "test")}
    y_test_multi = splits["y_test_multi"].to_numpy()
    benign_train = X["train"][y["train"] == 0]
    logger.info(f"Benign training rows: {len(benign_train)}")

    # 1. Supervised probability of ATTACK from the saved binary XGBoost model (computed in a child process).
    p = supervised_probabilities(args.config)

    # 2. Anomaly candidates, all fitted on benign training rows only.
    candidates: Dict[str, Dict[str, np.ndarray]] = {}
    fit_seconds: Dict[str, float] = {}

    t0 = time.perf_counter()
    if_raw = IsolationForestDetector(seed=seed).fit(benign_train)
    fit_seconds["isolation_forest_raw"] = round(time.perf_counter() - t0, 2)
    candidates["isolation_forest_raw"] = {s: if_raw.score(X[s]) for s in ("val", "test")}

    t0 = time.perf_counter()
    if_log = IsolationForestDetector(seed=seed).fit(signed_log1p(benign_train))
    fit_seconds["isolation_forest_log"] = round(time.perf_counter() - t0, 2)
    candidates["isolation_forest_log"] = {s: if_log.score(signed_log1p(X[s])) for s in ("val", "test")}
    if_log.calibrate(signed_log1p(X["val"][y["val"] == 0]), target_fpr=args.target_fpr)
    if_log.save(IF_LOG_ARTIFACT)

    from src.anomaly_detection.autoencoder import AutoencoderDetector  # imported here: torch must not load in the XGBoost child

    t0 = time.perf_counter()
    ae = AutoencoderDetector(seed=seed).fit(benign_train)
    fit_seconds["autoencoder"] = round(time.perf_counter() - t0, 2)
    candidates["autoencoder"] = {s: ae.score(X[s]) for s in ("val", "test")}
    ae.save(AE_ARTIFACT)

    # 3. Pick the anomaly candidate on validation data only (ROC-AUC for attack vs benign).
    candidate_auc = {name: float(roc_auc_score(y["val"], c["val"])) for name, c in candidates.items()}
    chosen = max(candidate_auc, key=candidate_auc.get)
    logger.info(f"Validation ROC-AUC per anomaly candidate: {candidate_auc}. Chosen: {chosen}")
    a = candidates[chosen]

    # 4. Standalone results on the test split. Every threshold is set on benign validation flows.
    tests: Dict[str, Dict[str, Any]] = {}
    sup_thr = threshold_for_fpr(p["val"][y["val"] == 0], args.target_fpr)
    tests["supervised_xgboost"] = report("supervised_xgboost", y["test"], p["test"], sup_thr)
    for name, c in candidates.items():
        thr = threshold_for_fpr(c["val"][y["val"] == 0], args.target_fpr)
        tests[name] = report(name, y["test"], c["test"], thr)

    # 5. Fusion of the supervised probability with the chosen anomaly score.
    normalize = rank_normalizer(a["val"][y["val"] == 0])
    a_norm = {s: normalize(a[s]) for s in ("val", "test")}
    choice = select_fusion(p["val"], a_norm["val"], y["val"], ALPHA_GRID, args.target_fpr)
    fused_test = fuse(p["test"], a_norm["test"], choice["alpha"])
    tests["hybrid_fusion"] = report("hybrid_fusion", y["test"], fused_test, choice["threshold"])
    logger.info(f"Fusion selected alpha={choice['alpha']} (validation F1 {choice['val_f1']:.4f}).")

    # 6. Share of each test class flagged by each rule (benign row = false-positive rate).
    per_class = []
    for label in pd.unique(y_test_multi):
        mask = y_test_multi == label
        row = {"class": str(label), "test_rows": int(mask.sum()), "is_attack": label != BENIGN}
        for name in ("supervised_xgboost", chosen, "hybrid_fusion"):
            row[name] = round(float(tests[name]["_flagged"][mask].mean()), 6)
        per_class.append(row)
    per_class_df = pd.DataFrame(per_class).sort_values(["is_attack", "test_rows"], ascending=[True, False])

    for m in tests.values():
        m.pop("_flagged", None)
    summary = {
        "target_fpr": args.target_fpr,
        "seed": seed,
        "anomaly_candidate_validation_roc_auc": candidate_auc,
        "anomaly_candidate_chosen": chosen,
        "fusion": {**choice, "alpha_grid": ALPHA_GRID.tolist(), "anomaly_source": chosen},
        "fit_seconds": fit_seconds,
        "autoencoder_loss_history": ae.loss_history_,
        "test_results": tests,
        "per_class_flagged_share": "results/tables/cicids2017_hybrid_per_class_detection.csv",
        "artifacts": {"autoencoder": AE_ARTIFACT, "isolation_forest_log": IF_LOG_ARTIFACT, "supervised": SUPERVISED_ARTIFACT},
        "elapsed_seconds": round(time.perf_counter() - started, 1),
    }
    save_json(summary, get_absolute_path(METRICS_PATH))
    get_absolute_path(PER_CLASS_PATH).parent.mkdir(parents=True, exist_ok=True)
    per_class_df.to_csv(get_absolute_path(PER_CLASS_PATH), index=False)

    for name, m in tests.items():
        logger.info(f"[test] {name}: recall={m['recall']:.4f} precision={m['precision']:.4f} "
                    f"f1={m['f1']:.4f} fpr={m['fpr']:.4f} roc_auc={m['roc_auc']:.4f}")
    logger.info(f"Done in {summary['elapsed_seconds']} s. Metrics: {METRICS_PATH}")


if __name__ == "__main__":
    main()
