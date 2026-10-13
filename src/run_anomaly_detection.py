"""
Anomaly detection experiment for CIC-IDS2017: an Isolation Forest trained on benign flows only.

It uses the same deduplicated, stratified split as src.run_cicids2017, so its numbers are comparable with the
supervised models. Per attack type, it reports the share of test flows flagged as anomalies. The detector never
saw attack labels, so these rates show how well it generalises to attacks it was not trained on.

Outputs:
    models/artifacts/cicids2017_anomaly_isolation_forest.joblib
    results/metrics/cicids2017_anomaly_isolation_forest.json
    results/tables/cicids2017_anomaly_per_class_detection.csv

Run from the repository root:
    python -m src.run_anomaly_detection
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.anomaly_detection.isolation_forest import IsolationForestDetector
from src.evaluation.detection_metrics import binary_detection_metrics
from src.data.splits import prepare_splits
from src.utils.config import get_absolute_path, load_config
from src.utils.io_utils import save_json
from src.utils.logging import setup_logger
from src.utils.reproducibility import set_seed

logger = setup_logger("run_anomaly_detection")

BENIGN = "BENIGN"
MODEL_ARTIFACT = "models/artifacts/cicids2017_anomaly_isolation_forest.joblib"
METRICS_PATH = "results/metrics/cicids2017_anomaly_isolation_forest.json"
PER_CLASS_PATH = "results/tables/cicids2017_anomaly_per_class_detection.csv"


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Isolation Forest anomaly detection on CIC-IDS2017.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--target-fpr", type=float, default=0.01,
                        help="Share of benign validation flows allowed to be flagged (sets the threshold).")
    parser.add_argument("--n-estimators", type=int, default=200)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    seed = int(cfg.system.seed)
    set_seed(seed)
    started = time.perf_counter()

    prepared = prepare_splits(cfg)["splits"]
    X_train = prepared["X_train"].to_numpy(dtype=np.float32)
    X_val = prepared["X_val"].to_numpy(dtype=np.float32)
    X_test = prepared["X_test"].to_numpy(dtype=np.float32)
    y_train_bin = prepared["y_train_binary"].to_numpy(dtype=int)
    y_val_bin = prepared["y_val_binary"].to_numpy(dtype=int)
    y_test_bin = prepared["y_test_binary"].to_numpy(dtype=int)
    y_test_multi = prepared["y_test_multi"].to_numpy()

    # Training uses benign rows only. Attack labels are never passed to the detector.
    detector = IsolationForestDetector(n_estimators=args.n_estimators, seed=seed)
    t0 = time.perf_counter()
    detector.fit(X_train[y_train_bin == 0])
    fit_seconds = time.perf_counter() - t0

    # The threshold is set on benign validation flows only, so the test set is not used for tuning.
    detector.calibrate(X_val[y_val_bin == 0], target_fpr=args.target_fpr)

    scores = detector.score(X_test)
    flagged = (scores > detector.threshold).astype(int)
    metrics = binary_detection_metrics(y_test_bin, flagged, scores)
    metrics.update({
        "model": "isolation_forest",
        "detector_type": "unsupervised (trained on BENIGN only)",
        "threshold": detector.threshold,
        "calibration_target_fpr": args.target_fpr,
        "seed": seed,
        "n_train_benign_rows": int((y_train_bin == 0).sum()),
        "n_test_rows": int(len(y_test_bin)),
        "n_features": int(X_test.shape[1]),
        "train_fit_seconds": round(fit_seconds, 2),
        "model_artifact": MODEL_ARTIFACT,
    })

    # Share of each attack type flagged as anomalous. Benign shows the false-positive rate.
    rows = []
    for label in pd.unique(y_test_multi):
        mask = y_test_multi == label
        rows.append({
            "class": str(label),
            "test_rows": int(mask.sum()),
            "flagged_share": round(float(flagged[mask].mean()), 6),
            "is_attack": label != BENIGN,
        })
    per_class = pd.DataFrame(rows).sort_values(["is_attack", "test_rows"], ascending=[True, False])

    detector.save(MODEL_ARTIFACT)
    save_json(metrics, get_absolute_path(METRICS_PATH))
    get_absolute_path(PER_CLASS_PATH).parent.mkdir(parents=True, exist_ok=True)
    per_class.to_csv(get_absolute_path(PER_CLASS_PATH), index=False)

    logger.info(
        f"Anomaly detection test results: recall={metrics['recall']:.4f} precision={metrics['precision']:.4f} "
        f"f1={metrics['f1']:.4f} fpr={metrics['fpr']:.4f} roc_auc={metrics['roc_auc']:.4f}"
    )
    logger.info(f"Done in {time.perf_counter() - started:.1f}s. Metrics: {METRICS_PATH}")


if __name__ == "__main__":
    main()
