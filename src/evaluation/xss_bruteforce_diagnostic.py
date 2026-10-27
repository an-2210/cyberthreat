"""
Diagnostic: can flow features separate Web Attack - XSS from Web Attack - Brute Force?

Uses only the CIC-IDS2017 file that contains the web attacks (Thursday morning). Compares pairwise
separability (per-feature ROC-AUC, and a cross-validated random forest) for:
    XSS vs Brute Force, XSS vs BENIGN, Brute Force vs BENIGN.
Diagnostic only: it does not change any model or artifact.

Run from the repository root:
    python -m src.evaluation.xss_bruteforce_diagnostic
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from src.data.dataset_adapters import CICIDS2017Adapter
from src.data.loaders import read_cicids2017_csv
from src.preprocessing.leakage import drop_identifier_columns
from src.utils.config import get_absolute_path, load_config
from src.utils.io_utils import save_json
from src.utils.logging import setup_logger

logger = setup_logger("xss_diagnostic")

WEB_FILE = "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"
PAYLOAD_FEATURES = [
    "Total Length of Fwd Packets", "Total Length of Bwd Packets",
    "Fwd Packet Length Max", "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std",
    "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean", "Bwd Packet Length Std",
    "Average Packet Size", "Avg Fwd Segment Size", "Avg Bwd Segment Size", "Packet Length Mean",
    "Flow Bytes/s", "Flow Packets/s",
]
PAIRS = {
    "XSS_vs_BruteForce": ("Web Attack - XSS", "Web Attack - Brute Force"),
    "XSS_vs_BENIGN": ("Web Attack - XSS", "BENIGN"),
    "BruteForce_vs_BENIGN": ("Web Attack - Brute Force", "BENIGN"),
}


def feature_auc(X: pd.DataFrame, y: np.ndarray) -> pd.Series:
    scores = {}
    for col in X.columns:
        v = X[col].to_numpy(dtype=np.float64)
        if np.nanmax(v) == np.nanmin(v):
            continue
        auc = roc_auc_score(y, np.nan_to_num(v, nan=np.nanmedian(v)))
        scores[col] = max(auc, 1 - auc)
    return pd.Series(scores).sort_values(ascending=False)


def cv_random_forest(X: pd.DataFrame, y: np.ndarray, seed: int = 42) -> dict:
    model = RandomForestClassifier(n_estimators=200, min_samples_leaf=2, random_state=seed, n_jobs=-1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    proba = cross_val_predict(model, X.to_numpy(np.float32), y, cv=cv, method="predict_proba")[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y, proba)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "n_rows": int(len(y)),
        "n_positive": int(y.sum()),
    }


def run(cfg=None, out_dir=None) -> dict:
    cfg = cfg or load_config("configs/config.yaml")
    raw_path = Path(get_absolute_path(cfg.paths.cicids2017_dir)) / "MachineLearningCVE" / WEB_FILE
    if not raw_path.exists():
        raw_path = Path(get_absolute_path(cfg.paths.cicids2017_dir)) / WEB_FILE
    df = read_cicids2017_csv(raw_path)
    X, _, y_multi = CICIDS2017Adapter(cfg.data.target_column_cicids2017).extract_labels(df)
    X, _ = drop_identifier_columns(X, cfg.data.cicids2017.identifier_patterns)
    keep = ~pd.concat([X, y_multi.rename("__label__")], axis=1).duplicated(keep="first").to_numpy()
    X, y_multi = X[keep].reset_index(drop=True), y_multi[keep].reset_index(drop=True)
    # Diagnostic-only cleaning: infinities -> NaN, then median imputation (computed on this file; no model is kept).
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0)
    logger.info(f"Web-attack file rows after dedupe: {len(X)}; labels: {y_multi.value_counts().to_dict()}")

    result = {"file": raw_path.name, "rows_after_dedupe": int(len(X)),
              "label_counts": {k: int(v) for k, v in y_multi.value_counts().items()},
              "pairs": {}, "payload_feature_auc": {}}
    for name, (pos, neg) in PAIRS.items():
        mask = y_multi.isin([pos, neg]).to_numpy()
        Xp = X[mask]
        yp = (y_multi[mask] == pos).astype(int).to_numpy()
        aucs = feature_auc(Xp, yp)
        result["pairs"][name] = {
            "positive": pos, "negative": neg,
            "rows": int(mask.sum()),
            "top_single_feature_auc": {k: round(float(v), 4) for k, v in aucs.head(5).items()},
            "features_with_auc_ge_0.9": int((aucs >= 0.9).sum()),
            "random_forest_5fold": cv_random_forest(Xp, yp),
        }
        result["payload_feature_auc"][name] = {f: round(float(aucs[f]), 4) for f in PAYLOAD_FEATURES if f in aucs}
        logger.info(f"{name}: {result['pairs'][name]['random_forest_5fold']}")

    out_dir = get_absolute_path(out_dir or cfg.paths.results_dir) / "diagnostics"
    save_json(result, out_dir / "xss_vs_bruteforce_diagnostic.json")
    return result


if __name__ == "__main__":
    r = run()
    for name, p in r["pairs"].items():
        print(f"{name:22s} rows={p['rows']:5d} RF roc_auc={p['random_forest_5fold']['roc_auc']:.4f} "
              f"bal_acc={p['random_forest_5fold']['balanced_accuracy']:.4f} "
              f"features_auc>=0.9: {p['features_with_auc_ge_0.9']}")
