"""
Cross-dataset generalization: train on CIC-IDS2017, test on UNSW-NB15 (binary BENIGN vs ATTACK).

The two datasets share no feature names, so the test is restricted to a small set of flow features whose
meaning is close in both datasets. Each mapping is an approximation and is listed in SHARED_FEATURES:

    UNSW-NB15 column   ->  CIC-IDS2017 column           conversion
    dur                ->  Flow Duration                 seconds -> microseconds (x 1e6)
    spkts              ->  Total Fwd Packets             none
    dpkts              ->  Total Backward Packets        none
    sbytes             ->  Total Length of Fwd Packets   none
    dbytes             ->  Total Length of Bwd Packets   none
    smean            ->  Fwd Packet Length Mean        none
    dmean            ->  Bwd Packet Length Mean        none
    swin               ->  Init_Win_bytes_forward        none
    dwin               ->  Init_Win_bytes_backward       none

A new XGBoost model is trained on CIC-IDS2017 using only these columns (imputation and clipping use CIC training
rows only). Its scores are NOT comparable with the 77-feature production model.

Needs the official UNSW-NB15 CSVs in data/raw/UNSW-NB15/ (UNSW_NB15_training-set.csv, UNSW_NB15_testing-set.csv).
They are not downloaded automatically.

Run from the repository root:
    python -m src.evaluation.cross_dataset
"""

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from src.evaluation.detection_metrics import binary_detection_metrics
from src.utils.config import get_absolute_path, load_config
from src.utils.io_utils import save_json
from src.utils.logging import setup_logger

logger = setup_logger("cross_dataset")

# UNSW column -> (CIC column, multiplier applied to the UNSW values)
SHARED_FEATURES: Dict[str, Tuple[str, float]] = {
    "dur": ("Flow Duration", 1e6),
    "spkts": ("Total Fwd Packets", 1.0),
    "dpkts": ("Total Backward Packets", 1.0),
    "sbytes": ("Total Length of Fwd Packets", 1.0),
    "dbytes": ("Total Length of Bwd Packets", 1.0),
    "smean": ("Fwd Packet Length Mean", 1.0),
    "dmean": ("Bwd Packet Length Mean", 1.0),
    "swin": ("Init_Win_bytes_forward", 1.0),
    "dwin": ("Init_Win_bytes_backward", 1.0),
}
CIC_COLUMNS = [cic for cic, _ in SHARED_FEATURES.values()]


def unsw_to_shared(df: pd.DataFrame) -> pd.DataFrame:
    """Selects the shared UNSW columns and renames/rescales them to the CIC-IDS2017 names."""
    missing = [c for c in SHARED_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"UNSW-NB15 frame is missing shared feature columns: {missing}")
    out = pd.DataFrame({cic: pd.to_numeric(df[u], errors="coerce").astype(float) * mult
                        for u, (cic, mult) in SHARED_FEATURES.items()})
    return out.replace([np.inf, -np.inf], np.nan)


def cic_shared_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Selects the shared CIC columns (infinities are turned into NaN)."""
    missing = [c for c in CIC_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CIC-IDS2017 frame is missing shared feature columns: {missing}")
    return df[CIC_COLUMNS].astype(float).replace([np.inf, -np.inf], np.nan)


def fit_shared_model(X_train: pd.DataFrame, y_train: np.ndarray, X_val: pd.DataFrame, y_val: np.ndarray,
                     seed: int, params: Dict[str, Any]) -> Tuple[XGBClassifier, pd.Series]:
    """Imputes with CIC training medians (and clips to CIC training 0.1%/99.9% quantiles), then fits XGBoost."""
    medians = X_train.median()
    lo, hi = X_train.quantile(0.001), X_train.quantile(0.999)

    def prep(X: pd.DataFrame) -> np.ndarray:
        return X.clip(lower=lo, upper=hi, axis=1).fillna(medians).fillna(0.0).to_numpy(dtype=np.float32)

    # random_state and n_jobs come from the config too; they are set once here, not passed twice.
    model_params = {k: v for k, v in params.items() if k not in ("random_state", "n_jobs", "early_stopping_rounds")}
    model = XGBClassifier(random_state=seed, n_jobs=-1, eval_metric="logloss",
                          early_stopping_rounds=params.get("early_stopping_rounds", 15), **model_params)
    model.fit(prep(X_train), y_train, eval_set=[(prep(X_val), y_val)], verbose=False)
    model._cross_prep = prep  # kept with the model so the same transform is applied at test time
    return model, medians


def evaluate_transfer(model: XGBClassifier, X: pd.DataFrame, y: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    proba = model.predict_proba(model._cross_prep(X))[:, 1]
    pred = (proba >= threshold).astype(int)
    out = binary_detection_metrics(np.asarray(y).astype(int), pred, proba)
    out["threshold"] = float(threshold)
    return out


def feature_shift(X_cic: pd.DataFrame, y_cic: np.ndarray, X_unsw: pd.DataFrame, y_unsw: np.ndarray,
                  fill: pd.Series) -> pd.DataFrame:
    """Per shared feature: how well it separates attack from benign on each dataset, and the medians.

    A feature that separates well on CIC but not on UNSW explains part of the transfer failure.
    """
    from sklearn.metrics import roc_auc_score

    rows = []
    for col in CIC_COLUMNS:
        row = {"feature": col}
        for name, X, y in (("cic", X_cic, y_cic), ("unsw", X_unsw, y_unsw)):
            v = X[col].fillna(fill[col]).to_numpy(dtype=float)
            y = np.asarray(y).astype(int)
            row[f"auc_{name}"] = round(float(roc_auc_score(y, v)), 4) if len(np.unique(y)) == 2 else None
            row[f"median_benign_{name}"] = float(np.median(v[y == 0]))
            row[f"median_attack_{name}"] = float(np.median(v[y == 1]))
        rows.append(row)
    return pd.DataFrame(rows)


def run(cfg=None) -> Dict[str, Any]:
    from src.anomaly_detection.common import threshold_for_fpr
    from src.data.dataset_adapters import CICIDS2017Adapter, UNSWNB15Adapter
    from src.data.loaders import CICIDS2017Loader, UNSWNB15Loader
    from src.preprocessing.leakage import drop_identifier_columns
    from sklearn.model_selection import train_test_split

    cfg = cfg or load_config("configs/config.yaml")
    seed = int(cfg.system.seed)
    target_fpr = 0.01
    unsw_dir = get_absolute_path(cfg.paths.unsw_nb15_dir)
    unsw_loader = UNSWNB15Loader(unsw_dir)
    if not (unsw_dir / "UNSW_NB15_training-set.csv").exists() or not (unsw_dir / "UNSW_NB15_testing-set.csv").exists():
        raise FileNotFoundError(
            f"UNSW-NB15 CSVs not found in {unsw_dir}. Place UNSW_NB15_training-set.csv and "
            "UNSW_NB15_testing-set.csv there (they are not downloaded automatically)."
        )

    # CIC-IDS2017: dedupe, stratified split, shared columns only.
    raw = CICIDS2017Loader(cfg.paths.cicids2017_dir).load_merged_dataset()
    X, y_bin, _ = CICIDS2017Adapter(cfg.data.target_column_cicids2017).extract_labels(raw)
    del raw
    X, _ = drop_identifier_columns(X, cfg.data.cicids2017.identifier_patterns)
    X = cic_shared_frame(X)
    keep = ~X.duplicated(keep="first").to_numpy()
    X, y = X[keep].reset_index(drop=True), y_bin[keep].reset_index(drop=True).to_numpy()
    X_tmp, X_te, y_tmp, y_te = train_test_split(X, y, test_size=cfg.data.test_size, stratify=y, random_state=seed)
    X_tr, X_va, y_tr, y_va = train_test_split(X_tmp, y_tmp, test_size=cfg.data.val_size / (1 - cfg.data.test_size),
                                              stratify=y_tmp, random_state=seed)

    model, fill = fit_shared_model(X_tr, y_tr, X_va, y_va, seed, dict(cfg.models.advanced.xgboost))
    in_domain = evaluate_transfer(model, X_te, y_te)

    # Threshold chosen on CIC VALIDATION rows only: flags 1% of benign validation flows.
    val_proba = model.predict_proba(model._cross_prep(X_va))[:, 1]
    calibrated_thr = float(threshold_for_fpr(val_proba[y_va == 0], target_fpr))

    # UNSW-NB15: the official test file is the transfer target. Nothing is tuned on it.
    unsw_test = unsw_loader.load_dataset("test")
    Xu, yu_bin, _ = UNSWNB15Adapter().extract_labels(unsw_test)
    Xu_shared = unsw_to_shared(Xu)
    yu = yu_bin.to_numpy()
    transfer_default = evaluate_transfer(model, Xu_shared, yu, threshold=0.5)
    transfer_calibrated = evaluate_transfer(model, Xu_shared, yu, threshold=calibrated_thr)

    shift = feature_shift(X_te, y_te, Xu_shared, yu, fill)

    result = {
        "data_source": "real CIC-IDS2017 (train/validation/test) and UNSW-NB15 (test)",
        "shared_features": {u: {"cic": c, "multiplier": m} for u, (c, m) in SHARED_FEATURES.items()},
        "model": "xgboost (binary) trained on shared features only",
        "in_domain_cic_test": in_domain,
        "transfer_unsw_nb15_test_threshold_0_5": transfer_default,
        "calibrated_threshold": {"value": calibrated_thr, "chosen_on": "CIC-IDS2017 validation, 1% benign FPR",
                                 "tuned_on_unsw": False},
        "transfer_unsw_nb15_test_calibrated": transfer_calibrated,
        "generalization_gap_f1_calibrated": round(in_domain["f1"] - transfer_calibrated["f1"], 4),
        "feature_shift": shift.to_dict(orient="records"),
        "caveat": "Shared-feature approximation: a small subset of flow features, not the full CIC feature set.",
    }
    save_json(result, get_absolute_path(cfg.paths.results_dir) / "metrics" / "cross_dataset_cic_to_unsw.json")
    logger.info(f"Transfer F1 at 0.5: {transfer_default['f1']:.4f}; at calibrated {calibrated_thr:.4g}: {transfer_calibrated['f1']:.4f}")
    return result


if __name__ == "__main__" and "--unsw-in-domain" not in __import__("sys").argv:
    r = run()
    print({k: round(r["transfer_unsw_nb15_test_calibrated"][k], 4) for k in ("precision", "recall", "f1", "fpr", "roc_auc")})


def unsw_in_domain(cfg=None) -> Dict[str, Any]:
    """Diagnostic baseline: the same nine shared features, trained and tested inside UNSW-NB15.

    Trained on the UNSW training file (with a validation slice for the threshold); the UNSW test file is
    scored once. If this is also weak, the shared features lack signal on UNSW, not only under transfer.
    """
    from sklearn.model_selection import train_test_split
    from src.anomaly_detection.common import threshold_for_fpr
    from src.data.dataset_adapters import UNSWNB15Adapter
    from src.data.loaders import UNSWNB15Loader

    cfg = cfg or load_config("configs/config.yaml")
    seed = int(cfg.system.seed)
    loader = UNSWNB15Loader(get_absolute_path(cfg.paths.unsw_nb15_dir))
    Xtr_raw, ytr, _ = UNSWNB15Adapter().extract_labels(loader.load_dataset("train"))
    Xte_raw, yte, _ = UNSWNB15Adapter().extract_labels(loader.load_dataset("test"))
    X = unsw_to_shared(Xtr_raw)
    y = ytr.to_numpy()
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.1, stratify=y, random_state=seed)
    model, fill = fit_shared_model(X_tr, y_tr, X_va, y_va, seed, dict(cfg.models.advanced.xgboost))
    val_proba = model.predict_proba(model._cross_prep(X_va))[:, 1]
    thr = float(threshold_for_fpr(val_proba[y_va == 0], 0.01))
    test = evaluate_transfer(model, unsw_to_shared(Xte_raw), yte.to_numpy(), threshold=thr)
    out = {"model": "xgboost (binary), shared features, trained and tested in UNSW-NB15",
           "threshold": {"value": thr, "chosen_on": "UNSW training validation slice, 1% benign FPR"},
           "unsw_test": test}
    save_json(out, get_absolute_path(cfg.paths.results_dir) / "metrics" / "unsw_in_domain_shared_features.json")
    return out


if __name__ == "__main__" and "--unsw-in-domain" in __import__("sys").argv:
    r = unsw_in_domain()
    print({k: round(r["unsw_test"][k], 4) for k in ("precision", "recall", "f1", "fpr", "roc_auc")})
