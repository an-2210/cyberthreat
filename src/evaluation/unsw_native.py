"""
UNSW-NB15 native baseline: XGBoost trained and tested inside UNSW-NB15 on all of its features.

This is separate from the CIC-to-UNSW transfer in cross_dataset.py. It asks a different question: how well
UNSW-NB15 is detected when its own feature set is used. Nothing here is tuned on the UNSW test file.

Preprocessing is fitted on the UNSW training file only:
  - srcip and dstip are dropped (no IP addresses as features, the same policy as the CIC pipeline)
  - sport and dsport are kept; hexadecimal strings such as 0x0303 are parsed to integers
  - proto, service and state become integer codes learned from the training file; unseen values become -1
  - other columns are coerced to numeric, infinities become NaN, values are clipped to the training
    0.1% / 99.9% quantiles and missing values are filled with training medians
  - the training file is split 90/10 (stratified); early stopping and the threshold use the validation slice only
  - threshold: 1% benign false-positive rate on validation rows

Run from the repository root:
    python -m src.evaluation.unsw_native
"""

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.anomaly_detection.common import threshold_for_fpr
from src.data.dataset_adapters import UNSWNB15Adapter
from src.data.loaders import UNSWNB15Loader
from src.evaluation.cross_dataset import evaluate_transfer, fit_shared_model
from src.evaluation.detection_metrics import binary_detection_metrics
from src.preprocessing.leakage import single_feature_leakage_scan
from src.utils.config import get_absolute_path, load_config
from src.utils.io_utils import save_json
from src.utils.logging import setup_logger

logger = setup_logger("unsw_native")

IDENTIFIER_COLUMNS = ("srcip", "dstip")
PORT_COLUMNS = ("sport", "dsport")
CATEGORICAL_COLUMNS = ("proto", "service", "state")


def _parse_port(value: Any) -> float:
    """Decimal ports pass through; hexadecimal strings such as '0x0303' are converted; anything else is NaN."""
    if isinstance(value, str) and value.strip().lower().startswith("0x"):
        try:
            return float(int(value.strip(), 16))
        except ValueError:
            return np.nan
    return pd.to_numeric(value, errors="coerce")


def numeric_base(X: pd.DataFrame) -> pd.DataFrame:
    """Drops identifiers, parses ports and coerces every non-categorical column to numeric."""
    X = X.drop(columns=[c for c in IDENTIFIER_COLUMNS if c in X.columns]).copy()
    for c in PORT_COLUMNS:
        if c in X.columns:
            X[c] = X[c].map(_parse_port).astype(float)
    for c in X.columns:
        if c not in CATEGORICAL_COLUMNS:
            X[c] = pd.to_numeric(X[c], errors="coerce").replace([np.inf, -np.inf], np.nan).astype(float)
    return X


def fit_category_codes(X_train: pd.DataFrame) -> Dict[str, list]:
    """Category values seen in the training file, in sorted order (their position is the integer code)."""
    return {c: sorted(X_train[c].dropna().astype(str).unique()) for c in CATEGORICAL_COLUMNS if c in X_train.columns}


def apply_category_codes(X: pd.DataFrame, codes: Dict[str, list]) -> pd.DataFrame:
    """Replaces categorical text with the training codes. Values not seen in training become -1."""
    out = X.copy()
    for c, values in codes.items():
        mapping = {v: float(i) for i, v in enumerate(values)}
        out[c] = out[c].astype(str).map(mapping).fillna(-1.0).astype(float)
    return out


def run(cfg=None) -> Dict[str, Any]:
    cfg = cfg or load_config("configs/config.yaml")
    seed = int(cfg.system.seed)
    loader = UNSWNB15Loader(get_absolute_path(cfg.paths.unsw_nb15_dir))

    Xtr_raw, ytr, _ = UNSWNB15Adapter().extract_labels(loader.load_dataset("train"))
    Xte_raw, yte, _ = UNSWNB15Adapter().extract_labels(loader.load_dataset("test"))
    Xtr_base, Xte_base = numeric_base(Xtr_raw), numeric_base(Xte_raw)
    codes = fit_category_codes(Xtr_base)
    X = apply_category_codes(Xtr_base, codes)
    Xte = apply_category_codes(Xte_base, codes)
    y = ytr.to_numpy()
    yt = yte.to_numpy()

    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.1, stratify=y, random_state=seed)

    # Leakage scan on the training split only, after imputation with training medians (the scan needs no NaN).
    medians = X_tr.median()
    scan = single_feature_leakage_scan(X_tr.fillna(medians).fillna(0.0), y_tr,
                                       sample_size=int(cfg.data.cicids2017.leakage_scan_rows),
                                       threshold=float(cfg.data.cicids2017.leakage_auc_threshold), seed=seed)

    model, _ = fit_shared_model(X_tr, y_tr, X_va, y_va, seed, dict(cfg.models.advanced.xgboost))
    val_proba = model.predict_proba(model._cross_prep(X_va))[:, 1]
    thr = float(threshold_for_fpr(val_proba[y_va == 0], 0.01))
    test_default = evaluate_transfer(model, Xte, yt, threshold=0.5)
    test = evaluate_transfer(model, Xte, yt, threshold=thr)

    importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    out = {
        "data_source": "real UNSW-NB15 (training file for fitting and validation, official test file for scoring)",
        "model": "xgboost (binary), all UNSW-NB15 features except identifiers, trained and tested in UNSW-NB15",
        "features_used": list(X.columns),
        "dropped_identifiers": [c for c in IDENTIFIER_COLUMNS if c in Xtr_raw.columns],
        "categorical_codes": codes,
        "threshold": {"value": thr, "chosen_on": "UNSW training validation slice, 1% benign FPR"},
        "unsw_test": test,
        "unsw_test_at_threshold_0_5": test_default,
        "top_features_by_importance": [{"feature": f, "importance": round(float(v), 5)}
                                       for f, v in importance.head(10).items()],
        "leakage_scan_top": scan.head(10).to_dict(orient="records"),
        "leakage_scan_flagged": scan.loc[scan["flagged"], "feature"].tolist(),
        "shape": {"train_rows": int(len(X_tr)), "validation_rows": int(len(X_va)), "test_rows": int(len(Xte))},
    }
    save_json(out, get_absolute_path(cfg.paths.results_dir) / "metrics" / "unsw_native_all_features.json")
    logger.info(f"UNSW native test: F1 {test['f1']:.4f}, ROC-AUC {test['roc_auc']:.4f}, threshold {thr:.4g}")
    return out


TTL_COLUMNS = ("sttl", "dttl", "ct_state_ttl")


def per_category_detection(attack_cat: np.ndarray, y_bin: np.ndarray, pred: np.ndarray) -> Dict[str, Dict[str, Any]]:
    """Detection rate per UNSW attack category, over attack rows of the test file only."""
    cats = pd.Series(attack_cat).astype(str).str.strip().to_numpy()
    out = {}
    for cat in sorted(set(cats[y_bin == 1])):
        m = (cats == cat) & (y_bin == 1)
        out[cat] = {"support": int(m.sum()), "detected": int(pred[m].sum()),
                    "recall": round(float(pred[m].mean()), 4)}
    return out


def _fit_variant(X: pd.DataFrame, y: np.ndarray, Xte: pd.DataFrame, yt: np.ndarray, cat_te: np.ndarray,
                 seed: int, params: Dict[str, Any]) -> Dict[str, Any]:
    """Trains on the training slice, picks the 1%-FPR threshold on validation, then scores the test file once."""
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.1, stratify=y, random_state=seed)
    model, _ = fit_shared_model(X_tr, y_tr, X_va, y_va, seed, params)
    val_proba = model.predict_proba(model._cross_prep(X_va))[:, 1]
    thr = float(threshold_for_fpr(val_proba[y_va == 0], 0.01))
    test_proba = model.predict_proba(model._cross_prep(Xte))[:, 1]
    pred = (test_proba >= thr).astype(int)
    metrics = binary_detection_metrics(yt.astype(int), pred, test_proba)
    return {
        "features": list(X.columns),
        "threshold": thr,
        "test": {k: round(float(metrics[k]), 4) for k in ("accuracy", "precision", "recall", "f1", "fpr", "roc_auc")},
        "confusion_matrix": metrics["confusion_matrix"],
        "per_attack_category": per_category_detection(cat_te, yt.astype(int), pred),
        "top_features_by_importance": [
            {"feature": f, "importance": round(float(v), 5)}
            for f, v in pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False).head(5).items()
        ],
    }


def ttl_ablation(cfg=None) -> Dict[str, Any]:
    """Tests whether the result depends on TTL and connection-state-TTL fields.

    Variants (all trained and scored the same way as run()):
      all_features     : the baseline above
      without_ttl      : sttl, dttl and ct_state_ttl removed
      only_ttl         : only sttl, dttl and ct_state_ttl kept
    """
    cfg = cfg or load_config("configs/config.yaml")
    seed = int(cfg.system.seed)
    params = dict(cfg.models.advanced.xgboost)
    loader = UNSWNB15Loader(get_absolute_path(cfg.paths.unsw_nb15_dir))

    Xtr_raw, ytr, _ = UNSWNB15Adapter().extract_labels(loader.load_dataset("train"))
    Xte_raw, yte, cat_te = UNSWNB15Adapter().extract_labels(loader.load_dataset("test"))
    Xtr_base, Xte_base = numeric_base(Xtr_raw), numeric_base(Xte_raw)
    codes = fit_category_codes(Xtr_base)
    X = apply_category_codes(Xtr_base, codes)
    Xte = apply_category_codes(Xte_base, codes)
    y, yt = ytr.to_numpy(), yte.to_numpy()
    cat = cat_te.to_numpy()

    no_ttl = [c for c in X.columns if c not in TTL_COLUMNS]
    variants = {
        "all_features": _fit_variant(X, y, Xte, yt, cat, seed, params),
        "without_ttl": _fit_variant(X[no_ttl], y, Xte[no_ttl], yt, cat, seed, params),
        "only_ttl": _fit_variant(X[list(TTL_COLUMNS)], y, Xte[list(TTL_COLUMNS)], yt, cat, seed, params),
    }
    out = {
        "question": "Does detection depend on the TTL fields (sttl, dttl, ct_state_ttl)?",
        "ttl_columns": list(TTL_COLUMNS),
        "variants": variants,
        "note": "Threshold per variant is chosen on the training validation slice; the test file is scored once per variant.",
    }
    save_json(out, get_absolute_path(cfg.paths.results_dir) / "metrics" / "unsw_native_ttl_ablation.json")
    for name, v in variants.items():
        logger.info(f"{name}: F1 {v['test']['f1']:.4f}, ROC-AUC {v['test']['roc_auc']:.4f}")
    return out


def _clip_impute_fn(X_train: pd.DataFrame):
    """Same transform as fit_shared_model: training 0.1%/99.9% clip and training-median fill."""
    medians = X_train.median()
    lo, hi = X_train.quantile(0.001), X_train.quantile(0.999)
    return lambda X: X.clip(lower=lo, upper=hi, axis=1).fillna(medians).fillna(0.0).to_numpy(dtype=np.float32)


def _category_labels(y_bin: np.ndarray, attack_cat: np.ndarray) -> np.ndarray:
    """'BENIGN' for benign rows, otherwise the stripped UNSW attack category."""
    cats = pd.Series(attack_cat).astype(str).str.strip().to_numpy()
    return np.where(np.asarray(y_bin).astype(int) == 1, cats, "BENIGN")


def _fit_multiclass(X_tr: pd.DataFrame, lab_tr: np.ndarray, X_va: pd.DataFrame, lab_va: np.ndarray,
                    seed: int, params: Dict[str, Any]):
    """XGBoost over BENIGN + attack categories. Early stopping uses validation rows of known categories only."""
    classes = sorted(set(lab_tr))
    index = {c: i for i, c in enumerate(classes)}
    prep = _clip_impute_fn(X_tr)
    known = np.isin(lab_va, classes)
    model_params = {k: v for k, v in params.items()
                    if k not in ("random_state", "n_jobs", "early_stopping_rounds", "eval_metric", "objective")}
    model = XGBClassifier(objective="multi:softprob", num_class=len(classes), random_state=seed, n_jobs=-1,
                          eval_metric="mlogloss", early_stopping_rounds=params.get("early_stopping_rounds", 15),
                          **model_params)
    model.fit(prep(X_tr), np.array([index[c] for c in lab_tr]),
              eval_set=[(prep(X_va[known]), np.array([index[c] for c in lab_va[known]]))], verbose=False)
    benign_col = index["BENIGN"]

    def attack_score(X: pd.DataFrame) -> np.ndarray:
        return 1.0 - model.predict_proba(prep(X))[:, benign_col]

    return attack_score


def _binary_scores_and_metrics(model, X: pd.DataFrame, y: np.ndarray, thr: float, cat: np.ndarray) -> Dict[str, Any]:
    proba = model.predict_proba(model._cross_prep(X))[:, 1]
    pred = (proba >= thr).astype(int)
    m = binary_detection_metrics(y.astype(int), pred, proba)
    return {"threshold": thr, "test": {k: round(float(m[k]), 4) for k in ("accuracy", "precision", "recall", "f1", "fpr", "roc_auc")},
            "confusion_matrix": m["confusion_matrix"], "per_attack_category": per_category_detection(cat, y.astype(int), pred)}


def fuzzers_fix(cfg=None) -> Dict[str, Any]:
    """Compares the binary detector with a multiclass detector (attack score = 1 - P(BENIGN)).

    Selection uses VALIDATION rows only: the variant with the higher mean per-category detection rate
    at the 1%-FPR threshold is chosen. The UNSW test file is then scored once per variant.
    """
    cfg = cfg or load_config("configs/config.yaml")
    seed = int(cfg.system.seed)
    params = dict(cfg.models.advanced.xgboost)
    loader = UNSWNB15Loader(get_absolute_path(cfg.paths.unsw_nb15_dir))

    Xtr_raw, ytr, cat_tr = UNSWNB15Adapter().extract_labels(loader.load_dataset("train"))
    Xte_raw, yte, cat_te = UNSWNB15Adapter().extract_labels(loader.load_dataset("test"))
    Xtr_base, Xte_base = numeric_base(Xtr_raw), numeric_base(Xte_raw)
    codes = fit_category_codes(Xtr_base)
    X = apply_category_codes(Xtr_base, codes)
    Xte = apply_category_codes(Xte_base, codes)
    y = ytr.to_numpy()
    lab = _category_labels(y, cat_tr.to_numpy())
    yt, cat_t = yte.to_numpy(), cat_te.to_numpy()
    lab_t = _category_labels(yt, cat_t)

    X_tr, X_va, y_tr, y_va, lab_tr, lab_va = train_test_split(X, y, lab, test_size=0.1, stratify=y, random_state=seed)

    # Binary detector (the baseline from run()).
    bin_model, _ = fit_shared_model(X_tr, y_tr, X_va, y_va, seed, params)
    bin_va = bin_model.predict_proba(bin_model._cross_prep(X_va))[:, 1]
    bin_thr = float(threshold_for_fpr(bin_va[y_va == 0], 0.01))
    bin_val = {"per_attack_category": per_category_detection(lab_va, y_va, (bin_va >= bin_thr).astype(int))}

    # Multiclass detector.
    mc_score = _fit_multiclass(X_tr, lab_tr, X_va, lab_va, seed, params)
    mc_va = mc_score(X_va)
    mc_thr = float(threshold_for_fpr(mc_va[y_va == 0], 0.01))
    mc_val = {"per_attack_category": per_category_detection(lab_va, y_va, (mc_va >= mc_thr).astype(int))}

    def mean_recall(v: Dict[str, Any]) -> float:
        return float(np.mean([d["recall"] for d in v["per_attack_category"].values()]))

    selection = {
        "binary_xgboost": {"validation_mean_category_recall": round(mean_recall(bin_val), 4)},
        "multiclass_xgboost": {"validation_mean_category_recall": round(mean_recall(mc_val), 4)},
    }
    chosen = max(selection, key=lambda k: selection[k]["validation_mean_category_recall"])

    # Test: each variant scored once, at its validation threshold.
    mc_te = mc_score(Xte)
    pred_mc = (mc_te >= mc_thr).astype(int)
    m = binary_detection_metrics(yt.astype(int), pred_mc, mc_te)
    multi_test = {"threshold": mc_thr,
                  "test": {k: round(float(m[k]), 4) for k in ("accuracy", "precision", "recall", "f1", "fpr", "roc_auc")},
                  "confusion_matrix": m["confusion_matrix"],
                  "per_attack_category": per_category_detection(cat_t, yt.astype(int), pred_mc)}
    binary_test = _binary_scores_and_metrics(bin_model, Xte, yt, bin_thr, cat_t)

    out = {
        "question": "Why is Fuzzers detection low, and does a multiclass detector improve it?",
        "selection_rule": "mean per-category detection rate on validation rows at 1% benign FPR (test file not used)",
        "validation_selection": selection,
        "chosen_on_validation": chosen,
        "variants": {"binary_xgboost": binary_test, "multiclass_xgboost": multi_test},
        "note": "Multiclass attack score = 1 - P(BENIGN). Per-category recall is over attack rows of the test file.",
    }
    save_json(out, get_absolute_path(cfg.paths.results_dir) / "metrics" / "unsw_native_fuzzers_fix.json")
    for name, v in out["variants"].items():
        logger.info(f"{name}: Fuzzers recall {v['per_attack_category']['Fuzzers']['recall']:.4f}, F1 {v['test']['f1']:.4f}")
    return out


def add_derived_features(X: pd.DataFrame) -> pd.DataFrame:
    """Per-flow ratios of existing columns. No timestamps or IP addresses exist in the UNSW training file,
    so windowed per-destination counts cannot be built here; the dataset's ct_* counters already cover that.
    Each ratio uses only its own row, so nothing is fitted and nothing leaks between splits."""
    eps = 1.0
    out = X.copy()
    out["d_loss_rate_src"] = X["sloss"] / (X["spkts"] + eps)
    out["d_loss_rate_dst"] = X["dloss"] / (X["dpkts"] + eps)
    out["d_bytes_per_pkt_src"] = X["sbytes"] / (X["spkts"] + eps)
    out["d_bytes_per_pkt_dst"] = X["dbytes"] / (X["dpkts"] + eps)
    out["d_jitter_ratio_src"] = X["sjit"] / (X["sinpkt"] + eps)
    out["d_jitter_ratio_dst"] = X["djit"] / (X["dinpkt"] + eps)
    out["d_response_share"] = X["response_body_len"] / (X["dbytes"] + eps)
    out["d_handshake_ratio"] = X["synack"] / (X["ackdat"] + eps)
    out["d_src_vs_dst_ct"] = X["ct_src_ltm"] / (X["ct_dst_ltm"] + eps)
    return out


def _run_one_seed(X: pd.DataFrame, y: np.ndarray, lab: np.ndarray, Xte: pd.DataFrame, yt: np.ndarray,
                  cat_t: np.ndarray, seed: int, params: Dict[str, Any]) -> Dict[str, Any]:
    X_tr, X_va, y_tr, y_va, _, lab_va = train_test_split(X, y, lab, test_size=0.1, stratify=y, random_state=seed)
    model, _ = fit_shared_model(X_tr, y_tr, X_va, y_va, seed, params)
    p_va = model.predict_proba(model._cross_prep(X_va))[:, 1]
    thr = float(threshold_for_fpr(p_va[y_va == 0], 0.01))
    val_pred = (p_va >= thr).astype(int)
    val = binary_detection_metrics(y_va.astype(int), val_pred, p_va)
    p_te = model.predict_proba(model._cross_prep(Xte))[:, 1]
    te_pred = (p_te >= thr).astype(int)
    test = binary_detection_metrics(yt.astype(int), te_pred, p_te)
    return {
        "seed": seed,
        "threshold": thr,
        "validation": {"f1": val["f1"], "fuzzers_recall": per_category_detection(lab_va, y_va, val_pred)["Fuzzers"]["recall"]},
        "test": {k: round(float(test[k]), 4) for k in ("precision", "recall", "f1", "fpr", "roc_auc")},
        "test_fuzzers_recall": per_category_detection(cat_t, yt.astype(int), te_pred)["Fuzzers"]["recall"],
    }


def derived_feature_check(cfg=None) -> Dict[str, Any]:
    """Binary XGBoost with and without the derived ratio features, over three seeds.

    Rule fixed before running: choose the variant with the higher mean VALIDATION Fuzzers recall, provided its
    mean validation F1 is not more than 0.01 below the other variant. The test file is only reported.
    """
    cfg = cfg or load_config("configs/config.yaml")
    base_seed = int(cfg.system.seed)
    params = dict(cfg.models.advanced.xgboost)
    loader = UNSWNB15Loader(get_absolute_path(cfg.paths.unsw_nb15_dir))

    Xtr_raw, ytr, cat_tr = UNSWNB15Adapter().extract_labels(loader.load_dataset("train"))
    Xte_raw, yte, cat_te = UNSWNB15Adapter().extract_labels(loader.load_dataset("test"))
    codes = fit_category_codes(numeric_base(Xtr_raw))
    X = apply_category_codes(numeric_base(Xtr_raw), codes)
    Xte = apply_category_codes(numeric_base(Xte_raw), codes)
    y = ytr.to_numpy()
    lab = _category_labels(y, cat_tr.to_numpy())
    yt, cat_t = yte.to_numpy(), cat_te.to_numpy()

    seeds = [base_seed, base_seed + 1, base_seed + 2]
    feature_sets = {"baseline": (X, Xte), "with_derived": (add_derived_features(X), add_derived_features(Xte))}
    runs = {name: [_run_one_seed(Xa, y, lab, Xb, yt, cat_t, s, params) for s in seeds]
            for name, (Xa, Xb) in feature_sets.items()}

    def summary(rs: list) -> Dict[str, Any]:
        def ms(vals):
            return {"mean": round(float(np.mean(vals)), 4), "std": round(float(np.std(vals, ddof=1)), 4)}
        return {"validation_fuzzers_recall": ms([r["validation"]["fuzzers_recall"] for r in rs]),
                "validation_f1": ms([r["validation"]["f1"] for r in rs]),
                "test_f1": ms([r["test"]["f1"] for r in rs]),
                "test_roc_auc": ms([r["test"]["roc_auc"] for r in rs]),
                "test_fuzzers_recall": ms([r["test_fuzzers_recall"] for r in rs])}

    summaries = {name: summary(rs) for name, rs in runs.items()}
    b, d = summaries["baseline"], summaries["with_derived"]
    if d["validation_f1"]["mean"] >= b["validation_f1"]["mean"] - 0.01 and \
            d["validation_fuzzers_recall"]["mean"] > b["validation_fuzzers_recall"]["mean"]:
        chosen = "with_derived"
    else:
        chosen = "baseline"
    out = {
        "question": "Do per-flow ratio features improve Fuzzers detection?",
        "seeds": seeds,
        "rule": "higher mean validation Fuzzers recall, if mean validation F1 is not more than 0.01 lower",
        "chosen_on_validation": chosen,
        "summary": summaries,
        "runs": runs,
        "derived_features": [c for c in add_derived_features(X.head(1)).columns if c.startswith("d_")],
    }
    save_json(out, get_absolute_path(cfg.paths.results_dir) / "metrics" / "unsw_native_derived_features.json")
    return out


if __name__ == "__main__":
    import sys

    if "--ttl-ablation" in sys.argv:
        r = ttl_ablation()
        for name, v in r["variants"].items():
            print(name, v["test"])
    elif "--derived-features" in sys.argv:
        r = derived_feature_check()
        print("chosen on validation:", r["chosen_on_validation"])
        for name, s in r["summary"].items():
            print(name, {k: v["mean"] for k, v in s.items()})
    elif "--fuzzers-fix" in sys.argv:
        r = fuzzers_fix()
        print("chosen on validation:", r["chosen_on_validation"])
        for name, v in r["variants"].items():
            print(name, v["test"], "Fuzzers recall:", v["per_attack_category"]["Fuzzers"]["recall"])
    else:
        r = run()
        print({k: round(r["unsw_test"][k], 4) for k in ("precision", "recall", "f1", "fpr", "roc_auc")})
