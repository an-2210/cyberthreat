"""
Multiclass-only rare-class experiment for CIC-IDS2017.

Reproduces the existing deduplicated stratified split (same seed, same test set), then trains
Random Forest and XGBoost on a per-class capped training set, with and without sqrt-inverse-frequency
class weights. Validation rows are used for XGBoost early stopping only. Hyperparameters are the
existing config values (no tuning). The test set is evaluated once per model.

All outputs go to new directories:
    models/artifacts/rare_class_experiment/<variant>/
    results/rare_class_experiment/

Run from the repository root:
    python -m src.run_rare_class_experiment
"""

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import pandas as pd
from src.data.dataset_adapters import CICIDS2017Adapter
from src.data.loaders import CICIDS2017Loader
from src.evaluation.detection_metrics import multiclass_metrics
from src.models.inference import CICIDS2017Detector, PIPELINE_FILENAME, write_schema
from src.models.supervised import build_classifiers, model_feature_importances, predict_full_proba
from src.preprocessing.leakage import drop_identifier_columns
from src.preprocessing.pipeline import CyberthreatPreprocessingPipeline
from src.preprocessing.sampling import cap_per_class_positions
from src.utils.config import get_absolute_path, load_config
from src.utils.io_utils import save_json
from src.utils.logging import setup_logger
from src.utils.reproducibility import set_seed

logger = setup_logger("rare_class_experiment")

EXPERIMENT_NAME = "rare_class_experiment"
MAX_PER_CLASS = 60_000
VARIANTS = ("capped", "capped_weighted")
MODEL_NAMES = ("random_forest", "xgboost")


def prepare_splits(cfg) -> Dict[str, Any]:
    """Recreates the deduplicated, stratified split used by src.run_cicids2017 (same seed, no sampling)."""
    seed = int(cfg.system.seed)
    raw = CICIDS2017Loader(cfg.paths.cicids2017_dir).load_merged_dataset(add_source_column=True)
    raw.pop("source_file")
    X, y_binary, y_multi = CICIDS2017Adapter(cfg.data.target_column_cicids2017).extract_labels(raw)
    del raw
    X, _ = drop_identifier_columns(X, cfg.data.cicids2017.identifier_patterns)
    keep = ~pd.concat([X, y_multi.rename("__label__")], axis=1).duplicated(keep="first").to_numpy()
    X, y_binary, y_multi = X[keep].reset_index(drop=True), y_binary[keep].reset_index(drop=True), y_multi[keep].reset_index(drop=True)

    pipeline = CyberthreatPreprocessingPipeline(
        impute_strategy=cfg.data.handle_missing,
        remove_constant=cfg.preprocessing.remove_constant_features,
        scaling_method=cfg.data.scaling_method,
        encoding_type=cfg.data.categorical_encoding,
        test_size=cfg.data.test_size,
        val_size=cfg.data.val_size,
        random_state=seed,
    )
    splits = pipeline.fit_transform_splits(X, y_binary, y_multi)
    return {"pipeline": pipeline, "splits": splits}


def class_weights(y: np.ndarray, n_classes: int) -> np.ndarray:
    """Per-row sqrt-inverse-frequency weights, normalized to mean 1 over the training rows."""
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    per_class = np.sqrt(len(y) / (n_classes * np.maximum(counts, 1.0)))
    weights = per_class[y]
    return weights / weights.mean()


def train_variant(variant: str, cfg, X: np.ndarray, y: np.ndarray, Xv: np.ndarray, yv: np.ndarray,
                  n_classes: int, out_dir: Path, seed: int) -> Dict[str, Any]:
    """Trains RF and XGBoost for one variant with the existing model wrappers and config values."""
    sample_weight = class_weights(y, n_classes) if variant == "capped_weighted" else None
    fitted: Dict[str, Any] = {}
    for name, model in build_classifiers(cfg.models, "multiclass", seed).items():
        t0 = time.perf_counter()
        if name == "xgboost":
            known = np.isin(yv, np.unique(y))
            model.model.fit(X, y, sample_weight=sample_weight, eval_set=[(Xv[known], yv[known])], verbose=False)
            model.best_iteration = int(model.model.best_iteration)
        else:
            model.model.fit(X, y, sample_weight=sample_weight)
        model.is_fitted = True
        model.feature_importances_ = model.model.feature_importances_
        fit_seconds = time.perf_counter() - t0
        model.save_model(out_dir / f"cicids2017_multiclass_{name}.joblib")
        fitted[name] = {"model": model, "fit_seconds": round(fit_seconds, 2)}
        logger.info(f"[{variant}/{name}] trained in {fit_seconds:.1f}s")
    return fitted


def main(argv: List[str] = None) -> None:
    parser = argparse.ArgumentParser(description="Multiclass rare-class experiment for CIC-IDS2017.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--max-per-class", type=int, default=MAX_PER_CLASS)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    seed = int(cfg.system.seed)
    set_seed(seed)
    started = time.perf_counter()

    models_root = get_absolute_path(cfg.paths.artifacts_dir)
    exp_models = models_root / EXPERIMENT_NAME
    exp_results = get_absolute_path(cfg.paths.results_dir) / EXPERIMENT_NAME
    exp_results.mkdir(parents=True, exist_ok=True)

    summary = json.loads((get_absolute_path(cfg.paths.results_dir) / "reports" / "cicids2017_run_summary.json").read_text())
    class_names = summary["class_names"]
    name_to_id = {n: i for i, n in enumerate(class_names)}
    K = len(class_names)

    prep = prepare_splits(cfg)
    pipeline, splits = prep["pipeline"], prep["splits"]
    X_tr_all = splits["X_train"].to_numpy(dtype=np.float32)
    y_tr_all = pd.Series(splits["y_train_multi"]).map(name_to_id).to_numpy(dtype=int)
    X_va = splits["X_val"].to_numpy(dtype=np.float32)
    y_va = pd.Series(splits["y_val_multi"]).map(name_to_id).to_numpy(dtype=int)
    X_te = splits["X_test"].to_numpy(dtype=np.float32)
    y_te = pd.Series(splits["y_test_multi"]).map(name_to_id).to_numpy(dtype=int)
    logger.info(f"Split sizes: train={len(y_tr_all)} val={len(y_va)} test={len(y_te)}")

    cap_pos = cap_per_class_positions(y_tr_all, args.max_per_class, seed=seed)
    X_tr, y_tr = X_tr_all[cap_pos], y_tr_all[cap_pos]
    train_counts = {class_names[i]: int(c) for i, c in enumerate(np.bincount(y_tr, minlength=K))}
    logger.info(f"Capped training set: {len(y_tr)} rows (from {len(y_tr_all)}), per-class {train_counts}")

    config_record = {
        "experiment": EXPERIMENT_NAME,
        "max_per_class": args.max_per_class,
        "variants": {
            "capped": "per-class cap, no class weights",
            "capped_weighted": "per-class cap + sqrt-inverse-frequency sample weights (mean 1)",
        },
        "hyperparameters": "unchanged from configs/config.yaml (no tuning)",
        "early_stopping": "XGBoost on validation split only",
        "seed": seed,
        "split": {"train_before_cap": int(len(y_tr_all)), "train_after_cap": int(len(y_tr)),
                  "validation": int(len(y_va)), "test": int(len(y_te))},
        "train_counts_after_cap": train_counts,
        "class_names": class_names,
        "data_source": "real CIC-IDS2017 CSVs",
    }
    save_json(config_record, exp_results / "config.json")

    # Original multiclass metrics, for comparison (read-only).
    originals = {m: json.loads((get_absolute_path(cfg.paths.results_dir) / "metrics" / f"cicids2017_multiclass_{m}.json").read_text())
                 for m in MODEL_NAMES}

    rows: List[Dict[str, Any]] = []
    per_class_rows: List[Dict[str, Any]] = []
    for variant in VARIANTS:
        out_dir = exp_models / variant
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(models_root / PIPELINE_FILENAME, out_dir / PIPELINE_FILENAME)
        write_schema(class_names, pipeline, out_dir)
        fitted = train_variant(variant, cfg, X_tr, y_tr, X_va, y_va, K, out_dir, seed)

        for name in MODEL_NAMES:
            model, fit_s = fitted[name]["model"], fitted[name]["fit_seconds"]
            proba_te = predict_full_proba(model, X_te, K)
            pred_te = np.asarray(model.predict(X_te)).astype(int)
            test = multiclass_metrics(y_te, pred_te, proba_te, class_names)
            test.update({"variant": variant, "model": name, "fit_seconds": fit_s,
                         "n_train_rows": int(len(y_tr)), "data_source": "real CIC-IDS2017 CSVs",
                         "top_feature_importances": model_feature_importances(model, tuple(splits["X_train"].columns))})
            if name == "xgboost":
                test["xgboost_best_iteration"] = model.best_iteration
            val = multiclass_metrics(y_va, np.asarray(model.predict(X_va)).astype(int),
                                     predict_full_proba(model, X_va, K), class_names)
            test["validation_macro_f1"] = val["f1_macro"]
            save_json(test, exp_results / f"test_metrics_{variant}_{name}.json")
            rows.append({"variant": variant, "model": name, "macro_f1": test["f1_macro"],
                         "weighted_f1": test["f1_weighted"], "accuracy": test["accuracy"],
                         "validation_macro_f1": val["f1_macro"], "roc_auc_ovr_macro": test["roc_auc_ovr_macro"]})
            for cname, stats in test["per_class"].items():
                per_class_rows.append({"variant": variant, "model": name, "class": cname, **stats})

            # Compatibility check: the detector reloads this variant and must give the same predictions.
            det = CICIDS2017Detector.load("multiclass", name, artifacts_dir=out_dir)
            assert det.labels == class_names
            reloaded = np.asarray(det.model.predict_proba(X_te[:5000]))
            assert np.allclose(reloaded, proba_te[:5000][:, np.asarray(model.model.classes_).astype(int)]), "reload mismatch"
        logger.info(f"Finished variant '{variant}'.")

    for name in MODEL_NAMES:
        orig = originals[name]
        rows.insert(0, {"variant": "original", "model": name, "macro_f1": orig["f1_macro"],
                        "weighted_f1": orig["f1_weighted"], "accuracy": orig["accuracy"],
                        "validation_macro_f1": None, "roc_auc_ovr_macro": orig["roc_auc_ovr_macro"]})
        for cname, stats in orig["per_class"].items():
            per_class_rows.append({"variant": "original", "model": name, "class": cname, **stats})

    comparison = pd.DataFrame(rows)
    comparison.to_csv(exp_results / "comparison_summary.csv", index=False)
    pd.DataFrame(per_class_rows).to_csv(exp_results / "comparison_per_class.csv", index=False)
    save_json({"runtime_seconds": round(time.perf_counter() - started, 1)}, exp_results / "run_info.json")
    print(comparison.to_string(index=False))
    logger.info(f"Experiment outputs: {exp_models} and {exp_results}")


if __name__ == "__main__":
    main()
