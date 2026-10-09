"""
End-to-end CIC-IDS2017 pipeline: load -> audit -> leakage guard -> split -> fit preprocessing on
train only -> leakage scan -> sample training rows -> train Random Forest and XGBoost -> evaluate on test.

Run from the repository root:
    python -m src.run_cicids2017 --config configs/config.yaml
    python -m src.run_cicids2017 --tasks binary --max-train-rows 100000
"""

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from src.data.audit import audit_cicids2017, save_audit
from src.data.dataset_adapters import CICIDS2017Adapter
from src.data.eda import run_eda_plots
from src.data.loaders import CICIDS2017Loader
from src.evaluation.detection_metrics import binary_detection_metrics, multiclass_metrics
from src.evaluation.plots import plot_binary_roc, plot_confusion_matrix
from src.models.inference import write_schema
from src.models.supervised import build_classifiers, fit_classifier, model_feature_importances, predict_full_proba
from src.preprocessing.leakage import drop_identifier_columns, single_feature_leakage_scan
from src.preprocessing.pipeline import CyberthreatPreprocessingPipeline
from src.preprocessing.sampling import sample_positions
from src.utils.config import get_absolute_path, load_config
from src.utils.io_utils import save_json
from src.utils.logging import setup_logger
from src.utils.reproducibility import set_seed

logger = setup_logger("run_cicids2017")

REAL_DATA_SOURCE = "real CIC-IDS2017 CSVs"
BENIGN = "BENIGN"


def run_pipeline(
    cfg: Dict[str, Any],
    raw_dir: Optional[str] = None,
    results_dir: Optional[str] = None,
    models_dir: Optional[str] = None,
    tasks: Optional[List[str]] = None,
    sampling_strategy: Optional[str] = None,
    max_train_rows: Optional[int] = None,
    data_source: str = REAL_DATA_SOURCE,
) -> Dict[str, Any]:
    """Runs the full CIC-IDS2017 pipeline and writes metrics, reports, figures and model artifacts.

    Args:
        cfg: Loaded configuration (see configs/config.yaml).
        raw_dir: Directory with the CIC-IDS2017 CSVs (default: config paths.cicids2017_dir).
        results_dir: Root for results/ outputs (default: config paths.results_dir).
        models_dir: Directory for model artifacts (default: config paths.artifacts_dir).
        tasks: Subset of ['binary', 'multiclass'] (default: config training.tasks).
        sampling_strategy: 'none' or 'stratified' (default: config training.sampling.strategy).
        max_train_rows: Training cap for 'stratified' sampling. None = config value; 0 = no cap.
        data_source: Label recorded in every output to distinguish real runs from synthetic test runs.

    Returns:
        The run summary dictionary (also saved to results/reports/cicids2017_run_summary.json).
    """
    started = time.perf_counter()
    seed = int(cfg.system.seed)
    set_seed(seed)

    raw_dir = raw_dir or cfg.paths.cicids2017_dir
    results_root = get_absolute_path(results_dir or cfg.paths.results_dir)
    models_root = get_absolute_path(models_dir or cfg.paths.artifacts_dir)
    reports_dir = results_root / "reports"
    tables_dir = results_root / "tables"
    metrics_dir = results_root / "metrics"
    figures_dir = results_root / "figures"

    tasks = list(tasks or cfg.training.tasks)
    for task in tasks:
        if task not in ("binary", "multiclass"):
            raise ValueError(f"Unknown task '{task}'. Use 'binary' and/or 'multiclass'.")
    strategy = sampling_strategy or cfg.training.sampling.strategy
    if max_train_rows is None:
        cap = cfg.training.sampling.max_train_rows
    else:
        cap = None if max_train_rows == 0 else int(max_train_rows)
    if cfg.data.handle_missing not in ("median", "mean"):
        raise ValueError("Only 'median' or 'mean' imputation is supported by the preprocessing pipeline.")

    # 1. Load every available CSV (strips headers, drops identical duplicate columns, float32 downcast).
    loader = CICIDS2017Loader(raw_dir)
    raw = loader.load_merged_dataset(add_source_column=True)
    source_files = raw.pop("source_file")
    adapter = CICIDS2017Adapter(target_column=cfg.data.target_column_cicids2017)
    X, y_binary, y_multi = adapter.extract_labels(raw)
    del raw

    # 2. Audit the merged data as delivered (before any cleaning) and produce EDA figures.
    audit = audit_cicids2017(X, y_multi, source_files, loader.load_report_)
    audit_paths = save_audit(audit, reports_dir, tables_dir)
    eda_paths = run_eda_plots(audit, figures_dir)
    del source_files

    # 3. Remove identifier-like columns (leakage guard) and exact duplicate rows (label included).
    X, dropped_ids = drop_identifier_columns(X, cfg.data.cicids2017.identifier_patterns)
    removed_duplicates = 0
    if cfg.preprocessing.remove_duplicate_rows:
        keep = ~pd.concat([X, y_multi.rename("__label__")], axis=1).duplicated(keep="first").to_numpy()
        removed_duplicates = int((~keep).sum())
        X = X[keep].reset_index(drop=True)
        y_binary = y_binary[keep].reset_index(drop=True)
        y_multi = y_multi[keep].reset_index(drop=True)
        logger.info(f"Removed {removed_duplicates} exact duplicate rows before splitting.")

    # 4. Stratified split first; cleaning, feature engineering, encoding and scaling are fitted on train only.
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
    del X, y_binary, y_multi
    feature_names = list(splits["X_train"].columns)
    pipeline_path = pipeline.save_pipeline(models_root / "cicids2017_preprocessing_pipeline.joblib")

    # 5. Leakage scan on the TRAINING split only (report; features are not removed automatically).
    leak = single_feature_leakage_scan(
        splits["X_train"],
        splits["y_train_binary"].to_numpy(),
        sample_size=cfg.data.cicids2017.leakage_scan_rows,
        threshold=cfg.data.cicids2017.leakage_auc_threshold,
        seed=seed,
    )
    tables_dir.mkdir(parents=True, exist_ok=True)
    leak_path = tables_dir / "cicids2017_single_feature_leakage_scan.csv"
    leak.to_csv(leak_path, index=False)
    flagged_features = leak.loc[leak["flagged"], "feature"].tolist() if len(leak) else []

    # 6. Class ids: BENIGN is always id 0 so the binary and multiclass tasks share one encoding.
    label_values = sorted(set(audit["class_distribution"]["label"]) - {BENIGN})
    class_names = ([BENIGN] if BENIGN in set(audit["class_distribution"]["label"]) else []) + label_values
    name_to_id = {name: i for i, name in enumerate(class_names)}
    schema_path = write_schema(class_names, pipeline, models_root)

    def to_id(labels: np.ndarray) -> np.ndarray:
        return pd.Series(labels).map(name_to_id).to_numpy(dtype=int)

    # 7. Training subsample (configurable). The same rows are used for every model and task.
    tr_ids_all = to_id(splits["y_train_multi"].to_numpy())
    positions, sample_meta = sample_positions(tr_ids_all, strategy=strategy, max_rows=cap, seed=seed)
    X_train = splits["X_train"].iloc[positions].to_numpy(dtype=np.float32)
    y_train = {
        "binary": splits["y_train_binary"].to_numpy(dtype=int)[positions],
        "multiclass": tr_ids_all[positions],
    }
    X_val = splits["X_val"].to_numpy(dtype=np.float32)
    y_val = {
        "binary": splits["y_val_binary"].to_numpy(dtype=int),
        "multiclass": to_id(splits["y_val_multi"].to_numpy()),
    }
    X_test = splits["X_test"].to_numpy(dtype=np.float32)
    y_test = {
        "binary": splits["y_test_binary"].to_numpy(dtype=int),
        "multiclass": to_id(splits["y_test_multi"].to_numpy()),
    }
    del splits, tr_ids_all

    # 8. Train and evaluate on the held-out test split.
    comparison: List[Dict[str, Any]] = []
    for task in tasks:
        n_classes = 2 if task == "binary" else len(class_names)
        task_labels = ["BENIGN", "ATTACK"] if task == "binary" else class_names
        roc_inputs: Dict[str, Dict[str, np.ndarray]] = {}

        for name, model in build_classifiers(cfg.models, task, seed).items():
            t0 = time.perf_counter()
            fit_classifier(model, X_train, y_train[task], X_val, y_val[task])
            fit_seconds = time.perf_counter() - t0

            model_path = models_root / f"cicids2017_{task}_{name}.joblib"
            model.save_model(model_path)

            y_pred = np.asarray(model.predict(X_test)).astype(int)
            proba = predict_full_proba(model, X_test, n_classes)
            if task == "binary":
                metrics = binary_detection_metrics(y_test[task], y_pred, proba[:, 1])
                roc_inputs[name] = {"y_true": y_test[task], "y_score": proba[:, 1]}
            else:
                metrics = multiclass_metrics(y_test[task], y_pred, proba, class_names)

            metrics.update({
                "model": name,
                "data_source": data_source,
                "seed": seed,
                "n_train_rows": int(len(positions)),
                "n_validation_rows": int(len(y_val[task])),
                "n_test_rows": int(len(y_test[task])),
                "n_features": int(X_train.shape[1]),
                "train_fit_seconds": round(fit_seconds, 2),
                "sampling": sample_meta,
                "top_feature_importances": model_feature_importances(model, tuple(feature_names)),
                "model_artifact": str(model_path),
            })
            if name == "xgboost" and getattr(model.model, "best_iteration", None) is not None:
                metrics["xgboost_best_iteration"] = int(model.model.best_iteration)

            save_json(metrics, metrics_dir / f"cicids2017_{task}_{name}.json")
            plot_confusion_matrix(
                metrics["confusion_matrix"]["matrix"],
                task_labels,
                figures_dir / f"cicids2017_{task}_{name}_confusion_matrix.png",
                title=f"{name} | {task} | test set ({data_source})",
            )

            row = {"task": task, "model": name}
            if task == "binary":
                row.update({k: metrics[k] for k in ("precision", "recall", "f1", "fpr", "roc_auc", "mcc", "accuracy")})
            else:
                row.update({k: metrics[k] for k in ("precision_macro", "recall_macro", "f1_macro", "f1_weighted",
                                                   "roc_auc_ovr_macro", "mcc", "accuracy")})
            comparison.append(row)
            logger.info(f"[{task}/{name}] test metrics: {row}")

        if task == "binary" and roc_inputs:
            plot_binary_roc(roc_inputs, figures_dir / "cicids2017_binary_roc.png",
                            title=f"ROC on test set ({data_source})")

    comparison_df = pd.DataFrame(comparison)
    comparison_path = tables_dir / "cicids2017_model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)

    summary = {
        "data_source": data_source,
        "seed": seed,
        "tasks": tasks,
        "sampling": sample_meta,
        "split_sizes": {
            "train_before_sampling": int(sample_meta["rows_before"]),
            "train_used": int(len(positions)),
            "validation": int(len(y_val["binary"])),
            "test": int(len(y_test["binary"])),
        },
        "duplicate_rows_removed_before_split": removed_duplicates,
        "identifier_columns_dropped": dropped_ids,
        "single_feature_leakage_flagged": flagged_features,
        "class_names": class_names,
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "loader": loader.load_report_,
        "audit_outputs": {k: str(v) for k, v in audit_paths.items()},
        "figures": {k: str(v) for k, v in eda_paths.items()},
        "preprocessing_pipeline": str(pipeline_path),
        "leakage_scan_table": str(leak_path),
        "model_comparison_table": str(comparison_path),
        "model_comparison": comparison,
        "elapsed_seconds": round(time.perf_counter() - started, 1),
    }
    save_json(summary, reports_dir / "cicids2017_run_summary.json")
    logger.info(f"Run complete in {summary['elapsed_seconds']} s. Summary: {reports_dir / 'cicids2017_run_summary.json'}")
    return summary


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run the CIC-IDS2017 detection pipeline.")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to the YAML configuration file.")
    parser.add_argument("--raw-dir", default=None, help="Directory containing the CIC-IDS2017 CSV files.")
    parser.add_argument("--results-dir", default=None, help="Output root for results/ (default from config).")
    parser.add_argument("--models-dir", default=None, help="Directory for model artifacts (default from config).")
    parser.add_argument("--tasks", nargs="+", choices=["binary", "multiclass"], default=None)
    parser.add_argument("--sampling", choices=["none", "stratified"], default=None,
                        help="Training-row sampling strategy (default from config).")
    parser.add_argument("--max-train-rows", type=int, default=None,
                        help="Cap on training rows for stratified sampling; 0 means no cap.")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    run_pipeline(
        cfg,
        raw_dir=args.raw_dir,
        results_dir=args.results_dir,
        models_dir=args.models_dir,
        tasks=args.tasks,
        sampling_strategy=args.sampling,
        max_train_rows=args.max_train_rows,
    )


if __name__ == "__main__":
    main()
