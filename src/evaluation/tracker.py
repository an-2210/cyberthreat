"""
One table of every experiment in results/metrics, written to results/metrics/experiments.csv.

Each row is one model evaluated on one test split. Values come only from the saved JSON files and the saved
latency benchmark. A field that was not measured for a run is left empty, never estimated.

Columns (spec section 23): experiment_id, dataset, model, task, n_features, parameters, training_time_s,
inference_time_ms_per_row, accuracy, precision, recall, macro_f1, weighted_f1, mcc, fpr, fnr, roc_auc, source.

Run from the repository root:
    python -m src.evaluation.tracker
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.utils.config import get_absolute_path, load_config
from src.utils.logging import setup_logger

logger = setup_logger("tracker")

COLUMNS = [
    "experiment_id", "dataset", "model", "task", "n_features", "parameters", "training_time_s",
    "inference_time_ms_per_row", "accuracy", "precision", "recall", "macro_f1", "weighted_f1", "mcc",
    "fpr", "fnr", "roc_auc", "source",
]


def _row(**kwargs: Any) -> Dict[str, Any]:
    base = {c: None for c in COLUMNS}
    base.update(kwargs)
    return base


def _round(v: Optional[float]) -> Optional[float]:
    return None if v is None else round(float(v), 6)


def _from_cic_model(path: Path, dataset: str, cfg) -> Dict[str, Any]:
    """Random Forest and XGBoost files written by src.run_cicids2017 (multiclass files use macro/weighted averages)."""
    m = json.loads(path.read_text(encoding="utf-8"))
    task, model = m["task"], m["model"]
    params = cfg.models.advanced.xgboost if model == "xgboost" else cfg.models.baselines.random_forest
    if task == "binary":
        return _row(
            experiment_id=f"{dataset}-{model}-{task}", dataset=dataset, model=model, task=task,
            n_features=m["n_features"], parameters=json.dumps(dict(params)), training_time_s=_round(m["train_fit_seconds"]),
            accuracy=_round(m["accuracy"]), precision=_round(m["precision"]), recall=_round(m["recall"]),
            mcc=_round(m["mcc"]), fpr=_round(m["fpr"]), fnr=_round(m["fnr"]), roc_auc=_round(m["roc_auc"]),
            source=path.name,
        )
    return _row(
        experiment_id=f"{dataset}-{model}-{task}", dataset=dataset, model=model, task=task,
        n_features=m["n_features"], parameters=json.dumps(dict(params)), training_time_s=_round(m["train_fit_seconds"]),
        accuracy=_round(m["accuracy"]), precision=_round(m["precision_macro"]), recall=_round(m["recall_macro"]),
        macro_f1=_round(m["f1_macro"]), weighted_f1=_round(m["f1_weighted"]), mcc=_round(m["mcc"]),
        roc_auc=_round(m["roc_auc_ovr_macro"]), source=path.name,
    )


def _from_baseline(path: Path, cfg) -> Dict[str, Any]:
    m = json.loads(path.read_text(encoding="utf-8"))
    name, task = m["model"], m["task"]
    per_row_ms = round(1000.0 * m["test_inference_seconds"] / m["n_test_rows"], 6) if m.get("n_test_rows") else None
    params = cfg.models.baselines.logistic_regression if name == "logistic_regression" else {"C": 1.0, "max_iter": 2000, "dual": False}
    if task == "binary":
        return _row(
            experiment_id=f"cicids2017-{name}-{task}", dataset="CIC-IDS2017", model=name, task=task,
            n_features=None, parameters=json.dumps(dict(params)), training_time_s=_round(m["train_fit_seconds"]),
            inference_time_ms_per_row=per_row_ms, accuracy=_round(m["accuracy"]), precision=_round(m["precision"]),
            recall=_round(m["recall"]), mcc=_round(m["mcc"]), fpr=_round(m["fpr"]), fnr=_round(m["fnr"]),
            roc_auc=_round(m.get("roc_auc")), source=path.name,
        )
    return _row(
        experiment_id=f"cicids2017-{name}-{task}", dataset="CIC-IDS2017", model=name, task=task,
        parameters=json.dumps(dict(params)), training_time_s=_round(m["train_fit_seconds"]),
        inference_time_ms_per_row=per_row_ms, accuracy=_round(m["accuracy"]), precision=_round(m["precision_macro"]),
        recall=_round(m["recall_macro"]), macro_f1=_round(m["f1_macro"]), weighted_f1=_round(m["f1_weighted"]),
        mcc=_round(m["mcc"]), roc_auc=_round(m.get("roc_auc_ovr_macro")), source=path.name,
    )


def _from_hybrid(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for key, m in data["test_results"].items():
        rows.append(_row(
            experiment_id=f"cicids2017-{key}-binary", dataset="CIC-IDS2017", model=key, task="binary",
            parameters=json.dumps({"target_fpr": data["target_fpr"]}), accuracy=_round(m.get("accuracy")),
            precision=_round(m["precision"]), recall=_round(m["recall"]), mcc=_round(m.get("mcc")),
            fpr=_round(m["fpr"]), fnr=_round(m["fnr"]), roc_auc=_round(m.get("roc_auc")), source=path.name,
        ))
    return rows


def _from_unsw(path: Path, experiment: str, block: str) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    m = data[block]
    return _row(
        experiment_id=f"unsw-{experiment}", dataset="UNSW-NB15", model=experiment, task="binary",
        parameters=json.dumps({"threshold": data.get("threshold", {}).get("value")}),
        accuracy=_round(m["accuracy"]), precision=_round(m["precision"]), recall=_round(m["recall"]),
        mcc=_round(m.get("mcc")), fpr=_round(m["fpr"]), fnr=_round(m.get("fnr")), roc_auc=_round(m["roc_auc"]),
        source=path.name,
    )


def build_table(cfg) -> pd.DataFrame:
    metrics = get_absolute_path(cfg.paths.results_dir) / "metrics"
    rows: List[Dict[str, Any]] = []
    for task in ("binary", "multiclass"):
        for model in ("random_forest", "xgboost"):
            path = metrics / f"cicids2017_{task}_{model}.json"
            if path.exists():
                rows.append(_from_cic_model(path, "CIC-IDS2017", cfg))
    for path in sorted(metrics.glob("cicids2017_baseline_*.json")):
        rows.append(_from_baseline(path, cfg))
    if (metrics / "cicids2017_hybrid_detection.json").exists():
        rows.extend(_from_hybrid(metrics / "cicids2017_hybrid_detection.json"))
    if (metrics / "unsw_native_all_features.json").exists():
        rows.append(_from_unsw(metrics / "unsw_native_all_features.json", "unsw-native-all-features", "unsw_test"))
    if (metrics / "unsw_in_domain_shared_features.json").exists():
        rows.append(_from_unsw(metrics / "unsw_in_domain_shared_features.json", "unsw-in-domain-shared-features", "unsw_test"))
    if (metrics / "cross_dataset_cic_to_unsw.json").exists():
        data = json.loads((metrics / "cross_dataset_cic_to_unsw.json").read_text(encoding="utf-8"))
        rows.append(_row(
            experiment_id="cic-to-unsw-transfer-calibrated", dataset="UNSW-NB15", model="CIC-trained XGBoost (shared features)",
            task="binary", parameters=json.dumps({"threshold": data["calibrated_threshold"]["value"]}),
            accuracy=_round(data["transfer_unsw_nb15_test_calibrated"]["accuracy"]),
            precision=_round(data["transfer_unsw_nb15_test_calibrated"]["precision"]),
            recall=_round(data["transfer_unsw_nb15_test_calibrated"]["recall"]),
            mcc=_round(data["transfer_unsw_nb15_test_calibrated"]["mcc"]),
            fpr=_round(data["transfer_unsw_nb15_test_calibrated"]["fpr"]),
            fnr=_round(data["transfer_unsw_nb15_test_calibrated"]["fnr"]),
            roc_auc=_round(data["transfer_unsw_nb15_test_calibrated"]["roc_auc"]), source="cross_dataset_cic_to_unsw.json",
        ))

    # Inference time per row for the XGBoost detector, from the single-request latency benchmark.
    latency = metrics.parent / "reports" / "cicids2017_latency_benchmark.json"
    if latency.exists():
        bench = json.loads(latency.read_text(encoding="utf-8"))
        p50 = next((r["p50_ms"] for r in bench["table"] if r["stage"].startswith("XGBoost end-to-end")), None)
        for r in rows:
            if r["model"] == "xgboost" and r["task"] == "binary" and p50 is not None:
                r["inference_time_ms_per_row"] = p50
    return pd.DataFrame(rows, columns=COLUMNS)


def main() -> None:
    cfg = load_config("configs/config.yaml")
    df = build_table(cfg)
    out = get_absolute_path(cfg.paths.results_dir) / "metrics" / "experiments.csv"
    df.to_csv(out, index=False)
    logger.info(f"Wrote {len(df)} experiment rows to {out}")


if __name__ == "__main__":
    main()
