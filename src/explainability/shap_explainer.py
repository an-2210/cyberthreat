"""
SHAP explanations for the saved XGBoost binary detector (BENIGN vs ATTACK).

Uses TreeExplainer with an interventional background drawn from the TRAINING split, and explains a
random sample of TEST rows. SHAP values are in the model's log-odds margin space; positive values
push a row towards ATTACK.

Run from the repository root:
    python -m src.explainability.shap_explainer
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from src.models.inference import CICIDS2017Detector
from src.run_rare_class_experiment import prepare_splits
from src.utils.config import get_absolute_path, load_config
from src.utils.io_utils import save_json
from src.utils.logging import setup_logger
from src.utils.plotting import new_figure, save_figure

logger = setup_logger("shap_explainer")

EXPLAIN_SAMPLE_SIZE = 2000
EXAMPLES_PER_GROUP = 3
BACKGROUND_FILE = "shap_background.npy"
SAMPLE_FILE = "shap_binary_xgboost_sample.npz"


def explain_rows(shap_values: np.ndarray, X: np.ndarray, feature_names: List[str], row_index: np.ndarray,
                 top_k: int) -> List[Dict[str, Any]]:
    """Top-k contributions for each row, sorted by absolute SHAP value."""
    out = []
    for r in range(shap_values.shape[0]):
        order = np.argsort(np.abs(shap_values[r]))[::-1][:top_k]
        out.append({
            "test_row": int(row_index[r]),
            "shap_margin_total": float(shap_values[r].sum()),
            "top_features": [
                {"feature": feature_names[i], "value": float(X[r, i]), "shap": float(shap_values[r, i])}
                for i in order
            ],
        })
    return out


def run_shap(cfg, results_dir: Path = None, models_dir: Path = None, sample_size: int = EXPLAIN_SAMPLE_SIZE,
             seed: int = None) -> Dict[str, Any]:
    seed = int(cfg.system.seed) if seed is None else seed
    shap_cfg = cfg.explainability.shap
    results_dir = get_absolute_path(results_dir or cfg.paths.results_dir) / "explainability"
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir = get_absolute_path(models_dir or cfg.paths.artifacts_dir)

    detector = CICIDS2017Detector.load("binary", "xgboost", artifacts_dir=models_dir)
    feature_names = list(detector.pipeline.feature_names_out_)

    # Same deduplicated, stratified split as the trained models (no retraining; pipeline refit is deterministic).
    splits = prepare_splits(cfg)["splits"]
    X_train = splits["X_train"].to_numpy(dtype=np.float32)
    X_test = splits["X_test"].to_numpy(dtype=np.float32)
    y_test = splits["y_test_binary"].to_numpy(dtype=int)

    rng = np.random.default_rng(seed)
    bg_idx = rng.choice(len(X_train), size=int(shap_cfg.background_samples), replace=False)
    background = X_train[bg_idx]
    np.save(results_dir / BACKGROUND_FILE, background)  # reused for single-row explanations in the reports and dashboard
    row_idx = np.sort(rng.choice(len(X_test), size=min(sample_size, len(X_test)), replace=False))
    X_s, y_s = X_test[row_idx], y_test[row_idx]

    # SHAP 0.52 rejects the sklearn wrapper of XGBoost 3.x (categorical flag), so use the raw booster.
    booster = detector.model.model.get_booster()
    # Passing the masker explicitly keeps all background rows (shap otherwise caps them at 100).
    masker = shap.maskers.Independent(background, max_samples=len(background))
    explainer = shap.TreeExplainer(booster, data=masker, feature_perturbation="interventional")
    sv = np.asarray(explainer.shap_values(X_s, check_additivity=False))
    np.savez_compressed(results_dir / SAMPLE_FILE, shap_values=sv, X=X_s, y=y_s, feature_names=np.array(feature_names))
    base = float(np.atleast_1d(explainer.expected_value)[0])

    # Additivity check: base value + sum(SHAP) must reproduce the model's margin output.
    margin = booster.predict(xgb.DMatrix(X_s), output_margin=True)
    additivity_max_abs_error = float(np.max(np.abs(base + sv.sum(axis=1) - margin)))

    mean_abs = np.abs(sv).mean(axis=0)
    global_rank = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs}).sort_values(
        "mean_abs_shap", ascending=False).reset_index(drop=True)
    global_rank.to_csv(results_dir / "shap_binary_xgboost_mean_abs.csv", index=False)

    top_k = int(shap_cfg.top_k_features)
    top = global_rank.head(top_k)
    fig = new_figure((8, 5))
    ax = fig.add_subplot()
    ax.barh(top["feature"][::-1], top["mean_abs_shap"][::-1], color="#4C72B0")
    ax.set_xlabel("mean |SHAP value| (log-odds, test sample)")
    ax.set_title(f"Top {top_k} features, XGBoost binary detector (n={len(X_s)} test rows)", fontsize=10)
    fig_path = save_figure(fig, results_dir / "shap_binary_xgboost_top_features.png")

    preds = (margin > 0).astype(int)
    tp = np.flatnonzero((y_s == 1) & (preds == 1))[:EXAMPLES_PER_GROUP]
    fn = np.flatnonzero((y_s == 1) & (preds == 0))[:EXAMPLES_PER_GROUP]
    examples = {
        "true_positive_attacks": explain_rows(sv[tp], X_s[tp], feature_names, row_idx[tp], top_k),
        "false_negative_attacks": explain_rows(sv[fn], X_s[fn], feature_names, row_idx[fn], top_k),
    }

    summary = {
        "model": "xgboost (binary)",
        "data_source": "real CIC-IDS2017 CSVs",
        "explainer": "shap.TreeExplainer, interventional, background from training split",
        "n_test_rows_explained": int(len(X_s)),
        "n_background_rows": int(len(background)),
        "expected_value_log_odds": base,
        "additivity_max_abs_error": additivity_max_abs_error,
        "sample_attack_rate": float(y_s.mean()),
        "sample_detection_counts": {"tp": int(((y_s == 1) & (preds == 1)).sum()),
                                    "fn": int(((y_s == 1) & (preds == 0)).sum()),
                                    "fp": int(((y_s == 0) & (preds == 1)).sum()),
                                    "tn": int(((y_s == 0) & (preds == 0)).sum())},
        "top_features_global": global_rank.head(top_k).to_dict(orient="records"),
        "examples": examples,
        "figure": str(fig_path),
        "seed": seed,
    }
    save_json(summary, results_dir / "shap_binary_xgboost_summary.json")
    logger.info(f"SHAP additivity max abs error: {additivity_max_abs_error:.2e}")
    return summary


def build_explainer(booster: Any, background: np.ndarray) -> shap.TreeExplainer:
    """TreeExplainer on the XGBoost booster with an explicit interventional background (all rows kept)."""
    masker = shap.maskers.Independent(background, max_samples=len(background))
    return shap.TreeExplainer(booster, data=masker, feature_perturbation="interventional")


def explain_single(explainer: shap.TreeExplainer, x: np.ndarray, feature_names: List[str], top_k: int) -> Dict[str, Any]:
    """Why one row scored as it did: its top SHAP contributions, in log-odds, signed by direction."""
    sv = np.asarray(explainer.shap_values(np.asarray(x, dtype=np.float32).reshape(1, -1), check_additivity=False))[0]
    order = np.argsort(np.abs(sv))[::-1][:top_k]
    return {
        "expected_value_log_odds": float(np.atleast_1d(explainer.expected_value)[0]),
        "margin_total_log_odds": float(sv.sum()),
        "top_features": [
            {"feature": feature_names[i], "value": float(x[i]), "shap": round(float(sv[i]), 5),
             "direction": "toward ATTACK" if sv[i] > 0 else "toward BENIGN"}
            for i in order
        ],
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="SHAP explanations for the XGBoost binary detector.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--sample-size", type=int, default=EXPLAIN_SAMPLE_SIZE)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    s = run_shap(cfg, sample_size=args.sample_size)
    print(json.dumps({k: s[k] for k in ("n_test_rows_explained", "additivity_max_abs_error", "sample_detection_counts")}, indent=2))
    print("top features:", [f["feature"] for f in s["top_features_global"]])


if __name__ == "__main__":
    main()
