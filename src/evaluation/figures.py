"""
Research figures for CIC-IDS2017, all drawn from saved results (no figure is hand-made or hard-coded).

Three groups, each run in its own process because XGBoost, PyTorch and SHAP do not share a process well on macOS:

    python -m src.evaluation.figures metrics   # precision-recall, ROC, model comparison, known vs unseen, ablation
    python -m src.evaluation.figures shap      # SHAP summary (beeswarm) from the saved SHAP sample
    python -m src.evaluation.figures anomaly   # autoencoder score distribution, benign vs attack (needs torch)

Inputs: data/interim/cicids2017_binary_test_scores.npz (run_baselines_cicids2017), results/metrics/*.json,
results/tables/cicids2017_ablation_detection.csv, results/explainability/shap_binary_xgboost_sample.npz,
data/interim/latency_cache/xgboost_stage.npz and models/artifacts/cicids2017_anomaly_autoencoder.pt.
"""

import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from src.utils.config import get_absolute_path, load_config
from src.utils.logging import setup_logger
from src.utils.plotting import new_figure, save_figure

logger = setup_logger("figures")

SCORES = "data/interim/cicids2017_binary_test_scores.npz"
MODEL_LABELS = {
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "logistic_regression": "Logistic Regression",
    "linear_svm": "Linear SVM",
}


def _metrics_dir(cfg) -> Path:
    return get_absolute_path(cfg.paths.results_dir) / "metrics"


def _figures_dir(cfg) -> Path:
    path = get_absolute_path(cfg.paths.results_dir) / "figures"
    path.mkdir(parents=True, exist_ok=True)
    return path


def precision_recall_figure(cfg) -> Path:
    data = np.load(get_absolute_path(SCORES))
    y = data["y_test"]
    fig = new_figure((7, 5))
    ax = fig.add_subplot()
    for key, label in MODEL_LABELS.items():
        if key not in data.files:
            continue
        p, r, _ = precision_recall_curve(y, data[key])
        ap = average_precision_score(y, data[key])
        ax.plot(r, p, label=f"{label} (AP {ap:.3f})")
    ax.set_xlabel("recall (attack class)")
    ax.set_ylabel("precision (attack class)")
    ax.set_title(f"Precision-recall, binary BENIGN vs ATTACK (n={len(y):,} test rows)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return save_figure(fig, _figures_dir(cfg) / "cicids2017_binary_precision_recall.png")


def roc_figure(cfg) -> Path:
    data = np.load(get_absolute_path(SCORES))
    y = data["y_test"]
    fig = new_figure((7, 5))
    ax = fig.add_subplot()
    for key, label in MODEL_LABELS.items():
        if key not in data.files:
            continue
        fpr, tpr, _ = roc_curve(y, data[key])
        ax.plot(fpr, tpr, label=f"{label} (AUC {roc_auc_score(y, data[key]):.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#9ca3af", label="chance")
    ax.set_xlabel("false positive rate (benign flows flagged)")
    ax.set_ylabel("true positive rate (attacks flagged)")
    ax.set_title("ROC, binary BENIGN vs ATTACK (test split)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return save_figure(fig, _figures_dir(cfg) / "cicids2017_binary_roc_all_models.png")


def model_comparison_figure(cfg) -> Path:
    """Binary test metrics of every saved and baseline model: F1, MCC, FPR and FNR side by side."""
    rows = []
    for key in ("random_forest", "xgboost", "logistic_regression", "linear_svm"):
        path = _metrics_dir(cfg) / (f"cicids2017_binary_{key}.json" if key in ("random_forest", "xgboost")
                                    else f"cicids2017_baseline_{key}_binary.json")
        m = json.loads(path.read_text(encoding="utf-8"))
        rows.append({"model": MODEL_LABELS[key], "F1": m["f1"], "MCC": m["mcc"], "FPR": m["fpr"], "FNR": m["fnr"]})
    df = pd.DataFrame(rows).set_index("model")
    fig = new_figure((9, 5))
    ax = fig.add_subplot()
    df.plot.bar(ax=ax, rot=0, color=["#2563eb", "#059669", "#f59e0b", "#dc2626"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("value")
    ax.set_title("Binary detection on the CIC-IDS2017 test split (same rows for every model)", fontsize=10)
    ax.legend(fontsize=8, ncol=4)
    ax.grid(axis="y", alpha=0.3)
    return save_figure(fig, _figures_dir(cfg) / "cicids2017_binary_model_comparison.png")


def known_vs_unseen_figure(cfg) -> Path:
    """Detection of each attack type when it is held out of supervised training, against the autoencoder."""
    loao = json.loads((_metrics_dir(cfg) / "cicids2017_leave_one_attack_out.json").read_text(encoding="utf-8"))
    folds = [f for f in loao["folds"] if f["held_out"] != "none"]
    folds.sort(key=lambda f: f["supervised_detection"])
    names = [f["held_out"] for f in folds]
    fig = new_figure((10, 5.5))
    ax = fig.add_subplot()
    idx = np.arange(len(names))
    ax.barh(idx - 0.2, [f["supervised_detection"] for f in folds], height=0.4, label="supervised (retrained without this attack)",
            color="#2563eb")
    ax.barh(idx + 0.2, [f["autoencoder_detection"] for f in folds], height=0.4, label="autoencoder (never saw attacks)",
            color="#f59e0b")
    ax.set_yticks(idx, names, fontsize=8)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel(f"share of held-out flows flagged (threshold at {loao['summary']['target_fpr']:.0%} benign FPR)")
    ax.set_title("Known vs unseen attack types: leave-one-attack-out", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    return save_figure(fig, _figures_dir(cfg) / "cicids2017_known_vs_unseen.png")


def ablation_figure(cfg) -> Path:
    table = pd.read_csv(get_absolute_path(cfg.paths.results_dir) / "tables" / "cicids2017_ablation_detection.csv")
    fig = new_figure((10, 5))
    ax = fig.add_subplot()
    ax.barh(table["variant"][::-1], table["f1"][::-1], color="#4f46e5")
    for y, (f1, fnr) in enumerate(zip(table["f1"][::-1], table["fnr"][::-1])):
        ax.text(f1 + 0.005, y, f"F1 {f1:.3f} · FNR {fnr:.3f}", va="center", fontsize=8)
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("F1 on the CIC-IDS2017 test split (attack = positive class)")
    ax.set_title("Ablation: detection with and without each component", fontsize=10)
    ax.grid(axis="x", alpha=0.3)
    return save_figure(fig, _figures_dir(cfg) / "cicids2017_ablation_detection.png")


def shap_summary_figure(cfg) -> Path:
    import matplotlib.pyplot as plt
    import shap

    sample = np.load(get_absolute_path(cfg.paths.results_dir) / "explainability" / "shap_binary_xgboost_sample.npz")
    names = list(sample["feature_names"])
    plt.figure(figsize=(9, 7))
    shap.summary_plot(sample["shap_values"], sample["X"], feature_names=names, max_display=15, show=False)
    fig = plt.gcf()
    fig.suptitle("SHAP summary, XGBoost binary detector (log-odds toward ATTACK)", fontsize=10)
    path = _figures_dir(cfg) / "cicids2017_shap_summary.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def anomaly_score_figure(cfg) -> Path:
    """Autoencoder reconstruction-error distribution on the test rows, benign vs attack.

    Uses the test split cached by the latency stage, so no split is rebuilt here.
    """
    import torch
    from src.anomaly_detection.autoencoder import AutoencoderDetector

    artifacts = get_absolute_path(cfg.paths.artifacts_dir)
    ck = torch.load(artifacts / "cicids2017_anomaly_autoencoder.pt", map_location="cpu", weights_only=False)
    ae = AutoencoderDetector(hidden_dims=tuple(ck["hidden_dims"]))
    ae.network = ae._build(int(ck["n_features"]))
    ae.network.load_state_dict(ck["state_dict"])
    ae.network.eval()
    ae.mean_, ae.std_ = np.asarray(ck["mean"]), np.asarray(ck["std"])

    cache = np.load(get_absolute_path("data/interim/latency_cache") / "xgboost_stage.npz")
    scores = ae.score(cache["X_test"])
    y = cache["y_test"]
    fig = new_figure((8, 5))
    ax = fig.add_subplot()
    bins = np.linspace(np.percentile(scores, 0.5), np.percentile(scores, 99.5), 80)
    ax.hist(scores[y == 0], bins=bins, alpha=0.6, density=True, label=f"BENIGN (n={int((y == 0).sum()):,})", color="#059669")
    ax.hist(scores[y == 1], bins=bins, alpha=0.6, density=True, label=f"ATTACK (n={int((y == 1).sum()):,})", color="#dc2626")
    ax.set_xlabel("autoencoder reconstruction error (higher = more unusual)")
    ax.set_ylabel("density")
    ax.set_title(f"Anomaly score distribution, test split (ROC-AUC {roc_auc_score(y, scores):.3f})", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return save_figure(fig, _figures_dir(cfg) / "cicids2017_anomaly_score_distribution.png")


def main(argv=None) -> None:
    group = (argv or sys.argv[1:] or ["metrics"])[0]
    cfg = load_config("configs/config.yaml")
    if group == "metrics":
        paths = [precision_recall_figure(cfg), roc_figure(cfg), model_comparison_figure(cfg),
                 known_vs_unseen_figure(cfg), ablation_figure(cfg)]
    elif group == "shap":
        paths = [shap_summary_figure(cfg)]
    elif group == "anomaly":
        paths = [anomaly_score_figure(cfg)]
    else:
        raise SystemExit(f"Unknown group '{group}'. Use metrics, shap or anomaly.")
    for p in paths:
        logger.info(f"Wrote {p}")


if __name__ == "__main__":
    main()
