"""
Detection ablation table for CIC-IDS2017 (binary BENIGN vs ATTACK, held-out test split).

Every row comes from results/metrics/cicids2017_hybrid_detection.json, which evaluates all rules at the
same operating point: the threshold that flags 1% of benign validation flows. Comparing rules at
different thresholds would not be a fair ablation, so no other rows are mixed in.

Outputs:
    results/tables/cicids2017_ablation_detection.csv
    results/tables/cicids2017_ablation_detection.md
    results/reports/cicids2017_ablation_detection.json

Run from the repository root:
    python -m src.evaluation.ablation
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from src.utils.config import get_absolute_path, load_config
from src.utils.io_utils import save_json
from src.utils.logging import setup_logger

logger = setup_logger("ablation")

# variant key in the hybrid results -> (display name, what the row removes or adds)
VARIANTS = [
    ("supervised_xgboost", "XGBoost only", "Supervised component alone (no anomaly component)"),
    ("autoencoder", "Autoencoder only", "Removes the supervised component"),
    ("isolation_forest_log", "Isolation Forest (log) only", "Anomaly alternative, no supervised component"),
    ("isolation_forest_raw", "Isolation Forest (raw) only", "Anomaly alternative, no supervised component"),
    ("hybrid_fusion", "Hybrid: XGBoost + Autoencoder fusion", "Full hybrid system (alpha and threshold chosen on validation)"),
]


def ablation_rows(hybrid: Dict[str, Any]) -> List[Dict[str, Any]]:
    tests = hybrid["test_results"]
    base = tests["supervised_xgboost"]
    rows = []
    for key, name, role in VARIANTS:
        m = tests[key]
        cm = m["confusion_matrix"]["matrix"]
        rows.append({
            "variant": name,
            "role": role,
            "precision": round(m["precision"], 4),
            "recall": round(m["recall"], 4),
            "f1": round(m["f1"], 4),
            "fpr": round(m["fpr"], 4),
            "fnr": round(m["fnr"], 4),
            "mcc": round(m["mcc"], 4),
            "roc_auc": round(m["roc_auc"], 4),
            "tp": cm[1][1],
            "fn": cm[1][0],
            "fp": cm[0][1],
            "tn": cm[0][0],
            "delta_f1_vs_xgboost": round(m["f1"] - base["f1"], 4),
            "delta_fnr_vs_xgboost": round(m["fnr"] - base["fnr"], 4),
            "delta_fpr_vs_xgboost": round(m["fpr"] - base["fpr"], 4),
        })
    return rows


def unseen_attack_rows(loao: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Leave-one-attack-out: each attack family is held out of training, then detected at the 1%-FPR threshold."""
    s = loao["summary"]
    mean = s["mean_detection_over_held_out_attacks"]
    count = s["held_out_attacks_detected_at_least_50pct"]
    names = [("supervised", "XGBoost only"), ("autoencoder", "Autoencoder only"), ("fusion", "Hybrid: XGBoost + Autoencoder fusion")]
    return [{"variant": name, "mean_detection_unseen": round(mean[k], 4),
             "families_detected_at_least_50pct": f"{count[k]} of {s['n_held_out_attacks']}",
             "delta_vs_xgboost": round(mean[k] - mean["supervised"], 4)} for k, name in names]


def to_markdown(df: pd.DataFrame, fusion_threshold: float, alpha: float, target_fpr: float,
                unseen: List[Dict[str, Any]] = None) -> str:
    cols = ["variant", "precision", "recall", "f1", "fpr", "fnr", "mcc", "roc_auc", "delta_f1_vs_xgboost", "delta_fnr_vs_xgboost"]
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    body = ["| " + " | ".join(str(r[c]) for c in cols) + " |" for _, r in df.iterrows()]
    note = (f"All rows use the threshold that flags {target_fpr:.0%} of benign validation flows. "
            f"Fusion alpha = {alpha}, fusion threshold = {fusion_threshold:.4f}. "
            "Deltas are relative to XGBoost only.")
    out = [header, sep, *body, "", note, ""]
    if unseen:
        ucols = ["variant", "mean_detection_unseen", "families_detected_at_least_50pct", "delta_vs_xgboost"]
        out += ["### Unseen attacks (leave-one-attack-out)", "",
                "| " + " | ".join(ucols) + " |", "|" + "|".join("---" for _ in ucols) + "|",
                *["| " + " | ".join(str(r[c]) for c in ucols) + " |" for r in unseen], "",
                "Each family is held out of training and scored at the same 1%-FPR operating point.", ""]
    return "\n".join(out)


def run(cfg=None) -> Dict[str, Any]:
    cfg = cfg or load_config("configs/config.yaml")
    results = get_absolute_path(cfg.paths.results_dir)
    hybrid = json.loads((results / "metrics" / "cicids2017_hybrid_detection.json").read_text())
    df = pd.DataFrame(ablation_rows(hybrid))
    loao = json.loads((results / "metrics" / "cicids2017_leave_one_attack_out.json").read_text())
    unseen = unseen_attack_rows(loao)

    tables = results / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    df.to_csv(tables / "cicids2017_ablation_detection.csv", index=False)
    (tables / "cicids2017_ablation_detection.md").write_text(
        to_markdown(df, hybrid["fusion"]["threshold"], hybrid["fusion"]["alpha"], hybrid["target_fpr"], unseen), encoding="utf-8")
    summary = {
        "data_source": "real CIC-IDS2017 CSVs (held-out test split)",
        "operating_point": f"threshold flags {hybrid['target_fpr']:.0%} of benign validation flows",
        "fusion": {"alpha": hybrid["fusion"]["alpha"], "threshold": hybrid["fusion"]["threshold"],
                   "anomaly_source": hybrid["fusion"]["anomaly_source"]},
        "rows": df.to_dict(orient="records"),
        "unseen_attacks": unseen,
        "not_included": {
            "random_forest_binary": "Saved binary RF metrics use the default 0.5 threshold, not the 1%-FPR operating point, so they are not comparable here.",
            "isolation_forest_fused": "Only the chosen anomaly candidate (autoencoder) was fused; fusion with Isolation Forest scores was not computed.",
        },
    }
    save_json(summary, results / "reports" / "cicids2017_ablation_detection.json")
    logger.info("Wrote detection ablation table.")
    return summary


if __name__ == "__main__":
    s = run()
    print(pd.DataFrame(s["rows"])[["variant", "f1", "fpr", "fnr", "roc_auc", "delta_f1_vs_xgboost"]].to_string(index=False))
