"""
Unseen-attack experiments for CIC-IDS2017: hold out attack types from the supervised training, then measure
how often their test flows are still detected.

Two modes:
  single  (default): one attack type held out per fold (leave-one-attack-out)
  family:            a whole attack family held out together, e.g. all four DoS types

In every fold the supervised binary XGBoost model is retrained with the held-out rows removed from training and
validation. Then:
  - supervised: the retrained XGBoost probability of ATTACK
  - autoencoder: trained on benign rows only, so it is the same in every fold (it never saw attacks)
  - fusion: supervised + autoencoder, with the weight and thresholds chosen on validation rows without the held-out types

The "none" fold holds out nothing and is the baseline. Thresholds are set on benign validation flows (1% flagged),
so false-positive rates are comparable across folds.

Two processes, for the same reason as src/run_hybrid_detection.py: XGBoost must not share a process with PyTorch.
    child:  trains the XGBoost fold models and caches their probabilities
    parent: trains the autoencoder, evaluates every fold, writes the results

Outputs (single):  results/metrics/cicids2017_leave_one_attack_out.json, results/tables/cicids2017_leave_one_attack_out.csv
Outputs (family):  results/metrics/cicids2017_family_holdout.json,      results/tables/cicids2017_family_holdout.csv

Run from the repository root:
    python -m src.run_leave_one_attack_out
    python -m src.run_leave_one_attack_out --holdout family
"""

import argparse
import subprocess
import sys
import time
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from src.anomaly_detection.common import threshold_for_fpr
from src.data.splits import prepare_splits
from src.fusion.decision_fusion import fuse, rank_normalizer, select_fusion
from src.preprocessing.sampling import sample_positions
from src.utils.config import get_absolute_path, load_config
from src.utils.io_utils import save_json
from src.utils.logging import setup_logger
from src.utils.reproducibility import set_seed

logger = setup_logger("run_leave_one_attack_out")

BENIGN = "BENIGN"
NONE = "none"  # baseline fold: nothing held out
ALPHA_GRID = np.round(np.linspace(0.0, 1.0, 21), 2)
LOW_SAMPLE_WARNING = 50  # held-out groups with fewer test rows than this are flagged as low-sample

# Family folds. Labels are the exact CIC-IDS2017 label strings.
DOS_FAMILY = ("DoS GoldenEye", "DoS Hulk", "DoS Slowhttptest", "DoS slowloris")
FAMILIES: Dict[str, Tuple[str, ...]] = {
    "DoS family (4 types)": DOS_FAMILY,
    "DoS + DDoS (5 types)": DOS_FAMILY + ("DDoS",),
    "Patator family (2 types)": ("FTP-Patator", "SSH-Patator"),
    "Web attacks (3 types)": ("Web Attack - Brute Force", "Web Attack - XSS", "Web Attack - Sql Injection"),
}

MODES = {
    "single": {
        "cache": "results/loao_cache/xgboost_fold_probabilities_single.npz",
        "metrics": "results/metrics/cicids2017_leave_one_attack_out.json",
        "table": "results/tables/cicids2017_leave_one_attack_out.csv",
    },
    "family": {
        "cache": "results/loao_cache/xgboost_fold_probabilities_family.npz",
        "metrics": "results/metrics/cicids2017_family_holdout.json",
        "table": "results/tables/cicids2017_family_holdout.csv",
    },
}


def fold_specs(mode: str, attacks: Sequence[str]) -> List[Tuple[str, Tuple[str, ...]]]:
    """Returns (fold name, held-out labels) pairs. The baseline fold holds out no labels."""
    if mode == "single":
        return [(NONE, ())] + [(a, (a,)) for a in attacks]
    return [(NONE, ())] + [(name, labels) for name, labels in FAMILIES.items()]


def fold_detection(
    held_out: str,
    held_labels: Sequence[str],
    p_val: np.ndarray, a_val: np.ndarray, y_val_bin: np.ndarray, y_val_multi: np.ndarray,
    p_test: np.ndarray, a_test: np.ndarray, y_test_bin: np.ndarray, y_test_multi: np.ndarray,
    target_fpr: float,
) -> Dict[str, Any]:
    """Evaluates one fold. Validation rows of the held-out labels are removed before any choice is made."""
    held = np.isin(y_val_multi, list(held_labels))
    keep_val = ~held
    benign_val = keep_val & (y_val_bin == 0)

    # Each rule's threshold is set on benign validation rows of this fold.
    thr_sup = threshold_for_fpr(p_val[benign_val], target_fpr)
    thr_ae = threshold_for_fpr(a_val[benign_val], target_fpr)
    normalize = rank_normalizer(a_val[benign_val])
    a_norm_val, a_norm_test = normalize(a_val), normalize(a_test)
    choice = select_fusion(p_val[keep_val], a_norm_val[keep_val], y_val_bin[keep_val], ALPHA_GRID, target_fpr)
    fused_test = fuse(p_test, a_norm_test, choice["alpha"])

    rules = {
        "supervised": (p_test, thr_sup),
        "autoencoder": (a_test, thr_ae),
        "fusion": (fused_test, choice["threshold"]),
    }
    if len(held_labels) == 0:
        attack_rows = y_test_bin == 1
    else:
        attack_rows = np.isin(y_test_multi, list(held_labels))
    benign_rows = (y_test_bin == 0)

    result: Dict[str, Any] = {
        "held_out": held_out,
        "held_out_labels": list(held_labels),
        "test_rows_held_out": int(attack_rows.sum()),
        "fusion_alpha": choice["alpha"],
    }
    for name, (score, thr) in rules.items():
        result[f"{name}_detection"] = round(float((score[attack_rows] > thr).mean()), 6)
        result[f"{name}_fpr"] = round(float((score[benign_rows] > thr).mean()), 6)
    return result


def xgboost_child(config: str, mode: str) -> None:
    """Child process: trains one binary XGBoost per fold and caches the probabilities for every validation/test row."""
    from src.models.supervised import build_classifiers, fit_classifier, predict_full_proba

    cfg = load_config(config)
    seed = int(cfg.system.seed)
    set_seed(seed)
    splits = prepare_splits(cfg)["splits"]

    X_train = splits["X_train"].to_numpy(dtype=np.float32)
    y_train_bin = splits["y_train_binary"].to_numpy(dtype=int)
    y_train_multi = splits["y_train_multi"].to_numpy()
    X_val = splits["X_val"].to_numpy(dtype=np.float32)
    y_val_bin = splits["y_val_binary"].to_numpy(dtype=int)
    y_val_multi = splits["y_val_multi"].to_numpy()
    X_test = splits["X_test"].to_numpy(dtype=np.float32)
    y_test_multi = splits["y_test_multi"].to_numpy()

    # Same stratified training subsample as src.run_cicids2017, so the baseline fold matches the main run.
    ids = pd.factorize(y_train_multi)[0]
    positions, _ = sample_positions(ids, strategy=cfg.training.sampling.strategy,
                                    max_rows=cfg.training.sampling.max_train_rows, seed=seed)
    X_train, y_train_bin, y_train_multi = X_train[positions], y_train_bin[positions], y_train_multi[positions]

    attacks = sorted(set(np.unique(y_test_multi)) - {BENIGN})
    specs = fold_specs(mode, attacks)
    arrays: Dict[str, np.ndarray] = {"folds": np.array([name for name, _ in specs])}
    for i, (name, labels) in enumerate(specs):
        train_keep = ~np.isin(y_train_multi, list(labels))
        val_keep = ~np.isin(y_val_multi, list(labels))
        model = build_classifiers(cfg.models, "binary", seed)["xgboost"]
        t0 = time.perf_counter()
        fit_classifier(model, X_train[train_keep], y_train_bin[train_keep], X_val[val_keep], y_val_bin[val_keep])
        logger.info(f"[fold {name}] XGBoost trained on {int(train_keep.sum())} rows in {time.perf_counter() - t0:.1f}s")
        arrays[f"val_{i}"] = predict_full_proba(model, X_val, 2)[:, 1].astype(np.float32)
        arrays[f"test_{i}"] = predict_full_proba(model, X_test, 2)[:, 1].astype(np.float32)

    out = get_absolute_path(MODES[mode]["cache"])
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **arrays)
    logger.info(f"Cached fold probabilities to {out}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Unseen-attack evaluation on CIC-IDS2017.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--holdout", choices=sorted(MODES), default="single",
                        help="single: one attack type per fold. family: whole attack families per fold.")
    parser.add_argument("--target-fpr", type=float, default=0.01)
    parser.add_argument("--xgboost-child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.xgboost_child:
        xgboost_child(args.config, args.holdout)
        return

    from src.anomaly_detection.autoencoder import AutoencoderDetector  # torch only in this process

    cfg = load_config(args.config)
    seed = int(cfg.system.seed)
    set_seed(seed)
    started = time.perf_counter()

    splits = prepare_splits(cfg)["splits"]
    X_train = splits["X_train"].to_numpy(dtype=np.float32)
    y_train_bin = splits["y_train_binary"].to_numpy(dtype=int)
    X_val = splits["X_val"].to_numpy(dtype=np.float32)
    y_val_bin = splits["y_val_binary"].to_numpy(dtype=int)
    y_val_multi = splits["y_val_multi"].to_numpy()
    X_test = splits["X_test"].to_numpy(dtype=np.float32)
    y_test_bin = splits["y_test_binary"].to_numpy(dtype=int)
    y_test_multi = splits["y_test_multi"].to_numpy()

    # Autoencoder: benign training rows only, identical to src.run_hybrid_detection. Same in every fold.
    ae = AutoencoderDetector(seed=seed).fit(X_train[y_train_bin == 0])
    a_val, a_test = ae.score(X_val), ae.score(X_test)

    # XGBoost fold models come from the child process.
    subprocess.run([sys.executable, "-m", "src.run_leave_one_attack_out", "--xgboost-child",
                    "--holdout", args.holdout, "--config", args.config], check=True)
    cache = np.load(get_absolute_path(MODES[args.holdout]["cache"]))
    specs = fold_specs(args.holdout, sorted(set(np.unique(y_test_multi)) - {BENIGN}))

    records = []
    for i, (name, labels) in enumerate(specs):
        assert str(cache["folds"][i]) == name, "fold order changed between child and parent"
        rec = fold_detection(name, labels, cache[f"val_{i}"], a_val, y_val_bin, y_val_multi,
                             cache[f"test_{i}"], a_test, y_test_bin, y_test_multi, args.target_fpr)
        records.append(rec)
        logger.info(f"[{name}] held-out test rows={rec['test_rows_held_out']} detection: "
                    f"supervised={rec['supervised_detection']:.3f} autoencoder={rec['autoencoder_detection']:.3f} "
                    f"fusion={rec['fusion_detection']:.3f}")

    table = pd.DataFrame(records)
    table["held_out_labels"] = table["held_out_labels"].apply(lambda xs: "; ".join(xs))
    table["low_sample"] = table["test_rows_held_out"] < LOW_SAMPLE_WARNING
    table = table.sort_values("held_out").reset_index(drop=True)

    held_table = table[table["held_out"] != NONE]
    summary = {
        "holdout_mode": args.holdout,
        "target_fpr": args.target_fpr,
        "seed": seed,
        "baseline_fold_none": table[table["held_out"] == NONE].drop(columns=["held_out", "held_out_labels"]).to_dict(orient="records")[0],
        "mean_detection_over_held_out_groups": {
            name: round(float(held_table[f"{name}_detection"].mean()), 6)
            for name in ("supervised", "autoencoder", "fusion")
        },
        "held_out_groups_detected_at_least_50pct": {
            name: int((held_table[f"{name}_detection"] >= 0.5).sum())
            for name in ("supervised", "autoencoder", "fusion")
        },
        "n_held_out_groups": int(len(held_table)),
        "n_low_sample_held_out": int(held_table["low_sample"].sum()),
        "elapsed_seconds": round(time.perf_counter() - started, 1),
    }
    paths = MODES[args.holdout]
    save_json({"summary": summary, "folds": records}, get_absolute_path(paths["metrics"]))
    get_absolute_path(paths["table"]).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(get_absolute_path(paths["table"]), index=False)
    logger.info(f"Done in {summary['elapsed_seconds']} s. Summary: {summary['mean_detection_over_held_out_groups']}")


if __name__ == "__main__":
    main()
