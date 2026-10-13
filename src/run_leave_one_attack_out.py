"""
Leave-one-attack-out (unseen attack) experiment for CIC-IDS2017.

For each attack type A, the supervised binary XGBoost model is retrained with every row of A removed from training
and validation. Then we measure how often A is still detected in the test split:

  - supervised: the retrained XGBoost probability of ATTACK
  - autoencoder: trained on benign rows only, so it is the same in every fold (it never saw A)
  - fusion: supervised + autoencoder, with the weight and thresholds chosen on validation rows without A

The "none" fold holds out nothing and is the baseline. Thresholds are set on benign validation flows (1% flagged),
so false-positive rates are comparable across folds.

Two processes, for the same reason as src/run_hybrid_detection.py: XGBoost must not share a process with PyTorch.
    child:  trains the XGBoost fold models and caches their probabilities
    parent: trains the autoencoder, evaluates every fold, writes the results

Outputs:
    results/metrics/cicids2017_leave_one_attack_out.json
    results/tables/cicids2017_leave_one_attack_out.csv

Run from the repository root:
    python -m src.run_leave_one_attack_out
"""

import argparse
import subprocess
import sys
import time
from typing import Any, Dict, Optional

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
CACHE = "results/loao_cache/xgboost_fold_probabilities.npz"
METRICS_PATH = "results/metrics/cicids2017_leave_one_attack_out.json"
TABLE_PATH = "results/tables/cicids2017_leave_one_attack_out.csv"
ALPHA_GRID = np.round(np.linspace(0.0, 1.0, 21), 2)
LOW_SAMPLE_WARNING = 50  # held-out attacks with fewer test rows than this are flagged as low-sample


def fold_detection(
    held_out: str,
    p_val: np.ndarray, a_val: np.ndarray, y_val_bin: np.ndarray, y_val_multi: np.ndarray,
    p_test: np.ndarray, a_test: np.ndarray, y_test_bin: np.ndarray, y_test_multi: np.ndarray,
    target_fpr: float,
) -> Dict[str, Any]:
    """Evaluates one fold. Validation rows of `held_out` are removed before any choice is made."""
    keep_val = np.ones(len(y_val_bin), dtype=bool) if held_out == NONE else y_val_multi != held_out
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
    if held_out == NONE:
        attack_rows = y_test_bin == 1
    else:
        attack_rows = y_test_multi == held_out
    benign_rows = (y_test_bin == 0)

    result: Dict[str, Any] = {
        "held_out": held_out,
        "test_rows_held_out": int(attack_rows.sum()),
        "fusion_alpha": choice["alpha"],
    }
    for name, (score, thr) in rules.items():
        result[f"{name}_detection"] = round(float((score[attack_rows] > thr).mean()), 6)
        result[f"{name}_fpr"] = round(float((score[benign_rows] > thr).mean()), 6)
    return result


def xgboost_child(config: str) -> None:
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
    folds = [NONE] + attacks
    arrays: Dict[str, np.ndarray] = {"folds": np.array(folds)}
    for i, held in enumerate(folds):
        train_keep = np.ones(len(y_train_bin), dtype=bool) if held == NONE else y_train_multi != held
        val_keep = np.ones(len(y_val_bin), dtype=bool) if held == NONE else y_val_multi != held
        model = build_classifiers(cfg.models, "binary", seed)["xgboost"]
        t0 = time.perf_counter()
        fit_classifier(model, X_train[train_keep], y_train_bin[train_keep], X_val[val_keep], y_val_bin[val_keep])
        logger.info(f"[fold {held}] XGBoost trained on {int(train_keep.sum())} rows in {time.perf_counter() - t0:.1f}s")
        arrays[f"val_{i}"] = predict_full_proba(model, X_val, 2)[:, 1].astype(np.float32)
        arrays[f"test_{i}"] = predict_full_proba(model, X_test, 2)[:, 1].astype(np.float32)

    out = get_absolute_path(CACHE)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **arrays)
    logger.info(f"Cached fold probabilities to {out}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Leave-one-attack-out evaluation on CIC-IDS2017.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--target-fpr", type=float, default=0.01)
    parser.add_argument("--xgboost-child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.xgboost_child:
        xgboost_child(args.config)
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
    subprocess.run([sys.executable, "-m", "src.run_leave_one_attack_out", "--xgboost-child", "--config", args.config],
                   check=True)
    cache = np.load(get_absolute_path(CACHE))
    folds = [str(f) for f in cache["folds"]]

    records = []
    for i, held in enumerate(folds):
        rec = fold_detection(held, cache[f"val_{i}"], a_val, y_val_bin, y_val_multi,
                             cache[f"test_{i}"], a_test, y_test_bin, y_test_multi, args.target_fpr)
        records.append(rec)
        logger.info(f"[{held}] held-out test rows={rec['test_rows_held_out']} detection: "
                    f"supervised={rec['supervised_detection']:.3f} autoencoder={rec['autoencoder_detection']:.3f} "
                    f"fusion={rec['fusion_detection']:.3f}")

    table = pd.DataFrame(records)
    table["low_sample"] = table["test_rows_held_out"] < LOW_SAMPLE_WARNING
    table = table.sort_values("held_out").reset_index(drop=True)

    attack_table = table[table["held_out"] != NONE]
    summary = {
        "target_fpr": args.target_fpr,
        "seed": seed,
        "baseline_fold_none": table[table["held_out"] == NONE].drop(columns=["held_out"]).to_dict(orient="records")[0],
        "mean_detection_over_held_out_attacks": {
            name: round(float(attack_table[f"{name}_detection"].mean()), 6)
            for name in ("supervised", "autoencoder", "fusion")
        },
        "held_out_attacks_detected_at_least_50pct": {
            name: int((attack_table[f"{name}_detection"] >= 0.5).sum())
            for name in ("supervised", "autoencoder", "fusion")
        },
        "n_held_out_attacks": int(len(attack_table)),
        "n_low_sample_held_out": int(attack_table["low_sample"].sum()),
        "elapsed_seconds": round(time.perf_counter() - started, 1),
    }
    save_json({"summary": summary, "folds": records}, get_absolute_path(METRICS_PATH))
    get_absolute_path(TABLE_PATH).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(get_absolute_path(TABLE_PATH), index=False)
    logger.info(f"Done in {summary['elapsed_seconds']} s. Summary: {summary['mean_detection_over_held_out_attacks']}")


if __name__ == "__main__":
    main()
