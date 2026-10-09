"""
Configurable training-set sampling. Sampling is applied ONLY to the training split, after the
split and after preprocessing have been fitted. Validation and test sets are never subsampled.
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np
from sklearn.model_selection import train_test_split
from src.utils.logging import setup_logger

logger = setup_logger("training_sampler")

SAMPLING_STRATEGIES = ("none", "stratified")


def sample_positions(
    strata: np.ndarray,
    strategy: str = "stratified",
    max_rows: Optional[int] = None,
    seed: int = 42,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Chooses row positions for a training subsample.

    Args:
        strata: Per-row class labels used to keep class proportions (typically the multiclass label,
            so rare attack families are not dropped entirely).
        strategy: 'none' keeps every row; 'stratified' caps the size at `max_rows` while keeping
            class proportions.
        max_rows: Maximum rows to keep for 'stratified' (None or >= n means no cap).
        seed: Seed for reproducible sampling.

    Returns:
        (sorted row positions, metadata dictionary describing what was done).
    """
    if strategy not in SAMPLING_STRATEGIES:
        raise ValueError(f"Unknown sampling strategy '{strategy}'. Use one of {SAMPLING_STRATEGIES}.")

    n = len(strata)
    meta: Dict[str, Any] = {"strategy": strategy, "rows_before": int(n), "max_rows": max_rows, "seed": seed}
    if strategy == "none" or max_rows is None or max_rows >= n:
        meta.update(method="all_rows", rows_after=int(n))
        return np.arange(n), meta

    positions = np.arange(n)
    values, counts = np.unique(strata, return_counts=True)
    try:
        if counts.min() < 2 or max_rows < len(values):
            raise ValueError("stratification impossible with these class counts")
        keep, _ = train_test_split(positions, train_size=max_rows, stratify=strata, random_state=seed)
        method = "stratified_cap"
    except ValueError as exc:
        logger.warning(f"Stratified subsampling not possible ({exc}); falling back to uniform random sampling.")
        keep = np.random.default_rng(seed).choice(n, size=max_rows, replace=False)
        method = "uniform_random_cap"

    keep = np.sort(keep)
    meta.update(method=method, rows_after=int(len(keep)))
    logger.info(f"Training subsample: {n} -> {len(keep)} rows ({method}, seed={seed}).")
    return keep, meta


def cap_per_class_positions(strata: np.ndarray, max_per_class: int, seed: int = 42) -> np.ndarray:
    """Keeps every row of classes with <= max_per_class rows and a seeded random subset of larger classes.

    Used so that rare attack families keep all their training rows while dominant classes are capped.
    """
    strata = np.asarray(strata)
    rng = np.random.default_rng(seed)
    keep = []
    for value in np.unique(strata):
        idx = np.flatnonzero(strata == value)
        if len(idx) > max_per_class:
            idx = rng.choice(idx, size=max_per_class, replace=False)
        keep.append(idx)
    return np.sort(np.concatenate(keep))
