"""
Target and identifier leakage guards.

Two defenses are applied before modelling:
  1. Identifier-like columns (flow IDs, IP addresses, timestamps, label-derived names) are removed
     by name. The MachineLearningCVE CSVs do not contain these, so this is a safety net for other
     CSV variants of the dataset.
  2. A single-feature scan on the TRAINING split flags any feature whose one-column ROC-AUC is
     near-perfect. Such a feature may be a leaked label proxy. The scan only reports; it never
     removes features, because a strong feature can also be a genuine signal.
"""

from typing import Iterable, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from src.utils.logging import setup_logger

logger = setup_logger("leakage_guard")

DEFAULT_IDENTIFIER_PATTERNS: Tuple[str, ...] = (
    "flow id",
    "source ip",
    "src ip",
    "destination ip",
    "dst ip",
    "timestamp",
    "label",
    "attack_cat",
)


def find_identifier_columns(columns: Iterable[str], patterns: Sequence[str] = DEFAULT_IDENTIFIER_PATTERNS) -> List[str]:
    """Returns columns whose lower-cased name contains any of the identifier patterns."""
    lowered = [p.lower() for p in patterns]
    return [c for c in columns if any(p in str(c).strip().lower() for p in lowered)]


def drop_identifier_columns(
    X: pd.DataFrame, patterns: Sequence[str] = DEFAULT_IDENTIFIER_PATTERNS
) -> Tuple[pd.DataFrame, List[str]]:
    """Drops identifier-like columns. Returns (reduced DataFrame, dropped column names)."""
    dropped = find_identifier_columns(X.columns, patterns)
    if dropped:
        logger.warning(f"Dropping identifier/label-derived columns to prevent leakage: {dropped}")
    return X.drop(columns=dropped), dropped


def single_feature_leakage_scan(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    sample_size: Optional[int] = 200_000,
    threshold: float = 0.99,
    seed: int = 42,
) -> pd.DataFrame:
    """Scores each numeric feature by its one-column ROC-AUC on (a sample of) the training split.

    Args:
        X_train: Preprocessed TRAINING features only (never validation or test rows).
        y_train: Binary training labels (1 = attack).
        sample_size: Rows used for the scan (None = all rows).
        threshold: Features with separability score >= threshold are flagged.
        seed: Seed for the row sample.

    Returns:
        DataFrame with columns feature, auc_separability (max(AUC, 1-AUC)), flagged; sorted descending.
    """
    y = np.asarray(y_train).astype(int)
    n = len(y)
    if sample_size is not None and sample_size < n:
        idx = np.random.default_rng(seed).choice(n, size=sample_size, replace=False)
        X_s, y_s = X_train.iloc[idx], y[idx]
    else:
        X_s, y_s = X_train, y

    rows = []
    if len(np.unique(y_s)) < 2:
        return pd.DataFrame(columns=["feature", "auc_separability", "flagged"])

    for col in X_s.select_dtypes(include=[np.number]).columns:
        values = X_s[col].to_numpy(dtype=np.float64)
        if np.nanmax(values) == np.nanmin(values):
            continue
        auc = roc_auc_score(y_s, values)
        rows.append({"feature": col, "auc_separability": max(auc, 1.0 - auc)})

    report = pd.DataFrame(rows, columns=["feature", "auc_separability"])
    report["flagged"] = report["auc_separability"] >= threshold
    report = report.sort_values("auc_separability", ascending=False).reset_index(drop=True)
    flagged = report.loc[report["flagged"], "feature"].tolist()
    if flagged:
        logger.warning(f"Single-feature leakage scan flagged {len(flagged)} feature(s) with AUC >= {threshold}: {flagged}")
    else:
        logger.info(f"Single-feature leakage scan: no feature reached AUC >= {threshold}.")
    return report
