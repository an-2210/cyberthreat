"""Helpers shared by the anomaly detectors."""

import numpy as np


def signed_log1p(X: np.ndarray) -> np.ndarray:
    """sign(x) * log(1 + |x|). Compresses heavy tails while keeping negative values.

    Flow features such as byte and packet counts span many orders of magnitude; this keeps one extreme
    flow from dominating the anomaly score.
    """
    return np.sign(X) * np.log1p(np.abs(X))


def threshold_for_fpr(benign_scores: np.ndarray, target_fpr: float) -> float:
    """Score threshold that flags about `target_fpr` of benign rows (higher score = more anomalous)."""
    if not 0 < target_fpr < 1:
        raise ValueError("target_fpr must be between 0 and 1.")
    return float(np.quantile(benign_scores, 1 - target_fpr))
