"""
Decision-level fusion of a supervised attack probability and an anomaly score.

    fused = alpha * P_supervised + (1 - alpha) * A_normalized

The anomaly score is turned into a value in [0, 1] by its rank among benign validation flows, so the two inputs
are on the same scale. The weight alpha and the decision threshold are both chosen on the validation split only.
The test split is used once, to report the final numbers.
"""

from typing import Callable, Dict, Sequence

import numpy as np
from sklearn.metrics import f1_score

from src.anomaly_detection.common import threshold_for_fpr


def rank_normalizer(benign_reference: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """Maps a score to the share of benign reference scores at or below it (an empirical CDF, in [0, 1])."""
    reference = np.sort(np.asarray(benign_reference, dtype=np.float64))

    def normalize(scores: np.ndarray) -> np.ndarray:
        return np.searchsorted(reference, np.asarray(scores, dtype=np.float64), side="right") / len(reference)

    return normalize


def fuse(p_supervised: np.ndarray, a_normalized: np.ndarray, alpha: float) -> np.ndarray:
    """Weighted combination of the two scores."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1.")
    return alpha * np.asarray(p_supervised) + (1.0 - alpha) * np.asarray(a_normalized)


def select_fusion(
    p_val: np.ndarray,
    a_val: np.ndarray,
    y_val: np.ndarray,
    alphas: Sequence[float],
    target_fpr: float,
) -> Dict[str, float]:
    """Chooses alpha and the threshold on validation data.

    For each alpha, the threshold is the one that flags `target_fpr` of benign validation flows. The alpha with
    the best validation F1 for the ATTACK class wins (first one on ties).
    """
    y_val = np.asarray(y_val).astype(int)
    best = None
    for alpha in alphas:
        fused = fuse(p_val, a_val, alpha)
        threshold = threshold_for_fpr(fused[y_val == 0], target_fpr)
        f1 = f1_score(y_val, (fused > threshold).astype(int), pos_label=1, zero_division=0)
        if best is None or f1 > best["val_f1"]:
            best = {"alpha": float(alpha), "threshold": threshold, "val_f1": float(f1)}
    return best
