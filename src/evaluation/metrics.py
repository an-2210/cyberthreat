"""
Cybersecurity classification evaluation metrics computation.
"""

from typing import Any, Dict, Optional, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from src.utils.logging import setup_logger

logger = setup_logger("evaluation_metrics")


def evaluate_classification(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    y_prob: Optional[Union[pd.Series, np.ndarray]] = None,
    pos_label: int = 1,
) -> Dict[str, Any]:
    """Computes comprehensive security evaluation metrics for binary/multiclass predictions.

    Args:
        y_true: Ground truth target labels.
        y_pred: Predicted class labels.
        y_prob: Optional predicted probability for positive class (or probability array for multiclass).
        pos_label: Positive class label indicator (default=1).

    Returns:
        Dictionary containing metric scores:
        accuracy, precision, recall, macro_f1, weighted_f1, mcc, fpr, fnr, roc_auc, confusion_matrix.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    acc = float(accuracy_score(y_true, y_pred))
    prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    mcc = float(matthews_corrcoef(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred)

    # Compute binary-specific FPR and FNR
    fpr = 0.0
    fnr = 0.0
    roc_auc = None

    if len(np.unique(y_true)) <= 2 and cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

        if y_prob is not None:
            try:
                if y_prob.ndim == 2:
                    y_prob_pos = y_prob[:, 1]
                else:
                    y_prob_pos = y_prob
                roc_auc = float(roc_auc_score(y_true, y_prob_pos))
            except Exception as e:
                logger.warning(f"Could not compute ROC-AUC: {e}")
                roc_auc = None
    elif y_prob is not None:
        try:
            roc_auc = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
        except Exception:
            roc_auc = None

    metrics_dict = {
        "accuracy": round(acc, 6),
        "precision_macro": round(prec_macro, 6),
        "recall_macro": round(rec_macro, 6),
        "macro_f1": round(macro_f1, 6),
        "weighted_f1": round(weighted_f1, 6),
        "mcc": round(mcc, 6),
        "fpr": round(fpr, 6),
        "fnr": round(fnr, 6),
        "roc_auc": round(roc_auc, 6) if roc_auc is not None else None,
        "confusion_matrix": cm.tolist(),
    }

    return metrics_dict
