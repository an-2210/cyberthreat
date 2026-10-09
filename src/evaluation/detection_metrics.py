"""
Test-set metrics for CIC-IDS2017 classifiers: binary attack detection and multiclass attack typing.

Binary precision/recall/F1 are reported for the ATTACK class (label 1), which is what an
intrusion detector is judged on. Macro and weighted averages are included for multiclass.
"""

from typing import Any, Dict, List, Optional, Sequence
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)

BINARY_CLASS_NAMES = ("BENIGN", "ATTACK")


def binary_detection_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: Optional[np.ndarray] = None,
    pos_label: int = 1,
) -> Dict[str, Any]:
    """Metrics for BENIGN (0) vs ATTACK (1).

    Args:
        y_true: True binary labels.
        y_pred: Predicted binary labels.
        y_score: Attack probability/score for each row (needed for ROC-AUC).
        pos_label: Label treated as positive (attack).

    Returns:
        Dictionary with precision, recall (detection rate), f1, fpr, fnr, accuracy, mcc, roc_auc,
        confusion matrix, and support counts.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    roc_auc = None
    if y_score is not None and len(np.unique(y_true)) == 2:
        roc_auc = float(roc_auc_score(y_true, np.asarray(y_score, dtype=np.float64)))

    return {
        "task": "binary",
        "positive_class": "ATTACK",
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0)),
        "fpr": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "fnr": float(fn / (fn + tp)) if (fn + tp) else 0.0,
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "roc_auc": roc_auc,
        "confusion_matrix": {"labels": list(BINARY_CLASS_NAMES), "matrix": cm.tolist()},
        "support": {"benign": int(tn + fp), "attack": int(fn + tp)},
    }


def multiclass_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
    class_names: Sequence[str],
) -> Dict[str, Any]:
    """Metrics for multi-attack classification.

    Args:
        y_true: True class ids (0..K-1, indexing `class_names`).
        y_pred: Predicted class ids.
        y_proba: Probability matrix (n_rows x K) with columns aligned to `class_names`.
        class_names: Human-readable names for ids 0..K-1.

    Returns:
        Dictionary with accuracy, macro/weighted precision/recall/F1, per-class precision/recall/F1/support,
        one-vs-rest macro ROC-AUC (over classes that have both positive and negative test rows),
        and the K x K confusion matrix.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    k = len(class_names)
    labels = np.arange(k)

    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    per_class = {
        name: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
        for i, name in enumerate(class_names)
    }

    per_class_auc: Dict[str, Optional[float]] = {}
    if y_proba is not None:
        y_proba = np.asarray(y_proba, dtype=np.float64)
        for i, name in enumerate(class_names):
            positives = y_true == i
            if positives.any() and (~positives).any():
                per_class_auc[name] = float(roc_auc_score(positives.astype(int), y_proba[:, i]))
            else:
                per_class_auc[name] = None
    valid_auc = [v for v in per_class_auc.values() if v is not None]
    roc_auc_macro = float(np.mean(valid_auc)) if valid_auc else None

    return {
        "task": "multiclass",
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "roc_auc_ovr_macro": roc_auc_macro,
        "roc_auc_ovr_per_class": per_class_auc,
        "per_class": per_class,
        "confusion_matrix": {"labels": list(class_names), "matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist()},
        "classes_absent_from_test": [name for i, name in enumerate(class_names) if s[i] == 0],
    }
