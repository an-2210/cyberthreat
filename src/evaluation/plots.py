"""
Evaluation figures: confusion matrices and binary ROC curves for test-set results.
"""

from pathlib import Path
from typing import Dict, Sequence
import numpy as np
from sklearn.metrics import roc_curve
from src.utils.plotting import new_figure, save_figure


def plot_confusion_matrix(
    matrix: Sequence[Sequence[int]],
    labels: Sequence[str],
    path: Path,
    title: str,
    normalize: bool = True,
) -> Path:
    """Heatmap of a confusion matrix. Cells show counts; colour shows the row-normalized rate."""
    cm = np.asarray(matrix, dtype=np.float64)
    row_sums = cm.sum(axis=1, keepdims=True)
    rates = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0)
    size = max(6, 0.6 * len(labels) + 3)
    fig = new_figure((size, size * 0.85))
    ax = fig.add_subplot()
    im = ax.imshow(rates if normalize else cm, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="row rate" if normalize else "count")
    ax.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)), labels=labels, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=10)
    threshold = (rates.max() if rates.size else 0) / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{int(cm[i, j]):,}", ha="center", va="center", fontsize=7,
                    color="white" if rates[i, j] > threshold else "black")
    return save_figure(fig, path)


def plot_binary_roc(curves: Dict[str, Dict[str, np.ndarray]], path: Path, title: str) -> Path:
    """Overlays ROC curves. `curves` maps a model name to {'y_true': ..., 'y_score': ...}."""
    fig = new_figure((6.5, 6))
    ax = fig.add_subplot()
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1, label="chance")
    for name, data in curves.items():
        fpr, tpr, _ = roc_curve(np.asarray(data["y_true"]).astype(int), np.asarray(data["y_score"], dtype=np.float64))
        ax.plot(fpr, tpr, label=name)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (detection rate)")
    ax.set_title(title, fontsize=10)
    ax.legend(loc="lower right")
    return save_figure(fig, path)
