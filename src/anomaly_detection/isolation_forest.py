"""
Unsupervised anomaly detector for network flows, trained on benign traffic only.

The model never sees attack labels during training. It learns what normal flows look like and scores
how far a new flow sits from that norm. Flows scoring above a threshold are reported as anomalies, which
lets the system flag attack types it was never trained on.

The threshold is chosen on the validation split so that a fixed share of benign flows (the target false-positive
rate) is flagged. Scores are higher for more anomalous flows.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest

from src.utils.config import get_absolute_path
from src.utils.logging import setup_logger

logger = setup_logger("isolation_forest")


class IsolationForestDetector:
    """Isolation Forest with a score threshold calibrated on benign validation data."""

    def __init__(self, n_estimators: int = 200, max_samples: Union[int, str] = 256, seed: int = 42):
        self.model = IsolationForest(n_estimators=n_estimators, max_samples=max_samples, random_state=seed, n_jobs=-1)
        self.threshold: Optional[float] = None
        self.target_fpr: Optional[float] = None

    def fit(self, X_benign: np.ndarray) -> "IsolationForestDetector":
        """Fits on benign rows only. Labels are never used here."""
        self.model.fit(X_benign)
        logger.info(f"Fitted Isolation Forest on {len(X_benign)} benign rows.")
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Anomaly score per row: higher means more anomalous (the negated sklearn decision function)."""
        return -self.model.score_samples(X)

    def calibrate(self, X_val_benign: np.ndarray, target_fpr: float = 0.01) -> float:
        """Sets the threshold so that about `target_fpr` of benign validation rows are flagged."""
        if not 0 < target_fpr < 1:
            raise ValueError("target_fpr must be between 0 and 1.")
        self.target_fpr = float(target_fpr)
        self.threshold = float(np.quantile(self.score(X_val_benign), 1 - target_fpr))
        logger.info(f"Threshold {self.threshold:.4f} gives benign validation FPR ~ {target_fpr:.3f}.")
        return self.threshold

    def predict(self, X: np.ndarray) -> np.ndarray:
        """1 = anomaly (flagged), 0 = normal. Requires calibrate() first."""
        if self.threshold is None:
            raise RuntimeError("Call calibrate() before predict().")
        return (self.score(X) > self.threshold).astype(int)

    def save(self, path: Union[str, Path]) -> Path:
        path = get_absolute_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "threshold": self.threshold, "target_fpr": self.target_fpr}, path)
        return path

    @classmethod
    def load(cls, path: Union[str, Path]) -> "IsolationForestDetector":
        payload: Dict[str, Any] = joblib.load(get_absolute_path(path))
        detector = cls()
        detector.model = payload["model"]
        detector.threshold = payload["threshold"]
        detector.target_fpr = payload["target_fpr"]
        return detector
