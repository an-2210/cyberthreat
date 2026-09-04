"""
XGBoost advanced supervised classifier for cyber attack detection.
"""

from typing import Any, Dict, Optional, Union
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from src.models.baselines import BaseCyberModel
from src.utils.logging import setup_logger

logger = setup_logger("xgboost_model")


class XGBoostCyberClassifier(BaseCyberModel):
    """Empirically justified strong gradient boosted tree classifier for attack classification."""

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 8,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        n_jobs: int = -1,
        eval_metric: str = "logloss",
        early_stopping_rounds: Optional[int] = 15,
    ):
        super().__init__("XGBoost")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.eval_metric = eval_metric
        self.early_stopping_rounds = early_stopping_rounds

        self.model = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            eval_metric=self.eval_metric,
            early_stopping_rounds=self.early_stopping_rounds,
        )
        self.feature_importances_: Optional[np.ndarray] = None

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        eval_set: Optional[list] = None,
    ):
        """Fits XGBoost classifier with optional early stopping validation set."""
        logger.info(f"Training XGBoost classifier on {X.shape[0]} samples with max_depth={self.max_depth}...")

        if eval_set is not None:
            self.model.fit(X, y, eval_set=eval_set, verbose=False)
        else:
            # Disable early stopping if no eval_set is supplied
            self.model.early_stopping_rounds = None
            self.model.fit(X, y, verbose=False)

        self.is_fitted = True
        self.feature_importances_ = self.model.feature_importances_
        logger.info("XGBoost classifier fit completed.")
        return self

    def get_feature_importances(self, feature_names: Optional[list] = None) -> pd.Series:
        """Returns sorted feature importance pandas Series."""
        if not self.is_fitted or self.feature_importances_ is None:
            raise RuntimeError("Model must be fitted before fetching feature importances.")
        names = feature_names if feature_names is not None else [f"feature_{i}" for i in range(len(self.feature_importances_))]
        s = pd.Series(self.feature_importances_, index=names).sort_values(ascending=False)
        return s
