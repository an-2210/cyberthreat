"""
Hyperparameter optimization tuner for XGBoost classifier using cross-validation.
"""

from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from src.models.xgboost_model import XGBoostCyberClassifier
from src.utils.logging import setup_logger

logger = setup_logger("hyperparam_tuner")


class XGBoostTuner:
    """Hyperparameter search and optimization for XGBoost model."""

    def __init__(
        self,
        param_distributions: Optional[Dict[str, list]] = None,
        n_iter: int = 10,
        cv: int = 3,
        scoring: str = "f1_macro",
        random_state: int = 42,
    ):
        self.param_distributions = param_distributions or {
            "n_estimators": [100, 200, 300],
            "max_depth": [4, 6, 8, 10],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.7, 0.8, 0.9],
            "colsample_bytree": [0.7, 0.8, 0.9],
        }
        self.n_iter = n_iter
        self.cv = cv
        self.scoring = scoring
        self.random_state = random_state

        self.best_params_: Optional[Dict[str, Any]] = None
        self.best_score_: Optional[float] = None
        self.search_results_df_: Optional[pd.DataFrame] = None

    def tune(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> Tuple[XGBoostCyberClassifier, Dict[str, Any]]:
        """Executes randomized search cross-validation to discover optimal hyperparameters.

        Args:
            X_train: Training feature DataFrame.
            y_train: Training target label Series.

        Returns:
            Tuple of (fitted_best_model, best_params_dict).
        """
        logger.info(f"Starting XGBoost hyperparameter tuning ({self.n_iter} iterations, {self.cv}-fold CV)...")
        base_xgb = XGBClassifier(random_state=self.random_state, n_jobs=-1, eval_metric="logloss")

        search = RandomizedSearchCV(
            estimator=base_xgb,
            param_distributions=self.param_distributions,
            n_iter=self.n_iter,
            cv=self.cv,
            scoring=self.scoring,
            random_state=self.random_state,
            n_jobs=-1,
        )

        search.fit(X_train, y_train)

        self.best_params_ = search.best_params_
        self.best_score_ = float(search.best_score_)
        self.search_results_df_ = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")

        logger.info(f"Best CV {self.scoring} score: {self.best_score_:.6f}")
        logger.info(f"Best Hyperparameters: {self.best_params_}")

        best_model = XGBoostCyberClassifier(**self.best_params_, random_state=self.random_state)
        best_model.fit(X_train, y_train)

        return best_model, self.best_params_
