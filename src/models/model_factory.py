"""
Model factory for instantiating baseline and advanced classifiers.
"""

from typing import Any, Dict, Optional
from src.models.baselines import (
    BaseCyberModel,
    DecisionTreeModel,
    LogisticRegressionModel,
    RandomForestModel,
    SVMModel,
)
from src.utils.logging import setup_logger

logger = setup_logger("model_factory")


class ModelFactory:
    """Factory class to create machine learning models using string identifiers and config dictionaries."""

    @staticmethod
    def create_model(model_type: str, config: Optional[Dict[str, Any]] = None) -> BaseCyberModel:
        """Instantiates and returns a model instance.

        Args:
            model_type: Identifier name (e.g. 'logistic_regression', 'random_forest', 'svm', 'decision_tree').
            config: Optional dictionary of hyperparameter overrides.

        Returns:
            Instance of BaseCyberModel subclass.
        """
        cfg = config or {}
        model_type_norm = model_type.strip().lower()

        if model_type_norm in ["logistic_regression", "logisticregression", "lr"]:
            return LogisticRegressionModel(
                max_iter=cfg.get("max_iter", 1000),
                C=cfg.get("C", 1.0),
                solver=cfg.get("solver", "lbfgs"),
                random_state=cfg.get("random_state", 42),
            )
        elif model_type_norm in ["random_forest", "randomforest", "rf"]:
            return RandomForestModel(
                n_estimators=cfg.get("n_estimators", 100),
                max_depth=cfg.get("max_depth", 20),
                random_state=cfg.get("random_state", 42),
                n_jobs=cfg.get("n_jobs", -1),
            )
        elif model_type_norm in ["svm", "svc"]:
            return SVMModel(
                C=cfg.get("C", 1.0),
                kernel=cfg.get("kernel", "rbf"),
                max_iter=cfg.get("max_iter", 2000),
                random_state=cfg.get("random_state", 42),
            )
        elif model_type_norm in ["decision_tree", "decisiontree", "dt"]:
            return DecisionTreeModel(
                max_depth=cfg.get("max_depth", 15),
                random_state=cfg.get("random_state", 42),
            )
        else:
            raise ValueError(f"Unknown model_type: '{model_type}'. Available: ['logistic_regression', 'random_forest', 'svm', 'decision_tree']")
