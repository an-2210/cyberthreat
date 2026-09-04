"""
Standardized machine learning baseline classifier wrappers.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from src.utils.config import get_absolute_path
from src.utils.logging import setup_logger

logger = setup_logger("baseline_models")


class BaseCyberModel(ABC):
    """Abstract Base Class for security detection models."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.is_fitted = False
        self.model: Any = None

    @abstractmethod
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]):
        """Fits the underlying estimator."""
        pass

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predicts class labels."""
        if not self.is_fitted or self.model is None:
            raise RuntimeError(f"Model '{self.model_name}' must be fitted before calling predict.")
        return self.model.predict(X)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predicts class probabilities if supported."""
        if not self.is_fitted or self.model is None:
            raise RuntimeError(f"Model '{self.model_name}' must be fitted before calling predict_proba.")
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        elif hasattr(self.model, "decision_function"):
            df = self.model.decision_function(X)
            # Sigmoid conversion for 1D decision function
            proba_pos = 1.0 / (1.0 + np.exp(-df))
            return np.vstack([1.0 - proba_pos, proba_pos]).T
        else:
            raise NotImplementedError(f"Model '{self.model_name}' does not support probability estimation.")

    def save_model(self, artifact_path: Union[str, Path]) -> Path:
        """Saves fitted model checkpoint to disk via joblib."""
        path = get_absolute_path(artifact_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"Saved model '{self.model_name}' to: {path}")
        return path

    @classmethod
    def load_model(cls, artifact_path: Union[str, Path]) -> "BaseCyberModel":
        """Loads fitted model checkpoint from disk via joblib."""
        path = get_absolute_path(artifact_path)
        if not path.exists():
            raise FileNotFoundError(f"Model checkpoint not found at: {path}")
        model_obj = joblib.load(path)
        logger.info(f"Loaded model checkpoint from: {path}")
        return model_obj


class LogisticRegressionModel(BaseCyberModel):
    """Logistic Regression baseline classifier."""

    def __init__(self, max_iter: int = 1000, C: float = 1.0, solver: str = "lbfgs", random_state: int = 42):
        super().__init__("LogisticRegression")
        self.model = LogisticRegression(max_iter=max_iter, C=C, solver=solver, random_state=random_state)

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]):
        logger.info(f"Training {self.model_name} on {X.shape[0]} samples...")
        self.model.fit(X, y)
        self.is_fitted = True
        return self


class RandomForestModel(BaseCyberModel):
    """Random Forest baseline classifier."""

    def __init__(self, n_estimators: int = 100, max_depth: Optional[int] = 20, random_state: int = 42, n_jobs: int = -1):
        super().__init__("RandomForest")
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, n_jobs=n_jobs
        )

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]):
        logger.info(f"Training {self.model_name} on {X.shape[0]} samples...")
        self.model.fit(X, y)
        self.is_fitted = True
        return self


class SVMModel(BaseCyberModel):
    """Support Vector Machine baseline classifier."""

    def __init__(self, C: float = 1.0, kernel: str = "rbf", probability: bool = True, max_iter: int = 2000, random_state: int = 42):
        super().__init__("SVM")
        self.model = SVC(C=C, kernel=kernel, probability=probability, max_iter=max_iter, random_state=random_state)

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]):
        logger.info(f"Training {self.model_name} on {X.shape[0]} samples...")
        self.model.fit(X, y)
        self.is_fitted = True
        return self


class DecisionTreeModel(BaseCyberModel):
    """Decision Tree baseline classifier."""

    def __init__(self, max_depth: Optional[int] = 15, random_state: int = 42):
        super().__init__("DecisionTree")
        self.model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]):
        logger.info(f"Training {self.model_name} on {X.shape[0]} samples...")
        self.model.fit(X, y)
        self.is_fitted = True
        return self
