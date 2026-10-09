"""
Supervised classifiers for CIC-IDS2017: Random Forest and XGBoost, built from configs/config.yaml.

Binary task: BENIGN (0) vs ATTACK (1). Multiclass task: one id per label value.
XGBoost uses the VALIDATION split for early stopping; the test split is never touched during training.
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np
from src.models.baselines import BaseCyberModel, RandomForestModel
from src.models.xgboost_model import XGBoostCyberClassifier
from src.utils.logging import setup_logger

logger = setup_logger("supervised_models")

MODEL_NAMES = ("random_forest", "xgboost")


def build_classifiers(models_cfg: Dict[str, Any], task: str, seed: int = 42) -> Dict[str, BaseCyberModel]:
    """Instantiates the Random Forest and XGBoost classifiers for a task.

    Args:
        models_cfg: The `models` section of the config (ConfigDict or dict).
        task: 'binary' or 'multiclass'.
        seed: Random seed applied to both models (reproducibility).

    Returns:
        Dictionary mapping model name to an untrained model wrapper.
    """
    if task not in ("binary", "multiclass"):
        raise ValueError(f"Unknown task '{task}'. Use 'binary' or 'multiclass'.")

    rf_cfg = models_cfg["baselines"]["random_forest"]
    xgb_cfg = models_cfg["advanced"]["xgboost"]

    return {
        "random_forest": RandomForestModel(
            n_estimators=int(rf_cfg["n_estimators"]),
            max_depth=rf_cfg.get("max_depth"),
            random_state=seed,
            n_jobs=int(rf_cfg.get("n_jobs", -1)),
        ),
        "xgboost": XGBoostCyberClassifier(
            n_estimators=int(xgb_cfg["n_estimators"]),
            max_depth=int(xgb_cfg["max_depth"]),
            learning_rate=float(xgb_cfg["learning_rate"]),
            subsample=float(xgb_cfg["subsample"]),
            colsample_bytree=float(xgb_cfg["colsample_bytree"]),
            random_state=seed,
            n_jobs=int(xgb_cfg.get("n_jobs", -1)),
            eval_metric="logloss" if task == "binary" else "mlogloss",
            early_stopping_rounds=xgb_cfg.get("early_stopping_rounds", 15),
        ),
    }


def fit_classifier(
    model: BaseCyberModel,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
) -> BaseCyberModel:
    """Fits a classifier. XGBoost early-stops on the validation rows whose labels appear in training."""
    if isinstance(model, XGBoostCyberClassifier) and X_val is not None:
        known = np.isin(y_val, np.unique(y_train))
        if not known.all():
            logger.warning(f"Dropping {int((~known).sum())} validation rows whose labels are absent from training (early stopping only).")
        model.fit(X_train, y_train, eval_set=[(X_val[known], y_val[known])])
    else:
        model.fit(X_train, y_train)
    return model


def predict_full_proba(model: BaseCyberModel, X: np.ndarray, n_classes: int) -> np.ndarray:
    """Returns probabilities with one column per class 0..n_classes-1.

    A model trained on a subsample may not have seen every class; those columns are filled with zeros.
    """
    proba = model.predict_proba(X)
    classes = np.asarray(model.model.classes_).astype(int)
    full = np.zeros((proba.shape[0], n_classes), dtype=np.float64)
    full[:, classes] = proba
    return full


def model_feature_importances(model: BaseCyberModel, feature_names: Tuple[str, ...], top_k: int = 15) -> Dict[str, float]:
    """Returns the top-k feature importances (impurity-based for RF, gain-based for XGBoost)."""
    importances = getattr(model.model, "feature_importances_", None)
    if importances is None:
        return {}
    order = np.argsort(importances)[::-1][:top_k]
    return {feature_names[i]: float(importances[i]) for i in order}
