"""
Inference interface for the saved CIC-IDS2017 models.

A detector takes RAW CIC-IDS2017-style rows (stripped column names, the 77 feature columns listed in
the schema, no label column), applies the saved preprocessing pipeline fitted on the training split,
and returns predictions from the saved classifier. Predictions and probabilities always use the
labels listed in the schema, in that order.

Example:
    from src.models.inference import CICIDS2017Detector
    detector = CICIDS2017Detector.load("binary", "xgboost")
    labels = detector.predict(raw_df)          # array of 'BENIGN' / 'ATTACK'
    proba = detector.predict_proba(raw_df)     # shape (n_rows, 2), columns = detector.labels
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
import joblib
import numpy as np
import pandas as pd
from src.utils.config import get_absolute_path
from src.utils.io_utils import save_json
from src.utils.logging import setup_logger

logger = setup_logger("cicids_inference")

SCHEMA_FILENAME = "cicids2017_schema.json"
PIPELINE_FILENAME = "cicids2017_preprocessing_pipeline.joblib"
TASKS = ("binary", "multiclass")
MODEL_NAMES = ("random_forest", "xgboost")
BINARY_LABELS = ["BENIGN", "ATTACK"]


def build_schema(class_names: Sequence[str], pipeline: Any, artifacts_dir: Union[str, Path]) -> Dict[str, Any]:
    """Describes the saved artifacts: input columns, model features, labels and file paths."""
    artifacts_dir = Path(artifacts_dir)
    return {
        "dataset": "CIC-IDS2017 (MachineLearningCVE)",
        "input_columns": list(pipeline.cleaner.feature_names_in_),
        "model_features": list(pipeline.feature_names_out_),
        "label_column_expected_in_input": False,
        "tasks": {
            "binary": {"labels": BINARY_LABELS, "positive_label": "ATTACK"},
            "multiclass": {"labels": list(class_names), "positive_label": None},
        },
        "artifacts": {
            "preprocessing_pipeline": PIPELINE_FILENAME,
            **{f"{t}_{m}": f"cicids2017_{t}_{m}.joblib" for t in TASKS for m in MODEL_NAMES},
        },
        "artifacts_dir": str(artifacts_dir),
        "notes": (
            "Preprocessing (cleaning, feature engineering, encoding, scaling) is fitted on the training split only. "
            "Inputs are converted to float32 by the detector. Duplicate-free training data; no identifier columns."
        ),
    }


def write_schema(class_names: Sequence[str], pipeline: Any, artifacts_dir: Union[str, Path]) -> Path:
    """Writes models/artifacts/cicids2017_schema.json."""
    path = get_absolute_path(artifacts_dir) / SCHEMA_FILENAME
    save_json(build_schema(class_names, pipeline, artifacts_dir), path)
    logger.info(f"Wrote model schema to {path}")
    return path


class CICIDS2017Detector:
    """Raw-row inference wrapper around a saved preprocessing pipeline and classifier."""

    def __init__(self, pipeline: Any, model: Any, task: str, labels: List[str], input_columns: List[str]):
        if task not in TASKS:
            raise ValueError(f"Unknown task '{task}'. Use one of {TASKS}.")
        self.pipeline = pipeline
        self.model = model
        self.task = task
        self.labels = list(labels)
        self.input_columns = list(input_columns)

    @classmethod
    def load(
        cls,
        task: str,
        model_name: str,
        artifacts_dir: Union[str, Path] = "models/artifacts",
    ) -> "CICIDS2017Detector":
        """Loads the saved pipeline, classifier and schema for one task and model."""
        artifacts_dir = get_absolute_path(artifacts_dir)
        if model_name not in MODEL_NAMES:
            raise ValueError(f"Unknown model '{model_name}'. Use one of {MODEL_NAMES}.")
        schema_path = artifacts_dir / SCHEMA_FILENAME
        if not schema_path.exists():
            raise FileNotFoundError(f"Schema not found at {schema_path}. Run src.run_cicids2017 or write_schema first.")
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        pipeline = joblib.load(artifacts_dir / PIPELINE_FILENAME)
        model = joblib.load(artifacts_dir / f"cicids2017_{task}_{model_name}.joblib")
        labels = schema["tasks"][task]["labels"]
        return cls(pipeline, model, task, labels, schema["input_columns"])

    def _prepare(self, X: pd.DataFrame) -> np.ndarray:
        """Validates columns, orders them as in training, and applies the fitted preprocessing."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame with the CIC-IDS2017 feature columns.")
        X = X.copy()
        X.columns = X.columns.str.strip()
        missing = [c for c in self.input_columns if c not in X.columns]
        if missing:
            raise ValueError(f"Input is missing {len(missing)} required column(s): {missing[:10]}")
        processed = self.pipeline.transform_new_data(X[self.input_columns])
        return processed.to_numpy(dtype=np.float32)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Probabilities with one column per label in `self.labels` (zeros for labels unseen in training)."""
        proba = self.model.predict_proba(self._prepare(X))
        classes = np.asarray(self.model.model.classes_).astype(int)
        full = np.zeros((proba.shape[0], len(self.labels)), dtype=np.float64)
        full[:, classes] = proba
        return full

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predicted labels as strings from `self.labels` (binary: 'BENIGN' or 'ATTACK')."""
        ids = np.asarray(self.model.predict(self._prepare(X))).astype(int)
        return np.asarray(self.labels, dtype=object)[ids]
