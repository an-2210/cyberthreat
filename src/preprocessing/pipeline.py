"""
End-to-end leakage-safe preprocessing pipeline orchestrator for cyber threat datasets.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.preprocessing.cleaning import DataFrameCleaner
from src.preprocessing.encoding import CategoricalEncoder
from src.preprocessing.feature_engineering import CybersecurityFeatureEngineer
from src.preprocessing.scaling import FeatureScaler
from src.utils.config import get_absolute_path
from src.utils.logging import setup_logger

logger = setup_logger("preprocessing_pipeline")


class CyberthreatPreprocessingPipeline:
    """Orchestrates stratified dataset splitting and sequential leakage-safe transformations."""

    def __init__(
        self,
        impute_strategy: str = "median",
        remove_constant: bool = True,
        scaling_method: str = "robust",
        encoding_type: str = "onehot",
        test_size: float = 0.20,
        val_size: float = 0.10,
        random_state: int = 42,
    ):
        self.cleaner = DataFrameCleaner(impute_strategy=impute_strategy, remove_constant=remove_constant)
        self.feature_engineer = CybersecurityFeatureEngineer()
        self.encoder = CategoricalEncoder(encoding_type=encoding_type)
        self.scaler = FeatureScaler(scaling_method=scaling_method)

        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.is_fitted = False
        self.feature_names_out_: list = []

    def fit_transform_splits(
        self,
        X: pd.DataFrame,
        y_binary: pd.Series,
        y_multi: Optional[pd.Series] = None,
    ) -> Dict[str, Union[pd.DataFrame, pd.Series, np.ndarray]]:
        """Splits raw dataset into train/val/test first, then fits transformers ONLY on X_train.

        Args:
            X: Input feature DataFrame.
            y_binary: Binary label Series (0=BENIGN, 1=ATTACK).
            y_multi: Optional multiclass label Series.

        Returns:
            Dictionary containing processed splits:
            X_train, y_train_bin, y_train_multi,
            X_val, y_val_bin, y_val_multi,
            X_test, y_test_bin, y_test_multi.
        """
        logger.info("Executing stratified train/val/test dataset split before fitting preprocessing...")
        
        # 1. Stratified Train / Test Split
        X_train_raw, X_temp, y_train_bin, y_temp_bin = train_test_split(
            X, y_binary, test_size=(self.test_size + self.val_size), stratify=y_binary, random_state=self.random_state
        )

        val_ratio = self.val_size / (self.test_size + self.val_size)
        X_val_raw, X_test_raw, y_val_bin, y_test_bin = train_test_split(
            X_temp, y_temp_bin, test_size=(1.0 - val_ratio), stratify=y_temp_bin, random_state=self.random_state
        )

        y_train_multi, y_val_multi, y_test_multi = None, None, None
        if y_multi is not None:
            y_train_multi = y_multi.loc[X_train_raw.index]
            y_val_multi = y_multi.loc[X_val_raw.index]
            y_test_multi = y_multi.loc[X_test_raw.index]

        logger.info(f"Split sizes: Train={len(X_train_raw)}, Val={len(X_val_raw)}, Test={len(X_test_raw)}")

        # 2. Sequential Fit & Transform strictly on Train set
        logger.info("Fitting preprocessing transformers strictly on X_train...")
        X_train_clean = self.cleaner.fit_transform(X_train_raw)
        X_train_fe = self.feature_engineer.fit_transform(X_train_clean)
        X_train_enc = self.encoder.fit_transform(X_train_fe)
        X_train_proc = self.scaler.fit_transform(X_train_enc)

        self.feature_names_out_ = list(X_train_proc.columns)
        self.is_fitted = True

        # 3. Transform Val and Test sets using fitted train statistics
        logger.info("Transforming X_val and X_test using training-fitted statistics...")
        X_val_proc = self._transform_steps(X_val_raw)
        X_test_proc = self._transform_steps(X_test_raw)

        return {
            "X_train": X_train_proc,
            "y_train_binary": y_train_bin.reset_index(drop=True),
            "y_train_multi": y_train_multi.reset_index(drop=True) if y_train_multi is not None else None,
            "X_val": X_val_proc,
            "y_val_binary": y_val_bin.reset_index(drop=True),
            "y_val_multi": y_val_multi.reset_index(drop=True) if y_val_multi is not None else None,
            "X_test": X_test_proc,
            "y_test_binary": y_test_bin.reset_index(drop=True),
            "y_test_multi": y_test_multi.reset_index(drop=True) if y_test_multi is not None else None,
        }

    def _transform_steps(self, X: pd.DataFrame) -> pd.DataFrame:
        """Internal transform sequence using fitted transformers."""
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted before calling transform!")
        X_clean = self.cleaner.transform(X)
        X_fe = self.feature_engineer.transform(X_clean)
        X_enc = self.encoder.transform(X_fe)
        X_proc = self.scaler.transform(X_enc)
        return X_proc

    def transform_new_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transforms unseen test data or real-time security events.

        Args:
            X: Raw input DataFrame.

        Returns:
            Preprocessed pandas DataFrame matching training schema.
        """
        return self._transform_steps(X)

    def save_pipeline(self, artifact_path: Union[str, Path] = "models/artifacts/preprocessing_pipeline.joblib") -> Path:
        """Saves fitted pipeline state and transformers to disk using joblib.

        Args:
            artifact_path: Output file path.

        Returns:
            Path object to saved file.
        """
        path = get_absolute_path(artifact_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"Saved fitted preprocessing pipeline to: {path}")
        return path

    @staticmethod
    def load_pipeline(artifact_path: Union[str, Path] = "models/artifacts/preprocessing_pipeline.joblib") -> "CyberthreatPreprocessingPipeline":
        """Loads fitted pipeline state from joblib artifact file.

        Args:
            artifact_path: Saved joblib file path.

        Returns:
            Fitted CyberthreatPreprocessingPipeline instance.
        """
        path = get_absolute_path(artifact_path)
        if not path.exists():
            raise FileNotFoundError(f"Preprocessing artifact not found at: {path}")
        pipeline = joblib.load(path)
        logger.info(f"Loaded preprocessing pipeline from: {path}")
        return pipeline
