"""
Numerical feature scaling transformer preserving tabular DataFrame structure.
"""

from typing import List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from src.utils.logging import setup_logger

logger = setup_logger("feature_scaler")


class FeatureScaler(BaseEstimator, TransformerMixin):
    """Scales numerical features using robust, standard, or min-max scaling."""

    def __init__(self, scaling_method: str = "robust"):
        """
        Args:
            scaling_method: 'robust', 'standard', or 'minmax'.
        """
        self.scaling_method = scaling_method
        self.numeric_cols_: List[str] = []
        self.non_numeric_cols_: List[str] = []
        self.scaler_: Optional[Union[RobustScaler, StandardScaler, MinMaxScaler]] = None
        self.feature_names_out_: List[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[Union[pd.Series, np.ndarray]] = None):
        """Fits numerical scaler strictly on training DataFrame X.

        Args:
            X: Input training DataFrame.
            y: Ignored.

        Returns:
            self
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.numeric_cols_ = list(X.select_dtypes(include=[np.number]).columns)
        self.non_numeric_cols_ = list(X.select_dtypes(exclude=[np.number]).columns)
        self.feature_names_out_ = list(X.columns)

        if self.numeric_cols_:
            if self.scaling_method == "robust":
                self.scaler_ = RobustScaler()
            elif self.scaling_method == "standard":
                self.scaler_ = StandardScaler()
            elif self.scaling_method == "minmax":
                self.scaler_ = MinMaxScaler()
            else:
                raise ValueError(f"Unsupported scaling_method: '{self.scaling_method}'")

            self.scaler_.fit(X[self.numeric_cols_])

        logger.info(f"FeatureScaler fit completed using '{self.scaling_method}' scaling on {len(self.numeric_cols_)} numeric features.")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Scales numerical columns of DataFrame X using fitted scaler parameters.

        Args:
            X: Input DataFrame.

        Returns:
            Scaled pandas DataFrame.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_out_)

        X_out = X.copy()
        if self.numeric_cols_ and self.scaler_ is not None:
            scaled_arr = self.scaler_.transform(X_out[self.numeric_cols_])
            scaled_df = pd.DataFrame(scaled_arr, columns=self.numeric_cols_, index=X_out.index)
            for col in self.numeric_cols_:
                X_out[col] = scaled_df[col].astype(np.float32)

        return X_out

    def get_feature_names_out(self, input_features=None) -> List[str]:
        """Returns feature names after scaling."""
        return self.feature_names_out_
