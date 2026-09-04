"""
Data cleaning transformer for tabular network traffic data.
"""

from typing import List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from src.utils.logging import setup_logger

logger = setup_logger("cleaning_transformer")


class DataFrameCleaner(BaseEstimator, TransformerMixin):
    """Transformer for handling infinite values, missing value imputation, and constant feature removal safely."""

    def __init__(
        self,
        impute_strategy: str = "median",
        remove_constant: bool = True,
        constant_variance_threshold: float = 0.0,
    ):
        """
        Args:
            impute_strategy: Imputation strategy for numerical missing values ('median' or 'mean').
            remove_constant: Whether to identify and drop zero-variance constant columns.
            constant_variance_threshold: Threshold below which features are considered constant.
        """
        self.impute_strategy = impute_strategy
        self.remove_constant = remove_constant
        self.constant_variance_threshold = constant_variance_threshold

        self.constant_cols_: List[str] = []
        self.impute_values_: dict = {}
        self.numeric_cols_: List[str] = []
        self.categorical_cols_: List[str] = []
        self.feature_names_in_: List[str] = []
        self.feature_names_out_: List[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[Union[pd.Series, np.ndarray]] = None):
        """Learns imputation values and constant columns strictly from training DataFrame X.

        Args:
            X: Input training feature DataFrame.
            y: Ignored.

        Returns:
            self
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.feature_names_in_ = list(X.columns)

        # Identify numeric vs categorical columns
        self.numeric_cols_ = list(X.select_dtypes(include=[np.number]).columns)
        self.categorical_cols_ = list(X.select_dtypes(exclude=[np.number]).columns)

        # Clean infs to compute stats safely
        X_clean = X.copy()
        for col in self.numeric_cols_:
            X_clean[col] = X_clean[col].replace([np.inf, -np.inf], np.nan)

        # Calculate numerical imputation values strictly on train set
        self.impute_values_ = {}
        for col in self.numeric_cols_:
            if self.impute_strategy == "median":
                val = X_clean[col].median()
            else:
                val = X_clean[col].mean()
            self.impute_values_[col] = 0.0 if pd.isna(val) else float(val)

        # Categorical mode imputation
        for col in self.categorical_cols_:
            mode_series = X_clean[col].mode()
            self.impute_values_[col] = mode_series.iloc[0] if len(mode_series) > 0 else "UNKNOWN"

        # Identify constant columns strictly from train set
        self.constant_cols_ = []
        if self.remove_constant:
            for col in X_clean.columns:
                n_unique = X_clean[col].nunique(dropna=True)
                if n_unique <= 1:
                    self.constant_cols_.append(col)
                elif col in self.numeric_cols_:
                    var = X_clean[col].var()
                    if pd.isna(var) or var <= self.constant_variance_threshold:
                        if col not in self.constant_cols_:
                            self.constant_cols_.append(col)

        self.feature_names_out_ = [col for col in self.feature_names_in_ if col not in self.constant_cols_]
        logger.info(
            f"DataFrameCleaner fit completed: {len(self.constant_cols_)} constant columns identified out of {len(self.feature_names_in_)} total."
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transforms input DataFrame X using precomputed training statistics.

        Args:
            X: Input feature DataFrame.

        Returns:
            Cleaned pandas DataFrame.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_in_)

        X_out = X.copy()

        # 1. Replace infinite values with NaN
        num_cols = [c for c in self.numeric_cols_ if c in X_out.columns]
        for col in num_cols:
            X_out[col] = X_out[col].replace([np.inf, -np.inf], np.nan)

        # 2. Impute NaNs using training-computed imputation dictionary
        for col, fill_val in self.impute_values_.items():
            if col in X_out.columns:
                X_out[col] = X_out[col].fillna(fill_val)

        # 3. Drop constant features identified during fit
        if self.constant_cols_:
            cols_to_drop = [c for c in self.constant_cols_ if c in X_out.columns]
            X_out = X_out.drop(columns=cols_to_drop)

        return X_out

    def get_feature_names_out(self, input_features=None) -> List[str]:
        """Returns feature names after cleaning transformations."""
        return self.feature_names_out_
