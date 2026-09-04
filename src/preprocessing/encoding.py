"""
Categorical feature encoder transformer with unknown category safety.
"""

from typing import List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from src.utils.logging import setup_logger

logger = setup_logger("categorical_encoder")


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """Encodes categorical columns while keeping numeric columns unchanged."""

    def __init__(self, encoding_type: str = "onehot"):
        """
        Args:
            encoding_type: 'onehot' or 'ordinal'.
        """
        self.encoding_type = encoding_type
        self.categorical_cols_: List[str] = []
        self.numeric_cols_: List[str] = []
        self.encoder_: Optional[Union[OneHotEncoder, OrdinalEncoder]] = None
        self.feature_names_out_: List[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[Union[pd.Series, np.ndarray]] = None):
        """Fits categorical encoder strictly on training DataFrame X.

        Args:
            X: Input training DataFrame.
            y: Ignored.

        Returns:
            self
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.categorical_cols_ = list(X.select_dtypes(exclude=[np.number]).columns)
        self.numeric_cols_ = list(X.select_dtypes(include=[np.number]).columns)

        if self.categorical_cols_:
            if self.encoding_type == "onehot":
                self.encoder_ = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
                self.encoder_.fit(X[self.categorical_cols_])
                encoded_names = list(self.encoder_.get_feature_names_out(self.categorical_cols_))
            else:
                self.encoder_ = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
                self.encoder_.fit(X[self.categorical_cols_])
                encoded_names = self.categorical_cols_
            self.feature_names_out_ = self.numeric_cols_ + encoded_names
        else:
            self.encoder_ = None
            self.feature_names_out_ = list(X.columns)

        logger.info(f"CategoricalEncoder fit completed on {len(self.categorical_cols_)} categorical feature(s). Total output features: {len(self.feature_names_out_)}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transforms categorical columns using fitted encoder.

        Args:
            X: Input DataFrame.

        Returns:
            Encoded pandas DataFrame.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        if not self.categorical_cols_ or self.encoder_ is None:
            return X.copy()

        X_num = X[self.numeric_cols_].reset_index(drop=True)
        cat_data = X[self.categorical_cols_]

        if self.encoding_type == "onehot":
            encoded_arr = self.encoder_.transform(cat_data)
            encoded_names = list(self.encoder_.get_feature_names_out(self.categorical_cols_))
            X_cat = pd.DataFrame(encoded_arr, columns=encoded_names)
        else:
            encoded_arr = self.encoder_.transform(cat_data)
            X_cat = pd.DataFrame(encoded_arr, columns=self.categorical_cols_)

        X_out = pd.concat([X_num, X_cat], axis=1)
        return X_out

    def get_feature_names_out(self, input_features=None) -> List[str]:
        """Returns feature names after categorical encoding."""
        return self.feature_names_out_
