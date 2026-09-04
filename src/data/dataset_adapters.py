"""
Dataset adapters to standardize column names, labels, and schemas across datasets.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
import pandas as pd
from src.utils.logging import setup_logger

logger = setup_logger("dataset_adapter")


class BaseDatasetAdapter(ABC):
    """Abstract Base Adapter for network intrusion detection datasets."""

    @abstractmethod
    def clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Strips whitespace, normalizes non-printable characters in column headers."""
        pass

    @abstractmethod
    def extract_labels(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Separates features X, binary labels y_binary (0=BENIGN, 1=ATTACK), and multiclass labels y_multi."""
        pass


class CICIDS2017Adapter(BaseDatasetAdapter):
    """Adapter for CIC-IDS2017 dataset."""

    def __init__(self, target_column: str = "Label"):
        self.target_column = target_column

    def clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Removes leading/trailing spaces from column names (a common bug in CIC-IDS2017 CSV files)."""
        df = df.copy()
        df.columns = df.columns.str.strip()
        return df

    def extract_labels(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Extracts binary and multiclass target labels for CIC-IDS2017."""
        df_clean = self.clean_column_names(df)
        if self.target_column not in df_clean.columns:
            raise KeyError(f"Target column '{self.target_column}' not found in DataFrame. Existing columns: {list(df_clean.columns)}")

        y_multi = df_clean[self.target_column].astype(str).str.strip()
        y_binary = (y_multi.str.upper() != "BENIGN").astype(int)

        X = df_clean.drop(columns=[self.target_column])
        return X, y_binary, y_multi


class UNSWNB15Adapter(BaseDatasetAdapter):
    """Adapter for UNSW-NB15 dataset."""

    def __init__(self, target_column: str = "label", cat_target_column: str = "attack_cat"):
        self.target_column = target_column
        self.cat_target_column = cat_target_column

    def clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Strips whitespace and normalizes column headers for UNSW-NB15."""
        df = df.copy()
        df.columns = df.columns.str.strip().str.lower()
        return df

    def extract_labels(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Extracts binary and multiclass target labels for UNSW-NB15."""
        df_clean = self.clean_column_names(df)
        
        # Check target column names
        target_col = self.target_column.lower()
        cat_col = self.cat_target_column.lower()

        if target_col not in df_clean.columns:
            raise KeyError(f"Target binary column '{target_col}' not found in UNSW-NB15 DataFrame.")

        y_binary = df_clean[target_col].astype(int)
        
        if cat_col in df_clean.columns:
            y_multi = df_clean[cat_col].astype(str).str.strip()
            drop_cols = [target_col, cat_col]
        else:
            y_multi = y_binary.astype(str)
            drop_cols = [target_col]

        # Exclude 'id' column if present
        if "id" in df_clean.columns:
            drop_cols.append("id")

        X = df_clean.drop(columns=[c for c in drop_cols if c in df_clean.columns])
        return X, y_binary, y_multi
