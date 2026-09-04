"""
Data quality validation module for cybersecurity tabular datasets.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from src.utils.logging import setup_logger

logger = setup_logger("data_validator")


class DataValidator:
    """Validates dataframe integrity, column data types, missing/infinite values, and class distributions."""

    def __init__(self, df: pd.DataFrame, dataset_name: str = "Dataset", target_col: Optional[str] = None):
        """
        Args:
            df: Input pandas DataFrame to validate.
            dataset_name: Identifier name for logging.
            target_col: Target label column name if present.
        """
        self.df = df
        self.dataset_name = dataset_name
        self.target_col = target_col

    def run_full_validation(self) -> Dict[str, Any]:
        """Runs comprehensive validation checks and returns a summary report dictionary.

        Returns:
            Dictionary containing metrics for shape, data types, missing values,
            infinite values, duplicate rows, constant features, and label distribution.
        """
        logger.info(f"Starting data quality validation for dataset: {self.dataset_name}")
        report: Dict[str, Any] = {
            "dataset_name": self.dataset_name,
            "n_rows": int(self.df.shape[0]),
            "n_columns": int(self.df.shape[1]),
            "column_names": list(self.df.columns),
            "dtypes_summary": self._check_dtypes(),
            "missing_values": self._check_missing_values(),
            "infinite_values": self._check_infinite_values(),
            "duplicate_rows": self._check_duplicate_rows(),
            "constant_features": self._check_constant_features(),
            "label_distribution": self._check_label_distribution(),
        }

        self._log_summary(report)
        return report

    def _check_dtypes(self) -> Dict[str, int]:
        dtypes_counts = self.df.dtypes.value_counts()
        return {str(k): int(v) for k, v in dtypes_counts.items()}

    def _check_missing_values(self) -> Dict[str, Any]:
        null_counts = self.df.isnull().sum()
        cols_with_nulls = null_counts[null_counts > 0]
        total_missing = int(null_counts.sum())
        return {
            "total_missing_cells": total_missing,
            "features_with_missing": {col: int(cnt) for col, cnt in cols_with_nulls.items()},
        }

    def _check_infinite_values(self) -> Dict[str, Any]:
        num_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(num_cols) == 0:
            return {"total_infinite_cells": 0, "features_with_inf": {}}

        inf_counts = np.isinf(self.df[num_cols]).sum()
        cols_with_inf = inf_counts[inf_counts > 0]
        total_inf = int(inf_counts.sum())
        return {
            "total_infinite_cells": total_inf,
            "features_with_inf": {col: int(cnt) for col, cnt in cols_with_inf.items()},
        }

    def _check_duplicate_rows(self) -> Dict[str, Any]:
        n_duplicates = int(self.df.duplicated().sum())
        return {
            "count": n_duplicates,
            "percentage": float(round((n_duplicates / len(self.df)) * 100, 4)) if len(self.df) > 0 else 0.0,
        }

    def _check_constant_features(self) -> List[str]:
        n_unique = self.df.nunique(dropna=False)
        constant_cols = n_unique[n_unique <= 1].index.tolist()
        return list(constant_cols)

    def _check_label_distribution(self) -> Dict[str, Any]:
        if not self.target_col or self.target_col not in self.df.columns:
            return {"status": "Target column not specified or not present in DataFrame"}

        counts = self.df[self.target_col].value_counts(dropna=False)
        proportions = self.df[self.target_col].value_counts(dropna=False, normalize=True)
        return {
            "counts": {str(k): int(v) for k, v in counts.items()},
            "proportions": {str(k): float(round(v, 6)) for k, v in proportions.items()},
            "n_classes": int(self.df[self.target_col].nunique()),
        }

    def _log_summary(self, report: Dict[str, Any]) -> None:
        logger.info(f"--- Data Quality Validation Summary [{self.dataset_name}] ---")
        logger.info(f"Shape: {report['n_rows']} rows x {report['n_columns']} columns")
        logger.info(f"Duplicate Rows: {report['duplicate_rows']['count']} ({report['duplicate_rows']['percentage']}%)")
        logger.info(f"Total Missing Cells: {report['missing_values']['total_missing_cells']}")
        logger.info(f"Total Infinite Cells: {report['infinite_values']['total_infinite_cells']}")
        logger.info(f"Constant Features ({len(report['constant_features'])}): {report['constant_features']}")
        if "counts" in report["label_distribution"]:
            logger.info(f"Class Count ({report['label_distribution']['n_classes']} classes): {report['label_distribution']['counts']}")
