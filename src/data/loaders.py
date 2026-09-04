"""
Data loading utilities for CIC-IDS2017 and UNSW-NB15 datasets.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from src.utils.config import get_absolute_path, load_config
from src.utils.logging import setup_logger

logger = setup_logger("data_loader")


class DatasetNotFoundError(FileNotFoundError):
    """Exception raised when required dataset raw files are missing."""
    pass


def downcast_numeric_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Optimizes memory usage by downcasting numeric columns to lower precision.

    Args:
        df: Input DataFrame.

    Returns:
        Memory-optimized DataFrame.
    """
    df = df.copy()
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype(np.float32)
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = df[col].astype(np.int32)
    return df


class CICIDS2017Loader:
    """DataLoader for CIC-IDS2017 network traffic dataset."""

    def __init__(self, raw_dir: Union[str, Path] = "data/raw/CICIDS2017"):
        self.raw_dir = get_absolute_path(raw_dir)

    def list_available_files(self) -> List[Path]:
        """Lists CSV files in the raw CIC-IDS2017 directory."""
        if not self.raw_dir.exists():
            return []
        return sorted(list(self.raw_dir.glob("*.csv")))

    def load_merged_dataset(
        self, sample_frac: Optional[float] = None, optimize_memory: bool = True
    ) -> pd.DataFrame:
        """Loads and concatenates all CSV files present in the raw directory.

        Args:
            sample_frac: Optional fraction (0.0 to 1.0) to subsample for fast EDA/testing.
            optimize_memory: Whether to downcast float64/int64 data types.

        Returns:
            Merged pandas DataFrame.
        """
        csv_files = self.list_available_files()
        if not csv_files:
            raise DatasetNotFoundError(
                f"No CSV files found in directory: {self.raw_dir}.\n"
                f"Please place official CIC-IDS2017 CSV files (e.g., 'Monday-WorkingHours.pcap_ISCX.csv') "
                f"into '{self.raw_dir}'."
            )

        logger.info(f"Loading {len(csv_files)} CIC-IDS2017 dataset CSV file(s) from {self.raw_dir}...")
        dfs = []
        for file_path in csv_files:
            logger.info(f"  Reading {file_path.name}...")
            df = pd.read_csv(file_path, low_memory=False)
            if sample_frac and 0.0 < sample_frac < 1.0:
                df = df.sample(frac=sample_frac, random_state=42)
            if optimize_memory:
                df = downcast_numeric_dtypes(df)
            dfs.append(df)

        merged_df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Successfully loaded CIC-IDS2017 dataset. Total shape: {merged_df.shape}")
        return merged_df


class UNSWNB15Loader:
    """DataLoader for UNSW-NB15 dataset."""

    def __init__(self, raw_dir: Union[str, Path] = "data/raw/UNSW-NB15"):
        self.raw_dir = get_absolute_path(raw_dir)

    def list_available_files(self) -> List[Path]:
        """Lists CSV files in the raw UNSW-NB15 directory."""
        if not self.raw_dir.exists():
            return []
        return sorted(list(self.raw_dir.glob("*.csv")))

    def load_dataset(
        self, train_or_test: Optional[str] = None, optimize_memory: bool = True
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        """Loads UNSW-NB15 train/test CSV files.

        Args:
            train_or_test: 'train', 'test', or None (returns both (train_df, test_df)).
            optimize_memory: Downcast precision for memory savings.

        Returns:
            Single DataFrame or tuple of (train_df, test_df).
        """
        csv_files = self.list_available_files()
        if not csv_files:
            raise DatasetNotFoundError(
                f"No CSV files found in directory: {self.raw_dir}.\n"
                f"Please place official UNSW-NB15 CSV files (e.g., 'UNSW_NB15_training-set.csv', "
                f"'UNSW_NB15_testing-set.csv') into '{self.raw_dir}'."
            )

        train_file = self.raw_dir / "UNSW_NB15_training-set.csv"
        test_file = self.raw_dir / "UNSW_NB15_testing-set.csv"

        if train_or_test == "train":
            if not train_file.exists():
                raise DatasetNotFoundError(f"Training file missing: {train_file}")
            df = pd.read_csv(train_file, low_memory=False)
            return downcast_numeric_dtypes(df) if optimize_memory else df

        if train_or_test == "test":
            if not test_file.exists():
                raise DatasetNotFoundError(f"Testing file missing: {test_file}")
            df = pd.read_csv(test_file, low_memory=False)
            return downcast_numeric_dtypes(df) if optimize_memory else df

        # Default: load all available CSVs
        dfs = []
        for f in csv_files:
            logger.info(f"Reading {f.name}...")
            df = pd.read_csv(f, low_memory=False)
            if optimize_memory:
                df = downcast_numeric_dtypes(df)
            dfs.append(df)

        merged_df = pd.concat(dfs, ignore_index=True)
        return merged_df
