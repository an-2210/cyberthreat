"""
Data loading utilities for CIC-IDS2017 and UNSW-NB15 datasets.
"""

import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
from src.utils.config import get_absolute_path, load_config
from src.utils.logging import setup_logger

logger = setup_logger("data_loader")

# The eight MachineLearningCVE CSVs distributed with CIC-IDS2017 (the 79-column version without flow identifiers).
CICIDS2017_EXPECTED_FILES: Tuple[str, ...] = (
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
)


class DatasetNotFoundError(FileNotFoundError):
    """Exception raised when required dataset raw files are missing."""
    pass


def read_cicids2017_csv(file_path: Union[str, Path]) -> pd.DataFrame:
    """Reads one CIC-IDS2017 CSV with stripped column names.

    The raw headers carry leading spaces (e.g. ' Destination Port'). Undecodable bytes
    (the en dash in 'Web Attack – Brute Force' is stored as U+FFFD) are replaced, not fatal.
    """
    df = pd.read_csv(file_path, low_memory=False, encoding="utf-8", encoding_errors="replace")
    df.columns = df.columns.str.strip()
    return df


def drop_identical_duplicate_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Removes pandas-mangled duplicate headers ('X.1') whose values equal the original column 'X'.

    CIC-IDS2017 contains 'Fwd Header Length' twice in every file. Identical copies are dropped;
    non-identical copies are kept so that no information is silently lost.

    Returns:
        (DataFrame, dropped column names, duplicate columns kept because they differ).
    """
    dropped, kept_different = [], []
    for col in list(df.columns):
        match = re.fullmatch(r"(.+)\.(\d+)", col)
        if not match or match.group(1) not in df.columns:
            continue
        if df[col].equals(df[match.group(1)]):
            df = df.drop(columns=col)
            dropped.append(col)
        else:
            kept_different.append(col)
    return df, dropped, kept_different


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
    """DataLoader for CIC-IDS2017 network traffic dataset.

    Reads each CSV separately (bounded peak memory), strips header whitespace, removes
    identical duplicate columns and downcasts numerics to float32/int32 before concatenation.
    After `load_merged_dataset`, `load_report_` describes what was read, skipped or repaired.
    """

    def __init__(
        self,
        raw_dir: Union[str, Path] = "data/raw/CICIDS2017",
        expected_files: Sequence[str] = CICIDS2017_EXPECTED_FILES,
    ):
        self.raw_dir = get_absolute_path(raw_dir)
        self.expected_files = tuple(expected_files)
        self.load_report_: Dict[str, Any] = {}

    def list_available_files(self) -> List[Path]:
        """Lists CIC-IDS2017 CSV files under the raw directory, including sub-folders such as MachineLearningCVE/.

        If the same file name exists in two places, only the first (sorted) copy is kept, so no rows are loaded twice.
        """
        if not self.raw_dir.exists():
            return []
        unique: Dict[str, Path] = {}
        for path in sorted(self.raw_dir.rglob("*.csv")):
            if path.name in unique:
                logger.warning(f"Ignoring duplicate copy of {path.name} at {path}; using {unique[path.name]}.")
                continue
            unique[path.name] = path
        return list(unique.values())

    def find_missing_files(self) -> List[str]:
        """Returns expected CIC-IDS2017 file names that are not present in the raw directory."""
        present = {p.name for p in self.list_available_files()}
        return [name for name in self.expected_files if name not in present]

    def load_merged_dataset(
        self,
        sample_frac: Optional[float] = None,
        optimize_memory: bool = True,
        random_state: int = 42,
        add_source_column: bool = False,
    ) -> pd.DataFrame:
        """Loads and concatenates all CIC-IDS2017 CSV files present in the raw directory.

        Missing expected files are reported in `load_report_` and logged; loading only fails
        when no CSV file at all is present.

        Args:
            sample_frac: Optional fraction (0.0 to 1.0) to subsample each file, for quick EDA/debugging.
            optimize_memory: Whether to downcast float64/int64 data types.
            random_state: Seed used by `sample_frac`.
            add_source_column: Adds a 'source_file' column (used by the audit; drop before modelling).

        Returns:
            Merged pandas DataFrame with stripped column names.
        """
        csv_files = self.list_available_files()
        if not csv_files:
            raise DatasetNotFoundError(
                f"No CSV files found in directory: {self.raw_dir}.\n"
                f"Please place official CIC-IDS2017 CSV files (e.g., 'Monday-WorkingHours.pcap_ISCX.csv') "
                f"into '{self.raw_dir}'."
            )

        missing = self.find_missing_files()
        unexpected = [p.name for p in csv_files if p.name not in self.expected_files]
        if missing:
            logger.warning(f"Missing {len(missing)} expected CIC-IDS2017 file(s): {missing}. Continuing with available files.")
        if unexpected:
            logger.warning(f"Unexpected CSV file(s) found and included: {unexpected}")

        logger.info(f"Loading {len(csv_files)} CIC-IDS2017 dataset CSV file(s) from {self.raw_dir}...")
        dfs: List[pd.DataFrame] = []
        per_file: List[Dict[str, Any]] = []
        reference_columns: Optional[List[str]] = None
        column_mismatch: List[str] = []

        for file_path in csv_files:
            started = time.perf_counter()
            logger.info(f"  Reading {file_path.name}...")
            df = read_cicids2017_csv(file_path)
            df, dropped, kept_different = drop_identical_duplicate_columns(df)
            rows_raw = int(len(df))

            if sample_frac and 0.0 < sample_frac < 1.0:
                df = df.sample(frac=sample_frac, random_state=random_state)
            if optimize_memory:
                df = downcast_numeric_dtypes(df)
            if add_source_column:
                df["source_file"] = file_path.name

            if reference_columns is None:
                reference_columns = list(df.columns)
            elif list(df.columns) != reference_columns:
                column_mismatch.append(file_path.name)
                logger.warning(f"Column layout of {file_path.name} differs from the first file; unmatched columns become NaN.")

            per_file.append({
                "file": file_path.name,
                "rows": rows_raw,
                "rows_loaded": int(len(df)),
                "columns": int(df.shape[1]),
                "duplicate_columns_dropped": dropped,
                "duplicate_columns_kept_different_values": kept_different,
                "read_seconds": round(time.perf_counter() - started, 3),
            })
            dfs.append(df)

        merged_df = pd.concat(dfs, ignore_index=True)
        del dfs
        self.load_report_ = {
            "raw_dir": str(self.raw_dir),
            "expected_files": list(self.expected_files),
            "files_loaded": [p.name for p in csv_files],
            "missing_files": missing,
            "unexpected_files": unexpected,
            "column_layout_mismatch_files": column_mismatch,
            "per_file": per_file,
            "total_rows": int(len(merged_df)),
            "total_columns": int(merged_df.shape[1]),
        }
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
