"""
CIC-IDS2017 data quality audit: missing values, infinities, duplicates, invalid values,
constant features and class distribution.

The audit runs on the merged raw data before any cleaning, so the reports describe the
dataset as delivered. It reports problems; it does not remove rows (cleaning happens later,
fitted on training data only).
"""

from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
from src.data.validators import DataValidator
from src.utils.io_utils import save_json
from src.utils.logging import setup_logger

logger = setup_logger("cicids_audit")

# The 15 label values published with the MachineLearningCVE CSVs (after label normalization).
CICIDS2017_KNOWN_LABELS = frozenset({
    "BENIGN", "DoS Hulk", "PortScan", "DDoS", "DoS GoldenEye", "FTP-Patator", "SSH-Patator",
    "DoS slowloris", "DoS Slowhttptest", "Bot", "Web Attack - Brute Force", "Web Attack - XSS",
    "Infiltration", "Web Attack - Sql Injection", "Heartbleed",
})

PORT_COLUMN = "Destination Port"


def audit_cicids2017(
    features: pd.DataFrame,
    labels: pd.Series,
    source_files: Optional[pd.Series] = None,
    loader_report: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Audits merged CIC-IDS2017 data.

    Args:
        features: Feature columns with stripped names (no label column).
        labels: Normalized multiclass label per row (e.g. 'BENIGN', 'DoS Hulk').
        source_files: Optional per-row source file name, used for the per-file composition table.
        loader_report: Optional `CICIDS2017Loader.load_report_`, embedded in the report.

    Returns:
        Dictionary with keys 'report' (JSON-serializable), 'column_quality',
        'class_distribution' and 'class_composition' (DataFrames; composition is None without source_files).
    """
    n_rows = int(len(features))
    labels = labels.reset_index(drop=True)

    # 1. Global checks (missing, infinite, exact duplicates, constant columns) via the shared validator.
    validator_df = features.assign(label=labels.to_numpy())
    base = DataValidator(validator_df, dataset_name="CIC-IDS2017", target_col="label").run_full_validation()
    del validator_df

    # 2. Per-column quality table.
    numeric = features.select_dtypes(include=[np.number])
    infinite = pd.Series(0, index=features.columns, dtype="int64")
    negative = pd.Series(0, index=features.columns, dtype="int64")
    infinite[numeric.columns] = np.isinf(numeric).sum().astype("int64")
    negative[numeric.columns] = (numeric < 0).sum().astype("int64")
    missing = features.isna().sum().astype("int64")
    column_quality = pd.DataFrame({
        "dtype": features.dtypes.astype(str),
        "missing": missing,
        "missing_pct": (missing / max(n_rows, 1) * 100).round(4),
        "infinite": infinite,
        "negative": negative,
    })
    column_quality.index.name = "column"

    # 3. Invalid values: ports outside [0, 65535]; negatives are reported, not removed.
    invalid_port = 0
    if PORT_COLUMN in features.columns:
        ports = features[PORT_COLUMN]
        invalid_port = int(((ports < 0) | (ports > 65535)).sum())
    columns_with_negatives = {c: int(v) for c, v in negative[negative > 0].items()}

    # 4. Feature rows that appear with more than one label (label noise; not removed).
    hashes = pd.util.hash_pandas_object(features, index=False)
    dup_mask = hashes.duplicated(keep=False).to_numpy()
    conflict_groups, conflict_rows = 0, 0
    if dup_mask.any():
        tmp = pd.DataFrame({"h": hashes.to_numpy()[dup_mask], "y": labels.to_numpy()[dup_mask]})
        n_labels = tmp.groupby("h")["y"].nunique()
        conflicting = n_labels[n_labels > 1].index
        conflict_groups = int(len(conflicting))
        conflict_rows = int(tmp["h"].isin(conflicting).sum())
    del hashes

    # 5. Class distribution and per-file composition.
    counts = labels.value_counts(dropna=False)
    class_distribution = pd.DataFrame({
        "label": counts.index.astype(str),
        "count": counts.to_numpy(),
        "percent": (counts.to_numpy() / max(n_rows, 1) * 100).round(4),
    }).reset_index(drop=True)

    class_composition = None
    if source_files is not None:
        class_composition = pd.crosstab(
            source_files.reset_index(drop=True).to_numpy(), labels.to_numpy()
        )
        class_composition.index.name = "source_file"

    unexpected_labels = sorted(set(map(str, counts.index)) - CICIDS2017_KNOWN_LABELS)

    report = {
        "dataset": "CIC-IDS2017 (MachineLearningCVE CSVs, merged, before cleaning)",
        "n_rows": n_rows,
        "n_feature_columns": int(features.shape[1]),
        "dtypes_summary": base["dtypes_summary"],
        "loader": loader_report,
        "missing_values": {
            "total_missing_cells": int(missing.sum()),
            "features_with_missing": {c: int(v) for c, v in missing[missing > 0].items()},
        },
        "infinite_values": {
            "total_infinite_cells": int(infinite.sum()),
            "features_with_infinite": {c: int(v) for c, v in infinite[infinite > 0].items()},
        },
        "duplicate_rows_exact_including_label": base["duplicate_rows"],
        "duplicate_feature_rows_with_conflicting_labels": {
            "distinct_feature_vectors": conflict_groups,
            "rows_involved": conflict_rows,
        },
        "constant_features": base["constant_features"],
        "invalid_values": {
            "destination_port_outside_0_65535": invalid_port,
            "columns_with_negative_values": columns_with_negatives,
            "note": (
                "Negative values are reported, not removed. CIC-IDS2017 uses -1 as a sentinel in "
                "some columns (e.g. Init_Win_bytes_*)."
            ),
        },
        "label_distribution": {
            "counts": {str(k): int(v) for k, v in zip(class_distribution["label"], class_distribution["count"])},
            "n_classes": int(len(class_distribution)),
        },
        "unexpected_label_values": unexpected_labels,
    }

    return {
        "report": report,
        "column_quality": column_quality,
        "class_distribution": class_distribution,
        "class_composition": class_composition,
    }


def save_audit(result: Dict[str, Any], reports_dir: Path, tables_dir: Path) -> Dict[str, Path]:
    """Writes the audit JSON report and CSV tables. Returns the written paths."""
    reports_dir, tables_dir = Path(reports_dir), Path(tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "report": save_json(result["report"], reports_dir / "cicids2017_data_quality_report.json"),
        "column_quality": tables_dir / "cicids2017_column_quality.csv",
        "class_distribution": tables_dir / "cicids2017_class_distribution.csv",
    }
    result["column_quality"].to_csv(paths["column_quality"])
    result["class_distribution"].to_csv(paths["class_distribution"], index=False)
    if result["class_composition"] is not None:
        paths["class_composition"] = tables_dir / "cicids2017_class_by_source_file.csv"
        result["class_composition"].to_csv(paths["class_composition"])
    logger.info(f"Saved data quality audit to {reports_dir} and {tables_dir}")
    return paths
