"""
Builds the deduplicated, stratified train/validation/test split used by the CIC-IDS2017 experiments.

This module deliberately imports no supervised models: it is used by scripts that run PyTorch in the same process,
and XGBoost and PyTorch crash when both run there (see src/run_hybrid_detection.py).
"""

from typing import Any, Dict

import pandas as pd

from src.data.dataset_adapters import CICIDS2017Adapter
from src.data.loaders import CICIDS2017Loader
from src.preprocessing.leakage import drop_identifier_columns
from src.preprocessing.pipeline import CyberthreatPreprocessingPipeline


def prepare_splits(cfg) -> Dict[str, Any]:
    """Recreates the deduplicated, stratified split used by src.run_cicids2017 (same seed, no sampling)."""
    seed = int(cfg.system.seed)
    raw = CICIDS2017Loader(cfg.paths.cicids2017_dir).load_merged_dataset(add_source_column=True)
    raw.pop("source_file")
    X, y_binary, y_multi = CICIDS2017Adapter(cfg.data.target_column_cicids2017).extract_labels(raw)
    del raw
    X, _ = drop_identifier_columns(X, cfg.data.cicids2017.identifier_patterns)
    keep = ~pd.concat([X, y_multi.rename("__label__")], axis=1).duplicated(keep="first").to_numpy()
    X, y_binary, y_multi = X[keep].reset_index(drop=True), y_binary[keep].reset_index(drop=True), y_multi[keep].reset_index(drop=True)

    pipeline = CyberthreatPreprocessingPipeline(
        impute_strategy=cfg.data.handle_missing,
        remove_constant=cfg.preprocessing.remove_constant_features,
        scaling_method=cfg.data.scaling_method,
        encoding_type=cfg.data.categorical_encoding,
        test_size=cfg.data.test_size,
        val_size=cfg.data.val_size,
        random_state=seed,
    )
    splits = pipeline.fit_transform_splits(X, y_binary, y_multi)
    return {"pipeline": pipeline, "splits": splits}
