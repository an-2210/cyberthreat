"""
Cybersecurity domain feature engineering transformer for network traffic metrics.
"""

from typing import List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from src.utils.logging import setup_logger

logger = setup_logger("feature_engineer")


class CybersecurityFeatureEngineer(BaseEstimator, TransformerMixin):
    """Computes specialized domain ratios and flow metrics for network intrusion datasets."""

    def __init__(self, eps: float = 1e-5):
        """
        Args:
            eps: Epsilon parameter to prevent division by zero.
        """
        self.eps = eps
        self.feature_names_out_: List[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[Union[pd.Series, np.ndarray]] = None):
        """Learns output schema feature names.

        Args:
            X: Input feature DataFrame.
            y: Ignored.

        Returns:
            self
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X_sample = self.transform(X.iloc[:5])
        self.feature_names_out_ = list(X_sample.columns)
        logger.info(f"CybersecurityFeatureEngineer fit completed: created domain features. Output columns: {len(self.feature_names_out_)}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Engineers network security ratio and rate features.

        Args:
            X: Input feature DataFrame.

        Returns:
            DataFrame with original and new engineered features.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X_out = X.copy()
        cols = {col.strip().lower(): col for col in X_out.columns}

        # 1. Bytes per Packet Ratio
        fwd_pkts_col = cols.get("total fwd packets") or cols.get("spkts")
        bwd_pkts_col = cols.get("total backward packets") or cols.get("dpkts")
        flow_bytes_col = cols.get("flow bytes/s") or cols.get("sbytes")

        if fwd_pkts_col and bwd_pkts_col:
            total_pkts = X_out[fwd_pkts_col] + X_out[bwd_pkts_col]
            X_out["fe_packet_ratio"] = (X_out[fwd_pkts_col] / (X_out[bwd_pkts_col] + self.eps)).astype(np.float32)
            if flow_bytes_col:
                X_out["fe_bytes_per_packet"] = (X_out[flow_bytes_col] / (total_pkts + self.eps)).astype(np.float32)

        # 2. Byte Ratio
        fwd_bytes_col = cols.get("total length of fwd packets") or cols.get("sbytes")
        bwd_bytes_col = cols.get("total length of bwd packets") or cols.get("dbytes")

        if fwd_bytes_col and bwd_bytes_col:
            X_out["fe_byte_ratio"] = (X_out[fwd_bytes_col] / (X_out[bwd_bytes_col] + self.eps)).astype(np.float32)

        # 3. Flow Intensity (Packets / Duration)
        duration_col = cols.get("flow duration") or cols.get("dur")
        if duration_col and (fwd_pkts_col and bwd_pkts_col):
            total_pkts = X_out[fwd_pkts_col] + X_out[bwd_pkts_col]
            X_out["fe_flow_intensity"] = (total_pkts / (X_out[duration_col] + self.eps)).astype(np.float32)

        # 4. IAT Spread (Flow IAT Max - Flow IAT Min)
        iat_max_col = cols.get("flow iat max")
        iat_min_col = cols.get("flow iat min")
        if iat_max_col and iat_min_col:
            X_out["fe_iat_spread"] = (X_out[iat_max_col] - X_out[iat_min_col]).astype(np.float32)

        # Replace any residual np.inf / -np.inf created during ratio computation
        new_cols = [c for c in X_out.columns if c.startswith("fe_")]
        for col in new_cols:
            X_out[col] = X_out[col].replace([np.inf, -np.inf], 0.0).fillna(0.0)

        return X_out

    def get_feature_names_out(self, input_features=None) -> List[str]:
        """Returns list of engineered feature names."""
        return self.feature_names_out_
