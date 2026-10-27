"""
Tests for the UNSW-NB15 native preprocessing helpers. Literal input values only; no UNSW data is needed.
"""

import numpy as np
import pandas as pd

from src.evaluation.unsw_native import apply_category_codes, fit_category_codes, numeric_base


def test_numeric_base_drops_ip_identifiers_and_parses_ports():
    raw = pd.DataFrame({
        "srcip": ["59.166.0.2"], "dstip": ["10.40.170.1"],
        "sport": ["0x0303"], "dsport": ["443"],
        "proto": ["tcp"], "dur": [0.5], "ct_ftp_cmd": [" "],
    })
    base = numeric_base(raw)
    assert "srcip" not in base.columns and "dstip" not in base.columns
    assert base.loc[0, "sport"] == 771.0  # 0x0303
    assert base.loc[0, "dsport"] == 443.0
    assert np.isnan(base.loc[0, "ct_ftp_cmd"])  # blank text becomes missing, not an error
    assert base.loc[0, "proto"] == "tcp"  # categorical text is left for the encoder


def test_numeric_base_turns_infinite_values_into_missing():
    raw = pd.DataFrame({"rate": [np.inf], "proto": ["udp"]})
    assert np.isnan(numeric_base(raw).loc[0, "rate"])


def test_category_codes_come_from_training_and_unseen_values_become_minus_one():
    train = pd.DataFrame({"proto": ["tcp", "udp", "tcp"], "service": ["-", "dns", "-"], "state": ["FIN", "INT", "FIN"]})
    codes = fit_category_codes(train)
    assert codes["proto"] == ["tcp", "udp"]
    test = pd.DataFrame({"proto": ["udp", "icmp"], "service": ["dns", "-"], "state": ["INT", "CON"]})
    encoded = apply_category_codes(test, codes)
    assert encoded["proto"].tolist() == [1.0, -1.0]  # icmp was not seen in training
    assert encoded["state"].tolist() == [1.0, -1.0]  # CON was not seen in training
