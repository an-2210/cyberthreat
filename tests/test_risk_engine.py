"""
Tests for the risk engine, using the real risk section of configs/config.yaml.

Expected values are worked out by hand in each test (weights 0.5 / 0.2 / 0.3, DDoS impact 0.8, PortScan impact 0.3).
"""

import pytest

from src.threat_intelligence.risk_engine import risk_for_detection
from src.utils.config import load_config


@pytest.fixture(scope="module")
def cfg():
    return load_config("configs/config.yaml")


def test_benign_traffic_is_not_rated(cfg):
    out = risk_for_detection("BENIGN", 0.99, cfg.risk)
    assert out["level"] == "NONE" and out["score"] == 0.0 and out["terms"] == []


def test_ddos_without_anomaly_term_renormalizes_weights(cfg):
    # (0.5 * 1.0 + 0.3 * 0.8) / (0.5 + 0.3) = 0.925 -> CRITICAL
    out = risk_for_detection("DDoS", 1.0, cfg.risk)
    assert out["score"] == pytest.approx(0.925, abs=1e-4)
    assert out["level"] == "CRITICAL"
    assert {t["term"] for t in out["terms"]} == {"supervised_confidence", "impact"}


def test_anomaly_term_is_included_when_available(cfg):
    # (0.5 * 1.0 + 0.2 * 0.5 + 0.3 * 0.8) / 1.0 = 0.84 -> CRITICAL
    out = risk_for_detection("DDoS", 1.0, cfg.risk, anomaly_score=0.5)
    assert out["score"] == pytest.approx(0.84, abs=1e-4)
    assert sum(t["weight"] for t in out["terms"]) == pytest.approx(1.0, abs=1e-3)


def test_contributions_add_up_to_the_score(cfg):
    out = risk_for_detection("PortScan", 0.5, cfg.risk, anomaly_score=0.2)
    assert sum(t["contribution"] for t in out["terms"]) == pytest.approx(out["score"], abs=1e-3)


def test_low_confidence_portscan_is_medium(cfg):
    # (0.5 * 0.5 + 0.3 * 0.3) / 0.8 = 0.425 -> MEDIUM
    out = risk_for_detection("PortScan", 0.5, cfg.risk)
    assert out["score"] == pytest.approx(0.425, abs=1e-4)
    assert out["level"] == "MEDIUM"


def test_unknown_attack_label_is_an_error_not_a_default(cfg):
    with pytest.raises(KeyError, match="No impact value"):
        risk_for_detection("Made-up attack", 0.9, cfg.risk)


def test_out_of_range_confidence_is_rejected(cfg):
    with pytest.raises(ValueError, match="between 0 and 1"):
        risk_for_detection("DDoS", 1.2, cfg.risk)
