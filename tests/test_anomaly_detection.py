"""
Unit tests for the Isolation Forest anomaly detector, using synthetic data.
"""

import numpy as np
import pytest
from src.anomaly_detection.isolation_forest import IsolationForestDetector


@pytest.fixture
def synthetic():
    rng = np.random.default_rng(0)
    benign_train = rng.normal(0, 1, size=(2000, 5))
    benign_val = rng.normal(0, 1, size=(1000, 5))
    # Attacks sit far from the benign cloud, like unusual traffic patterns.
    attacks = rng.normal(8, 1, size=(200, 5))
    return benign_train, benign_val, attacks


def test_predict_requires_calibration(synthetic):
    benign_train, _, _ = synthetic
    detector = IsolationForestDetector(seed=1).fit(benign_train)
    with pytest.raises(RuntimeError):
        detector.predict(benign_train[:10])


def test_calibrated_threshold_hits_target_false_positive_rate(synthetic):
    benign_train, benign_val, _ = synthetic
    detector = IsolationForestDetector(seed=1).fit(benign_train)
    detector.calibrate(benign_val, target_fpr=0.05)
    flagged_share = detector.predict(benign_val).mean()
    assert flagged_share == pytest.approx(0.05, abs=0.01)


def test_far_away_attacks_are_flagged(synthetic):
    benign_train, benign_val, attacks = synthetic
    detector = IsolationForestDetector(seed=1).fit(benign_train)
    detector.calibrate(benign_val, target_fpr=0.01)
    assert detector.predict(attacks).mean() > 0.95


def test_scores_are_higher_for_anomalies(synthetic):
    benign_train, _, attacks = synthetic
    detector = IsolationForestDetector(seed=1).fit(benign_train)
    assert detector.score(attacks).mean() > detector.score(benign_train).mean()


def test_invalid_target_fpr_is_rejected(synthetic):
    benign_train, benign_val, _ = synthetic
    detector = IsolationForestDetector(seed=1).fit(benign_train)
    with pytest.raises(ValueError):
        detector.calibrate(benign_val, target_fpr=0.0)


def test_save_and_load_round_trip(synthetic, tmp_path):
    benign_train, benign_val, attacks = synthetic
    detector = IsolationForestDetector(seed=1).fit(benign_train)
    detector.calibrate(benign_val, target_fpr=0.01)
    path = detector.save(tmp_path / "detector.joblib")
    loaded = IsolationForestDetector.load(path)
    assert loaded.threshold == detector.threshold
    np.testing.assert_array_equal(loaded.predict(attacks), detector.predict(attacks))
