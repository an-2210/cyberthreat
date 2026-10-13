"""
Unit tests for decision fusion, the rank normalizer, the signed-log transform and the autoencoder.
Synthetic data only.
"""

import numpy as np
import pytest
from src.anomaly_detection.autoencoder import AutoencoderDetector
from src.anomaly_detection.common import signed_log1p, threshold_for_fpr
from src.fusion.decision_fusion import fuse, rank_normalizer, select_fusion


def test_signed_log1p_keeps_sign_and_compresses():
    x = np.array([-1000.0, -1.0, 0.0, 1.0, 1000.0])
    out = signed_log1p(x)
    assert np.sign(out).tolist() == [-1.0, -1.0, 0.0, 1.0, 1.0]
    assert abs(out[-1]) < 10 < abs(x[-1])


def test_threshold_for_fpr_flags_target_share():
    scores = np.arange(1000, dtype=float)
    thr = threshold_for_fpr(scores, 0.05)
    assert (scores > thr).mean() == pytest.approx(0.05, abs=0.002)


def test_rank_normalizer_maps_benign_reference_to_unit_interval():
    reference = np.array([1.0, 2.0, 3.0, 4.0])
    normalize = rank_normalizer(reference)
    np.testing.assert_allclose(normalize(np.array([0.0, 2.0, 10.0])), [0.0, 0.5, 1.0])


def test_fuse_is_convex_combination():
    p = np.array([1.0, 0.0])
    a = np.array([0.0, 1.0])
    np.testing.assert_allclose(fuse(p, a, 1.0), p)
    np.testing.assert_allclose(fuse(p, a, 0.0), a)
    np.testing.assert_allclose(fuse(p, a, 0.5), [0.5, 0.5])


def test_fuse_rejects_bad_alpha():
    with pytest.raises(ValueError):
        fuse(np.array([0.5]), np.array([0.5]), 1.5)


def test_select_fusion_picks_the_informative_signal():
    rng = np.random.default_rng(0)
    n = 4000
    y = rng.integers(0, 2, n)
    # The supervised score is informative; the anomaly score is pure noise.
    p = np.clip(y * 0.8 + rng.normal(0.1, 0.1, n), 0, 1)
    a = rng.uniform(0, 1, n)
    choice = select_fusion(p, a, y, np.linspace(0, 1, 11), target_fpr=0.01)
    assert choice["alpha"] >= 0.8
    assert choice["val_f1"] > 0.8


def test_autoencoder_scores_outliers_higher_than_benign():
    rng = np.random.default_rng(1)
    benign = rng.normal(0, 1, size=(3000, 6)).astype(np.float32)
    outliers = rng.normal(0, 1, size=(200, 6)).astype(np.float32) * 6 + 20
    ae = AutoencoderDetector(hidden_dims=(8, 3), epochs=5, batch_size=256, seed=3).fit(benign)
    assert ae.score(outliers).mean() > ae.score(benign[:500]).mean() * 10


def test_autoencoder_requires_fit():
    with pytest.raises(RuntimeError):
        AutoencoderDetector().score(np.zeros((3, 4), dtype=np.float32))
