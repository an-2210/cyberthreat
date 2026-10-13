"""
Unit tests for the leave-one-attack-out fold evaluation, on synthetic data.
"""

import numpy as np
from src.run_leave_one_attack_out import NONE, fold_detection


def _synthetic(seed=0):
    rng = np.random.default_rng(seed)
    n = 4000
    labels = rng.choice(["BENIGN", "KnownAttack", "HeldAttack"], size=n, p=[0.6, 0.2, 0.2])
    y_bin = (labels != "BENIGN").astype(int)
    # Supervised score knows KnownAttack but gives HeldAttack no signal (it was never in training).
    p = np.where(labels == "KnownAttack", 0.9, rng.uniform(0, 0.2, n))
    # The autoencoder flags both attack types, since they are both unlike benign traffic.
    a = np.where(y_bin == 1, rng.normal(5, 1, n), rng.normal(0, 1, n))
    return labels, y_bin, p, a


def test_held_out_attack_is_detected_by_autoencoder_not_supervised():
    labels, y_bin, p, a = _synthetic()
    half = len(labels) // 2
    rec = fold_detection(
        "HeldAttack",
        p[:half], a[:half], y_bin[:half], labels[:half],
        p[half:], a[half:], y_bin[half:], labels[half:],
        target_fpr=0.01,
    )
    assert rec["supervised_detection"] < 0.05
    assert rec["autoencoder_detection"] > 0.9
    assert rec["test_rows_held_out"] > 0


def test_baseline_fold_scores_all_attacks():
    labels, y_bin, p, a = _synthetic()
    half = len(labels) // 2
    rec = fold_detection(
        NONE,
        p[:half], a[:half], y_bin[:half], labels[:half],
        p[half:], a[half:], y_bin[half:], labels[half:],
        target_fpr=0.01,
    )
    assert rec["test_rows_held_out"] == int((y_bin[half:] == 1).sum())
    assert rec["supervised_detection"] > 0.5


def test_held_out_rows_are_removed_from_validation_choices():
    labels, y_bin, p, a = _synthetic()
    half = len(labels) // 2
    # Make the held-out class look perfectly separable on validation only. The fold must ignore it.
    p_val = p[:half].copy()
    held_mask = labels[:half] == "HeldAttack"
    p_val[held_mask] = 1.0
    rec = fold_detection(
        "HeldAttack",
        p_val, a[:half], y_bin[:half], labels[:half],
        p[half:], a[half:], y_bin[half:], labels[half:],
        target_fpr=0.01,
    )
    assert 0.0 <= rec["fusion_alpha"] <= 1.0
    assert rec["supervised_detection"] < 0.05
