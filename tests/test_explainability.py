"""
SYNTHETIC tests for SHAP explanations. A tiny XGBoost model is trained on random data; no real
dataset or saved artifacts are needed, and no detection results are measured here.
"""

import numpy as np
import shap
import xgboost as xgb
from xgboost import XGBClassifier

from src.explainability.shap_explainer import explain_rows


def _tiny_xgb(seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(300, 5)).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 2] > 0).astype(int)
    model = XGBClassifier(n_estimators=20, max_depth=3, random_state=seed, n_jobs=1)
    model.fit(X, y)
    return model, X, y


def test_tree_shap_is_additive_to_margin():
    model, X, _ = _tiny_xgb()
    background = X[:50]
    booster = model.get_booster()  # SHAP 0.52 rejects the sklearn wrapper for XGBoost 3.x
    explainer = shap.TreeExplainer(booster, data=background, feature_perturbation="interventional")
    sv = np.asarray(explainer.shap_values(X[:40], check_additivity=False))
    base = float(np.atleast_1d(explainer.expected_value)[0])
    margin = booster.predict(xgb.DMatrix(X[:40]), output_margin=True)
    assert sv.shape == (40, 5)
    assert np.allclose(base + sv.sum(axis=1), margin, atol=1e-3)


def test_explain_rows_sorts_top_contributions_by_absolute_value():
    sv = np.array([[0.1, -2.0, 0.5, 0.0, 1.0], [3.0, 0.0, 0.0, 0.0, 0.0]])
    X = np.zeros((2, 5))
    names = ["a", "b", "c", "d", "e"]
    rows = explain_rows(sv, X, names, row_index=np.array([7, 9]), top_k=2)
    assert rows[0]["test_row"] == 7
    assert [f["feature"] for f in rows[0]["top_features"]] == ["b", "e"]
    assert rows[1]["top_features"][0]["feature"] == "a"
    assert np.isclose(rows[0]["shap_margin_total"], sv[0].sum())
