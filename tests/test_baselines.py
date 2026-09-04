"""
Unit tests for evaluation metrics, baseline classifiers, ModelFactory, and ExperimentRunner.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.evaluation.experiments import ExperimentRunner
from src.evaluation.metrics import evaluate_classification
from src.models.baselines import DecisionTreeModel, LogisticRegressionModel, RandomForestModel
from src.models.model_factory import ModelFactory


@pytest.fixture
def mock_classification_data():
    """Generates synthetic dataset for testing model training and metrics."""
    np.random.seed(42)
    n = 100
    X = pd.DataFrame({
        "feature1": np.random.randn(n),
        "feature2": np.random.randn(n),
        "feature3": np.random.uniform(0, 10, n),
    })
    y = np.random.choice([0, 1], n, p=[0.6, 0.4])
    return X, y


def test_evaluate_classification_binary():
    y_true = np.array([0, 0, 1, 1, 1, 0])
    y_pred = np.array([0, 0, 1, 1, 0, 0])
    y_prob = np.array([0.1, 0.2, 0.9, 0.85, 0.4, 0.3])

    metrics = evaluate_classification(y_true, y_pred, y_prob)

    assert metrics["accuracy"] == pytest.approx(5 / 6, abs=1e-3)
    assert "macro_f1" in metrics
    assert "mcc" in metrics
    assert "fpr" in metrics
    assert "fnr" in metrics
    assert metrics["roc_auc"] is not None


def test_model_factory():
    lr = ModelFactory.create_model("logistic_regression", {"max_iter": 500})
    rf = ModelFactory.create_model("random_forest", {"n_estimators": 50})
    dt = ModelFactory.create_model("decision_tree", {"max_depth": 5})

    assert isinstance(lr, LogisticRegressionModel)
    assert isinstance(rf, RandomForestModel)
    assert isinstance(dt, DecisionTreeModel)

    with pytest.raises(ValueError):
        ModelFactory.create_model("invalid_model")


def test_baseline_models_fit_predict(mock_classification_data, tmp_path):
    X, y = mock_classification_data
    rf = RandomForestModel(n_estimators=10)
    rf.fit(X, y)

    preds = rf.predict(X)
    probas = rf.predict_proba(X)

    assert len(preds) == len(y)
    assert probas.shape == (len(y), 2)

    # Persistence test
    checkpoint_path = tmp_path / "rf_model.joblib"
    rf.save_model(checkpoint_path)
    loaded_rf = RandomForestModel.load_model(checkpoint_path)

    loaded_preds = loaded_rf.predict(X)
    np.testing.assert_array_equal(preds, loaded_preds)


def test_experiment_runner(mock_classification_data, tmp_path):
    X, y = mock_classification_data
    X_train, X_test = X.iloc[:70], X.iloc[70:]
    y_train, y_test = y[:70], y[70:]

    models = {
        "LogisticRegression": LogisticRegressionModel(max_iter=100),
        "RandomForest": RandomForestModel(n_estimators=10),
    }

    runner = ExperimentRunner(results_dir=tmp_path)
    df_summary = runner.run_baseline_comparison(models, X_train, y_train, X_test, y_test, experiment_name="test_exp")

    assert len(df_summary) == 2
    assert "accuracy" in df_summary.columns
    assert "training_time_sec" in df_summary.columns

    # Verify generated artifact files
    assert (tmp_path / "metrics" / "test_exp_metrics.csv").exists()
    assert (tmp_path / "metrics" / "test_exp_metrics.json").exists()
    assert (tmp_path / "tables" / "test_exp_table.md").exists()
    assert (tmp_path / "figures" / "test_exp_roc_curves.png").exists()
    assert (tmp_path / "figures" / "test_exp_metrics_barchart.png").exists()
