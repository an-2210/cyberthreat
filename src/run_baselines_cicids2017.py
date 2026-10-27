"""
Classical baselines for CIC-IDS2017: Logistic Regression and a linear SVM, on exactly the rows the main pipeline used.

The split, the preprocessing pipeline and the training subsample are recreated with the same seed and config
(see src.data.splits.prepare_splits and src.run_cicids2017). No saved model is retrained here.

Also stores the binary test-set scores of the saved Random Forest and XGBoost models in data/interim, so that
precision-recall curves can be drawn without loading the models again.

Outputs:
    results/metrics/cicids2017_baseline_<model>_<task>.json
    data/interim/cicids2017_binary_test_scores.npz

Run from the repository root (about 10-30 minutes on a laptop, mostly the linear models):
    python -m src.run_baselines_cicids2017
"""

import time
from typing import Any, Dict

import numpy as np
from sklearn.linear_model import LogisticRegression, SGDClassifier

from src.data.splits import prepare_splits
from src.evaluation.detection_metrics import binary_detection_metrics, multiclass_metrics
from src.models.inference import CICIDS2017Detector
from src.preprocessing.sampling import sample_positions
from src.utils.config import get_absolute_path, load_config
from src.utils.io_utils import save_json
from src.utils.logging import setup_logger

logger = setup_logger("run_baselines")

BENIGN = "BENIGN"
SCORES_PATH = "data/interim/cicids2017_binary_test_scores.npz"


def _models(cfg, seed: int) -> Dict[str, Any]:
    lr_cfg = cfg.models.baselines.logistic_regression
    return {
        "logistic_regression": lambda: LogisticRegression(max_iter=int(lr_cfg.max_iter), C=float(lr_cfg.C),
                                                          solver=str(lr_cfg.solver)),
        # Linear SVM (hinge loss) trained by SGD: the RBF SVM in the config is not feasible on ~500k rows, and the exact
        # LinearSVC one-vs-rest fit over 15 classes did not finish in a practical time.
        "linear_svm": lambda: SGDClassifier(loss="hinge", alpha=1e-4, max_iter=50, tol=1e-4, random_state=seed),
    }


def _scores(model, X: np.ndarray, n_classes: int, class_ids: np.ndarray) -> np.ndarray:
    """Per-class scores over all classes: probabilities when the model has them, otherwise decision values.

    A class the model never saw in training gets the lowest possible score, so it is never predicted and has no ROC area.
    """
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X)
    else:
        d = model.decision_function(X)
        scores = d if d.ndim == 2 else np.column_stack([-d, d])
    if scores.shape[1] == n_classes:
        return scores
    full = np.full((scores.shape[0], n_classes), -1e9)
    full[:, np.asarray(class_ids, dtype=int)] = scores
    return full


def main() -> None:
    cfg = load_config("configs/config.yaml")
    seed = int(cfg.system.seed)
    metrics_dir = get_absolute_path(cfg.paths.results_dir) / "metrics"

    det_multi = CICIDS2017Detector.load("multiclass", "xgboost", artifacts_dir=get_absolute_path(cfg.paths.artifacts_dir))
    class_names = list(det_multi.labels)
    name_to_id = {name: i for i, name in enumerate(class_names)}

    logger.info("Recreating the split and preprocessing (same seed and config as the main pipeline)...")
    out = prepare_splits(cfg)
    splits = out["splits"]

    def to_id(labels: np.ndarray) -> np.ndarray:
        return np.asarray([name_to_id[str(v)] for v in labels], dtype=int)

    tr_ids = to_id(splits["y_train_multi"].to_numpy())
    cap = cfg.training.sampling.max_train_rows
    positions, sample_meta = sample_positions(tr_ids, strategy=cfg.training.sampling.strategy,
                                              max_rows=cap, seed=seed)
    X_train = splits["X_train"].iloc[positions].to_numpy(dtype=np.float32)
    y_train = {"binary": splits["y_train_binary"].to_numpy(dtype=int)[positions], "multiclass": tr_ids[positions]}
    X_test = splits["X_test"].to_numpy(dtype=np.float32)
    y_test = {"binary": splits["y_test_binary"].to_numpy(dtype=int),
              "multiclass": to_id(splits["y_test_multi"].to_numpy())}
    logger.info(f"Training rows: {len(X_train)}; test rows: {len(X_test)}; classes: {len(class_names)}")

    binary_scores: Dict[str, np.ndarray] = {}
    for name, factory in _models(cfg, seed).items():
        for task in ("binary", "multiclass"):
            n_classes = 2 if task == "binary" else len(class_names)
            model = factory()
            fit_X, fit_y = X_train, y_train[task]
            t0 = time.perf_counter()
            model.fit(fit_X, fit_y)
            fit_seconds = time.perf_counter() - t0

            t1 = time.perf_counter()
            y_pred = np.asarray(model.predict(X_test)).astype(int)
            inference_seconds = time.perf_counter() - t1
            scores = _scores(model, X_test, n_classes, getattr(model, "classes_", np.arange(n_classes)))
            if task == "binary":
                metrics = binary_detection_metrics(y_test["binary"], y_pred, scores[:, 1])
                binary_scores[name] = scores[:, 1]
            else:
                metrics = multiclass_metrics(y_test["multiclass"], y_pred, scores, class_names)
            metrics.update({
                "model": name, "task": task, "seed": seed,
                "n_train_rows": int(len(fit_X)), "n_test_rows": int(len(X_test)),
                "classes_absent_from_training": (
                    [class_names[i] for i in range(n_classes) if i not in set(np.asarray(fit_y).tolist())]
                    if task == "multiclass" else []
                ),
                "train_fit_seconds": round(fit_seconds, 3),
                "test_inference_seconds": round(inference_seconds, 3),
                "sampling": sample_meta,
                "note": "Linear model on the same preprocessed features and training rows as the main pipeline.",
            })
            save_json(metrics, metrics_dir / f"cicids2017_baseline_{name}_{task}.json")
            logger.info(f"{name} {task}: done in {fit_seconds:.1f}s")

    # Binary test scores of the saved tree models, for the precision-recall figure (the linear scores were stored above).
    for name in ("random_forest", "xgboost"):
        det = CICIDS2017Detector.load("binary", name, artifacts_dir=get_absolute_path(cfg.paths.artifacts_dir))
        binary_scores[name] = det.model.predict_proba(X_test)[:, 1]
    scores_path = get_absolute_path(SCORES_PATH)
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(scores_path, y_test=y_test["binary"], **binary_scores)
    logger.info(f"Saved binary test scores to {scores_path}")


if __name__ == "__main__":
    main()
