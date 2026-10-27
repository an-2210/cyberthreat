"""
Links the saved multiclass XGBoost detector to the ATT&CK mapping on the held-out TEST split.

Run from the repository root:
    python -m src.threat_intelligence.detection_linkage
"""

import numpy as np
import pandas as pd
from src.models.inference import CICIDS2017Detector
from src.run_rare_class_experiment import prepare_splits
from src.threat_intelligence.attack_mapping import load_attack_catalog, summarize_detections
from src.utils.config import get_absolute_path, load_config
from src.utils.io_utils import save_json
from src.utils.logging import setup_logger

logger = setup_logger("detection_linkage")


def run(cfg=None) -> dict:
    cfg = cfg or load_config("configs/config.yaml")
    catalog = load_attack_catalog(get_absolute_path(cfg.mitre_attack.stix_data_path))

    detector = CICIDS2017Detector.load("multiclass", "xgboost", artifacts_dir=get_absolute_path(cfg.paths.artifacts_dir))
    splits = prepare_splits(cfg)["splits"]
    X_test = splits["X_test"].to_numpy(dtype=np.float32)
    names = np.asarray(detector.labels, dtype=object)
    pred_ids = np.asarray(detector.model.predict(X_test)).astype(int)
    predicted = names[pred_ids]
    true = pd.Series(splits["y_test_multi"]).to_numpy(dtype=object)

    summary = summarize_detections(true, predicted, catalog)
    summary.update({
        "model": "xgboost (multiclass, saved artifact)",
        "data_source": "real CIC-IDS2017 CSVs (held-out test split)",
        "attack_version": catalog["version"],
    })
    out = get_absolute_path(cfg.paths.results_dir) / "threat_intelligence" / "detection_technique_summary.json"
    save_json(summary, out)
    print(f"rows={summary['n_rows']} label_accuracy={summary['label_accuracy']:.4f} "
          f"technique_set_agreement={summary['technique_set_agreement']:.4f}")
    for r in summary["per_technique"]:
        print(f"{r['technique_id']:10s} true={r['true_rows']:7d} pred={r['predicted_rows']:7d} "
              f"correct={r['correct_rows']:7d} recall={r['technique_recall']} precision={r['technique_precision']}")
    logger.info(f"Wrote {out}")
    return summary


if __name__ == "__main__":
    run()
