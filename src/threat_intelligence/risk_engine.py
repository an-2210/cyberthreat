"""
Transparent risk scoring for one detection.

    score = sum(weight_i * term_i) / sum(weight_i over the terms that are available)

Terms:
    supervised_confidence  the classifier's probability for its predicted attack type (0 to 1)
    anomaly_score          the anomaly detector's normalized score (0 to 1), only when an anomaly detector ran
    impact                 analyst policy value for the attack type (configs/config.yaml: risk.impact)

Not used, because the CIC-IDS2017 flow records do not contain them: affected asset, frequency per source and
persistence over time. They are listed as not available in every result rather than being guessed.

Level: the highest level whose minimum score the risk score reaches (risk.levels). Benign traffic gets level NONE.

Every term, weight and contribution is returned, so the score can be audited without reading any prompt.
"""

from typing import Any, Dict, Optional

from src.utils.config import ConfigDict

BENIGN = "BENIGN"
NOT_AVAILABLE_TERMS = ("affected_asset", "frequency", "persistence")
LEVEL_ORDER = ("CRITICAL", "HIGH", "MEDIUM", "LOW")


def _check_unit(name: str, value: float) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"Risk term '{name}' must be between 0 and 1, got {value}.")
    return value


def risk_for_detection(label: str, confidence: float, risk_cfg: ConfigDict,
                       anomaly_score: Optional[float] = None) -> Dict[str, Any]:
    """Risk level and score for one detection, with every term shown.

    risk_cfg is the `risk` section of configs/config.yaml (pass cfg.risk).

    Raises:
        KeyError: the attack label has no impact value in configs/config.yaml (no silent default).
        ValueError: a term is outside [0, 1] or the configured weights are not positive.
    """
    if label == BENIGN:
        return {"label": label, "level": "NONE", "score": 0.0, "terms": [],
                "not_available": list(NOT_AVAILABLE_TERMS),
                "note": "Benign traffic is not rated."}
    if label not in risk_cfg.impact:
        raise KeyError(f"No impact value for '{label}' in configs/config.yaml (risk.impact).")

    weights = {k: float(v) for k, v in risk_cfg.weights.items()}
    if any(w <= 0 for w in weights.values()):
        raise ValueError("Risk weights must be positive.")

    terms = {
        "supervised_confidence": _check_unit("supervised_confidence", confidence),
        "impact": _check_unit("impact", risk_cfg.impact[label]),
    }
    if anomaly_score is not None:
        terms["anomaly_score"] = _check_unit("anomaly_score", anomaly_score)

    used = sum(weights[k] for k in terms)
    score = sum(weights[k] * v for k, v in terms.items()) / used
    level = next(lvl for lvl in LEVEL_ORDER if score >= float(risk_cfg.levels[lvl]))

    return {
        "label": label,
        "level": level,
        "score": round(score, 4),
        "terms": [
            {"term": k, "value": round(v, 4), "weight": round(weights[k] / used, 4),
             "contribution": round(weights[k] * v / used, 4)}
            for k, v in terms.items()
        ],
        "not_available": list(NOT_AVAILABLE_TERMS),
        "note": "Weights are renormalized over the terms that are available for this detection.",
    }
