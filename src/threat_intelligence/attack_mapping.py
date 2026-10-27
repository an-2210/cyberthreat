"""
Maps CIC-IDS2017 attack labels to MITRE ATT&CK Enterprise techniques.

The ATT&CK catalog (names, tactics, validity) is read from the official STIX bundle in
data/external/enterprise-attack.json. The label-to-technique assignments below are an analyst
mapping written for this project, not something MITRE publishes. Each entry carries a confidence
level and a rationale so that it can be reviewed and changed.

Run from the repository root:
    python -m src.threat_intelligence.attack_mapping
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union
import pandas as pd
from src.utils.config import get_absolute_path, load_config
from src.utils.io_utils import save_json
from src.utils.logging import setup_logger

logger = setup_logger("attack_mapping")

ATTACK_SOURCE_URL = (
    "https://raw.githubusercontent.com/mitre-attack/attack-stix-data/master/enterprise-attack/enterprise-attack.json"
)

# label -> list of (technique id, confidence, rationale).
# Rationales quote or paraphrase the ATT&CK technique descriptions in the bundle.
LABEL_TO_TECHNIQUES: Dict[str, List[tuple]] = {
    "BENIGN": [],
    "DoS Hulk": [("T1499.002", "high", "T1499.002 describes a simple HTTP flood that overwhelms a web server; Hulk is an HTTP request flood.")],
    "DoS GoldenEye": [("T1499.002", "high", "T1499.002 covers HTTP floods against web services; GoldenEye is an HTTP flood with keep-alive.")],
    "DoS slowloris": [("T1499", "medium", "Slow HTTP connections exhaust server resources (T1499). Parent technique used because the definitions do not separate slow-connection attacks from flood attacks.")],
    "DoS Slowhttptest": [("T1499", "medium", "Slow HTTP requests exhaust server resources (T1499). Parent technique used for the same reason as slowloris.")],
    "DDoS": [("T1498.001", "high", "T1498.001 describes a high-volume direct network flood sent by one or more systems at a target.")],
    "PortScan": [("T1046", "high", "Probing ports and services to discover which are reachable (Network Service Discovery).")],
    "FTP-Patator": [("T1110.001", "high", "T1110.001 describes systematic password guessing against a login.")],
    "SSH-Patator": [("T1110.001", "high", "T1110.001 describes systematic password guessing against a login.")],
    "Bot": [("T1071.001", "medium", "T1071.001 describes C2 traffic embedded in web protocols; the dataset's bot polls over HTTP.")],
    "Web Attack - Brute Force": [("T1110.001", "medium", "Repeated login attempts against a web application (password guessing).")],
    "Web Attack - XSS": [
        ("T1190", "medium", "T1190 covers exploiting a weakness in an Internet-facing web application; the XSS flaw is the weakness exploited."),
        ("T1189", "medium", "T1189 (Drive-by Compromise) names cross-site scripting explicitly as a way to insert malicious scripts through web application interfaces, which is the XSS payload's effect on the victim's browser."),
    ],
    "Web Attack - Sql Injection": [("T1190", "high", "T1190 explicitly lists databases and web applications as exploitable public-facing systems; SQL injection is such an exploit.")],
    "Heartbleed": [("T1190", "high", "Exploitation of an Internet-facing TLS service (CVE-2014-0160) is a software-weakness exploit covered by T1190.")],
}

# Labels with no defensible ATT&CK technique. They are reported as unmapped rather than guessed.
UNMAPPED_LABELS: Dict[str, str] = {
    "Infiltration": (
        "CIC-IDS2017 describes Infiltration as an exploit followed by backdoor activity. "
        "No single ATT&CK technique matches that end-to-end sequence, so no technique is assigned."
    ),
}

CIC_IDS2017_LABELS = tuple(LABEL_TO_TECHNIQUES.keys()) + tuple(UNMAPPED_LABELS.keys())


def sha256_file(path: Union[str, Path]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_attack_catalog(path: Union[str, Path]) -> Dict[str, Any]:
    """Reads active (not revoked, not deprecated) ATT&CK Enterprise techniques from a STIX bundle.

    Returns:
        {"version": ATT&CK version string or None, "techniques": {id: {name, tactics, is_subtechnique, url}}}
    """
    bundle = json.loads(Path(path).read_text(encoding="utf-8"))
    version = None
    techniques: Dict[str, Dict[str, Any]] = {}
    for obj in bundle.get("objects", []):
        if obj.get("type") == "x-mitre-collection" and version is None:
            version = obj.get("x_mitre_version")
        if obj.get("type") != "attack-pattern" or obj.get("revoked") or obj.get("x_mitre_deprecated"):
            continue
        refs = [r for r in obj.get("external_references", []) if r.get("source_name") == "mitre-attack"]
        if not refs:
            continue
        tid = refs[0]["external_id"]
        techniques[tid] = {
            "name": obj["name"],
            "tactics": [p["phase_name"] for p in obj.get("kill_chain_phases", [])],
            "is_subtechnique": bool(obj.get("x_mitre_is_subtechnique", False)),
            "url": refs[0].get("url"),
        }
    return {"version": version, "techniques": techniques}


def validate_mapping(catalog: Dict[str, Any], mapping: Dict[str, List[tuple]] = LABEL_TO_TECHNIQUES) -> None:
    """Raises if a technique ID is unknown to the catalog or a confidence value is invalid."""
    problems = []
    for label, entries in mapping.items():
        for tid, confidence, _ in entries:
            if tid not in catalog["techniques"]:
                problems.append(f"{label}: unknown technique {tid}")
            if confidence not in ("high", "medium", "low"):
                problems.append(f"{label}: invalid confidence '{confidence}' for {tid}")
    if problems:
        raise ValueError("ATT&CK mapping validation failed:\n" + "\n".join(problems))


def label_status(label: str) -> str:
    """'benign', 'mapped', or 'unmapped' (no defensible technique; see UNMAPPED_LABELS)."""
    if label == "BENIGN":
        return "benign"
    if label in UNMAPPED_LABELS:
        return "unmapped"
    return "mapped" if LABEL_TO_TECHNIQUES.get(label) else "unknown"


def techniques_for_label(label: str, catalog: Dict[str, Any], mapping: Dict[str, List[tuple]] = LABEL_TO_TECHNIQUES) -> List[Dict[str, Any]]:
    """Returns the enriched technique records for one CIC-IDS2017 label (empty for BENIGN or unknown labels)."""
    out = []
    for tid, confidence, rationale in mapping.get(label, []):
        info = catalog["techniques"][tid]
        out.append({
            "technique_id": tid,
            "technique_name": info["name"],
            "tactics": info["tactics"],
            "is_subtechnique": info["is_subtechnique"],
            "url": info["url"],
            "confidence": confidence,
            "rationale": rationale,
        })
    return out


def map_predictions(predicted_labels: Iterable[str], catalog: Dict[str, Any]) -> pd.Series:
    """Maps a sequence of predicted labels to a Series of technique lists (for report generation)."""
    cache: Dict[str, List[Dict[str, Any]]] = {}
    values = []
    for label in predicted_labels:
        if label not in cache:
            cache[label] = techniques_for_label(label, catalog)
        values.append(cache[label])
    return pd.Series(values, dtype=object)


def build_mapping_report(catalog_path: Union[str, Path], out_dir: Union[str, Path]) -> Dict[str, Any]:
    """Validates the mapping against the bundle and writes JSON and CSV outputs with provenance."""
    catalog_path = Path(catalog_path)
    catalog = load_attack_catalog(catalog_path)
    validate_mapping(catalog)
    overlap = set(LABEL_TO_TECHNIQUES) & set(UNMAPPED_LABELS)
    if overlap:
        raise ValueError(f"Labels listed as both mapped and unmapped: {sorted(overlap)}")
    if len(CIC_IDS2017_LABELS) != 15:
        raise ValueError(f"Expected 15 CIC-IDS2017 labels, found {len(CIC_IDS2017_LABELS)}")

    rows = []
    for label in CIC_IDS2017_LABELS:
        techs = techniques_for_label(label, catalog)
        if not techs:
            rows.append({"label": label, "technique_id": None, "technique_name": None, "tactics": None,
                         "confidence": None,
                         "rationale": UNMAPPED_LABELS.get(label, "Benign traffic; no technique.")})
        for t in techs:
            rows.append({"label": label, "technique_id": t["technique_id"], "technique_name": t["technique_name"],
                         "tactics": ", ".join(t["tactics"]), "confidence": t["confidence"], "rationale": t["rationale"]})

    report = {
        "catalog": {
            "source_url": ATTACK_SOURCE_URL,
            "local_path": str(catalog_path),
            "sha256": sha256_file(catalog_path),
            "attack_version": catalog["version"],
            "active_techniques_including_subtechniques": len(catalog["techniques"]),
        },
        "mapping_type": "analyst-defined lookup table; technique IDs and names validated against the catalog above",
        "labels": {l: {"status": label_status(l), "techniques": techniques_for_label(l, catalog),
                       "unmapped_reason": UNMAPPED_LABELS.get(l)} for l in CIC_IDS2017_LABELS},
    }
    out_dir = get_absolute_path(out_dir)
    save_json(report, out_dir / "attack_mapping.json")
    pd.DataFrame(rows).to_csv(out_dir / "attack_mapping.csv", index=False)
    logger.info(f"Wrote ATT&CK mapping ({len(rows)} rows) to {out_dir}")
    return report


def main() -> None:
    cfg = load_config("configs/config.yaml")
    report = build_mapping_report(get_absolute_path(cfg.mitre_attack.stix_data_path),
                                  get_absolute_path(cfg.paths.results_dir) / "threat_intelligence")
    print(f"ATT&CK version {report['catalog']['attack_version']}, sha256 {report['catalog']['sha256'][:12]}...")
    for label, info in report["labels"].items():
        ids = ", ".join(f"{t['technique_id']} ({t['confidence']})" for t in info["techniques"]) or info["status"]
        print(f"{label:28s} {ids}")


if __name__ == "__main__":
    main()


def summarize_detections(true_labels: Iterable[str], predicted_labels: Iterable[str], catalog: Dict[str, Any]) -> Dict[str, Any]:
    """Compares detector output with ground truth at the ATT&CK technique level.

    For each technique: how many true rows map to it, how many predicted rows map to it, and how many
    rows are predicted to it correctly. Rows whose label is unmapped or BENIGN contribute no technique.
    """
    true_labels = list(true_labels)
    predicted_labels = list(predicted_labels)
    if len(true_labels) != len(predicted_labels):
        raise ValueError("true and predicted label sequences must have the same length")

    def techs(label: str) -> set:
        return {t["technique_id"] for t in techniques_for_label(label, catalog)}

    per_tech: Dict[str, Dict[str, int]] = {}
    exact_technique_match = 0
    label_match = 0
    unmapped_true = unmapped_pred = 0
    for t_lab, p_lab in zip(true_labels, predicted_labels):
        label_match += int(t_lab == p_lab)
        t_set, p_set = techs(t_lab), techs(p_lab)
        unmapped_true += int(label_status(t_lab) == "unmapped")
        unmapped_pred += int(label_status(p_lab) == "unmapped")
        exact_technique_match += int(t_set == p_set)  # two empty sets (benign / unmapped) also agree
        for tid in t_set:
            per_tech.setdefault(tid, {"true_rows": 0, "predicted_rows": 0, "correct_rows": 0})["true_rows"] += 1
        for tid in p_set:
            per_tech.setdefault(tid, {"true_rows": 0, "predicted_rows": 0, "correct_rows": 0})["predicted_rows"] += 1
        for tid in t_set & p_set:
            per_tech[tid]["correct_rows"] += 1

    rows = []
    for tid, c in sorted(per_tech.items()):
        info = catalog["techniques"][tid]
        rows.append({
            "technique_id": tid,
            "technique_name": info["name"],
            **c,
            "technique_recall": (c["correct_rows"] / c["true_rows"]) if c["true_rows"] else None,
            "technique_precision": (c["correct_rows"] / c["predicted_rows"]) if c["predicted_rows"] else None,
        })
    n = len(true_labels)
    return {
        "n_rows": n,
        "label_accuracy": label_match / n if n else None,
        "technique_set_agreement": exact_technique_match / n if n else None,  # rows whose technique sets are identical
        "unmapped_true_rows": unmapped_true,
        "unmapped_predicted_rows": unmapped_pred,
        "per_technique": rows,
    }
