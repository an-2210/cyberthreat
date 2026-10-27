"""
Retrieval layer for the threat-intelligence reports.

Corpus (two sources, each document keeps its provenance):
  * ATTACK:<id>  - MITRE ATT&CK technique text from the official STIX bundle, for every technique the
                   label mapping uses (plus the parent of each mapped sub-technique).
  * PROFILE:<label> - project label profiles generated from saved results: mapping status and rationale,
                   training-set size, held-out test support and per-class F1 of the saved multiclass XGBoost
                   model, and the known confusion for each label.

Embeddings: sentence-transformers all-MiniLM-L6-v2 (local cache, loaded offline).
Vector store: FAISS (IndexFlatIP over L2-normalized vectors, i.e. cosine similarity).

Build the index:
    python -m src.threat_intelligence.rag
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np
from src.threat_intelligence.attack_mapping import (
    LABEL_TO_TECHNIQUES,
    UNMAPPED_LABELS,
    load_attack_catalog,
    label_status,
)
from src.utils.config import get_absolute_path, load_config
from src.utils.io_utils import save_json
from src.utils.logging import setup_logger

logger = setup_logger("rag_index")

INDEX_FILENAME = "rag_index.faiss"
DOCS_FILENAME = "rag_documents.json"
MAX_DOC_CHARS = 1200


def _faiss():
    # Imported lazily: on macOS, loading faiss before xgboost/scikit-learn makes both use different
    # OpenMP runtimes and the process segfaults during model fitting or prediction. Callers that use
    # the detector must load it before the first call here.
    import faiss

    return faiss


def _clean_attack_text(text: str) -> str:
    """Removes ATT&CK citation markup and markdown links, and collapses whitespace."""
    text = re.sub(r"\(Citation:[^)]*\)", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def build_attack_documents(catalog: Dict[str, Any], bundle_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """One document per technique used in the mapping (and the parent of each sub-technique)."""
    bundle = json.loads(Path(bundle_path).read_text(encoding="utf-8"))
    wanted = set()
    for entries in LABEL_TO_TECHNIQUES.values():
        for tid, _, _ in entries:
            wanted.add(tid)
            if "." in tid:
                wanted.add(tid.split(".")[0])

    docs = []
    for obj in bundle["objects"]:
        if obj.get("type") != "attack-pattern" or obj.get("revoked") or obj.get("x_mitre_deprecated"):
            continue
        refs = [r for r in obj.get("external_references", []) if r.get("source_name") == "mitre-attack"]
        if not refs or refs[0]["external_id"] not in wanted:
            continue
        tid = refs[0]["external_id"]
        body = _clean_attack_text(obj.get("description", ""))[:MAX_DOC_CHARS]
        docs.append({
            "id": f"ATTACK:{tid}",
            "kind": "attack_technique",
            "title": f"{tid} {obj['name']}",
            "tactics": [p["phase_name"] for p in obj.get("kill_chain_phases", [])],
            "text": f"MITRE ATT&CK {tid} ({obj['name']}). Tactics: {', '.join(p['phase_name'] for p in obj.get('kill_chain_phases', []))}. {body}",
            "source": f"MITRE ATT&CK v{catalog['version']}",
        })
    return sorted(docs, key=lambda d: d["id"])


def build_label_profiles(results_dir: Union[str, Path], catalog: Dict[str, Any]) -> List[Dict[str, Any]]:
    """One project-profile document per CIC-IDS2017 label, built from saved results and the mapping."""
    results_dir = Path(results_dir)
    audit = json.loads((results_dir / "reports" / "cicids2017_data_quality_report.json").read_text())
    metrics = json.loads((results_dir / "metrics" / "cicids2017_multiclass_xgboost.json").read_text())
    counts = audit["label_distribution"]["counts"]
    per_class = metrics["per_class"]
    labels = metrics["confusion_matrix"]["labels"]
    matrix = metrics["confusion_matrix"]["matrix"]

    docs = []
    for label in labels:
        status = label_status(label)
        if status == "benign":
            mapping_line = "Benign traffic; no ATT&CK technique."
        elif status == "unmapped":
            mapping_line = f"No ATT&CK technique assigned: {UNMAPPED_LABELS[label]}"
        else:
            parts = []
            for tid, conf, why in LABEL_TO_TECHNIQUES[label]:
                parts.append(f"{tid} ({catalog['techniques'][tid]['name']}, confidence {conf}): {why}")
            mapping_line = " ".join(parts)

        i = labels.index(label)
        total = sum(matrix[i])
        confusions = sorted(((matrix[i][j], labels[j]) for j in range(len(labels)) if j != i and matrix[i][j]), reverse=True)[:3]
        confusion_line = (
            "Most common errors on the held-out test split: "
            + ", ".join(f"{n} of {total} rows predicted as {name}" for n, name in confusions) + "."
            if confusions else "No errors on the held-out test split."
        )
        stats = per_class.get(label)
        perf_line = (
            f"Held-out test performance of the saved multiclass XGBoost model: precision {stats['precision']:.3f}, "
            f"recall {stats['recall']:.3f}, F1 {stats['f1']:.3f} on {stats['support']} test rows."
            if stats else "No held-out test statistics available."
        )
        text = (
            f"Attack label '{label}' in the CIC-IDS2017 dataset ({counts.get(label, 0)} rows after cleaning). "
            f"ATT&CK mapping: {mapping_line} {perf_line} {confusion_line}"
        )
        docs.append({
            "id": f"PROFILE:{label}",
            "kind": "label_profile",
            "title": f"CIC-IDS2017 label: {label}",
            "tactics": sorted({t for tid, _, _ in LABEL_TO_TECHNIQUES.get(label, []) for t in catalog["techniques"][tid]["tactics"]}),
            "text": text,
            "source": "project results (data quality report, saved multiclass XGBoost metrics, ATT&CK mapping)",
        })
    return docs


def build_documents(cfg=None) -> List[Dict[str, Any]]:
    cfg = cfg or load_config("configs/config.yaml")
    bundle = get_absolute_path(cfg.mitre_attack.stix_data_path)
    catalog = load_attack_catalog(bundle)
    results_dir = get_absolute_path(cfg.paths.results_dir)
    return build_attack_documents(catalog, bundle) + build_label_profiles(results_dir, catalog)


class ThreatIntelIndex:
    """FAISS index over the corpus with cosine-similarity retrieval."""

    def __init__(self, index: faiss.Index, documents: List[Dict[str, Any]], embedder):
        self.index = index
        self.documents = documents
        self._embedder = embedder

    @property
    def embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(self.model_name, local_files_only=True)
        return self._embedder

    model_name = "all-MiniLM-L6-v2"

    @classmethod
    def build(cls, documents: List[Dict[str, Any]], store_dir: Union[str, Path], model_name: str = "all-MiniLM-L6-v2") -> "ThreatIntelIndex":
        from sentence_transformers import SentenceTransformer

        embedder = SentenceTransformer(model_name, local_files_only=True)
        vectors = embedder.encode([d["text"] for d in documents], normalize_embeddings=True, convert_to_numpy=True)
        vectors = np.asarray(vectors, dtype=np.float32)
        index = _faiss().IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        store_dir = get_absolute_path(store_dir)
        store_dir.mkdir(parents=True, exist_ok=True)
        # Written as bytes: FAISS's path-based writer/reader fails intermittently on macOS (read error 0 != 1).
        (store_dir / INDEX_FILENAME).write_bytes(_faiss().serialize_index(index).tobytes())
        save_json({"model": model_name, "dimension": int(vectors.shape[1]), "documents": documents},
                  store_dir / DOCS_FILENAME)
        logger.info(f"Built FAISS index: {len(documents)} documents, dim {vectors.shape[1]}, at {store_dir}")
        return cls(index, documents, embedder)

    @classmethod
    def load(cls, store_dir: Union[str, Path]) -> "ThreatIntelIndex":
        store_dir = get_absolute_path(store_dir)
        meta = json.loads((store_dir / DOCS_FILENAME).read_text(encoding="utf-8"))
        raw = np.frombuffer((store_dir / INDEX_FILENAME).read_bytes(), dtype=np.uint8)
        index = _faiss().deserialize_index(raw)
        obj = cls(index, meta["documents"], None)
        obj.model_name = meta["model"]
        return obj

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        q = self.embedder.encode([query], normalize_embeddings=True, convert_to_numpy=True)
        scores, ids = self.index.search(np.asarray(q, dtype=np.float32), k)
        hits = []
        for score, i in zip(scores[0], ids[0]):
            if i < 0:
                continue
            doc = dict(self.documents[int(i)])
            doc["score"] = round(float(score), 4)
            hits.append(doc)
        return hits
