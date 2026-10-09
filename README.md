# AI-Based Cyber Threat Detection and Intelligence Generation

> **Research-Paper-Level Cybersecurity AI Architecture**  
> Supervised Classification + Anomaly Detection + Decision-Level Fusion + SHAP XAI + MITRE ATT&CK Mapping + RAG Threat Intelligence Report Generation.

---

## 📌 System Architecture

```text
RAW SECURITY DATA
        |
        v
DATA INGESTION & ADAPTERS
        |
        v
DATA VALIDATION & QUALITY AUDIT
        |
        v
LEAKAGE-SAFE PREPROCESSING
        |
        v
FEATURE ENGINEERING
        |
        +----------------------+
        |                      |
        v                      v
SUPERVISED CLASSIFIER    ANOMALY DETECTOR
(XGBoost / Random Forest) (Autoencoder / Isolation Forest)
        |                      |
        +----------+-----------+
                   |
                   v
             DECISION FUSION
     (Score = α * P_sup + (1-α) * A_score)
                   |
                   v
          FINAL THREAT DECISION & SEVERITY
                   |
                   v
             SHAP / XAI ENGINE
                   |
                   v
         MITRE ATT&CK MAPPING
                   |
                   v
        THREAT INTELLIGENCE KNOWLEDGE BASE
                   |
                   v
              RAG RETRIEVAL (FAISS)
                   |
                   v
                 LLM (Report Generator)
                   |
                   v
       STRUCTURED CTI INCIDENT REPORT
                   |
                   v
             SOC DASHBOARD
```

---

## 📊 Datasets & Placement Instructions

This system uses official network intrusion detection benchmark datasets:
1. **Primary Dataset**: [CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html) (UNB Canadian Institute for Cybersecurity)
2. **Secondary Dataset**: [UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset) (Cyber Range Lab of UNSW Canberra)

> **Important**: Raw datasets are NOT downloaded automatically from unverified mirrors. Place the official CSV files in the respective directories:

```text
data/raw/CICIDS2017/
├── Monday-WorkingHours.pcap_ISCX.csv
├── Tuesday-WorkingHours.pcap_ISCX.csv
├── Wednesday-workingHours.pcap_ISCX.csv
├── Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
├── Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
├── Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
├── Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
└── Friday-WorkingHours-Morning.pcap_ISCX.csv

data/raw/UNSW-NB15/
├── UNSW_NB15_training-set.csv
└── UNSW_NB15_testing-set.csv
```

---

## 📁 Repository Structure

```text
cyberthreat/
├── data/
│   ├── raw/
│   │   ├── CICIDS2017/
│   │   └── UNSW-NB15/
│   ├── interim/
│   ├── processed/
│   └── external/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_quality.ipynb
│   ├── 03_preprocessing.ipynb
│   ├── 04_baseline_models.ipynb
│   ├── 05_advanced_models.ipynb
│   ├── 06_anomaly_detection.ipynb
│   ├── 07_hybrid_fusion.ipynb
│   ├── 08_explainability.ipynb
│   ├── 09_mitre_mapping.ipynb
│   ├── 10_rag_cti.ipynb
│   └── 11_final_evaluation.ipynb
├── src/
│   ├── data/ (loaders.py, validators.py, dataset_adapters.py)
│   ├── preprocessing/ (cleaning.py, encoding.py, scaling.py, feature_engineering.py)
│   ├── models/ (baselines.py, xgboost_model.py, deep_models.py, model_factory.py)
│   ├── anomaly_detection/ (autoencoder.py, isolation_forest.py, anomaly_scoring.py)
│   ├── fusion/ (decision_fusion.py)
│   ├── explainability/ (shap_explainer.py)
│   ├── threat_intelligence/ (mitre_mapper.py, knowledge_base.py, retriever.py, rag_pipeline.py, report_generator.py)
│   ├── evaluation/ (metrics.py, experiments.py, ablation.py, cross_dataset.py)
│   └── utils/ (logging.py, config.py, reproducibility.py)
├── models/
│   ├── checkpoints/
│   └── artifacts/
├── configs/
│   └── config.yaml
├── results/
│   ├── figures/
│   ├── tables/
│   ├── metrics/
│   └── reports/
├── app/
│   ├── backend/
│   └── frontend/
├── tests/
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE
```

---

## ⚡ Quick Start & Setup

### 1. Environment Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Automated Verification Tests
```bash
PYTHONPATH=. pytest tests/ -v
```
The CIC-IDS2017 tests (`tests/test_cicids2017_synthetic.py`) use generated synthetic CSVs. They do **not** need the real dataset and do **not** measure detection performance.

### 3. Run Exploratory Data Analysis (EDA)
Launch Jupyter Notebook to inspect EDA pipelines and figures:
```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

> **macOS:** XGBoost needs the OpenMP runtime. If `import xgboost` fails with `Library not loaded: libomp.dylib`, run `brew install libomp`.

---

## 🧭 CIC-IDS2017 Detection Pipeline

Implemented scope: CIC-IDS2017 only, supervised Random Forest and XGBoost, binary (BENIGN vs ATTACK) and multiclass (one id per label) tasks.

### Run on the real CSVs
Place the official CSVs under `data/raw/CICIDS2017/`. Sub-folders are searched, so the `MachineLearningCVE/` folder from the official download works as-is. Then, from the repository root:
```bash
python -m src.run_cicids2017 --config configs/config.yaml
```
Useful options:
```bash
python -m src.run_cicids2017 --tasks binary                 # one task only
python -m src.run_cicids2017 --sampling none                # train on every training row (slower, more memory)
python -m src.run_cicids2017 --max-train-rows 100000        # cap the training split (0 = no cap)
python -m src.run_cicids2017 --raw-dir /path/to/csvs --results-dir /tmp/out
```
Missing expected files are reported and skipped, and the run continues with what is present. It fails only when no CSV is found at all.

### What the pipeline does
1. **Load**: reads each CSV separately, strips the padded header names, drops the duplicated `Fwd Header Length` column only when its values are identical to the original, and downcasts numerics to float32 to limit memory.
2. **Audit** (on the merged raw data, before cleaning): missing values, `Infinity`/infinite values, negatives and out-of-range ports, exact duplicates, feature rows with conflicting labels, constant columns, class distribution and per-file label composition.
3. **Leakage guard**: removes identifier-like columns by name (flow IDs, IP addresses, timestamps, label-derived names). Scans the **training** split for single features with near-perfect separability and reports them.
4. **Split, then fit**: removes exact duplicate rows, applies a stratified train/validation/test split, and fits imputation, constant-feature removal, scaling and encoding on the training split only.
5. **Sampling**: the training split can be capped at `training.sampling.max_train_rows`, keeping label proportions. The same rows are used for both models. Validation and test rows are never subsampled.
6. **Train**: Random Forest, and XGBoost with early stopping on the validation split.
7. **Evaluate** on the held-out test split: precision, recall, F1 and FPR/FNR for the attack class; macro/weighted scores, per-class results and one-vs-rest ROC-AUC for multiclass; confusion matrices; ROC curve for the binary task.

### Outputs
| Path | Content |
|---|---|
| `results/reports/cicids2017_data_quality_report.json` | Audit: missing, infinite, duplicates, invalid values, class counts |
| `results/reports/cicids2017_run_summary.json` | Run record: seed, sampling, split sizes, loader report, feature list, data source |
| `results/tables/cicids2017_column_quality.csv`, `cicids2017_class_distribution.csv`, `cicids2017_class_by_source_file.csv` | Audit tables |
| `results/tables/cicids2017_single_feature_leakage_scan.csv` | Per-feature separability on the training split |
| `results/tables/cicids2017_model_comparison.csv` | Test-set comparison of all models and tasks |
| `results/metrics/cicids2017_{binary,multiclass}_{random_forest,xgboost}.json` | Full test metrics, confusion matrix, top feature importances |
| `results/figures/*.png` | Class distribution, label composition by file, column quality, confusion matrices, binary ROC |
| `models/artifacts/cicids2017_preprocessing_pipeline.joblib`, `cicids2017_{task}_{model}.joblib` | Fitted preprocessing and classifiers |

Every output records `data_source`. Real runs say `real CIC-IDS2017 CSVs`; the synthetic tests say `synthetic test fixture`.

### Configuration
All settings are in `configs/config.yaml`: `training.tasks`, `training.sampling` (strategy and row cap), `models.baselines.random_forest`, `models.advanced.xgboost`, `data.cicids2017` (identifier patterns, leakage-scan size and threshold), and the split sizes under `data`. `system.seed` seeds the split, the sampler and both models.

### Using the saved models (inference)
Artifacts live in `models/artifacts/`; `cicids2017_schema.json` lists the input columns, model features and labels.
```python
from src.models.inference import CICIDS2017Detector
det = CICIDS2017Detector.load("binary", "xgboost")   # task: binary | multiclass; model: random_forest | xgboost
labels = det.predict(raw_df)        # 'BENIGN' / 'ATTACK' (multiclass: the 15 label names)
proba = det.predict_proba(raw_df)   # (n_rows, n_labels); columns follow det.labels
```
- **Input**: a DataFrame of the 77 raw feature columns (`schema["input_columns"]`), with stripped names and no `Label` column. Extra columns are ignored; missing ones raise `ValueError`.
- **Preprocessing**: the saved pipeline (`cicids2017_preprocessing_pipeline.joblib`) is fitted on the training split only. It is applied to inputs automatically.
- **Labels**: binary `[BENIGN, ATTACK]` (1 = attack); multiclass order = `schema["tasks"]["multiclass"]["labels"]`, which is also the column order of `predict_proba`.
- `predict` equals the argmax of `predict_proba` for every saved model (checked by tests and on real rows).

### Known limitations
- Metrics are only meaningful when produced by running the pipeline on the official CSVs. The synthetic tests check code behaviour only.
- The default training cap (500,000 rows) keeps runtime and model size manageable. Set `max_train_rows: null` to train on the full training split.
- Rare classes (Heartbleed, Infiltration, SQL injection) have very few rows. They can be absent from the test split, and their per-class scores are then undefined.
- Duplicate removal is exact (all feature values and the label must match).
- Not yet implemented, by design: UNSW-NB15, anomaly detection, SHAP, MITRE mapping, RAG and the dashboard.

---

## 🧪 Scientific Rigor & Reproducibility

- **Centralized Configuration**: All hyperparameter and pipeline settings are controlled via `configs/config.yaml`.
- **Global Seed Control**: Global random seeds fixed via `set_seed(42)` across Python, NumPy, and PyTorch.
- **Leakage Prevention**: Standardizers, scalers, and encoders are fit **strictly** on training splits.
- **No Hallucinated Results**: All metrics, figures, and CTI mappings are generated dynamically from executed code.

---

## 🛡️ Security Statement

This software is strictly intended for **defensive cybersecurity research, intrusion detection, and automated threat intelligence analysis**. It does not contain offensive exploitation capabilities.
