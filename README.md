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

### 3. Run Exploratory Data Analysis (EDA)
Launch Jupyter Notebook to inspect EDA pipelines and figures:
```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

---

## 🧪 Scientific Rigor & Reproducibility

- **Centralized Configuration**: All hyperparameter and pipeline settings are controlled via `configs/config.yaml`.
- **Global Seed Control**: Global random seeds fixed via `set_seed(42)` across Python, NumPy, and PyTorch.
- **Leakage Prevention**: Standardizers, scalers, and encoders are fit **strictly** on training splits.
- **No Hallucinated Results**: All metrics, figures, and CTI mappings are generated dynamically from executed code.

---

## 🛡️ Security Statement

This software is strictly intended for **defensive cybersecurity research, intrusion detection, and automated threat intelligence analysis**. It does not contain offensive exploitation capabilities.
