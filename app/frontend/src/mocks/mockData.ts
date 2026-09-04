import { DatasetOverview, DatasetQualityAudit, ClassDistributionItem } from '../types/dataset';
import { DetectionOverview, DetectionEvent, MLModelInfo } from '../types/detection';
import { ModelOverview, ModelMetrics, ConfusionMatrixData } from '../types/evaluation';
import { ExperimentRecord, RoadmapPhase } from '../types/experiment';

export const DEMO_DATA_TAG = "DEMO DATA (Phase 1-3 Baseline Experiments)";

// Benchmark Datasets Overview (CIC-IDS2017 baseline demo)
export const mockDatasetOverview: DatasetOverview = {
  datasetName: "CIC-IDS2017 (Processed Subset)",
  totalRecords: 2830743,
  totalFeatures: 78,
  totalClasses: 15,
  dataQualityScore: 99.4,
  lastAnalyzedTimestamp: "2026-09-04 10:30:00 UTC",
};

export const mockDatasetQualityAudit: DatasetQualityAudit = {
  filename: "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
  datasetType: "CIC-IDS2017",
  totalRows: 225745,
  totalColumns: 79,
  missingValuesCount: 288,
  missingValuesPercentage: 0.0016,
  infiniteValuesCount: 142,
  duplicateRowsCount: 3120,
  constantFeaturesCount: 8,
  categoricalFeaturesCount: 2,
  numericalFeaturesCount: 77,
};

export const mockClassDistribution: ClassDistributionItem[] = [
  { className: "BENIGN", sampleCount: 2273097, percentage: 80.3, isAttack: false },
  { className: "DoS / DDoS", sampleCount: 379772, percentage: 13.4, isAttack: true },
  { className: "PortScan", sampleCount: 158930, percentage: 5.6, isAttack: true },
  { className: "Brute Force", sampleCount: 13835, percentage: 0.49, isAttack: true },
  { className: "Web Attacks", sampleCount: 2180, percentage: 0.077, isAttack: true },
  { className: "Botnet", sampleCount: 1966, percentage: 0.069, isAttack: true },
  { className: "Infiltration", sampleCount: 963, percentage: 0.034, isAttack: true },
];

export const mockDetectionOverview: DetectionOverview = {
  totalEvents: 14500,
  benignEvents: 11800,
  maliciousEvents: 2700,
  attackClassesCount: 6,
  recentDetections: [
    {
      id: "EVT-89201",
      timestamp: "2026-09-04 12:45:12",
      sourceIp: "192.168.10.50",
      sourcePort: 49152,
      destinationIp: "172.16.0.1",
      destinationPort: 80,
      protocol: "TCP",
      prediction: "DDoS-LOIC",
      confidence: 0.994,
      severity: "CRITICAL",
      status: "FLAGGED",
    },
    {
      id: "EVT-89202",
      timestamp: "2026-09-04 12:45:10",
      sourceIp: "192.168.10.51",
      sourcePort: 54112,
      destinationIp: "172.16.0.1",
      destinationPort: 443,
      protocol: "TCP",
      prediction: "BENIGN",
      confidence: 0.998,
      severity: "BENIGN",
      status: "ANALYZED",
    },
    {
      id: "EVT-89203",
      timestamp: "2026-09-04 12:44:58",
      sourceIp: "192.168.10.120",
      sourcePort: 38920,
      destinationIp: "172.16.0.4",
      destinationPort: 22,
      protocol: "SSH",
      prediction: "SSH-Patator",
      confidence: 0.962,
      severity: "HIGH",
      status: "FLAGGED",
    },
    {
      id: "EVT-89204",
      timestamp: "2026-09-04 12:44:30",
      sourceIp: "192.168.10.88",
      sourcePort: 60234,
      destinationIp: "172.16.0.2",
      destinationPort: 8080,
      protocol: "HTTP",
      prediction: "PortScan",
      confidence: 0.941,
      severity: "MEDIUM",
      status: "ANALYZED",
    },
  ]
};

export const mockAvailableModels: MLModelInfo[] = [
  {
    id: "rf_baseline",
    name: "Random Forest (Baseline)",
    type: "Baseline Supervised",
    phase: 3,
    isSupportedByBackend: true,
    accuracy: 0.9972,
    macroF1: 0.9841,
    status: "READY"
  },
  {
    id: "xgboost_baseline",
    name: "XGBoost Classifier",
    type: "Baseline Supervised",
    phase: 3,
    isSupportedByBackend: true,
    accuracy: 0.9984,
    macroF1: 0.9892,
    status: "READY"
  },
  {
    id: "lr_baseline",
    name: "Logistic Regression",
    type: "Baseline Supervised",
    phase: 3,
    isSupportedByBackend: true,
    accuracy: 0.9420,
    macroF1: 0.8120,
    status: "READY"
  },
  {
    id: "svm_baseline",
    name: "Support Vector Machine (Linear)",
    type: "Baseline Supervised",
    phase: 3,
    isSupportedByBackend: true,
    accuracy: 0.9250,
    macroF1: 0.7840,
    status: "READY"
  },
  {
    id: "autoencoder_anomaly",
    name: "Deep Autoencoder (Anomaly)",
    type: "Anomaly Detection",
    phase: 4,
    isSupportedByBackend: false, // Disabled until Phase 4
    status: "NOT IMPLEMENTED"
  },
  {
    id: "hybrid_fusion",
    name: "Decision Fusion Engine",
    type: "Fusion",
    phase: 5,
    isSupportedByBackend: false, // Disabled until Phase 5
    status: "NOT IMPLEMENTED"
  }
];

export const mockModelOverview: ModelOverview = {
  bestModelName: "XGBoost Classifier (Baseline)",
  accuracy: 0.9984,
  macroF1: 0.9892,
  mcc: 0.9845,
  precision: 0.9912,
  recall: 0.9873,
  evaluatedModelsCount: 4,
};

export const mockModelMetrics: ModelMetrics[] = [
  {
    modelId: "xgboost_baseline",
    modelName: "XGBoost Classifier",
    datasetName: "CIC-IDS2017",
    accuracy: 0.9984,
    precision: 0.9912,
    recall: 0.9873,
    macroF1: 0.9892,
    weightedF1: 0.9982,
    mcc: 0.9845,
    fpr: 0.0012,
    fnr: 0.0031,
    trainingTimeSec: 142.5,
    inferenceTimeMsPerSample: 0.042,
    isEvaluated: true,
  },
  {
    modelId: "rf_baseline",
    modelName: "Random Forest",
    datasetName: "CIC-IDS2017",
    accuracy: 0.9972,
    precision: 0.9880,
    recall: 0.9802,
    macroF1: 0.9841,
    weightedF1: 0.9970,
    mcc: 0.9782,
    fpr: 0.0018,
    fnr: 0.0045,
    trainingTimeSec: 88.2,
    inferenceTimeMsPerSample: 0.038,
    isEvaluated: true,
  },
  {
    modelId: "lr_baseline",
    modelName: "Logistic Regression",
    datasetName: "CIC-IDS2017",
    accuracy: 0.9420,
    precision: 0.8340,
    recall: 0.7910,
    macroF1: 0.8120,
    weightedF1: 0.9380,
    mcc: 0.7510,
    fpr: 0.0240,
    fnr: 0.0890,
    trainingTimeSec: 18.4,
    inferenceTimeMsPerSample: 0.005,
    isEvaluated: true,
  },
  {
    modelId: "svm_baseline",
    modelName: "Support Vector Machine (Linear)",
    datasetName: "CIC-IDS2017",
    accuracy: 0.9250,
    precision: 0.8010,
    recall: 0.7680,
    macroF1: 0.7840,
    weightedF1: 0.9210,
    mcc: 0.7120,
    fpr: 0.0310,
    fnr: 0.1020,
    trainingTimeSec: 310.0,
    inferenceTimeMsPerSample: 0.120,
    isEvaluated: true,
  },
];

export const mockConfusionMatrix: ConfusionMatrixData = {
  modelId: "xgboost_baseline",
  modelName: "XGBoost Classifier",
  labels: ["BENIGN", "DDoS", "PortScan", "BruteForce", "WebAttack"],
  matrix: [
    [227000, 120, 80, 15, 5],
    [45, 37900, 20, 0, 0],
    [30, 10, 15850, 0, 0],
    [10, 0, 0, 1370, 5],
    [25, 0, 0, 10, 2125],
  ],
};

export const mockExperiments: ExperimentRecord[] = [
  {
    id: "EXP-2026-001",
    phase: 3,
    phaseName: "Baseline Model Comparison",
    datasetName: "CIC-IDS2017",
    modelName: "XGBoost Classifier",
    featuresCount: 78,
    hyperparameters: { n_estimators: 100, max_depth: 10, learning_rate: 0.1, random_state: 42 },
    trainingTimeSec: 142.5,
    inferenceTimeMs: 0.042,
    accuracy: 0.9984,
    precision: 0.9912,
    recall: 0.9873,
    macroF1: 0.9892,
    mcc: 0.9845,
    fpr: 0.0012,
    fnr: 0.0031,
    timestamp: "2026-09-03 16:30:00",
    status: "COMPLETED",
    notes: "Phase 3 primary benchmark run on CIC-IDS2017 preprocessed split."
  },
  {
    id: "EXP-2026-002",
    phase: 3,
    phaseName: "Baseline Model Comparison",
    datasetName: "CIC-IDS2017",
    modelName: "Random Forest",
    featuresCount: 78,
    hyperparameters: { n_estimators: 100, max_depth: 20, random_state: 42 },
    trainingTimeSec: 88.2,
    inferenceTimeMs: 0.038,
    accuracy: 0.9972,
    precision: 0.9880,
    recall: 0.9802,
    macroF1: 0.9841,
    mcc: 0.9782,
    fpr: 0.0018,
    fnr: 0.0045,
    timestamp: "2026-09-03 15:10:00",
    status: "COMPLETED",
    notes: "Leakage-safe standardization applied before cross-validation."
  },
  {
    id: "EXP-2026-003",
    phase: 3,
    phaseName: "Baseline Model Comparison",
    datasetName: "UNSW-NB15",
    modelName: "XGBoost Classifier",
    featuresCount: 42,
    hyperparameters: { n_estimators: 100, max_depth: 8, learning_rate: 0.1 },
    trainingTimeSec: 64.0,
    inferenceTimeMs: 0.035,
    accuracy: 0.9540,
    precision: 0.9410,
    recall: 0.9320,
    macroF1: 0.9365,
    mcc: 0.9120,
    fpr: 0.0120,
    fnr: 0.0240,
    timestamp: "2026-09-03 18:45:00",
    status: "COMPLETED",
    notes: "Cross-dataset validation split on UNSW-NB15 train/test splits."
  }
];

export const mockRoadmapPhases: RoadmapPhase[] = [
  {
    phaseNumber: 1,
    title: "Dataset Ingestion & Quality Audit",
    description: "Ingestion, validation, duplicate check, missing/inf values audit, class balance analysis.",
    status: "IMPLEMENTED",
    keyTechnologies: ["Pandas", "PyArrow", "Great Expectations", "NumPy"]
  },
  {
    phaseNumber: 2,
    title: "Leakage-Safe Preprocessing & Feature Engineering",
    description: "Fit-transform isolation on train split, variance thresholding, standard scaling, encodings.",
    status: "IMPLEMENTED",
    keyTechnologies: ["Scikit-learn", "StandardScaler", "OneHotEncoder"]
  },
  {
    phaseNumber: 3,
    title: "Baseline ML Models",
    description: "Supervised classification using Logistic Regression, Random Forest, XGBoost, SVM.",
    status: "IMPLEMENTED",
    keyTechnologies: ["XGBoost", "Scikit-Learn", "Optuna"]
  },
  {
    phaseNumber: 4,
    title: "Anomaly Detection Module",
    description: "Unsupervised anomaly detection via Deep Autoencoders & Isolation Forests for zero-day signals.",
    status: "NOT IMPLEMENTED",
    keyTechnologies: ["PyTorch", "Autoencoders", "IsolationForest"]
  },
  {
    phaseNumber: 5,
    title: "Decision-Level Fusion Engine",
    description: "Hybrid combination: Score = α * P_supervised + (1 - α) * Anomaly_score.",
    status: "NOT IMPLEMENTED",
    keyTechnologies: ["Custom Fusion Engine", "Calibrated Probabilities"]
  },
  {
    phaseNumber: 6,
    title: "Unknown Attack / Novelty Detection",
    description: "Statistical distance & clustering thresholding for unmapped network behavior.",
    status: "NOT IMPLEMENTED",
    keyTechnologies: ["HDBSCAN", "Mahalanobis Distance"]
  },
  {
    phaseNumber: 8,
    title: "SHAP Explainability Engine",
    description: "Feature attribution breakdown per alert using TreeSHAP and KernelSHAP.",
    status: "NOT IMPLEMENTED",
    keyTechnologies: ["SHAP", "TreeExplainer"]
  },
  {
    phaseNumber: 9,
    title: "MITRE ATT&CK Mapping",
    description: "Automated mapping of detected anomaly vectors to TTP IDs (e.g. T1498, T1110).",
    status: "NOT IMPLEMENTED",
    keyTechnologies: ["STIX/TAXII", "MITRE STIX API"]
  },
  {
    phaseNumber: 11,
    title: "RAG Threat Intelligence Retrieval",
    description: "FAISS vector database indexing CVE, NVD, and CTI advisories for context generation.",
    status: "NOT IMPLEMENTED",
    keyTechnologies: ["FAISS", "SentenceTransformers", "LangChain"]
  },
  {
    phaseNumber: 12,
    title: "Evidence-Grounded LLM CTI Reports",
    description: "Structured SOC Incident Report synthesis grounding threat context in retrieved RAG evidence.",
    status: "NOT IMPLEMENTED",
    keyTechnologies: ["LLaMA / Mistral", "Constrained Decoding"]
  }
];
