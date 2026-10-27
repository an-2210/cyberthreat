// Every call to the backend goes through this file, so the backend address lives in one place.
// Set VITE_API_URL in app/frontend/.env to point somewhere else (default: the local FastAPI server).

export const API_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000';

export interface Overview {
  dataSource: string;
  seed: number;
  elapsedSeconds: number;
  splitSizes: { train_before_sampling: number; train_used: number; validation: number; test: number };
  duplicateRowsRemoved: number;
  identifierColumnsDropped: string[];
  leakageFlaggedFeatures: string[];
  classNames: string[];
  featureCount: number;
  sampling: { strategy: string; rows_before: number; rows_after: number; max_rows: number | null };
}

export interface ClassCount {
  label: string;
  count: number;
  percent: number;
}

export interface DataSummary {
  rows: number;
  featureColumns: number;
  missingCells: number;
  missingByFeature: Record<string, number>;
  infiniteCells: number;
  infiniteByFeature: Record<string, number>;
  duplicateRows: number;
  conflictingFeatureRows: number;
  constantFeatures: string[];
  classes: ClassCount[];
}

export interface ConfusionMatrix {
  labels: string[];
  matrix: number[][];
}

export interface PerClass {
  precision: number;
  recall: number;
  f1: number;
  support: number;
}

export interface ModelResult {
  key: string;
  task: 'binary' | 'multiclass';
  model: 'random_forest' | 'xgboost';
  accuracy: number;
  mcc: number;
  precision: number;
  recall: number;
  f1: number;
  rocAuc: number | null;
  fpr: number | null;
  trainSeconds: number;
  trainRows: number;
  testRows: number;
  confusionMatrix: ConfusionMatrix;
  perClass: Record<string, PerClass> | null;
  topFeatures: [string, number][];
}

export interface RareClassRow {
  variant: string;
  model: string;
  macro_f1: number;
  weighted_f1: number;
  accuracy: number;
  roc_auc_ovr_macro: number;
}

export interface PredictionResult {
  modelKey: string;
  rowsClassified: number;
  counts: Record<string, number>;
  benignRows: number;
  rows: { row: number; prediction: string; confidence: number }[];
  milliseconds: number;
}

export interface DetectorRow {
  key: string;
  label: string;
  recall: number;
  precision: number;
  f1: number;
  fpr: number;
  rocAuc: number;
}

export interface DetectorsResponse {
  targetFpr: number;
  detectors: DetectorRow[];
  anomalyAucOnValidation: Record<string, number>;
  chosenAnomalyDetector: string;
  fusion: { alpha: number; threshold: number; validationF1: number };
}

export interface HeldOutFold {
  heldOut: string;
  heldOutLabels?: string[];
  testRows: number;
  supervised: number;
  autoencoder: number;
  fusion: number;
  fusionAlpha: number;
  falsePositiveRate: number;
}

export interface UnseenAttacksResponse {
  targetFpr: number;
  baseline: HeldOutFold;
  folds: HeldOutFold[];
  meanDetection: Record<string, number>;
  detectedAtLeastHalf: Record<string, number>;
  nHeldOut: number;
  nLowSample: number;
}

export interface BinaryMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  fpr: number;
  roc_auc: number;
}

export interface UnswResponse {
  native: {
    featureCount: number;
    threshold: number;
    test: BinaryMetrics;
    confusionMatrix: ConfusionMatrix;
    topFeatures: { feature: string; importance: number }[];
    rows: { train_rows: number; validation_rows: number; test_rows: number };
    leakageFlagged: string[];
    leakageTopFeature: { feature: string; auc_separability: number };
  };
  ttl: {
    variants: { key: string; label: string; featureCount: number; test: BinaryMetrics; fuzzersRecall: number }[];
    perCategory: { category: string; support: number; all_features: number; without_ttl: number; only_ttl: number }[];
  };
  fuzzers: {
    chosenOnValidation: string;
    variants: { key: string; label: string; fuzzersRecall: number; f1: number }[];
    seedRuns: { key: string; label: string; fuzzersRecall: { mean: number; std: number }; f1: { mean: number; std: number } }[];
  };
  transfer: {
    cicInDomain: BinaryMetrics;
    transferAtHalf: BinaryMetrics;
    calibrated: BinaryMetrics & { threshold: number };
    unswInDomain: BinaryMetrics & { threshold: number };
    featureShift: { feature: string; aucCic: number; aucUnsw: number }[];
  };
}

export interface OllamaStatus {
  reachable: boolean;
  model: string;
  host: string;
  modelInstalled: boolean;
  message: string;
}

export interface ReportFacts {
  rows: number;
  flaggedAttack: number;
  benign: number;
  highConfidenceAttack: number;
  borderlineAttack: number;
  meanAttackConfidence: number | null;
  attackTypes: { label: string; count: number; meanConfidence: number; risk: string }[];
  flaggedWithoutType: number;
}

export interface GeneratedReport {
  model: string;
  headline: string;
  summary: string;
  findings: string[];
  nextSteps: string[];
  referencesUsed: string[];
  facts: ReportFacts;
}

function withFile(file: File): FormData {
  const form = new FormData();
  form.append('file', file);
  return form;
}

export interface AnalyzedRow {
  row: number;
  prediction: 'ATTACK' | 'BENIGN';
  attackType: string | null;
  confidence: number;
  risk: string;
  riskScore: number | null;
}

export interface AnalysisResult {
  rowsAnalyzed: number;
  attackRows: number;
  benignRows: number;
  riskCounts: Record<string, number>;
  attackTypeCounts: Record<string, number>;
  mitre: Record<string, { technique: string; confidence: string }[]>;
  rows: AnalyzedRow[];
}

export interface FlowExplanation {
  row: number;
  prediction: 'ATTACK' | 'BENIGN';
  attackProbability: number;
  expected_value_log_odds: number;
  margin_total_log_odds: number;
  top_features: { feature: string; value: number; shap: number; direction: string }[];
}

export type ExperimentRow = Record<string, string | number | null>;

async function postForm<T>(path: string, form: FormData): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, { method: 'POST', body: form });
  if (!res.ok) throw new Error(await errorText(res));
  return res.json() as Promise<T>;
}

async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(`${API_URL}${path}`);
  if (!res.ok) throw new Error(await errorText(res));
  return res.json() as Promise<T>;
}

async function errorText(res: Response): Promise<string> {
  try {
    const body = await res.json();
    return typeof body.detail === 'string' ? body.detail : `Request failed (${res.status})`;
  } catch {
    return `Request failed (${res.status})`;
  }
}

export const api = {
  health: () => getJson<{ ok: boolean; pipelineRunFound: boolean }>('/api/health'),
  overview: () => getJson<Overview>('/api/overview'),
  data: () => getJson<DataSummary>('/api/data'),
  models: () => getJson<ModelResult[]>('/api/models'),
  rareClass: () => getJson<RareClassRow[]>('/api/rare-class'),
  figures: () => getJson<string[]>('/api/figures'),
  detectors: () => getJson<DetectorsResponse>('/api/detectors'),
  unseenAttacks: () => getJson<UnseenAttacksResponse>('/api/unseen-attacks'),
  familyHoldout: () => getJson<UnseenAttacksResponse>('/api/family-holdout'),
  unsw: () => getJson<UnswResponse>('/api/unsw'),
  ollama: () => getJson<OllamaStatus>('/api/ollama'),
  summarize: (file: File) => postForm<GeneratedReport>('/api/summarize', withFile(file)),
  analyze: (file: File) => postForm<AnalysisResult>('/api/analyze', withFile(file)),
  explain: (file: File, row: number) => {
    const form = withFile(file);
    form.append('row', String(row));
    return postForm<FlowExplanation>('/api/explain', form);
  },
  experiments: () => getJson<ExperimentRow[]>('/api/experiments'),
  figureUrl: (name: string) => `${API_URL}/figures/${name}`,

  predict: async (file: File, modelKey: string): Promise<PredictionResult> => {
    const form = new FormData();
    form.append('file', file);
    form.append('modelKey', modelKey);
    const res = await fetch(`${API_URL}/api/predict`, { method: 'POST', body: form });
    if (!res.ok) throw new Error(await errorText(res));
    return res.json() as Promise<PredictionResult>;
  },
};
