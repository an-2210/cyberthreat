export interface ModelMetrics {
  modelId: string;
  modelName: string;
  datasetName: string;
  accuracy: number | null;
  precision: number | null;
  recall: number | null;
  macroF1: number | null;
  weightedF1: number | null;
  mcc: number | null; // Matthews Correlation Coefficient
  fpr: number | null; // False Positive Rate
  fnr: number | null; // False Negative Rate
  trainingTimeSec: number | null;
  inferenceTimeMsPerSample: number | null;
  isEvaluated: boolean;
}

export interface ModelOverview {
  bestModelName: string | null;
  accuracy: number | null;
  macroF1: number | null;
  mcc: number | null;
  precision: number | null;
  recall: number | null;
  evaluatedModelsCount?: number;
}

export interface ConfusionMatrixData {
  modelId: string;
  modelName: string;
  labels: string[];
  matrix: number[][]; // Row: Actual, Column: Predicted
  normalizedMatrix?: number[][];
}

export interface CurvePoint {
  x: number; // e.g. FPR for ROC, Recall for PR
  y: number; // e.g. TPR for ROC, Precision for PR
  threshold?: number;
}

export interface EvaluationCurve {
  modelId: string;
  modelName: string;
  aucScore?: number;
  points: CurvePoint[];
}
