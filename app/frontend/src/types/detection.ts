export type SeverityLevel = 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL' | 'BENIGN' | 'UNKNOWN';

export interface DetectionEvent {
  id: string;
  timestamp: string;
  sourceIp?: string | null;
  sourcePort?: number | null;
  destinationIp?: string | null;
  destinationPort?: number | null;
  protocol?: string | null;
  prediction: string; // e.g. 'BENIGN', 'DDoS', 'PortScan', 'Bot', etc.
  confidence: number | null; // float 0 to 1
  severity: SeverityLevel;
  status: 'ANALYZED' | 'FLAGGED' | 'RESOLVED' | string;
  rawFeatures?: Record<string, number | string>;
}

export interface DetectionOverview {
  totalEvents: number | null;
  benignEvents: number | null;
  maliciousEvents: number | null;
  attackClassesCount: number | null;
  recentDetections: DetectionEvent[];
  classCounts?: Record<string, number>;
}

export interface MLModelInfo {
  id: string;
  name: string;
  type: 'Baseline Supervised' | 'Anomaly Detection' | 'Fusion';
  phase: number;
  isSupportedByBackend: boolean;
  accuracy?: number | null;
  macroF1?: number | null;
  status: 'READY' | 'TRAINING' | 'DISABLED' | 'NOT IMPLEMENTED';
}

export interface PredictionBatchRequest {
  modelId: string;
  csvFileName: string;
  recordsCount?: number;
}

export interface PredictionBatchResponse {
  totalPredicted: number;
  predictions: DetectionEvent[];
  modelUsed: string;
  executionTimeMs: number;
}
