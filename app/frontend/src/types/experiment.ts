export type ExperimentStatus = 'COMPLETED' | 'RUNNING' | 'FAILED' | 'PLANNED' | 'NOT IMPLEMENTED';

export interface ExperimentRecord {
  id: string; // e.g. 'EXP-2026-001'
  phase: number;
  phaseName: string;
  datasetName: string;
  modelName: string;
  featuresCount: number | null;
  hyperparameters: Record<string, string | number | boolean>;
  trainingTimeSec: number | null;
  inferenceTimeMs: number | null;
  accuracy: number | null;
  precision: number | null;
  recall: number | null;
  macroF1: number | null;
  mcc: number | null;
  fpr: number | null;
  fnr: number | null;
  timestamp: string;
  status: ExperimentStatus;
  notes?: string;
}

export interface RoadmapPhase {
  phaseNumber: number;
  title: string;
  description: string;
  status: 'IMPLEMENTED' | 'NOT IMPLEMENTED';
  keyTechnologies: string[];
}
