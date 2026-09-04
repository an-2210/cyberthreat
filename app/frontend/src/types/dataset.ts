export interface DatasetOverview {
  datasetName: string | null;
  totalRecords: number | null;
  totalFeatures: number | null;
  totalClasses: number | null;
  dataQualityScore: number | null; // e.g., percentage 0-100%
  lastAnalyzedTimestamp?: string | null;
}

export interface DatasetQualityAudit {
  filename: string | null;
  datasetType: 'CIC-IDS2017' | 'UNSW-NB15' | 'Custom CSV' | string;
  totalRows: number | null;
  totalColumns: number | null;
  missingValuesCount: number | null;
  missingValuesPercentage: number | null;
  infiniteValuesCount: number | null;
  duplicateRowsCount: number | null;
  constantFeaturesCount: number | null;
  categoricalFeaturesCount: number | null;
  numericalFeaturesCount: number | null;
}

export interface FeatureDistribution {
  featureName: string;
  type: 'numerical' | 'categorical';
  missingCount: number;
  uniqueValues: number;
  variance?: number;
  isConstant: boolean;
  min?: number;
  max?: number;
  mean?: number;
  std?: number;
}

export interface ClassDistributionItem {
  className: string;
  sampleCount: number;
  percentage: number;
  isAttack: boolean;
}

export interface CorrelationMatrixData {
  features: string[];
  matrix: number[][]; // 2D array of Pearson/Spearman coefficients
}
