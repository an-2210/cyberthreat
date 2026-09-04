import {
  mockDatasetOverview,
  mockDatasetQualityAudit,
  mockClassDistribution,
  mockDetectionOverview,
  mockAvailableModels,
  mockModelOverview,
  mockModelMetrics,
  mockConfusionMatrix,
  mockExperiments,
  mockRoadmapPhases
} from './mockData';

export const isDemoModeActive = (): boolean => {
  const stored = localStorage.getItem('CYBERTHREAT_DEMO_MODE');
  if (stored !== null) {
    return stored === 'true';
  }
  // Default to false for production empty-state API integrity
  return false;
};

export const setDemoMode = (enabled: boolean): void => {
  localStorage.setItem('CYBERTHREAT_DEMO_MODE', enabled ? 'true' : 'false');
  window.dispatchEvent(new Event('demo-mode-change'));
};

export const getMockDatasetOverview = () => mockDatasetOverview;
export const getMockDatasetQuality = () => mockDatasetQualityAudit;
export const getMockClassDistribution = () => mockClassDistribution;
export const getMockDetectionOverview = () => mockDetectionOverview;
export const getMockAvailableModels = () => mockAvailableModels;
export const getMockModelOverview = () => mockModelOverview;
export const getMockModelMetrics = () => mockModelMetrics;
export const getMockConfusionMatrix = () => mockConfusionMatrix;
export const getMockExperiments = () => mockExperiments;
export const getMockRoadmapPhases = () => mockRoadmapPhases;
