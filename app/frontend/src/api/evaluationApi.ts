import { apiClient } from './client';
import { ModelOverview, ModelMetrics, ConfusionMatrixData } from '../types/evaluation';
import { isDemoModeActive, getMockModelOverview, getMockModelMetrics, getMockConfusionMatrix } from '../mocks/mockAdapter';

export const evaluationApi = {
  getModelOverview: async (): Promise<ModelOverview | null> => {
    if (isDemoModeActive()) {
      return getMockModelOverview();
    }
    try {
      const res = await apiClient.get('/evaluation/overview');
      return res.data?.data || res.data || null;
    } catch {
      return null;
    }
  },

  getAllModelMetrics: async (): Promise<ModelMetrics[]> => {
    if (isDemoModeActive()) {
      return getMockModelMetrics();
    }
    try {
      const res = await apiClient.get('/evaluation/metrics');
      return res.data?.data || res.data || [];
    } catch {
      return [];
    }
  },

  getConfusionMatrix: async (modelId: string): Promise<ConfusionMatrixData | null> => {
    if (isDemoModeActive()) {
      return getMockConfusionMatrix();
    }
    try {
      const res = await apiClient.get(`/evaluation/confusion-matrix?modelId=${modelId}`);
      return res.data?.data || res.data || null;
    } catch {
      return null;
    }
  }
};
