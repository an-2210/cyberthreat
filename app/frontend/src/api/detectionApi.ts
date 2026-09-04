import { apiClient } from './client';
import { DetectionOverview, MLModelInfo, PredictionBatchResponse } from '../types/detection';
import { isDemoModeActive, getMockDetectionOverview, getMockAvailableModels } from '../mocks/mockAdapter';

export const detectionApi = {
  getOverview: async (): Promise<DetectionOverview | null> => {
    if (isDemoModeActive()) {
      return getMockDetectionOverview();
    }
    try {
      const res = await apiClient.get('/detection/overview');
      return res.data?.data || res.data || null;
    } catch {
      return null;
    }
  },

  getAvailableModels: async (): Promise<MLModelInfo[]> => {
    if (isDemoModeActive()) {
      return getMockAvailableModels();
    }
    try {
      const res = await apiClient.get('/detection/models');
      return res.data?.data || res.data || [];
    } catch {
      // Default to Phase 3 supported baseline models if backend is unreachable
      return [
        { id: 'rf_baseline', name: 'Random Forest', type: 'Baseline Supervised', phase: 3, isSupportedByBackend: true, status: 'READY' },
        { id: 'xgboost_baseline', name: 'XGBoost Classifier', type: 'Baseline Supervised', phase: 3, isSupportedByBackend: true, status: 'READY' },
        { id: 'lr_baseline', name: 'Logistic Regression', type: 'Baseline Supervised', phase: 3, isSupportedByBackend: true, status: 'READY' },
        { id: 'svm_baseline', name: 'SVM (Linear)', type: 'Baseline Supervised', phase: 3, isSupportedByBackend: true, status: 'READY' },
        { id: 'autoencoder', name: 'Autoencoder (Phase 4)', type: 'Anomaly Detection', phase: 4, isSupportedByBackend: false, status: 'NOT IMPLEMENTED' },
        { id: 'fusion', name: 'Decision Fusion (Phase 5)', type: 'Fusion', phase: 5, isSupportedByBackend: false, status: 'NOT IMPLEMENTED' },
      ];
    }
  },

  predictBatch: async (file: File, modelId: string): Promise<PredictionBatchResponse | null> => {
    if (isDemoModeActive()) {
      const mock = getMockDetectionOverview();
      return {
        totalPredicted: mock.recentDetections.length,
        predictions: mock.recentDetections,
        modelUsed: modelId,
        executionTimeMs: 145,
      };
    }

    const formData = new FormData();
    formData.append('file', file);
    formData.append('modelId', modelId);

    const res = await apiClient.post('/detection/predict', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return res.data?.data || res.data || null;
  }
};
