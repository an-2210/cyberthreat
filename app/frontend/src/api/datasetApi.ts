import { apiClient } from './client';
import { DatasetOverview, DatasetQualityAudit, ClassDistributionItem } from '../types/dataset';
import { isDemoModeActive, getMockDatasetOverview, getMockDatasetQuality, getMockClassDistribution } from '../mocks/mockAdapter';

export const datasetApi = {
  getOverview: async (): Promise<DatasetOverview | null> => {
    if (isDemoModeActive()) {
      return getMockDatasetOverview();
    }
    try {
      const res = await apiClient.get('/dataset/overview');
      return res.data?.data || res.data || null;
    } catch {
      return null; // Production empty state when API is offline
    }
  },

  getQualityAudit: async (): Promise<DatasetQualityAudit | null> => {
    if (isDemoModeActive()) {
      return getMockDatasetQuality();
    }
    try {
      const res = await apiClient.get('/dataset/quality');
      return res.data?.data || res.data || null;
    } catch {
      return null;
    }
  },

  getClassDistribution: async (): Promise<ClassDistributionItem[] | null> => {
    if (isDemoModeActive()) {
      return getMockClassDistribution();
    }
    try {
      const res = await apiClient.get('/dataset/class-distribution');
      return res.data?.data || res.data || null;
    } catch {
      return null;
    }
  },

  uploadDataset: async (file: File, datasetType: string): Promise<{ success: boolean; message: string }> => {
    if (isDemoModeActive()) {
      return { success: true, message: `[DEMO MODE] Dataset ${file.name} successfully analyzed for ${datasetType}.` };
    }
    const formData = new FormData();
    formData.append('file', file);
    formData.append('datasetType', datasetType);

    const res = await apiClient.post('/dataset/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return res.data;
  }
};
