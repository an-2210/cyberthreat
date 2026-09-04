import { apiClient } from './client';
import { ExperimentRecord, RoadmapPhase } from '../types/experiment';
import { isDemoModeActive, getMockExperiments, getMockRoadmapPhases } from '../mocks/mockAdapter';

export const experimentApi = {
  getExperiments: async (): Promise<ExperimentRecord[]> => {
    if (isDemoModeActive()) {
      return getMockExperiments();
    }
    try {
      const res = await apiClient.get('/experiments/list');
      return res.data?.data || res.data || [];
    } catch {
      return [];
    }
  },

  getRoadmapPhases: async (): Promise<RoadmapPhase[]> => {
    if (isDemoModeActive()) {
      return getMockRoadmapPhases();
    }
    try {
      const res = await apiClient.get('/experiments/roadmap');
      return res.data?.data || res.data || getMockRoadmapPhases();
    } catch {
      return getMockRoadmapPhases();
    }
  }
};
