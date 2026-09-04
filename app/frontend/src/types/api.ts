export interface ApiResponse<T> {
  success: boolean;
  message?: string;
  data: T | null;
  timestamp: string;
  error?: string;
}

export interface BackendHealthStatus {
  isOnline: boolean;
  version: string;
  maxSupportedPhase: number;
  activeDataset?: string;
  environment: string;
}
