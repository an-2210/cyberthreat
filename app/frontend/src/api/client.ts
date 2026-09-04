import axios, { AxiosInstance } from 'axios';

const getBaseUrl = (): string => {
  return import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api';
};

export const apiClient: AxiosInstance = axios.create({
  baseURL: getBaseUrl(),
  timeout: 15000,
  headers: {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
  },
});

// Update base URL dynamically when changed in Settings
export const updateApiBaseUrl = (newUrl: string): void => {
  apiClient.defaults.baseURL = newUrl;
  localStorage.setItem('CYBERTHREAT_API_BASE_URL', newUrl);
};

// Response Interceptor for Graceful Error Handling
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    // Log network/backend connection issues without throwing uncaught exceptions to React UI
    console.warn(`[API Client Warning] Backend endpoint unavailable: ${error.config?.url || 'Unknown'}`);
    return Promise.reject(error);
  }
);
