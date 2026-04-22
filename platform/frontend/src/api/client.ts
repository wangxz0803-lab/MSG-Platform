import axios, { AxiosInstance } from 'axios';

const baseURL = import.meta.env.VITE_API_BASE_URL ?? '';

export const apiClient: AxiosInstance = axios.create({
  baseURL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    // Normalize error messages for consumers.
    const detail =
      error?.response?.data?.detail ??
      error?.response?.data?.message ??
      error?.message ??
      'Unknown API error';
    return Promise.reject(new Error(typeof detail === 'string' ? detail : JSON.stringify(detail)));
  },
);

export default apiClient;
