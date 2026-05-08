import apiClient from './client';
import type {
  BatchJobCreateRequest,
  BatchJobCreateResponse,
  ChannelDetail,
  ChannelListResponse,
  CompareResponse,
  CreateCollectRequest,
  CreateJobRequest,
  CreateJobResponse,
  DatasetExportRequest,
  DatasetFilters,
  DatasetsResponse,
  ExportModelRequest,
  ExportsResponse,
  HealthResponse,
  Job,
  JobFilters,
  JobLogs,
  JobProgress,
  JobsResponse,
  LeaderboardResponse,
  ModelEvalRequest,
  ModelFilters,
  ModelInferRequest,
  ModelMeta,
  ModelsResponse,
  ModelUploadResponse,
  Run,
  RunFilters,
  RunMetrics,
  RunsResponse,
  SamplesResponse,
  SplitComputeRequest,
  SplitInfoResponse,
  TopologyPreviewRequest,
  TopologyPreviewResponse,
} from './types';

// --- Health -----------------------------------------------------------------
export async function getHealth(): Promise<HealthResponse> {
  const { data } = await apiClient.get<HealthResponse>('/api/health');
  return data;
}

// --- Datasets ---------------------------------------------------------------
export async function getDatasets(filters: DatasetFilters = {}): Promise<DatasetsResponse> {
  const { data } = await apiClient.get<DatasetsResponse>('/api/datasets', { params: filters });
  return data;
}

export async function getDatasetSamples(
  source: string,
  params: { link?: string; limit?: number; offset?: number } = {},
): Promise<SamplesResponse> {
  const { data } = await apiClient.get<SamplesResponse>(
    `/api/datasets/${encodeURIComponent(source)}/samples`,
    { params },
  );
  return data;
}

export async function collectDataset(req: CreateCollectRequest): Promise<CreateJobResponse> {
  const { data } = await apiClient.post<CreateJobResponse>('/api/datasets/collect', req);
  return data;
}

// --- Split management -------------------------------------------------------
export async function getSplitStatus(): Promise<SplitInfoResponse> {
  const { data } = await apiClient.get<SplitInfoResponse>('/api/datasets/split/status');
  return data;
}

export async function computeAndLockSplit(req: SplitComputeRequest): Promise<SplitInfoResponse> {
  const { data } = await apiClient.post<SplitInfoResponse>('/api/datasets/split', req);
  return data;
}

export async function unlockSplit(): Promise<void> {
  await apiClient.post('/api/datasets/split/unlock');
}

// --- Dataset export ---------------------------------------------------------
export async function exportDataset(req: DatasetExportRequest): Promise<CreateJobResponse> {
  const { data } = await apiClient.post<CreateJobResponse>('/api/datasets/export', req);
  return data;
}

export async function listExports(): Promise<ExportsResponse> {
  const { data } = await apiClient.get<ExportsResponse>('/api/datasets/exports');
  return data;
}

// --- Model upload -----------------------------------------------------------
export async function uploadModel(file: File, runId?: string, tags?: string): Promise<ModelUploadResponse> {
  const form = new FormData();
  form.append('file', file);
  if (runId) form.append('run_id', runId);
  if (tags) form.append('tags', tags);
  const { data } = await apiClient.post<ModelUploadResponse>('/api/models/upload', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 300000,
  });
  return data;
}

export async function evaluateModel(
  runId: string,
  req: ModelEvalRequest = {},
): Promise<CreateJobResponse> {
  const { data } = await apiClient.post<CreateJobResponse>(
    `/api/models/${encodeURIComponent(runId)}/evaluate`,
    req,
  );
  return data;
}

// --- Model inference --------------------------------------------------------
export async function inferModel(
  runId: string,
  req: ModelInferRequest = {},
): Promise<CreateJobResponse> {
  const { data } = await apiClient.post<CreateJobResponse>(
    `/api/models/${encodeURIComponent(runId)}/infer`,
    req,
  );
  return data;
}

// --- Model meta -------------------------------------------------------------
export async function getModelMeta(runId: string): Promise<ModelMeta> {
  const { data } = await apiClient.get<ModelMeta>(
    `/api/models/${encodeURIComponent(runId)}/meta`,
  );
  return data;
}

// --- Leaderboard ------------------------------------------------------------
export async function getLeaderboard(): Promise<LeaderboardResponse> {
  const { data } = await apiClient.get<LeaderboardResponse>('/api/models/leaderboard');
  return data;
}

// --- Jobs -------------------------------------------------------------------
export async function getJobs(filters: JobFilters = {}): Promise<JobsResponse> {
  const { data } = await apiClient.get<JobsResponse>('/api/jobs', { params: filters });
  return data;
}

export async function createJob(req: CreateJobRequest): Promise<Job> {
  const { data } = await apiClient.post<Job>('/api/jobs', req);
  return data;
}

export async function createBatchJobs(payload: BatchJobCreateRequest): Promise<BatchJobCreateResponse> {
  const res = await apiClient.post('/api/jobs/batch', payload);
  return res.data;
}

export async function getJob(jobId: string): Promise<Job> {
  const { data } = await apiClient.get<Job>(`/api/jobs/${encodeURIComponent(jobId)}`);
  return data;
}

export async function getJobProgress(jobId: string): Promise<JobProgress> {
  const { data } = await apiClient.get<JobProgress>(
    `/api/jobs/${encodeURIComponent(jobId)}/progress`,
  );
  return data;
}

export async function getJobLogs(jobId: string, tail = 500): Promise<JobLogs> {
  const { data } = await apiClient.get<JobLogs>(`/api/jobs/${encodeURIComponent(jobId)}/logs`, {
    params: { tail },
  });
  return data;
}

export async function cancelJob(jobId: string): Promise<void> {
  await apiClient.post(`/api/jobs/${encodeURIComponent(jobId)}/cancel`);
}

export async function deleteDataset(source: string): Promise<void> {
  await apiClient.delete(`/api/datasets/${encodeURIComponent(source)}`);
}

export async function deleteJob(jobId: string): Promise<void> {
  await apiClient.delete(`/api/jobs/${encodeURIComponent(jobId)}`);
}

export async function deleteRun(runId: string): Promise<void> {
  await apiClient.delete(`/api/runs/${encodeURIComponent(runId)}`);
}

// --- Runs -------------------------------------------------------------------
export async function getRuns(filters: RunFilters = {}): Promise<RunsResponse> {
  const { data } = await apiClient.get<RunsResponse>('/api/runs', { params: filters });
  return data;
}

export async function getRun(runId: string): Promise<Run> {
  const { data } = await apiClient.get<Run>(`/api/runs/${encodeURIComponent(runId)}`);
  return data;
}

export async function getRunMetrics(runId: string): Promise<RunMetrics> {
  const { data } = await apiClient.get<RunMetrics>(
    `/api/runs/${encodeURIComponent(runId)}/metrics`,
  );
  return data;
}

export async function compareRuns(ids: string[]): Promise<CompareResponse> {
  const { data } = await apiClient.get<CompareResponse>('/api/runs/compare', {
    params: { ids: ids.join(',') },
  });
  return data;
}

// --- Models -----------------------------------------------------------------
export async function getModels(filters: ModelFilters = {}): Promise<ModelsResponse> {
  const { data } = await apiClient.get<ModelsResponse>('/api/models', { params: filters });
  return data;
}

export async function exportModel(
  runId: string,
  req: ExportModelRequest,
): Promise<CreateJobResponse> {
  const { data } = await apiClient.post<CreateJobResponse>(
    `/api/models/${encodeURIComponent(runId)}/export`,
    req,
  );
  return data;
}

// --- Topology ---------------------------------------------------------------
export async function previewTopology(req: TopologyPreviewRequest): Promise<TopologyPreviewResponse> {
  const { data } = await apiClient.post<TopologyPreviewResponse>('/api/topology/preview', req);
  return data;
}

// --- Configs ----------------------------------------------------------------
export async function getConfigSchema(): Promise<Record<string, unknown>> {
  const { data } = await apiClient.get<Record<string, unknown>>('/api/configs/schema');
  return data;
}

export async function getConfigDefaults(section?: string): Promise<Record<string, unknown>> {
  const { data } = await apiClient.get<Record<string, unknown>>('/api/configs/defaults', {
    params: section ? { section } : undefined,
  });
  return data;
}

// --- Channels ---------------------------------------------------------------
export async function getChannels(params: { limit?: number; offset?: number } = {}): Promise<ChannelListResponse> {
  const { data } = await apiClient.get<ChannelListResponse>('/api/channels', { params });
  return data;
}

export async function getChannelDetail(index: number): Promise<ChannelDetail> {
  const { data } = await apiClient.get<ChannelDetail>(`/api/channels/${index}`);
  return data;
}
