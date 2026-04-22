import {
  useMutation,
  useQuery,
  useQueryClient,
  type UseQueryOptions,
} from '@tanstack/react-query';
import {
  cancelJob,
  collectDataset,
  compareRuns,
  createBatchJobs,
  createJob,
  deleteDataset,
  deleteJob,
  deleteRun,
  exportModel,
  getChannelDetail,
  getChannels,
  getConfigDefaults,
  getConfigSchema,
  getDatasetSamples,
  getDatasets,
  getHealth,
  getJob,
  getJobLogs,
  getJobProgress,
  getJobs,
  getModels,
  getRun,
  getRunMetrics,
  getRuns,
  previewTopology,
} from './endpoints';
import type {
  CreateCollectRequest,
  CreateJobRequest,
  DatasetFilters,
  ExportModelRequest,
  JobFilters,
  JobsResponse,
  ModelFilters,
  RunFilters,
  TopologyPreviewRequest,
} from './types';

// --- Health -----------------------------------------------------------------
export function useHealth() {
  return useQuery({
    queryKey: ['health'],
    queryFn: getHealth,
    refetchInterval: 15000,
    staleTime: 10000,
  });
}

// --- Datasets ---------------------------------------------------------------
export function useDatasets(filters: DatasetFilters = {}) {
  return useQuery({
    queryKey: ['datasets', filters],
    queryFn: () => getDatasets(filters),
  });
}

export function useDatasetSamples(
  source: string | undefined,
  params: { link?: string; limit?: number; offset?: number } = {},
) {
  return useQuery({
    queryKey: ['dataset-samples', source, params],
    queryFn: () => getDatasetSamples(source as string, params),
    enabled: Boolean(source),
  });
}

export function useCollectDataset() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (req: CreateCollectRequest) => collectDataset(req),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['jobs'] });
    },
  });
}

export function useDeleteDataset() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (source: string) => deleteDataset(source),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['datasets'] });
    },
  });
}

// --- Jobs -------------------------------------------------------------------
export function useJobs(filters: JobFilters = {}, options?: Partial<UseQueryOptions<JobsResponse>>) {
  return useQuery({
    queryKey: ['jobs', filters],
    queryFn: () => getJobs(filters),
    refetchInterval: 5000,
    ...options,
  });
}

export function useJob(jobId: string | undefined) {
  return useQuery({
    queryKey: ['job', jobId],
    queryFn: () => getJob(jobId as string),
    enabled: Boolean(jobId),
    refetchInterval: (query) => {
      const data = query.state.data;
      if (!data) return 2000;
      return data.status === 'running' || data.status === 'queued' ? 2000 : false;
    },
  });
}

export function useJobProgress(jobId: string | undefined) {
  return useQuery({
    queryKey: ['job-progress', jobId],
    queryFn: () => getJobProgress(jobId as string),
    enabled: Boolean(jobId),
    refetchInterval: (query) => {
      const data = query.state.data;
      if (!data) return 2000;
      return data.status === 'running' || data.status === 'queued' ? 2000 : false;
    },
  });
}

export function useJobLogs(jobId: string | undefined, tail = 500, enabled = true) {
  return useQuery({
    queryKey: ['job-logs', jobId, tail],
    queryFn: () => getJobLogs(jobId as string, tail),
    enabled: Boolean(jobId) && enabled,
    refetchInterval: 2000,
  });
}

export function useCreateJob() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (req: CreateJobRequest) => createJob(req),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['jobs'] });
    },
  });
}

export function useCreateBatchJobs() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: createBatchJobs,
    onSuccess: () => qc.invalidateQueries({ queryKey: ['jobs'] }),
  });
}

export function useCancelJob() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (jobId: string) => cancelJob(jobId),
    onSuccess: (_data, jobId) => {
      qc.invalidateQueries({ queryKey: ['jobs'] });
      qc.invalidateQueries({ queryKey: ['job', jobId] });
    },
  });
}

export function useDeleteJob() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (jobId: string) => deleteJob(jobId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['jobs'] });
    },
  });
}

// --- Runs -------------------------------------------------------------------
export function useRuns(filters: RunFilters = {}) {
  return useQuery({
    queryKey: ['runs', filters],
    queryFn: () => getRuns(filters),
  });
}

export function useRun(runId: string | undefined) {
  return useQuery({
    queryKey: ['run', runId],
    queryFn: () => getRun(runId as string),
    enabled: Boolean(runId),
  });
}

export function useRunMetrics(runId: string | undefined) {
  return useQuery({
    queryKey: ['run-metrics', runId],
    queryFn: () => getRunMetrics(runId as string),
    enabled: Boolean(runId),
  });
}

export function useDeleteRun() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (runId: string) => deleteRun(runId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['runs'] });
    },
  });
}

export function useCompareRuns(ids: string[]) {
  return useQuery({
    queryKey: ['runs-compare', ids],
    queryFn: () => compareRuns(ids),
    enabled: ids.length > 0,
  });
}

// --- Models -----------------------------------------------------------------
export function useModels(filters: ModelFilters = {}) {
  return useQuery({
    queryKey: ['models', filters],
    queryFn: () => getModels(filters),
  });
}

export function useExportModel() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ runId, req }: { runId: string; req: ExportModelRequest }) =>
      exportModel(runId, req),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['jobs'] });
      qc.invalidateQueries({ queryKey: ['models'] });
    },
  });
}

// --- Topology ---------------------------------------------------------------
export function useTopologyPreview(req: TopologyPreviewRequest | null) {
  return useQuery({
    queryKey: ['topology-preview', req],
    queryFn: () => previewTopology(req!),
    enabled: Boolean(req),
    staleTime: 30_000,
  });
}

// --- Configs ----------------------------------------------------------------
export function useConfigSchema() {
  return useQuery({
    queryKey: ['config-schema'],
    queryFn: getConfigSchema,
    staleTime: Infinity,
  });
}

export function useConfigDefaults(section?: string) {
  return useQuery({
    queryKey: ['config-defaults', section],
    queryFn: () => getConfigDefaults(section),
    staleTime: 60_000,
  });
}

// --- Channels ---------------------------------------------------------------
export function useChannels(params: { limit?: number; offset?: number } = {}) {
  return useQuery({
    queryKey: ['channels', params],
    queryFn: () => getChannels(params),
  });
}

export function useChannelDetail(index: number | null) {
  return useQuery({
    queryKey: ['channel-detail', index],
    queryFn: () => getChannelDetail(index as number),
    enabled: index !== null,
  });
}
