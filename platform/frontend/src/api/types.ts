// TypeScript types mirror backend Pydantic schemas.
// Keep in sync with platform/backend/app/schemas/*.py

export type JobStatus = 'queued' | 'running' | 'completed' | 'failed' | 'cancelled';
export type JobType =
  | 'convert'
  | 'bridge'
  | 'train'
  | 'eval'
  | 'infer'
  | 'export'
  | 'report'
  | 'simulate';
export type LinkType = 'UL' | 'DL';
export type LinkPairing = 'single' | 'paired';
export type ModelFormat = 'pt' | 'onnx' | 'torchscript';

export interface HealthResponse {
  status: 'ok' | 'degraded' | 'down';
  version: string;
  db: 'ok' | 'error';
}

export interface Sample {
  sample_id: string;
  source: string;
  link: LinkType;
  snr_dB: number;
  sir_dB: number | null;
  sinr_dB: number | null;
  ul_sir_dB: number | null;
  dl_sir_dB: number | null;
  num_interfering_ues: number | null;
  link_pairing: LinkPairing;
  timestamp: string;
  tags: string[];
  meta?: Record<string, unknown>;
}

export interface DatasetSummary {
  source: string;
  count: number;
  snr_mean: number;
  snr_std: number;
  sir_mean: number | null;
  sinr_mean: number | null;
  ul_sir_mean: number | null;
  dl_sir_mean: number | null;
  links: LinkType[];
  has_paired: boolean;
}

export interface DatasetsResponse {
  total: number;
  items: DatasetSummary[];
}

export interface SamplesResponse {
  total: number;
  items: Sample[];
}

export interface Job {
  job_id: string;
  type: JobType;
  status: JobStatus;
  display_name?: string | null;
  created_at: string;
  started_at?: string | null;
  finished_at?: string | null;
  progress_pct: number;
  current_step?: string | null;
  eta_seconds?: number | null;
  error?: string | null;
  run_id?: string | null;
  config_overrides?: Record<string, unknown>;
}

export interface JobsResponse {
  total: number;
  items: Job[];
}

export interface JobProgress {
  job_id: string;
  progress_pct: number;
  status: JobStatus;
  current_step?: string | null;
  eta_seconds?: number | null;
}

export interface JobLogs {
  job_id: string;
  lines: string[];
}

export interface RunMetrics {
  ct?: number;
  tw?: number;
  knn_acc?: number;
  nmse_dB?: number;
  [key: string]: number | undefined;
}

export interface ScalarSeries {
  tag: string;
  steps: number[];
  values: number[];
  wall_times?: number[];
}

export interface Run {
  run_id: string;
  created_at: string;
  git_sha?: string | null;
  config?: Record<string, unknown> | null;
  metrics?: RunMetrics | null;
  tags: string[];
  ckpt_path?: string | null;
  ckpt_best?: string | null;
  ckpt_last?: string | null;
  tb_scalars?: ScalarSeries[];
  artifacts?: ModelArtifact[];
}

export interface RunsResponse {
  total: number;
  items: Run[];
}

export interface CompareResponse {
  runs: { run_id: string; metrics: RunMetrics }[];
}

export interface ModelArtifact {
  id: number;
  run_id: string;
  format: ModelFormat;
  path: string;
  size_bytes: number;
  created_at: string;
  download_url?: string;
}

export interface ModelsResponse {
  total: number;
  items: ModelArtifact[];
}

export interface CreateJobRequest {
  type: JobType;
  config_overrides?: Record<string, unknown>;
  display_name?: string;
}

export interface BatchJobCreateRequest {
  type: JobType;
  configs: Record<string, unknown>[];
  display_name_prefix?: string;
}

export interface BatchJobCreateResponse {
  jobs: Job[];
}

export interface CreateCollectRequest {
  source: 'quadriga_multi' | 'quadriga_real' | 'sionna_rt' | 'internal_sim' | 'internal_upload';
  config_overrides?: Record<string, unknown>;
  output_dir?: string;
}

export interface SitePosition {
  site_id: number;
  x: number;
  y: number;
  z: number;
  sector_id: number;
  azimuth_deg: number;
  pci: number;
}

export interface UEPosition {
  ue_id: number;
  x: number;
  y: number;
  z: number;
}

export interface TopologyPreviewRequest {
  num_sites: number;
  isd_m: number;
  sectors_per_site: number;
  tx_height_m: number;
  num_ues: number;
  ue_distribution: 'uniform' | 'clustered' | 'hotspot';
  ue_speed_kmh: number;
}

export interface TopologyPreviewResponse {
  sites: SitePosition[];
  ues: UEPosition[];
  cell_radius_m: number;
  bounds: { min_x: number; max_x: number; min_y: number; max_y: number };
}

export interface CreateJobResponse {
  job_id: string;
}

export interface ExportModelRequest {
  format: 'onnx' | 'torchscript' | 'both';
}

export interface DatasetFilters {
  source?: string;
  link?: LinkType;
  min_snr?: number;
  max_snr?: number;
  limit?: number;
  offset?: number;
}

export interface JobFilters {
  status?: JobStatus;
  type?: JobType;
  limit?: number;
  offset?: number;
}

export interface RunFilters {
  tag?: string;
  limit?: number;
  offset?: number;
}

export interface ModelFilters {
  run_id?: string;
  format?: ModelFormat;
}

export interface ChannelSampleMeta {
  snr_dB: number | null;
  sir_dB: number | null;
  sinr_dB: number | null;
  ul_sir_dB: number | null;
  dl_sir_dB: number | null;
  source: string;
  link: string;
  link_pairing: LinkPairing | null;
  channel_est_mode: string;
  ue_position: number[] | null;
  ssb_rsrp_dBm: number[];
  serving_cell_id: number;
  num_interfering_ues: number | null;
}

export interface ChannelListItem {
  index: number;
  filename: string;
  meta: ChannelSampleMeta;
}

export interface ChannelListResponse {
  total: number;
  items: ChannelListItem[];
}

export interface FeatureInfo {
  name: string;
  shape: number[];
  values?: number[];
  magnitude?: number[];
  phase?: number[];
}

export interface ChannelDetail {
  index: number;
  shape: number[];
  channel_ideal: number[][];
  channel_est: number[][];
  channel_error: number[][];
  features: Record<string, FeatureInfo>;
  meta: ChannelSampleMeta;
}
