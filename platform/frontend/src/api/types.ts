// TypeScript types mirror backend Pydantic schemas.
// Keep in sync with platform/backend/app/schemas/*.py

export type JobStatus = 'queued' | 'running' | 'completed' | 'failed' | 'cancelled';
export type JobType =
  | 'convert'
  | 'bridge'
  | 'eval'
  | 'infer'
  | 'export'
  | 'report'
  | 'simulate'
  | 'dataset_export';
export type LinkType = 'UL' | 'DL';
export type LinkPairing = 'single' | 'paired';
export type MobilityMode = 'static' | 'linear' | 'random_walk' | 'random_waypoint';
export type ModelFormat = 'pt' | 'onnx' | 'torchscript';

export interface HealthResponse {
  status: 'ok' | 'degraded' | 'down';
  version: string;
  db: 'ok' | 'error';
}

export interface Sample {
  sample_id: string;
  uuid: string;
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
  stage: string;
  serving_cell_id: number | null;
  channel_est_mode: string | null;
  bridged_path: string | null;
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
  stage_counts: Record<string, number>;
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
  source: 'quadriga_real' | 'sionna_rt' | 'internal_sim' | 'internal_upload';
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
  topology_layout?: 'hexagonal' | 'linear';
  hypercell_size?: number;
  track_offset_m?: number;
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

// --- Split management -------------------------------------------------------

export interface SplitComputeRequest {
  strategy: 'random' | 'by_position' | 'by_beam';
  seed: number;
  ratios: [number, number, number];
  lock: boolean;
}

export interface SplitInfoResponse {
  locked: boolean;
  version: number;
  strategy: string | null;
  seed: number | null;
  ratios: number[] | null;
  locked_at: string | null;
  locked_test_uuids: number;
  counts: Record<string, number>;
}

// --- Dataset export ---------------------------------------------------------

export interface DatasetExportRequest {
  format: 'hdf5' | 'webdataset' | 'pt_dir';
  split?: string | null;
  source_filter?: string | null;
  link_filter?: LinkType | null;
  min_snr?: number | null;
  max_snr?: number | null;
  export_name?: string | null;
  shard_size?: number;
  include_interferers?: boolean;
}

export interface ExportInfo {
  name: string;
  format: string;
  num_samples: number;
  split: string;
  split_version: number;
  total_bytes: number;
  path: string;
  download_url?: string;
}

export interface ExportsResponse {
  exports: ExportInfo[];
}

// --- Model upload -----------------------------------------------------------

export interface ModelUploadResponse {
  run_id: string;
  artifact_id: number;
  path: string;
  format: ModelFormat;
  size_bytes: number;
  compatible: boolean;
  compatibility_detail: string | null;
}

export interface ModelEvalRequest {
  test_split?: string;
  limit?: number | null;
  device?: string;
}

export interface ModelInferRequest {
  input_path?: string | null;
  split?: string;
  limit?: number | null;
  device?: string;
  batch_size?: number;
  output_name?: string | null;
}

export interface ModelMeta {
  run_id: string;
  tags: string | null;
  created_at: string | null;
  ckpt_path: string | null;
  epoch?: number;
  best_loss?: number;
  global_step?: number;
  num_parameters?: number;
  has_optimizer_state?: boolean;
  training_config?: Record<string, unknown>;
  compatible: boolean;
  compatibility_detail: string;
  metrics?: Record<string, number>;
}

export interface LeaderboardEntry {
  run_id: string;
  tags: string | null;
  compatible: boolean;
  test_split_version: number | null;
  metrics: Record<string, number | null>;
  evaluated_at: string | null;
}

export interface LeaderboardResponse {
  entries: LeaderboardEntry[];
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
  mobility_mode: MobilityMode | null;
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
