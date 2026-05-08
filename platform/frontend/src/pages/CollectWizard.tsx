import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Badge,
  Breadcrumb,
  Button,
  Card,
  Col,
  Collapse,
  Descriptions,
  Divider,
  Form,
  Input,
  InputNumber,
  Radio,
  Row,
  Select,
  Space,
  Steps,
  Switch,
  Tag,
  Typography,
  Upload,
  message,
} from 'antd';
import {
  ThunderboltOutlined,
  AimOutlined,
  UploadOutlined,
  CheckCircleOutlined,
  ExperimentOutlined,
} from '@ant-design/icons';
import { Link, useNavigate } from 'react-router-dom';
import { useCollectDataset, useTopologyPreview } from '@/api/queries';
import type { TopologyPreviewRequest, SitePosition, UEPosition } from '@/api/types';
import TopologyCanvas from '@/components/Topology/TopologyCanvas';

const { Title, Text, Paragraph } = Typography;
const { Dragger } = Upload;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type DataSource = 'quadriga_real' | 'sionna_rt' | 'internal_sim' | 'internal_upload';

interface SourceOption {
  key: DataSource;
  title: string;
  description: string;
  icon: React.ReactNode;
  disabled?: boolean;
}

interface DeviceConfig {
  num_sites: number;
  isd_m: number;
  sectors_per_site: number;
  tx_height_m: number;
  carrier_freq_hz: number;
  bandwidth_hz: number;
  subcarrier_spacing: number;
  tx_power_dbm: number;
  bs_ant_h: number;
  bs_ant_v: number;
  bs_ant_p: number;
  ue_ant_h: number;
  ue_ant_v: number;
  ue_ant_p: number;
  xpd_db: number;
  num_ues: number;
  ue_distribution: 'uniform' | 'clustered' | 'hotspot';
  ue_speed_kmh: number;
  ue_tx_power_dbm: number;
  mobility_mode: 'static' | 'linear' | 'random_walk' | 'random_waypoint' | 'track';
  sample_interval_s: number;
  topology_layout: 'hexagonal' | 'linear';
  hypercell_size: number;
  track_offset_m: number;
  train_penetration_loss_db: number;
  train_length_m: number;
  train_width_m: number;
  ue_height_m: number;
  noise_figure_db: number;
  qr_scenario: string;
  num_snapshots: number;
  ues_per_shard: number;
  skip_generation: boolean;
}

interface ChannelConfigValues {
  link: 'UL' | 'DL' | 'both';
  channel_est_mode: string;
  pilot_type_dl: string;
  pilot_type_ul: string;
  num_samples: number;
  num_interfering_ues: number;
  scenario?: string;
  channel_model: string;
  tdd_pattern: string;
  num_ssb_beams: number;
  srs_group_hopping: boolean;
  srs_sequence_hopping: boolean;
  srs_periodicity: number;
  srs_b_hop: number;
  srs_comb: number;
  srs_c_srs: number;
  srs_b_srs: number;
  srs_n_rrc: number;
  max_rank: number;
  rank_threshold: number;
  apply_interferer_precoding: boolean;
  store_interferer_channels: boolean;
  osm_path?: string;
}

interface ScenePreset {
  key: string;
  label: string;
  desc: string;
  config: Partial<DeviceConfig>;
  sources?: DataSource[];
}

// ---------------------------------------------------------------------------
// Constants — data source options
// ---------------------------------------------------------------------------

const SOURCE_OPTIONS: SourceOption[] = [
  { key: 'quadriga_real', title: 'QuaDRiGa Real', description: 'MATLAB 实时生成多小区信道（需本地 MATLAB）', icon: <ExperimentOutlined style={{ fontSize: 36 }} /> },
  { key: 'sionna_rt', title: 'Sionna RT', description: '射线追踪仿真（真实场景精确建模，需 GPU）', icon: <AimOutlined style={{ fontSize: 36 }} /> },
  { key: 'internal_sim', title: 'Python 内置仿真', description: '3GPP 38.901 多小区统计模型（纯 Python，无需外部依赖）', icon: <ThunderboltOutlined style={{ fontSize: 36, color: '#52c41a' }} /> },
  { key: 'internal_upload', title: '内部数据上传', description: '上传已有数据集（.mat/.npy/.pt 文件）', icon: <UploadOutlined style={{ fontSize: 36 }} />, disabled: true },
];

// ---------------------------------------------------------------------------
// Constants — scene presets
// ---------------------------------------------------------------------------

const SCENE_PRESETS: ScenePreset[] = [
  {
    key: 'uma_64t', label: '城市宏站 64T', desc: 'UMa · 3.5G · 100M · ISD 500m',
    config: {
      num_sites: 7, isd_m: 500, sectors_per_site: 3, tx_height_m: 25,
      carrier_freq_hz: 3_500_000_000, bandwidth_hz: 100_000_000, subcarrier_spacing: 30_000, tx_power_dbm: 43,
      bs_ant_h: 8, bs_ant_v: 4, bs_ant_p: 2, ue_ant_h: 1, ue_ant_v: 1, ue_ant_p: 2, xpd_db: 8,
      num_ues: 50, ue_speed_kmh: 3, ue_tx_power_dbm: 23, mobility_mode: 'static' as const,
      qr_scenario: '3GPP_38.901_UMa_NLOS',
    },
  },
  {
    key: 'umi_32t', label: '城市微站 32T', desc: 'UMi · 3.5G · 100M · ISD 200m',
    config: {
      num_sites: 19, isd_m: 200, sectors_per_site: 3, tx_height_m: 10,
      carrier_freq_hz: 3_500_000_000, bandwidth_hz: 100_000_000, subcarrier_spacing: 30_000, tx_power_dbm: 38,
      bs_ant_h: 4, bs_ant_v: 4, bs_ant_p: 2, ue_ant_h: 1, ue_ant_v: 1, ue_ant_p: 2, xpd_db: 8,
      num_ues: 100, ue_speed_kmh: 3, ue_tx_power_dbm: 23, mobility_mode: 'static' as const,
      qr_scenario: '3GPP_38.901_UMi_NLOS',
    },
  },
  {
    key: 'inh_8t', label: '室内热点 8T', desc: 'InH · 3.5G · 50M · ISD 50m',
    sources: ['quadriga_real'],
    config: {
      num_sites: 3, isd_m: 50, sectors_per_site: 1, tx_height_m: 3,
      carrier_freq_hz: 3_500_000_000, bandwidth_hz: 50_000_000, subcarrier_spacing: 30_000, tx_power_dbm: 24,
      bs_ant_h: 2, bs_ant_v: 2, bs_ant_p: 2, ue_ant_h: 1, ue_ant_v: 1, ue_ant_p: 2, xpd_db: 8,
      num_ues: 20, ue_speed_kmh: 0, ue_tx_power_dbm: 23, mobility_mode: 'static' as const,
      qr_scenario: '3GPP_38.901_InH_NLOS',
    },
  },
  {
    key: 'rma_4t', label: '农村宏站 4T', desc: 'RMa · 700M · 20M · ISD 1732m',
    sources: ['quadriga_real'],
    config: {
      num_sites: 7, isd_m: 1732, sectors_per_site: 3, tx_height_m: 35,
      carrier_freq_hz: 700_000_000, bandwidth_hz: 20_000_000, subcarrier_spacing: 15_000, tx_power_dbm: 46,
      bs_ant_h: 2, bs_ant_v: 2, bs_ant_p: 1, ue_ant_h: 1, ue_ant_v: 1, ue_ant_p: 1, xpd_db: 8,
      num_ues: 30, ue_speed_kmh: 30, ue_tx_power_dbm: 23, mobility_mode: 'linear' as const,
      qr_scenario: '3GPP_38.901_RMa_NLOS',
    },
  },
  {
    key: 'mmwave', label: '毫米波 28G', desc: 'UMi · 28G · 100M · ISD 100m',
    config: {
      num_sites: 7, isd_m: 100, sectors_per_site: 3, tx_height_m: 10,
      carrier_freq_hz: 28_000_000_000, bandwidth_hz: 100_000_000, subcarrier_spacing: 120_000, tx_power_dbm: 35,
      bs_ant_h: 8, bs_ant_v: 4, bs_ant_p: 2, ue_ant_h: 2, ue_ant_v: 1, ue_ant_p: 2, xpd_db: 10,
      num_ues: 30, ue_speed_kmh: 3, ue_tx_power_dbm: 23, mobility_mode: 'static' as const,
      qr_scenario: '3GPP_38.901_UMi_NLOS',
    },
  },
  {
    key: 'hsr_350', label: '高铁 350km/h', desc: 'HyperCell · 2.6G · 100M · 线性部署',
    config: {
      num_sites: 12, isd_m: 1000, sectors_per_site: 3, tx_height_m: 30,
      carrier_freq_hz: 2_600_000_000, bandwidth_hz: 100_000_000, subcarrier_spacing: 30_000, tx_power_dbm: 46,
      bs_ant_h: 8, bs_ant_v: 4, bs_ant_p: 2, ue_ant_h: 1, ue_ant_v: 1, ue_ant_p: 2, xpd_db: 8,
      num_ues: 20, ue_speed_kmh: 350, ue_tx_power_dbm: 23, mobility_mode: 'track' as const,
      topology_layout: 'linear' as const, hypercell_size: 4, track_offset_m: 80, train_penetration_loss_db: 22,
      qr_scenario: '3GPP_38.901_UMa_LOS',
    },
  },
];

// ---------------------------------------------------------------------------
// Constants — antenna presets
// ---------------------------------------------------------------------------

const BS_PRESETS: Record<string, [number, number, number]> = {
  '8_4_2': [8, 4, 2], '4_4_2': [4, 4, 2], '4_2_2': [4, 2, 2],
  '2_2_2': [2, 2, 2], '4_1_1': [4, 1, 1], '2_1_1': [2, 1, 1],
};
const BS_PRESET_OPTIONS = [
  { label: '64T64R (8×4×2 双极化)', value: '8_4_2' },
  { label: '32T32R (4×4×2 双极化)', value: '4_4_2' },
  { label: '16T16R (4×2×2 双极化)', value: '4_2_2' },
  { label: '8T8R (2×2×2 双极化)', value: '2_2_2' },
  { label: '4T4R ULA (4×1×1)', value: '4_1_1' },
  { label: '2T2R ULA (2×1×1)', value: '2_1_1' },
];

const UE_PRESETS: Record<string, [number, number, number]> = {
  '2_1_2': [2, 1, 2], '1_1_2': [1, 1, 2], '2_1_1': [2, 1, 1], '1_1_1': [1, 1, 1],
};
const UE_PRESET_OPTIONS = [
  { label: '4 天线 (2×1×2 双极化)', value: '2_1_2' },
  { label: '2 天线 (1×1×2 双极化)', value: '1_1_2' },
  { label: '2 天线 ULA (2×1×1)', value: '2_1_1' },
  { label: '1 天线 (1×1×1)', value: '1_1_1' },
];

// ---------------------------------------------------------------------------
// Constants — select options
// ---------------------------------------------------------------------------

const SITE_OPTIONS = [1, 3, 7, 19, 37];
const SITE_OPTIONS_QR = [1, 3, 7, 19, 21];
const SECTOR_OPTIONS = [1, 3];
const BANDWIDTH_OPTIONS = [
  { label: '5 MHz', value: 5_000_000 }, { label: '10 MHz', value: 10_000_000 },
  { label: '15 MHz', value: 15_000_000 }, { label: '20 MHz', value: 20_000_000 },
  { label: '25 MHz', value: 25_000_000 }, { label: '30 MHz', value: 30_000_000 },
  { label: '40 MHz', value: 40_000_000 }, { label: '50 MHz', value: 50_000_000 },
  { label: '60 MHz', value: 60_000_000 }, { label: '70 MHz', value: 70_000_000 },
  { label: '80 MHz', value: 80_000_000 }, { label: '90 MHz', value: 90_000_000 },
  { label: '100 MHz', value: 100_000_000 },
];
const SCS_OPTIONS = [
  { label: '15 kHz', value: 15_000 }, { label: '30 kHz', value: 30_000 },
  { label: '60 kHz', value: 60_000 }, { label: '120 kHz', value: 120_000 },
];

const NR_RB_TABLE: Record<string, number> = {
  '5_15': 25, '10_15': 52, '15_15': 79, '20_15': 106, '25_15': 133,
  '30_15': 160, '40_15': 216, '50_15': 270,
  '5_30': 11, '10_30': 24, '15_30': 38, '20_30': 51, '25_30': 65,
  '30_30': 78, '40_30': 106, '50_30': 133, '60_30': 162, '70_30': 189,
  '80_30': 217, '90_30': 245, '100_30': 273,
  '10_60': 11, '15_60': 18, '20_60': 24, '25_60': 31, '30_60': 38,
  '40_60': 51, '50_60': 65, '60_60': 79, '70_60': 93, '80_60': 107,
  '90_60': 121, '100_60': 135,
  '50_120': 32, '100_120': 66, '200_120': 132, '400_120': 264,
};
function nrRbLookup(bwHz: number, scsHz: number): number | null {
  return NR_RB_TABLE[`${Math.round(bwHz / 1e6)}_${Math.round(scsHz / 1e3)}`] ?? null;
}

const CARRIER_FREQ_OPTIONS = [
  { label: '700 MHz', value: 700_000_000 }, { label: '2.1 GHz', value: 2_100_000_000 },
  { label: '2.6 GHz', value: 2_600_000_000 }, { label: '3.5 GHz', value: 3_500_000_000 },
  { label: '28 GHz', value: 28_000_000_000 }, { label: '39 GHz', value: 39_000_000_000 },
];
const UE_DISTRIBUTION_OPTIONS = [
  { label: '均匀分布', value: 'uniform' as const },
  { label: '簇状分布', value: 'clustered' as const },
  { label: '热点分布', value: 'hotspot' as const },
];
const MOBILITY_MODE_OPTIONS = [
  { label: '静止', value: 'static' as const }, { label: '匀速直线', value: 'linear' as const },
  { label: '随机游走', value: 'random_walk' as const }, { label: '随机航路点', value: 'random_waypoint' as const },
  { label: '轨道固定轨迹', value: 'track' as const },
];
const SAMPLE_INTERVAL_OPTIONS = [
  { label: '0.5 ms (1 slot)', value: 0.0005 }, { label: '1 ms (1 子帧)', value: 0.001 },
  { label: '5 ms', value: 0.005 }, { label: '10 ms (1 帧)', value: 0.01 },
  { label: '100 ms', value: 0.1 }, { label: '1 s', value: 1.0 },
];
const QUADRIGA_SCENARIO_OPTIONS = [
  { label: 'UMa NLOS (城市宏站)', value: '3GPP_38.901_UMa_NLOS' },
  { label: 'UMa LOS (城市宏站视距)', value: '3GPP_38.901_UMa_LOS' },
  { label: 'UMi NLOS (城市微站)', value: '3GPP_38.901_UMi_NLOS' },
  { label: 'UMi LOS (城市微站视距)', value: '3GPP_38.901_UMi_LOS' },
  { label: 'RMa NLOS (农村宏站)', value: '3GPP_38.901_RMa_NLOS' },
  { label: 'RMa LOS (农村宏站视距)', value: '3GPP_38.901_RMa_LOS' },
  { label: 'InH NLOS (室内热点)', value: '3GPP_38.901_InH_NLOS' },
  { label: 'InH LOS (室内热点视距)', value: '3GPP_38.901_InH_LOS' },
];
const CHANNEL_MODEL_OPTIONS = [
  { label: 'CDL-A (NLOS, 23 簇)', value: 'CDL-A' }, { label: 'CDL-B (NLOS, 23 簇)', value: 'CDL-B' },
  { label: 'CDL-C (NLOS, 24 簇)', value: 'CDL-C' }, { label: 'CDL-D (LOS, K=13.3dB)', value: 'CDL-D' },
  { label: 'CDL-E (LOS, K=22.0dB)', value: 'CDL-E' },
];
const TDD_PATTERN_OPTIONS = [
  { label: 'DDDSU (3D+S+1U)', value: 'DDDSU' }, { label: 'DDSUU (2D+S+2U)', value: 'DDSUU' },
  { label: 'DDDDDDDSUU (7D+S+2U)', value: 'DDDDDDDSUU' }, { label: 'DDDSUDDSUU (双周期)', value: 'DDDSUDDSUU' },
  { label: 'DSUUD (对称)', value: 'DSUUD' },
];
const DL_PILOT_OPTIONS = [{ label: 'CSI-RS (Gold 序列)', value: 'csi_rs_gold' }];
const UL_PILOT_OPTIONS = [{ label: 'SRS (ZC 序列)', value: 'srs_zc' }];

const SRS_PERIODICITY_OPTIONS = [1, 2, 4, 5, 8, 10, 16, 20, 32, 40, 64, 80, 160, 320, 640, 1280, 2560].map(v => ({ label: `${v} slots`, value: v }));
const SRS_B_HOP_OPTIONS = [
  { label: '0 (全频段跳频)', value: 0 }, { label: '1 (部分跳频)', value: 1 },
  { label: '2 (部分跳频)', value: 2 }, { label: '3 (不跳频)', value: 3 },
];
const SRS_C_SRS_OPTIONS = Array.from({ length: 18 }, (_, i) => ({ label: `C_SRS=${i}`, value: i }));
const SRS_B_SRS_OPTIONS = [
  { label: 'B_SRS=0 (全带宽)', value: 0 }, { label: 'B_SRS=1 (一级细分)', value: 1 },
  { label: 'B_SRS=2 (二级细分)', value: 2 }, { label: 'B_SRS=3 (三级细分)', value: 3 },
];

// ---------------------------------------------------------------------------
// Defaults
// ---------------------------------------------------------------------------

const DEFAULT_DEVICE: DeviceConfig = {
  num_sites: 7, isd_m: 500, sectors_per_site: 3, tx_height_m: 25,
  carrier_freq_hz: 3_500_000_000, bandwidth_hz: 100_000_000, subcarrier_spacing: 30_000, tx_power_dbm: 43,
  bs_ant_h: 8, bs_ant_v: 4, bs_ant_p: 2,
  ue_ant_h: 1, ue_ant_v: 1, ue_ant_p: 2, xpd_db: 8,
  num_ues: 50, ue_distribution: 'uniform', ue_speed_kmh: 3, ue_tx_power_dbm: 23,
  mobility_mode: 'static', sample_interval_s: 0.0005,
  topology_layout: 'hexagonal', hypercell_size: 1, track_offset_m: 80, train_penetration_loss_db: 0,
  train_length_m: 400, train_width_m: 3.4, ue_height_m: 1.5, noise_figure_db: 7.0,
  qr_scenario: '3GPP_38.901_UMa_NLOS', num_snapshots: 14, ues_per_shard: 50, skip_generation: false,
};

const DEFAULT_CHANNEL: ChannelConfigValues = {
  link: 'DL', channel_est_mode: 'ideal', pilot_type_dl: 'csi_rs_gold', pilot_type_ul: 'srs_zc',
  num_samples: 100, num_interfering_ues: 3, scenario: 'munich', channel_model: 'CDL-C',
  tdd_pattern: 'DDDSU', num_ssb_beams: 8,
  srs_group_hopping: false, srs_sequence_hopping: false, srs_periodicity: 10,
  srs_b_hop: 0, srs_comb: 2, srs_c_srs: 3, srs_b_srs: 1, srs_n_rrc: 0,
  max_rank: 4, rank_threshold: 0.1,
  apply_interferer_precoding: true, store_interferer_channels: false,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function freqLabel(hz: number) { return CARRIER_FREQ_OPTIONS.find(o => o.value === hz)?.label ?? `${hz} Hz`; }
function bwLabel(hz: number) { return BANDWIDTH_OPTIONS.find(o => o.value === hz)?.label ?? `${hz} Hz`; }
function distLabel(v: string) { return UE_DISTRIBUTION_OPTIONS.find(o => o.value === v)?.label ?? v; }
function mobilityLabel(v: string) { return MOBILITY_MODE_OPTIONS.find(o => o.value === v)?.label ?? v; }
function linkLabel(v: string) { return ({ UL: '上行', DL: '下行', both: '双向' } as Record<string, string>)[v] ?? v; }
function estLabel(v: string) { return ({ ideal: '理想', ls_linear: 'LS 线性插值', ls_mmse: 'LS+MMSE', ls_hop_concat: 'LS 逐跳拼接' } as Record<string, string>)[v] ?? v; }
function pilotLabel(v: string) { return ({ csi_rs_gold: 'CSI-RS (Gold)', srs_zc: 'SRS (ZC)' } as Record<string, string>)[v] ?? v; }
function sourceLabel(v: string) { return ({ quadriga_real: 'QuaDRiGa (MATLAB)', sionna_rt: 'Sionna RT', internal_sim: 'Python 内置仿真' } as Record<string, string>)[v] ?? v; }
function scenarioLabel(v: string) { return ({ munich: 'Munich', etoile: 'Etoile', custom_osm: '自定义 OSM' } as Record<string, string>)[v] ?? v; }
function panelText(h: number, v: number, p: number) { return `${h}H×${v}V×${p}P`; }

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function CollectWizard() {
  const navigate = useNavigate();
  const collect = useCollectDataset();

  const [currentStep, setCurrentStep] = useState(0);
  const [source, setSource] = useState<DataSource>('quadriga_real');
  const [device, setDevice] = useState<DeviceConfig>(DEFAULT_DEVICE);
  const [channelConfig, setChannelConfig] = useState<ChannelConfigValues>(DEFAULT_CHANNEL);
  const [activePreset, setActivePreset] = useState<string | null>('uma_64t');
  const [siteOverrides, setSiteOverrides] = useState<SitePosition[]>([]);
  const [ueOverrides, setUeOverrides] = useState<UEPosition[]>([]);

  const [deviceForm] = Form.useForm<DeviceConfig>();
  const [channelForm] = Form.useForm<ChannelConfigValues>();

  const presetRef = useRef<string | null>(null);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [previewReq, setPreviewReq] = useState<TopologyPreviewRequest | null>(null);

  // ---- Computed ----
  const bsTotal = device.bs_ant_h * device.bs_ant_v * device.bs_ant_p;
  const ueTotal = device.ue_ant_h * device.ue_ant_v * device.ue_ant_p;
  const siteOptions = source === 'quadriga_real' ? SITE_OPTIONS_QR : SITE_OPTIONS;
  const bsPresetKey = useMemo(() => {
    const k = `${device.bs_ant_h}_${device.bs_ant_v}_${device.bs_ant_p}`;
    return BS_PRESETS[k] ? k : undefined;
  }, [device.bs_ant_h, device.bs_ant_v, device.bs_ant_p]);
  const uePresetKey = useMemo(() => {
    const k = `${device.ue_ant_h}_${device.ue_ant_v}_${device.ue_ant_p}`;
    return UE_PRESETS[k] ? k : undefined;
  }, [device.ue_ant_h, device.ue_ant_v, device.ue_ant_p]);

  // ---- Preview ----
  const schedulePreview = useCallback((d: DeviceConfig) => {
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => {
      setPreviewReq({
        num_sites: d.num_sites, isd_m: d.isd_m,
        sectors_per_site: source === 'quadriga_real' ? 1 : d.sectors_per_site,
        tx_height_m: d.tx_height_m, num_ues: d.num_ues,
        ue_distribution: d.ue_distribution, ue_speed_kmh: d.ue_speed_kmh,
        topology_layout: d.topology_layout, hypercell_size: d.hypercell_size, track_offset_m: d.track_offset_m,
      });
    }, 500);
  }, [source]);

  useEffect(() => {
    if (currentStep === 1) schedulePreview(device);
    return () => { if (debounceRef.current) clearTimeout(debounceRef.current); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentStep]);

  const { data: previewData, isFetching: previewLoading } = useTopologyPreview(previewReq);

  // ---- Preset handlers ----
  const applyPreset = useCallback((preset: ScenePreset) => {
    const merged = { ...DEFAULT_DEVICE, ...preset.config };
    presetRef.current = preset.key;
    setActivePreset(preset.key);
    setDevice(merged);
    deviceForm.setFieldsValue(merged);
    schedulePreview(merged);
    setTimeout(() => { presetRef.current = null; }, 0);
  }, [deviceForm, schedulePreview]);

  const handleBsPreset = useCallback((key: string) => {
    if (!key || !BS_PRESETS[key]) return;
    const [h, v, p] = BS_PRESETS[key];
    deviceForm.setFieldsValue({ bs_ant_h: h, bs_ant_v: v, bs_ant_p: p });
  }, [deviceForm]);

  const handleUePreset = useCallback((key: string) => {
    if (!key || !UE_PRESETS[key]) return;
    const [h, v, p] = UE_PRESETS[key];
    deviceForm.setFieldsValue({ ue_ant_h: h, ue_ant_v: v, ue_ant_p: p });
  }, [deviceForm]);

  // ---- Callbacks ----
  const handleSiteDrag = useCallback((sites: SitePosition[]) => setSiteOverrides(sites), []);
  const handleUEDrag = useCallback((ues: UEPosition[]) => setUeOverrides(ues), []);
  const handleLinkChange = useCallback((link: 'UL' | 'DL' | 'both') => {
    setChannelConfig(prev => ({ ...prev, link }));
    channelForm.setFieldsValue({ link });
  }, [channelForm]);

  const handleDeviceChange = useCallback((_changed: Partial<DeviceConfig>, all: DeviceConfig) => {
    const vals = all as DeviceConfig;
    setDevice(vals);
    schedulePreview(vals);
    if (!presetRef.current) {
      setActivePreset(null);
    }
  }, [schedulePreview]);

  const validateStep = useCallback(async (): Promise<boolean> => {
    try {
      if (currentStep === 0) { if (source === 'internal_upload') { message.warning('内部数据上传功能即将支持'); return false; } return true; }
      if (currentStep === 1) { await deviceForm.validateFields(); return true; }
      if (currentStep === 2) { await channelForm.validateFields(); return true; }
      return true;
    } catch { return false; }
  }, [currentStep, source, deviceForm, channelForm]);

  const goNext = useCallback(async () => { if (await validateStep()) setCurrentStep(s => Math.min(s + 1, 3)); }, [validateStep]);
  const goPrev = useCallback(() => setCurrentStep(s => Math.max(s - 1, 0)), []);

  const hasCustomPositions = siteOverrides.length > 0 || ueOverrides.length > 0;

  // ---- Full config for submission ----
  const fullConfig = useMemo(() => {
    const d = device;
    const _bs = d.bs_ant_h * d.bs_ant_v * d.bs_ant_p;
    const _ue = d.ue_ant_h * d.ue_ant_v * d.ue_ant_p;
    const config: Record<string, unknown> = {
      num_sites: d.num_sites, num_cells: source === 'quadriga_real' ? d.num_sites : d.num_sites * d.sectors_per_site,
      isd_m: d.isd_m, sectors_per_site: d.sectors_per_site, tx_height_m: d.tx_height_m,
      carrier_freq_hz: d.carrier_freq_hz, bandwidth_hz: d.bandwidth_hz, subcarrier_spacing: d.subcarrier_spacing, tx_power_dbm: d.tx_power_dbm,
      bs_panel: [d.bs_ant_h, d.bs_ant_v, d.bs_ant_p], ue_panel: [d.ue_ant_h, d.ue_ant_v, d.ue_ant_p],
      num_bs_antennas: _bs, num_bs_tx_ant: _bs, num_bs_rx_ant: _bs,
      num_ue_antennas: _ue, num_ue_tx_ant: _ue, num_ue_rx_ant: _ue,
      bs_ant_h: d.bs_ant_h, bs_ant_v: d.bs_ant_v, bs_ant_p: d.bs_ant_p,
      ue_ant_h: d.ue_ant_h, ue_ant_v: d.ue_ant_v, ue_ant_p: d.ue_ant_p, xpd_db: d.xpd_db,
      num_ues: d.num_ues, ue_distribution: d.ue_distribution, ue_speed_kmh: d.ue_speed_kmh,
      ue_tx_power_dbm: d.ue_tx_power_dbm, mobility_mode: d.mobility_mode, sample_interval_s: d.sample_interval_s,
      ue_height_m: d.ue_height_m, noise_figure_db: d.noise_figure_db,
      topology_layout: d.topology_layout, hypercell_size: d.hypercell_size, track_offset_m: d.track_offset_m, train_penetration_loss_db: d.train_penetration_loss_db,
      train_length_m: d.train_length_m, train_width_m: d.train_width_m,
      link: channelConfig.link, channel_est_mode: channelConfig.channel_est_mode, channel_model: channelConfig.channel_model,
      tdd_pattern: channelConfig.tdd_pattern, num_ssb_beams: channelConfig.num_ssb_beams,
      srs_group_hopping: channelConfig.srs_group_hopping, srs_sequence_hopping: channelConfig.srs_sequence_hopping,
      srs_periodicity: channelConfig.srs_periodicity, srs_b_hop: channelConfig.srs_b_hop,
      srs_comb: channelConfig.srs_comb, srs_c_srs: channelConfig.srs_c_srs,
      srs_b_srs: channelConfig.srs_b_srs, srs_n_rrc: channelConfig.srs_n_rrc,
      max_rank: channelConfig.max_rank, rank_threshold: channelConfig.rank_threshold,
      apply_interferer_precoding: channelConfig.apply_interferer_precoding,
      store_interferer_channels: channelConfig.store_interferer_channels,
      num_samples: channelConfig.num_samples, num_interfering_ues: channelConfig.num_interfering_ues,
      pilot_type_dl: channelConfig.pilot_type_dl, pilot_type_ul: channelConfig.pilot_type_ul,
    };
    if (source === 'quadriga_real') { config.scenario = d.qr_scenario; config.num_snapshots = d.num_snapshots; config.ues_per_shard = d.ues_per_shard; config.skip_generation = d.skip_generation; }
    if (source === 'sionna_rt' && channelConfig.scenario) { config.scenario = channelConfig.scenario; if (channelConfig.scenario === 'custom_osm' && channelConfig.osm_path) config.osm_path = channelConfig.osm_path; }
    if (siteOverrides.length > 0) config.custom_site_positions = siteOverrides.map(s => ({ x: s.x, y: s.y, z: s.z }));
    if (ueOverrides.length > 0) config.custom_ue_positions = ueOverrides.map(u => ({ x: u.x, y: u.y, z: u.z }));
    return config;
  }, [device, channelConfig, source, siteOverrides, ueOverrides]);

  const handleSubmit = useCallback(async () => {
    try {
      const res = await collect.mutateAsync({ source, config_overrides: fullConfig });
      message.success(`采集任务 ${res.job_id} 已提交`);
      navigate(`/jobs/${res.job_id}`);
    } catch (e) { message.error((e as Error).message ?? '提交失败'); }
  }, [collect, source, fullConfig, navigate]);

  // ====================================================================
  // Render
  // ====================================================================
  return (
    <div className="msg-page">
      <Breadcrumb className="msg-breadcrumb" items={[{ title: <Link to="/datasets">数据集</Link> }, { title: '数据采集向导' }]} />
      <Title level={3} style={{ marginBottom: 24 }}>数据采集向导</Title>

      <Steps current={currentStep} style={{ marginBottom: 32 }} items={[
        { title: '数据来源' }, { title: '组网与设备配置' }, { title: '信道配置' }, { title: '确认提交' },
      ]} />

      {/* ================================================================ */}
      {/* Step 0: 数据来源                                                  */}
      {/* ================================================================ */}
      {currentStep === 0 && (
        <Row gutter={24} justify="center">
          {SOURCE_OPTIONS.map(opt => (
            <Col key={opt.key} xs={24} sm={8} style={{ marginBottom: 16 }}>
              {opt.disabled ? (
                <Badge.Ribbon text="即将支持" color="orange">
                  <Card hoverable style={{ textAlign: 'center', cursor: 'not-allowed', opacity: 0.55, minHeight: 220 }}>
                    <div style={{ fontSize: 36, marginBottom: 16, color: '#999' }}>{opt.icon}</div>
                    <Title level={4} style={{ marginBottom: 8 }}>{opt.title}</Title>
                    <Text type="secondary">{opt.description}</Text>
                    <div style={{ marginTop: 16 }}>
                      <Dragger disabled style={{ padding: 8 }} showUploadList={false}>
                        <Paragraph type="secondary" style={{ margin: 0 }}>拖拽或点击上传 .mat / .npy / .pt 文件</Paragraph>
                      </Dragger>
                    </div>
                  </Card>
                </Badge.Ribbon>
              ) : (
                <Card
                  hoverable
                  onClick={() => {
                    setSource(opt.key);
                    if (activePreset) {
                      const preset = SCENE_PRESETS.find(p => p.key === activePreset);
                      if (preset?.sources && !preset.sources.includes(opt.key)) setActivePreset(null);
                    }
                  }}
                  style={{
                    textAlign: 'center', minHeight: 220, display: 'flex', flexDirection: 'column', justifyContent: 'center',
                    borderColor: source === opt.key ? '#1677ff' : undefined,
                    borderWidth: source === opt.key ? 2 : 1,
                    boxShadow: source === opt.key ? '0 0 0 2px rgba(22,119,255,0.15)' : undefined,
                  }}
                  styles={{ body: { display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' } }}
                >
                  <div style={{ fontSize: 36, marginBottom: 16, color: source === opt.key ? '#1677ff' : '#595959' }}>{opt.icon}</div>
                  <Title level={4} style={{ marginBottom: 8 }}>{opt.title}</Title>
                  <Text type="secondary">{opt.description}</Text>
                  {source === opt.key && <CheckCircleOutlined style={{ color: '#1677ff', fontSize: 22, marginTop: 12 }} />}
                </Card>
              )}
            </Col>
          ))}
        </Row>
      )}

      {/* ================================================================ */}
      {/* Step 1: 组网与设备配置                                             */}
      {/* ================================================================ */}
      {currentStep === 1 && (
        <Row gutter={24}>
          <Col xs={24} lg={10}>
            {/* ---- Scene presets ---- */}
            <div style={{ marginBottom: 12 }}>
              <Text type="secondary" style={{ display: 'block', marginBottom: 8, fontSize: 13 }}>场景预设（点击快速填充）</Text>
              <Row gutter={[8, 8]}>
                {SCENE_PRESETS.filter(p => !p.sources || p.sources.includes(source)).map(p => (
                  <Col span={8} key={p.key}>
                    <Card
                      size="small" hoverable onClick={() => applyPreset(p)}
                      style={{
                        cursor: 'pointer',
                        borderColor: activePreset === p.key ? '#1677ff' : undefined,
                        borderWidth: activePreset === p.key ? 2 : 1,
                        background: activePreset === p.key ? '#e6f4ff' : undefined,
                      }}
                      styles={{ body: { padding: '6px 10px' } }}
                    >
                      <Text strong style={{ fontSize: 12 }}>{p.label}</Text>
                      <br />
                      <Text type="secondary" style={{ fontSize: 10, lineHeight: 1.3 }}>{p.desc}</Text>
                    </Card>
                  </Col>
                ))}
              </Row>
            </div>

            {/* ---- Config form in Collapse panels ---- */}
            <Form form={deviceForm} layout="vertical" initialValues={device} preserve onValuesChange={handleDeviceChange}>
              <Collapse
                defaultActiveKey={['topology', 'antenna']}
                style={{ background: '#fafafa' }}
                items={[
                  // ---- Panel: 拓扑配置 ----
                  {
                    key: 'topology', forceRender: true, label: <Text strong>拓扑配置</Text>,
                    extra: <Text type="secondary" style={{ fontSize: 12 }}>
                      {device.topology_layout === 'linear' ? '线性' : '蜂窝'} · {device.num_sites} 站 · ISD {device.isd_m}m
                      {device.hypercell_size > 1 ? ` · HC×${device.hypercell_size}` : ''}
                    </Text>,
                    children: (
                      <>
                        <Row gutter={16}>
                          <Col span={6}>
                            <Form.Item label="拓扑布局" name="topology_layout" rules={[{ required: true }]}>
                              <Select options={[
                                { value: 'hexagonal', label: '六边形蜂窝' },
                                { value: 'linear', label: '线性轨道 (HSR)' },
                              ]} />
                            </Form.Item>
                          </Col>
                          <Col span={device.topology_layout === 'linear' ? 4 : 6}>
                            <Form.Item label="站点数" name="num_sites" rules={[{ required: true }]}>
                              {device.topology_layout === 'linear'
                                ? <InputNumber min={2} max={57} style={{ width: '100%' }} />
                                : <Select options={siteOptions.map(v => ({ value: v, label: `${v}` }))} />
                              }
                            </Form.Item>
                          </Col>
                          <Col span={device.topology_layout === 'linear' ? 4 : 6}>
                            <Form.Item label="站间距 (m)" name="isd_m" rules={[{ required: true }]}>
                              <InputNumber min={50} max={5000} style={{ width: '100%' }} />
                            </Form.Item>
                          </Col>
                          {source !== 'quadriga_real' && (
                            <Col span={4}>
                              <Form.Item label="扇区/站" name="sectors_per_site" rules={[{ required: true }]}>
                                <Select options={SECTOR_OPTIONS.map(v => ({ value: v, label: `${v}` }))} />
                              </Form.Item>
                            </Col>
                          )}
                          <Col span={device.topology_layout === 'linear' ? 4 : 6}>
                            <Form.Item label="基站高度 (m)" name="tx_height_m" rules={[{ required: true }]}>
                              <InputNumber min={1} max={200} style={{ width: '100%' }} />
                            </Form.Item>
                          </Col>
                        </Row>
                        {device.topology_layout === 'linear' && (
                          <>
                            <Row gutter={16}>
                              <Col span={6}>
                                <Form.Item label="HyperCell 分组" name="hypercell_size" tooltip="每组内 RRH 共享 PCI，UE 不触发切换">
                                  <InputNumber min={1} max={20} style={{ width: '100%' }} />
                                </Form.Item>
                              </Col>
                              <Col span={6}>
                                <Form.Item label="轨道偏移距离 (m)" name="track_offset_m" tooltip="站点到轨道中心线的垂直距离">
                                  <InputNumber min={20} max={500} style={{ width: '100%' }} />
                                </Form.Item>
                              </Col>
                              <Col span={6}>
                                <Form.Item label="车体穿透损耗 (dB)" name="train_penetration_loss_db" tooltip="3GPP TR 38.913: 现代列车 ~22 dB">
                                  <InputNumber min={0} max={40} style={{ width: '100%' }} />
                                </Form.Item>
                              </Col>
                            </Row>
                            <Row gutter={16}>
                              <Col span={8}>
                                <Form.Item label="列车长度 (m)" name="train_length_m" tooltip="标准动车组 400m">
                                  <InputNumber min={50} max={1000} style={{ width: '100%' }} />
                                </Form.Item>
                              </Col>
                              <Col span={8}>
                                <Form.Item label="列车宽度 (m)" name="train_width_m" tooltip="标准动车组 3.4m">
                                  <InputNumber min={2} max={10} step={0.1} style={{ width: '100%' }} />
                                </Form.Item>
                              </Col>
                            </Row>
                          </>
                        )}
                      </>
                    ),
                  },
                  // ---- Panel: 天线阵列 ----
                  {
                    key: 'antenna', forceRender: true, label: <Text strong>天线阵列</Text>,
                    extra: <Text type="secondary" style={{ fontSize: 12 }}>BS {bsTotal}T{bsTotal}R · UE {ueTotal}</Text>,
                    children: (
                      <>
                        <Row gutter={16} style={{ marginBottom: 8 }}>
                          <Col span={12}>
                            <Form.Item label="BS 预设" style={{ marginBottom: 8 }}>
                              <Select placeholder="选择常用配置" options={BS_PRESET_OPTIONS} value={bsPresetKey} onChange={handleBsPreset} allowClear />
                            </Form.Item>
                          </Col>
                          <Col span={12}>
                            <Form.Item label="UE 预设" style={{ marginBottom: 8 }}>
                              <Select placeholder="选择常用配置" options={UE_PRESET_OPTIONS} value={uePresetKey} onChange={handleUePreset} allowClear />
                            </Form.Item>
                          </Col>
                        </Row>
                        <Row gutter={16}>
                          <Col span={4}><Form.Item label="BS H" name="bs_ant_h"><InputNumber min={1} max={16} style={{ width: '100%' }} /></Form.Item></Col>
                          <Col span={4}><Form.Item label="BS V" name="bs_ant_v"><InputNumber min={1} max={16} style={{ width: '100%' }} /></Form.Item></Col>
                          <Col span={4}><Form.Item label="BS P" name="bs_ant_p"><Select options={[{ value: 1, label: '1' }, { value: 2, label: '2' }]} /></Form.Item></Col>
                          <Col span={4}><Form.Item label="UE H" name="ue_ant_h"><InputNumber min={1} max={8} style={{ width: '100%' }} /></Form.Item></Col>
                          <Col span={4}><Form.Item label="UE V" name="ue_ant_v"><InputNumber min={1} max={8} style={{ width: '100%' }} /></Form.Item></Col>
                          <Col span={4}><Form.Item label="UE P" name="ue_ant_p"><Select options={[{ value: 1, label: '1' }, { value: 2, label: '2' }]} /></Form.Item></Col>
                        </Row>
                        <Row gutter={16}>
                          <Col span={8}><Form.Item label="XPD (dB)" name="xpd_db" tooltip="交叉极化鉴别度"><InputNumber min={0} max={30} style={{ width: '100%' }} /></Form.Item></Col>
                        </Row>
                        <Text type="secondary" style={{ fontSize: 12 }}>
                          BS: {panelText(device.bs_ant_h, device.bs_ant_v, device.bs_ant_p)} = {bsTotal}T{bsTotal}R{device.bs_ant_p === 2 ? ' 双极化' : ''}
                          ，UE: {panelText(device.ue_ant_h, device.ue_ant_v, device.ue_ant_p)} = {ueTotal} 端口
                        </Text>
                      </>
                    ),
                  },
                  // ---- Panel: 射频参数 ----
                  {
                    key: 'rf', forceRender: true, label: <Text strong>射频参数</Text>,
                    extra: <Text type="secondary" style={{ fontSize: 12 }}>{freqLabel(device.carrier_freq_hz)} · {bwLabel(device.bandwidth_hz)} · {device.tx_power_dbm} dBm</Text>,
                    children: (
                      <>
                        <Row gutter={16}>
                          <Col span={8}><Form.Item label="载波频率" name="carrier_freq_hz" rules={[{ required: true }]}><Select options={CARRIER_FREQ_OPTIONS} /></Form.Item></Col>
                          <Col span={8}><Form.Item label="系统带宽" name="bandwidth_hz" rules={[{ required: true }]}><Select options={BANDWIDTH_OPTIONS} /></Form.Item></Col>
                          <Col span={8}><Form.Item label="子载波间隔" name="subcarrier_spacing" rules={[{ required: true }]}><Select options={SCS_OPTIONS} /></Form.Item></Col>
                        </Row>
                        <Row gutter={16}>
                          <Col span={8}>
                            <Form.Item label="RB 数（自动）">
                              <InputNumber value={nrRbLookup(device.bandwidth_hz, device.subcarrier_spacing)} disabled style={{ width: '100%' }} />
                            </Form.Item>
                          </Col>
                          <Col span={8}><Form.Item label="BS 发射功率 (dBm)" name="tx_power_dbm" rules={[{ required: true }]}><InputNumber min={0} max={60} style={{ width: '100%' }} /></Form.Item></Col>
                        </Row>
                      </>
                    ),
                  },
                  // ---- Panel: 终端配置 ----
                  {
                    key: 'ue', forceRender: true, label: <Text strong>终端配置</Text>,
                    extra: <Text type="secondary" style={{ fontSize: 12 }}>{device.num_ues} UE · {device.ue_speed_kmh} km/h · {mobilityLabel(device.mobility_mode)}</Text>,
                    children: (
                      <>
                        <Row gutter={16}>
                          <Col span={8}><Form.Item label="用户数" name="num_ues" rules={[{ required: true }]}><InputNumber min={1} max={10000} style={{ width: '100%' }} /></Form.Item></Col>
                          <Col span={8}><Form.Item label="用户分布" name="ue_distribution" rules={[{ required: true }]}><Select options={UE_DISTRIBUTION_OPTIONS} /></Form.Item></Col>
                          <Col span={8}><Form.Item label="移动速度 (km/h)" name="ue_speed_kmh" rules={[{ required: true }]}><InputNumber min={0} max={500} style={{ width: '100%' }} /></Form.Item></Col>
                        </Row>
                        <Row gutter={16}>
                          <Col span={8}><Form.Item label="UE 功率 (dBm)" name="ue_tx_power_dbm" tooltip="3GPP PC3 = 23 dBm" rules={[{ required: true }]}><InputNumber min={-10} max={33} style={{ width: '100%' }} /></Form.Item></Col>
                          <Col span={8}><Form.Item label="运动模式" name="mobility_mode" rules={[{ required: true }]}><Select options={MOBILITY_MODE_OPTIONS} /></Form.Item></Col>
                          {device.mobility_mode !== 'static' && (
                            <Col span={8}><Form.Item label="采样间隔" name="sample_interval_s"><Select options={SAMPLE_INTERVAL_OPTIONS} /></Form.Item></Col>
                          )}
                        </Row>
                        <Row gutter={16}>
                          <Col span={8}><Form.Item label="UE 高度 (m)" name="ue_height_m" tooltip="室外步行 1.5m，车载 1.0m"><InputNumber min={0.5} max={10} step={0.1} style={{ width: '100%' }} /></Form.Item></Col>
                          <Col span={8}><Form.Item label="噪声系数 (dB)" name="noise_figure_db" tooltip="接收机噪声系数，典型 5~9 dB"><InputNumber min={0} max={15} step={0.5} style={{ width: '100%' }} /></Form.Item></Col>
                        </Row>
                      </>
                    ),
                  },
                  // ---- Panel: QuaDRiGa (conditional) ----
                  ...(source === 'quadriga_real' ? [{
                    key: 'qr', forceRender: true, label: <Text strong>QuaDRiGa 配置</Text>,
                    children: (
                      <>
                        <Row gutter={16}>
                          <Col span={12}><Form.Item label="3GPP 场景" name="qr_scenario" rules={[{ required: true }]}><Select options={QUADRIGA_SCENARIO_OPTIONS} /></Form.Item></Col>
                          <Col span={12}><Form.Item label="时间快照数" name="num_snapshots"><InputNumber min={1} max={100} style={{ width: '100%' }} /></Form.Item></Col>
                        </Row>
                        <Row gutter={16}>
                          <Col span={12}><Form.Item label="每分片 UE" name="ues_per_shard"><InputNumber min={10} max={500} style={{ width: '100%' }} /></Form.Item></Col>
                          <Col span={12}>
                            <Form.Item label="数据生成" name="skip_generation">
                              <Radio.Group>
                                <Radio.Button value={false}>生成新数据</Radio.Button>
                                <Radio.Button value={true}>仅读取已有</Radio.Button>
                              </Radio.Group>
                            </Form.Item>
                          </Col>
                        </Row>
                      </>
                    ),
                  }] : []),
                ]}
              />
            </Form>
          </Col>

          {/* RIGHT: topology preview */}
          <Col xs={24} lg={14}>
            <Card title="组网拓扑预览" loading={previewLoading} styles={{ body: { height: 480, padding: 0 } }}>
              <TopologyCanvas
                sites={previewData?.sites ?? []} ues={previewData?.ues ?? []}
                bounds={previewData?.bounds ?? { min_x: -500, max_x: 500, min_y: -500, max_y: 500 }}
                cellRadius={previewData?.cell_radius_m ?? 250}
                topologyLayout={device.topology_layout}
                onSiteDrag={handleSiteDrag} onUEDrag={handleUEDrag}
              />
            </Card>
          </Col>
        </Row>
      )}

      {/* ================================================================ */}
      {/* Step 2: 信道配置                                                  */}
      {/* ================================================================ */}
      {currentStep === 2 && (
        <Card style={{ maxWidth: 720 }}>
          <Form form={channelForm} layout="vertical" initialValues={channelConfig}
            onValuesChange={(_c, all) => setChannelConfig(all as ChannelConfigValues)}
          >
            {/* ---- Core channel params ---- */}
            <Form.Item label="链路方向" name="link" rules={[{ required: true }]}>
              <Radio.Group onChange={e => handleLinkChange(e.target.value)}>
                <Radio.Button value="UL">上行</Radio.Button>
                <Radio.Button value="DL">下行</Radio.Button>
                <Radio.Button value="both">双向</Radio.Button>
              </Radio.Group>
            </Form.Item>

            <Row gutter={16}>
              <Col span={source === 'internal_sim' ? 12 : 24}>
                <Form.Item label="信道估计模式" name="channel_est_mode" rules={[{ required: true }]}>
                  <Select options={[
                    { label: '理想', value: 'ideal' }, { label: 'LS 线性插值', value: 'ls_linear' },
                    { label: 'LS+MMSE', value: 'ls_mmse' }, { label: 'LS 逐跳拼接', value: 'ls_hop_concat' },
                  ]} />
                </Form.Item>
              </Col>
              {source === 'internal_sim' && (
                <Col span={12}>
                  <Form.Item label="信道模型 (CDL)" name="channel_model" rules={[{ required: true }]} tooltip="CDL 簇延迟线：NLOS(A/B/C)瑞利衰落，LOS(D/E)莱斯分量，含逐簇角度信息">
                    <Select options={CHANNEL_MODEL_OPTIONS} />
                  </Form.Item>
                </Col>
              )}
            </Row>

            {channelConfig.link === 'both' ? (
              <Row gutter={16}>
                <Col span={12}><Form.Item label="下行导频" name="pilot_type_dl" rules={[{ required: true }]}><Select options={DL_PILOT_OPTIONS} /></Form.Item></Col>
                <Col span={12}><Form.Item label="上行导频" name="pilot_type_ul" rules={[{ required: true }]}><Select options={UL_PILOT_OPTIONS} /></Form.Item></Col>
              </Row>
            ) : channelConfig.link === 'DL' ? (
              <Form.Item label="导频类型" name="pilot_type_dl" rules={[{ required: true }]}><Select options={DL_PILOT_OPTIONS} /></Form.Item>
            ) : (
              <Form.Item label="导频类型" name="pilot_type_ul" rules={[{ required: true }]}><Select options={UL_PILOT_OPTIONS} /></Form.Item>
            )}

            <Row gutter={16}>
              <Col span={8}>
                <Form.Item label="TDD 配比" name="tdd_pattern" rules={[{ required: true }]}><Select options={TDD_PATTERN_OPTIONS} /></Form.Item>
              </Col>
              <Col span={8}>
                <Form.Item label="采样数量" name="num_samples" rules={[{ required: true }]}><InputNumber min={1} max={100000} style={{ width: '100%' }} /></Form.Item>
              </Col>
              <Col span={8}>
                <Form.Item label="干扰 UE 上限" name="num_interfering_ues" tooltip="每个邻区最大干扰 UE 数">
                  <InputNumber min={0} max={20} style={{ width: '100%' }} />
                </Form.Item>
              </Col>
            </Row>

            <Row gutter={16}>
              <Col span={12}>
                <Form.Item label="邻区预编码投影" name="apply_interferer_precoding" valuePropName="checked"
                  tooltip="用邻区 DL 预编码 W_k 投影干扰信道，模拟调度对干扰的方向性影响（降低干扰秩）">
                  <Switch checkedChildren="开启" unCheckedChildren="关闭" />
                </Form.Item>
              </Col>
              <Col span={12}>
                <Form.Item label="存储干扰信道" name="store_interferer_channels" valuePropName="checked"
                  tooltip="默认关闭：h_interferers 仅作中间产物用于生成 h_serving_est，不持久存储。开启后可用于离线协方差分析">
                  <Switch checkedChildren="存储" unCheckedChildren="不存" />
                </Form.Item>
              </Col>
            </Row>

            {channelConfig.link === 'both' && (
              <Paragraph type="secondary" style={{ marginTop: -8 }}>
                双向配对模式：同时生成 UL（含 SRS 干扰）和 DL（含 CSI-RS 干扰）信道对。
              </Paragraph>
            )}

            {source === 'sionna_rt' && (
              <>
                <Form.Item label="场景" name="scenario" rules={[{ required: true }]}>
                  <Select options={[{ label: 'Munich', value: 'munich' }, { label: 'Etoile', value: 'etoile' }, { label: '自定义 OSM', value: 'custom_osm' }]} />
                </Form.Item>
                {channelConfig.scenario === 'custom_osm' && (
                  <Form.Item label="OSM 文件路径" name="osm_path" rules={[{ required: true }]}>
                    <Input placeholder="例如：D:\maps\custom_area.osm" />
                  </Form.Item>
                )}
              </>
            )}

            {/* ---- SRS advanced params (collapsed) ---- */}
            <Collapse
              ghost style={{ marginTop: 8, marginLeft: -16, marginRight: -16 }}
              items={[{
                key: 'srs', label: <Text type="secondary">SRS 高级参数</Text>,
                children: (
                  <>
                    <Row gutter={16}>
                      <Col span={8}><Form.Item label="SSB 波束数" name="num_ssb_beams"><Select options={[{ value: 4 }, { value: 8 }, { value: 16 }].map(o => ({ ...o, label: `${o.value}` }))} /></Form.Item></Col>
                      <Col span={8}>
                        <Form.Item label="SRS 组跳频" name="srs_group_hopping">
                          <Radio.Group><Radio.Button value={false}>关</Radio.Button><Radio.Button value={true}>开</Radio.Button></Radio.Group>
                        </Form.Item>
                      </Col>
                      <Col span={8}>
                        <Form.Item label="SRS 序列跳频" name="srs_sequence_hopping">
                          <Radio.Group><Radio.Button value={false}>关</Radio.Button><Radio.Button value={true}>开</Radio.Button></Radio.Group>
                        </Form.Item>
                      </Col>
                    </Row>
                    <Row gutter={16}>
                      <Col span={8}><Form.Item label="SRS 周期" name="srs_periodicity"><Select options={SRS_PERIODICITY_OPTIONS} /></Form.Item></Col>
                      <Col span={8}><Form.Item label="频域跳频 (b_hop)" name="srs_b_hop"><Select options={SRS_B_HOP_OPTIONS} /></Form.Item></Col>
                      <Col span={8}><Form.Item label="梳齿 (K_TC)" name="srs_comb"><Select options={[2, 4, 8].map(v => ({ value: v, label: `K_TC=${v}` }))} /></Form.Item></Col>
                    </Row>
                    <Row gutter={16}>
                      <Col span={8}><Form.Item label="C_SRS" name="srs_c_srs"><Select options={SRS_C_SRS_OPTIONS} /></Form.Item></Col>
                      <Col span={8}><Form.Item label="B_SRS" name="srs_b_srs"><Select options={SRS_B_SRS_OPTIONS} /></Form.Item></Col>
                      <Col span={8}><Form.Item label="n_RRC" name="srs_n_rrc"><InputNumber min={0} max={270} style={{ width: '100%' }} /></Form.Item></Col>
                    </Row>
                    <Row gutter={16}>
                      <Col span={8}><Form.Item label="最大秩" name="max_rank" tooltip="SVD 预编码允许的最大空间层数"><Select options={[1, 2, 3, 4].map(v => ({ value: v, label: `Rank ${v}` }))} /></Form.Item></Col>
                      <Col span={8}><Form.Item label="秩选择门限" name="rank_threshold" tooltip="奇异值低于最大值×门限时截断"><InputNumber min={0.01} max={0.5} step={0.01} style={{ width: '100%' }} /></Form.Item></Col>
                    </Row>
                  </>
                ),
              }]}
            />
          </Form>
        </Card>
      )}

      {/* ================================================================ */}
      {/* Step 3: 确认提交                                                  */}
      {/* ================================================================ */}
      {currentStep === 3 && (
        <>
          {/* Quick summary banner */}
          <Space size={[8, 8]} wrap style={{ marginBottom: 16 }}>
            <Tag color="blue">{sourceLabel(source)}</Tag>
            <Tag>{device.topology_layout === 'linear' ? '线性' : '蜂窝'} · {device.num_sites} 站 · ISD {device.isd_m}m</Tag>
            {device.hypercell_size > 1 && <Tag color="purple">HyperCell ×{device.hypercell_size}</Tag>}
            {device.train_penetration_loss_db > 0 && <Tag color="volcano">车体损耗 {device.train_penetration_loss_db}dB</Tag>}
            <Tag color="cyan">BS {bsTotal}T{bsTotal}R ({panelText(device.bs_ant_h, device.bs_ant_v, device.bs_ant_p)})</Tag>
            <Tag>UE {ueTotal} ant ({panelText(device.ue_ant_h, device.ue_ant_v, device.ue_ant_p)})</Tag>
            <Tag color="green">{freqLabel(device.carrier_freq_hz)} · {bwLabel(device.bandwidth_hz)}</Tag>
            <Tag>{nrRbLookup(device.bandwidth_hz, device.subcarrier_spacing) ?? '—'} RB</Tag>
            <Tag color="orange">{linkLabel(channelConfig.link)} · {channelConfig.num_samples} 样本</Tag>
            <Tag>{estLabel(channelConfig.channel_est_mode)}</Tag>
            <Tag>{source === 'internal_sim' ? `${channelConfig.channel_model} · ` : ''}{channelConfig.tdd_pattern}</Tag>
            {channelConfig.apply_interferer_precoding && <Tag color="geekblue">邻区预编码</Tag>}
            {channelConfig.store_interferer_channels && <Tag color="magenta">存储干扰信道</Tag>}
          </Space>

          <Row gutter={24}>
            <Col xs={24} lg={12}>
              <Card title="组网与设备" size="small" style={{ marginBottom: 16 }}>
                <Descriptions column={3} size="small" bordered>
                  <Descriptions.Item label="站点">{device.num_sites}</Descriptions.Item>
                  <Descriptions.Item label="ISD">{device.isd_m} m</Descriptions.Item>
                  {source !== 'quadriga_real' ? (
                    <Descriptions.Item label="扇区/站">{device.sectors_per_site}</Descriptions.Item>
                  ) : (
                    <Descriptions.Item label="站高">{device.tx_height_m} m</Descriptions.Item>
                  )}
                  <Descriptions.Item label="载波">{freqLabel(device.carrier_freq_hz)}</Descriptions.Item>
                  <Descriptions.Item label="带宽">{bwLabel(device.bandwidth_hz)}</Descriptions.Item>
                  <Descriptions.Item label="SCS">{Math.round(device.subcarrier_spacing / 1000)} kHz</Descriptions.Item>
                  <Descriptions.Item label="BS 天线">{panelText(device.bs_ant_h, device.bs_ant_v, device.bs_ant_p)}</Descriptions.Item>
                  <Descriptions.Item label="UE 天线">{panelText(device.ue_ant_h, device.ue_ant_v, device.ue_ant_p)}</Descriptions.Item>
                  <Descriptions.Item label="XPD">{device.xpd_db} dB</Descriptions.Item>
                  <Descriptions.Item label="BS 功率">{device.tx_power_dbm} dBm</Descriptions.Item>
                  <Descriptions.Item label="UE 功率">{device.ue_tx_power_dbm} dBm</Descriptions.Item>
                  <Descriptions.Item label="用户数">{device.num_ues}</Descriptions.Item>
                  <Descriptions.Item label="分布">{distLabel(device.ue_distribution)}</Descriptions.Item>
                  <Descriptions.Item label="速度">{device.ue_speed_kmh} km/h</Descriptions.Item>
                  <Descriptions.Item label="运动">{mobilityLabel(device.mobility_mode)}</Descriptions.Item>
                  <Descriptions.Item label="拓扑布局">{device.topology_layout === 'linear' ? '线性轨道' : '六边形蜂窝'}</Descriptions.Item>
                  <Descriptions.Item label="UE 高度">{device.ue_height_m} m</Descriptions.Item>
                  <Descriptions.Item label="噪声系数">{device.noise_figure_db} dB</Descriptions.Item>
                  {device.hypercell_size > 1 && <Descriptions.Item label="HyperCell">{device.hypercell_size} RRH/组</Descriptions.Item>}
                  {device.train_penetration_loss_db > 0 && <Descriptions.Item label="车体损耗">{device.train_penetration_loss_db} dB</Descriptions.Item>}
                  {device.topology_layout === 'linear' && <Descriptions.Item label="列车长度">{device.train_length_m} m</Descriptions.Item>}
                  {device.topology_layout === 'linear' && <Descriptions.Item label="列车宽度">{device.train_width_m} m</Descriptions.Item>}
                </Descriptions>

                {source === 'quadriga_real' && (
                  <>
                    <Divider orientation="left" orientationMargin={0} style={{ margin: '12px 0', fontSize: 13 }}>QuaDRiGa</Divider>
                    <Descriptions column={2} size="small" bordered>
                      <Descriptions.Item label="场景">{device.qr_scenario}</Descriptions.Item>
                      <Descriptions.Item label="快照">{device.num_snapshots}</Descriptions.Item>
                      <Descriptions.Item label="分片 UE">{device.ues_per_shard}</Descriptions.Item>
                      <Descriptions.Item label="生成">{device.skip_generation ? '仅读取' : '生成新数据'}</Descriptions.Item>
                    </Descriptions>
                  </>
                )}

                {hasCustomPositions && (
                  <Paragraph type="warning" style={{ marginTop: 8, marginBottom: 0, fontSize: 12 }}>
                    已手动调整站点/终端位置，仿真将使用自定义坐标。
                  </Paragraph>
                )}
              </Card>
            </Col>

            <Col xs={24} lg={12}>
              <Card title="信道配置" size="small" style={{ marginBottom: 16 }}>
                <Descriptions column={2} size="small" bordered>
                  <Descriptions.Item label="链路">{linkLabel(channelConfig.link)}</Descriptions.Item>
                  <Descriptions.Item label="估计">{estLabel(channelConfig.channel_est_mode)}</Descriptions.Item>
                  {source === 'internal_sim' && <Descriptions.Item label="模型">{channelConfig.channel_model}</Descriptions.Item>}
                  <Descriptions.Item label="TDD">{channelConfig.tdd_pattern}</Descriptions.Item>
                  <Descriptions.Item label="样本数">{channelConfig.num_samples}</Descriptions.Item>
                  <Descriptions.Item label="干扰 UE">{channelConfig.num_interfering_ues}</Descriptions.Item>
                  {channelConfig.link !== 'UL' && (
                    <Descriptions.Item label="DL 导频">{pilotLabel(channelConfig.pilot_type_dl)}</Descriptions.Item>
                  )}
                  {channelConfig.link !== 'DL' && (
                    <Descriptions.Item label="UL 导频">{pilotLabel(channelConfig.pilot_type_ul)}</Descriptions.Item>
                  )}
                  <Descriptions.Item label="SRS 周期">{channelConfig.srs_periodicity} slots</Descriptions.Item>
                  <Descriptions.Item label="SRS 跳频">{channelConfig.srs_b_hop >= channelConfig.srs_b_srs ? '关' : '开'}</Descriptions.Item>
                  <Descriptions.Item label="最大秩">Rank {channelConfig.max_rank}</Descriptions.Item>
                  <Descriptions.Item label="秩门限">{channelConfig.rank_threshold}</Descriptions.Item>
                  <Descriptions.Item label="邻区预编码">{channelConfig.apply_interferer_precoding ? '开启' : '关闭'}</Descriptions.Item>
                  <Descriptions.Item label="存储干扰信道">{channelConfig.store_interferer_channels ? '存储' : '不存'}</Descriptions.Item>
                  {source === 'sionna_rt' && channelConfig.scenario && (
                    <Descriptions.Item label="场景">{scenarioLabel(channelConfig.scenario)}</Descriptions.Item>
                  )}
                </Descriptions>
              </Card>

              <Card title="拓扑预览" size="small">
                <div style={{ height: 260 }}>
                  <TopologyCanvas
                    sites={previewData?.sites ?? []} ues={previewData?.ues ?? []}
                    bounds={previewData?.bounds ?? { min_x: -500, max_x: 500, min_y: -500, max_y: 500 }}
                    cellRadius={previewData?.cell_radius_m ?? 250} readOnly
                    topologyLayout={device.topology_layout}
                  />
                </div>
              </Card>
            </Col>
          </Row>
        </>
      )}

      {/* ================================================================ */}
      {/* Bottom nav                                                        */}
      {/* ================================================================ */}
      <div style={{ marginTop: 32, paddingTop: 16, borderTop: '1px solid #f0f0f0', display: 'flex', justifyContent: 'space-between' }}>
        <div>{currentStep > 0 && <Button onClick={goPrev}>上一步</Button>}</div>
        <div>
          {currentStep < 3 && <Button type="primary" onClick={goNext}>下一步</Button>}
          {currentStep === 3 && <Button type="primary" loading={collect.isPending} onClick={handleSubmit}>提交采集任务</Button>}
        </div>
      </div>
    </div>
  );
}
