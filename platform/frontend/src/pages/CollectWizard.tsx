import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Badge,
  Breadcrumb,
  Button,
  Card,
  Col,
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
  Tooltip,
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
// Constants
// ---------------------------------------------------------------------------

type DataSource = 'quadriga_real' | 'sionna_rt' | 'internal_sim' | 'internal_upload';

interface SourceOption {
  key: DataSource;
  title: string;
  description: string;
  icon: React.ReactNode;
  disabled?: boolean;
}

const SOURCE_OPTIONS: SourceOption[] = [
  {
    key: 'quadriga_real',
    title: 'QuaDRiGa Real',
    description: 'MATLAB 实时生成多小区信道（需本地 MATLAB）',
    icon: <ExperimentOutlined style={{ fontSize: 36 }} />,
  },
  {
    key: 'sionna_rt',
    title: 'Sionna RT',
    description: '射线追踪仿真（真实场景精确建模，需 GPU）',
    icon: <AimOutlined style={{ fontSize: 36 }} />,
  },
  {
    key: 'internal_sim',
    title: 'Python 内置仿真',
    description: '3GPP 38.901 多小区统计模型（纯 Python，无需外部依赖）',
    icon: <ThunderboltOutlined style={{ fontSize: 36, color: '#52c41a' }} />,
  },
  {
    key: 'internal_upload',
    title: '内部数据上传',
    description: '上传已有数据集（.mat/.npy/.pt 文件）',
    icon: <UploadOutlined style={{ fontSize: 36 }} />,
    disabled: true,
  },
];

const SITE_OPTIONS = [1, 3, 7, 19, 37];
const SECTOR_OPTIONS = [1, 3];
const BS_ANTENNA_OPTIONS = [2, 4, 8, 16, 32, 64, 128, 256];
const UE_ANTENNA_OPTIONS = [1, 2, 4, 8];
const BANDWIDTH_OPTIONS = [
  { label: '5 MHz', value: 5_000_000 },
  { label: '10 MHz', value: 10_000_000 },
  { label: '15 MHz', value: 15_000_000 },
  { label: '20 MHz', value: 20_000_000 },
  { label: '25 MHz', value: 25_000_000 },
  { label: '30 MHz', value: 30_000_000 },
  { label: '40 MHz', value: 40_000_000 },
  { label: '50 MHz', value: 50_000_000 },
  { label: '60 MHz', value: 60_000_000 },
  { label: '70 MHz', value: 70_000_000 },
  { label: '80 MHz', value: 80_000_000 },
  { label: '90 MHz', value: 90_000_000 },
  { label: '100 MHz', value: 100_000_000 },
];
const SCS_OPTIONS = [
  { label: '15 kHz', value: 15_000 },
  { label: '30 kHz', value: 30_000 },
  { label: '60 kHz', value: 60_000 },
  { label: '120 kHz', value: 120_000 },
];

// 3GPP TS 38.101 Table 5.3.2-1: (bandwidth_MHz, scs_kHz) → N_RB
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

function nrRbLookup(bandwidthHz: number, scsHz: number): number | null {
  const bwMhz = Math.round(bandwidthHz / 1e6);
  const scsKhz = Math.round(scsHz / 1e3);
  return NR_RB_TABLE[`${bwMhz}_${scsKhz}`] ?? null;
}

const CARRIER_FREQ_OPTIONS = [
  { label: '700 MHz', value: 700_000_000 },
  { label: '2.1 GHz', value: 2_100_000_000 },
  { label: '2.6 GHz', value: 2_600_000_000 },
  { label: '3.5 GHz', value: 3_500_000_000 },
  { label: '28 GHz', value: 28_000_000_000 },
  { label: '39 GHz', value: 39_000_000_000 },
];
const UE_DISTRIBUTION_OPTIONS = [
  { label: '均匀分布', value: 'uniform' as const },
  { label: '簇状分布', value: 'clustered' as const },
  { label: '热点分布', value: 'hotspot' as const },
];
const MOBILITY_MODE_OPTIONS = [
  { label: '静止', value: 'static' as const },
  { label: '匀速直线', value: 'linear' as const },
  { label: '随机游走', value: 'random_walk' as const },
  { label: '随机航路点', value: 'random_waypoint' as const },
];

// ---------------------------------------------------------------------------
// Form value interfaces
// ---------------------------------------------------------------------------

interface TopologyValues {
  num_sites: number;
  isd_m: number;
  sectors_per_site: number;
  tx_height_m: number;
}

interface BSConfigValues {
  num_bs_tx_ant: number;
  num_bs_rx_ant: number;
  carrier_freq_hz: number;
  bandwidth_hz: number;
  subcarrier_spacing: number;
  tx_power_dbm: number;
}

interface UEConfigValues {
  num_ues: number;
  num_ue_tx_ant: number;
  num_ue_rx_ant: number;
  ue_speed_kmh: number;
  mobility_mode: 'static' | 'linear' | 'random_walk' | 'random_waypoint';
  sample_interval_s: number;
  ue_distribution: 'uniform' | 'clustered' | 'hotspot';
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
  osm_path?: string;
}

interface QuadrigaRealValues {
  num_cells: number;
  isd_m: number;
  tx_height_m: number;
  ue_speed_kmh: number;
  mobility_mode: 'static' | 'linear' | 'random_walk' | 'random_waypoint';
  carrier_freq_hz: number;
  bandwidth_hz: number;
  subcarrier_spacing: number;
  scenario: string;
  num_snapshots: number;
  ues_per_shard: number;
  bs_ant_v: number;
  bs_ant_h: number;
  ue_ant_v: number;
  ue_ant_h: number;
  num_ues: number;
  ue_distribution: 'uniform' | 'clustered' | 'hotspot';
  skip_generation: boolean;
}

// ---------------------------------------------------------------------------
// Defaults
// ---------------------------------------------------------------------------

const DEFAULT_TOPOLOGY: TopologyValues = {
  num_sites: 7,
  isd_m: 500,
  sectors_per_site: 3,
  tx_height_m: 25,
};

const DEFAULT_BS: BSConfigValues = {
  num_bs_tx_ant: 64,
  num_bs_rx_ant: 64,
  carrier_freq_hz: 3_500_000_000,
  bandwidth_hz: 100_000_000,
  subcarrier_spacing: 30_000,
  tx_power_dbm: 43,
};

const DEFAULT_UE: UEConfigValues = {
  num_ues: 50,
  num_ue_tx_ant: 2,
  num_ue_rx_ant: 4,
  ue_speed_kmh: 3,
  mobility_mode: 'static',
  sample_interval_s: 0.0005,
  ue_distribution: 'uniform',
};

const DEFAULT_CHANNEL: ChannelConfigValues = {
  link: 'DL',
  channel_est_mode: 'ideal',
  pilot_type_dl: 'csi_rs_gold',
  pilot_type_ul: 'srs_zc',
  num_samples: 100,
  num_interfering_ues: 3,
  scenario: 'munich',
  channel_model: 'TDL-C',
  tdd_pattern: 'DDDSU',
  num_ssb_beams: 8,
  srs_group_hopping: false,
  srs_sequence_hopping: false,
  srs_periodicity: 10,
  srs_b_hop: 0,
  srs_comb: 2,
  srs_c_srs: 3,
  srs_b_srs: 1,
  srs_n_rrc: 0,
};

const DEFAULT_QUADRIGA_REAL: QuadrigaRealValues = {
  num_cells: 7,
  isd_m: 500,
  tx_height_m: 25,
  ue_speed_kmh: 3,
  mobility_mode: 'linear',
  carrier_freq_hz: 3_500_000_000,
  bandwidth_hz: 100_000_000,
  subcarrier_spacing: 30_000,
  scenario: '3GPP_38.901_UMa_NLOS',
  num_snapshots: 14,
  ues_per_shard: 50,
  bs_ant_v: 4,
  bs_ant_h: 8,
  ue_ant_v: 1,
  ue_ant_h: 2,
  num_ues: 50,
  ue_distribution: 'uniform',
  skip_generation: false,
};

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
  { label: 'TDL-A (NLOS, 23 taps)', value: 'TDL-A' },
  { label: 'TDL-B (NLOS, 23 taps)', value: 'TDL-B' },
  { label: 'TDL-C (NLOS, 24 taps)', value: 'TDL-C' },
  { label: 'TDL-D (LOS, K=13.3dB)', value: 'TDL-D' },
  { label: 'TDL-E (LOS, K=22.0dB)', value: 'TDL-E' },
];

const SRS_PERIODICITY_OPTIONS = [
  { label: '1 slot', value: 1 },
  { label: '2 slots', value: 2 },
  { label: '4 slots', value: 4 },
  { label: '5 slots', value: 5 },
  { label: '8 slots', value: 8 },
  { label: '10 slots', value: 10 },
  { label: '16 slots', value: 16 },
  { label: '20 slots', value: 20 },
  { label: '32 slots', value: 32 },
  { label: '40 slots', value: 40 },
  { label: '64 slots', value: 64 },
  { label: '80 slots', value: 80 },
  { label: '160 slots', value: 160 },
  { label: '320 slots', value: 320 },
  { label: '640 slots', value: 640 },
  { label: '1280 slots', value: 1280 },
  { label: '2560 slots', value: 2560 },
];

const SRS_B_HOP_OPTIONS = [
  { label: '0 (全频段跳频)', value: 0 },
  { label: '1 (部分跳频)', value: 1 },
  { label: '2 (部分跳频)', value: 2 },
  { label: '3 (不跳频)', value: 3 },
];

const SRS_C_SRS_OPTIONS = [
  { label: 'C_SRS=0 (4 RB)', value: 0 },
  { label: 'C_SRS=1 (8 RB, 2×)', value: 1 },
  { label: 'C_SRS=2 (12 RB, 3×)', value: 2 },
  { label: 'C_SRS=3 (16 RB, 4×)', value: 3 },
  { label: 'C_SRS=4 (16 RB, 2×2)', value: 4 },
  { label: 'C_SRS=5 (20 RB, 5×)', value: 5 },
  { label: 'C_SRS=6 (24 RB, 6×)', value: 6 },
  { label: 'C_SRS=7 (24 RB, 2×3)', value: 7 },
  { label: 'C_SRS=8 (28 RB, 7×)', value: 8 },
  { label: 'C_SRS=9 (32 RB, 2×2×2)', value: 9 },
  { label: 'C_SRS=10 (36 RB, 3×3)', value: 10 },
  { label: 'C_SRS=11 (40 RB, 2×5)', value: 11 },
  { label: 'C_SRS=12 (48 RB, 3×2×2)', value: 12 },
  { label: 'C_SRS=13 (48 RB, 2×2×3)', value: 13 },
  { label: 'C_SRS=14 (52 RB, 13×)', value: 14 },
  { label: 'C_SRS=15 (56 RB, 2×7)', value: 15 },
  { label: 'C_SRS=16 (60 RB, 3×5)', value: 16 },
  { label: 'C_SRS=17 (64 RB, 2×2×4)', value: 17 },
];

const SRS_B_SRS_OPTIONS = [
  { label: 'B_SRS=0 (全带宽发送)', value: 0 },
  { label: 'B_SRS=1 (一级细分)', value: 1 },
  { label: 'B_SRS=2 (二级细分)', value: 2 },
  { label: 'B_SRS=3 (三级细分)', value: 3 },
];

const TDD_PATTERN_OPTIONS = [
  { label: 'DDDSU (3D+S+1U, 5ms)', value: 'DDDSU' },
  { label: 'DDSUU (2D+S+2U, 5ms)', value: 'DDSUU' },
  { label: 'DDDDDDDSUU (7D+S+2U, 10ms)', value: 'DDDDDDDSUU' },
  { label: 'DDDSUDDSUU (双周期, 10ms)', value: 'DDDSUDDSUU' },
  { label: 'DSUUD (对称, 5ms)', value: 'DSUUD' },
];

// ---------------------------------------------------------------------------
// Helper: pilot options per link direction
// ---------------------------------------------------------------------------

const DL_PILOT_OPTIONS = [
  { label: 'CSI-RS (Gold 序列)', value: 'csi_rs_gold' },
];
const UL_PILOT_OPTIONS = [
  { label: 'SRS (ZC 序列)', value: 'srs_zc' },
];

// ---------------------------------------------------------------------------
// Helper: label for display values
// ---------------------------------------------------------------------------

function freqLabel(hz: number): string {
  return CARRIER_FREQ_OPTIONS.find((o) => o.value === hz)?.label ?? `${hz} Hz`;
}
function bwLabel(hz: number): string {
  return BANDWIDTH_OPTIONS.find((o) => o.value === hz)?.label ?? `${hz} Hz`;
}
function distLabel(val: string): string {
  return UE_DISTRIBUTION_OPTIONS.find((o) => o.value === val)?.label ?? val;
}
function mobilityLabel(val: string): string {
  return MOBILITY_MODE_OPTIONS.find((o) => o.value === val)?.label ?? val;
}
function linkLabel(val: string): string {
  const map: Record<string, string> = { UL: '上行', DL: '下行', both: '双向' };
  return map[val] ?? val;
}
function estLabel(val: string): string {
  const map: Record<string, string> = { ideal: '理想', ls_linear: 'LS 线性插值', ls_mmse: 'LS+MMSE' };
  return map[val] ?? val;
}
function pilotLabel(val: string): string {
  const map: Record<string, string> = { csi_rs_gold: 'CSI-RS (Gold 序列)', srs_zc: 'SRS (ZC 序列)', ssb: 'SSB (PSS/SSS)', dmrs_pusch: 'DMRS (PUSCH)' };
  return map[val] ?? val;
}
function sourceLabel(val: string): string {
  const map: Record<string, string> = { quadriga_real: 'QuaDRiGa Real (MATLAB)', sionna_rt: 'Sionna RT', internal_sim: 'Python 内置仿真', internal_upload: '内部数据上传' };
  return map[val] ?? val;
}
function scenarioLabel(val: string): string {
  const map: Record<string, string> = { munich: 'Munich', etoile: 'Etoile', custom_osm: '自定义 OSM' };
  return map[val] ?? val;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function CollectWizard() {
  const navigate = useNavigate();
  const collect = useCollectDataset();

  // Step state
  const [currentStep, setCurrentStep] = useState(0);

  // Step 1 state
  const [source, setSource] = useState<DataSource>('quadriga_real');
  const [qrConfig, setQrConfig] = useState<QuadrigaRealValues>(DEFAULT_QUADRIGA_REAL);

  // Step 2 state
  const [topology, setTopology] = useState<TopologyValues>(DEFAULT_TOPOLOGY);
  const [bsConfig, setBsConfig] = useState<BSConfigValues>(DEFAULT_BS);
  const [ueConfig, setUeConfig] = useState<UEConfigValues>(DEFAULT_UE);

  // Step 3 state
  const [channelConfig, setChannelConfig] = useState<ChannelConfigValues>(DEFAULT_CHANNEL);

  // Manual position overrides from canvas drag
  const [siteOverrides, setSiteOverrides] = useState<SitePosition[]>([]);
  const [ueOverrides, setUeOverrides] = useState<UEPosition[]>([]);

  // Forms
  const [topoForm] = Form.useForm<TopologyValues>();
  const [bsForm] = Form.useForm<BSConfigValues>();
  const [ueForm] = Form.useForm<UEConfigValues>();
  const [channelForm] = Form.useForm<ChannelConfigValues>();
  const [qrForm] = Form.useForm<QuadrigaRealValues>();

  // Debounced topology preview request
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [previewReq, setPreviewReq] = useState<TopologyPreviewRequest | null>(null);

  const buildPreviewReq = useCallback(
    (t: TopologyValues, u: UEConfigValues): TopologyPreviewRequest => ({
      num_sites: t.num_sites,
      isd_m: t.isd_m,
      sectors_per_site: t.sectors_per_site,
      tx_height_m: t.tx_height_m,
      num_ues: u.num_ues,
      ue_distribution: u.ue_distribution,
      ue_speed_kmh: u.ue_speed_kmh,
    }),
    [],
  );

  // Trigger preview on config change (debounced)
  const schedulePreview = useCallback(
    (t: TopologyValues, u: UEConfigValues) => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
      debounceRef.current = setTimeout(() => {
        setPreviewReq(buildPreviewReq(t, u));
      }, 500);
    },
    [buildPreviewReq],
  );

  const scheduleQrPreview = useCallback(
    (qr: QuadrigaRealValues) => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
      debounceRef.current = setTimeout(() => {
        setPreviewReq({
          num_sites: qr.num_cells,
          isd_m: qr.isd_m,
          sectors_per_site: 1,
          tx_height_m: qr.tx_height_m,
          num_ues: qr.num_ues,
          ue_distribution: qr.ue_distribution,
          ue_speed_kmh: qr.ue_speed_kmh,
        });
      }, 500);
    },
    [],
  );

  // Initial preview on mount / step enter
  useEffect(() => {
    if (currentStep === 1) {
      if (source === 'quadriga_real') {
        scheduleQrPreview(qrConfig);
      } else {
        schedulePreview(topology, ueConfig);
      }
    }
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentStep]);

  const { data: previewData, isFetching: previewLoading } = useTopologyPreview(previewReq);

  // Callbacks for drag on canvas
  const handleSiteDrag = useCallback((sites: SitePosition[]) => {
    setSiteOverrides(sites);
  }, []);

  const handleUEDrag = useCallback((ues: UEPosition[]) => {
    setUeOverrides(ues);
  }, []);

  const handleLinkChange = useCallback(
    (link: 'UL' | 'DL' | 'both') => {
      setChannelConfig((prev) => ({ ...prev, link }));
      channelForm.setFieldsValue({ link });
    },
    [channelForm],
  );

  // Step validation before proceeding
  const validateStep = useCallback(async (): Promise<boolean> => {
    try {
      if (currentStep === 0) {
        // Source must be selected (always is)
        if (source === 'internal_upload') {
          message.warning('内部数据上传功能即将支持');
          return false;
        }
        return true;
      }
      if (currentStep === 1) {
        if (source === 'quadriga_real') {
          await qrForm.validateFields();
          return true;
        }
        await topoForm.validateFields();
        await bsForm.validateFields();
        await ueForm.validateFields();
        return true;
      }
      if (currentStep === 2) {
        await channelForm.validateFields();
        return true;
      }
      return true;
    } catch {
      return false;
    }
  }, [currentStep, source, topoForm, bsForm, ueForm, channelForm]);

  const goNext = useCallback(async () => {
    const ok = await validateStep();
    if (ok) setCurrentStep((s) => Math.min(s + 1, 3));
  }, [validateStep]);

  const goPrev = useCallback(() => {
    setCurrentStep((s) => Math.max(s - 1, 0));
  }, []);

  const hasCustomPositions = siteOverrides.length > 0 || ueOverrides.length > 0;

  // Build full config for submission
  const fullConfig = useMemo(() => {
    const config: Record<string, unknown> = {};

    if (source === 'quadriga_real') {
      Object.assign(config, qrConfig);
    } else {
      Object.assign(config, topology, bsConfig, ueConfig);
    }

    config.link = channelConfig.link;
    config.channel_est_mode = channelConfig.channel_est_mode;
    config.channel_model = channelConfig.channel_model;
    config.tdd_pattern = channelConfig.tdd_pattern;
    config.num_ssb_beams = channelConfig.num_ssb_beams;
    config.srs_group_hopping = channelConfig.srs_group_hopping;
    config.srs_sequence_hopping = channelConfig.srs_sequence_hopping;
    config.srs_periodicity = channelConfig.srs_periodicity;
    config.srs_b_hop = channelConfig.srs_b_hop;
    config.srs_comb = channelConfig.srs_comb;
    config.srs_c_srs = channelConfig.srs_c_srs;
    config.srs_b_srs = channelConfig.srs_b_srs;
    config.srs_n_rrc = channelConfig.srs_n_rrc;
    config.num_samples = channelConfig.num_samples;
    config.num_interfering_ues = channelConfig.num_interfering_ues;

    config.pilot_type_dl = channelConfig.pilot_type_dl;
    config.pilot_type_ul = channelConfig.pilot_type_ul;
    if (source === 'sionna_rt' && channelConfig.scenario) {
      config.scenario = channelConfig.scenario;
      if (channelConfig.scenario === 'custom_osm' && channelConfig.osm_path) {
        config.osm_path = channelConfig.osm_path;
      }
    }
    if (siteOverrides.length > 0) {
      config.custom_site_positions = siteOverrides.map((s) => ({ x: s.x, y: s.y, z: s.z }));
    }
    if (ueOverrides.length > 0) {
      config.custom_ue_positions = ueOverrides.map((u) => ({ x: u.x, y: u.y, z: u.z }));
    }
    return config;
  }, [topology, bsConfig, ueConfig, channelConfig, source, qrConfig, siteOverrides, ueOverrides]);

  // Submit
  const handleSubmit = useCallback(async () => {
    try {
      const res = await collect.mutateAsync({
        source,
        config_overrides: fullConfig,
      });
      message.success(`采集任务 ${res.job_id} 已提交`);
      navigate(`/jobs/${res.job_id}`);
    } catch (e) {
      message.error((e as Error).message ?? '提交失败');
    }
  }, [collect, source, fullConfig, navigate]);

  // ------- Render -------

  return (
    <div className="msg-page">
      <Breadcrumb
        className="msg-breadcrumb"
        items={[
          { title: <Link to="/datasets">数据集</Link> },
          { title: '数据采集向导' },
        ]}
      />

      <Title level={3} style={{ marginBottom: 24 }}>
        数据采集向导
      </Title>

      <Steps
        current={currentStep}
        style={{ marginBottom: 32 }}
        items={[
          { title: '数据来源' },
          { title: source === 'quadriga_real' ? 'MATLAB 生成配置' : '组网与设备配置' },
          { title: '信道配置' },
          { title: '确认提交' },
        ]}
      />

      {/* ====== Step 1: 数据来源 ====== */}
      {currentStep === 0 && (
        <Row gutter={24} justify="center">
          {SOURCE_OPTIONS.map((opt) => (
            <Col key={opt.key} xs={24} sm={8} style={{ marginBottom: 16 }}>
              {opt.disabled ? (
                <Badge.Ribbon text="即将支持" color="orange">
                  <Card
                    hoverable
                    style={{
                      textAlign: 'center',
                      cursor: 'not-allowed',
                      opacity: 0.55,
                      minHeight: 220,
                    }}
                  >
                    <div style={{ fontSize: 36, marginBottom: 16, color: '#999' }}>
                      {opt.icon}
                    </div>
                    <Title level={4} style={{ marginBottom: 8 }}>
                      {opt.title}
                    </Title>
                    <Text type="secondary">{opt.description}</Text>
                    <div style={{ marginTop: 16 }}>
                      <Dragger
                        disabled
                        style={{ padding: 8 }}
                        showUploadList={false}
                      >
                        <Paragraph type="secondary" style={{ margin: 0 }}>
                          拖拽或点击上传 .mat / .npy / .pt 文件
                        </Paragraph>
                      </Dragger>
                    </div>
                  </Card>
                </Badge.Ribbon>
              ) : (
                <Card
                  hoverable
                  onClick={() => setSource(opt.key)}
                  style={{
                    textAlign: 'center',
                    borderColor: source === opt.key ? '#1677ff' : undefined,
                    borderWidth: source === opt.key ? 2 : 1,
                    boxShadow: source === opt.key ? '0 0 0 2px rgba(22,119,255,0.15)' : undefined,
                    minHeight: 220,
                    display: 'flex',
                    flexDirection: 'column',
                    justifyContent: 'center',
                  }}
                  styles={{ body: { display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' } }}
                >
                  <div
                    style={{
                      fontSize: 36,
                      marginBottom: 16,
                      color: source === opt.key ? '#1677ff' : '#595959',
                    }}
                  >
                    {opt.icon}
                  </div>
                  <Title level={4} style={{ marginBottom: 8 }}>
                    {opt.title}
                  </Title>
                  <Text type="secondary">{opt.description}</Text>
                  {source === opt.key && (
                    <CheckCircleOutlined
                      style={{ color: '#1677ff', fontSize: 22, marginTop: 12 }}
                    />
                  )}
                </Card>
              )}
            </Col>
          ))}
        </Row>
      )}

      {/* ====== Step 2: MATLAB 生成配置 (quadriga_real) ====== */}
      {currentStep === 1 && source === 'quadriga_real' && (
        <Row gutter={24}>
          {/* LEFT: config forms */}
          <Col xs={24} lg={10}>
            <Form
              form={qrForm}
              layout="vertical"
              initialValues={qrConfig}
              onValuesChange={(_changed, all) => {
                const vals = all as QuadrigaRealValues;
                setQrConfig(vals);
                scheduleQrPreview(vals);
              }}
            >
              <Space direction="vertical" size="middle" style={{ width: '100%' }}>
                <Card title="场景与拓扑" size="small">
                  <Row gutter={16}>
                    <Col span={12}>
                      <Form.Item
                        label="3GPP 场景"
                        name="scenario"
                        rules={[{ required: true, message: '请选择场景' }]}
                      >
                        <Select options={QUADRIGA_SCENARIO_OPTIONS} />
                      </Form.Item>
                    </Col>
                    <Col span={12}>
                      <Form.Item
                        label="小区数"
                        name="num_cells"
                        rules={[{ required: true }]}
                      >
                        <Select options={[1, 3, 7, 19, 21].map((v) => ({ value: v, label: `${v}` }))} />
                      </Form.Item>
                    </Col>
                  </Row>
                  <Row gutter={16}>
                    <Col span={12}>
                      <Form.Item label="站间距 (m)" name="isd_m" rules={[{ required: true }]}>
                        <InputNumber min={50} max={5000} style={{ width: '100%' }} />
                      </Form.Item>
                    </Col>
                    <Col span={12}>
                      <Form.Item label="基站高度 (m)" name="tx_height_m" rules={[{ required: true }]}>
                        <InputNumber min={1} max={200} style={{ width: '100%' }} />
                      </Form.Item>
                    </Col>
                  </Row>
                  <Row gutter={16}>
                    <Col span={12}>
                      <Form.Item label="载波频率" name="carrier_freq_hz" rules={[{ required: true }]}>
                        <Select options={CARRIER_FREQ_OPTIONS} />
                      </Form.Item>
                    </Col>
                    <Col span={12}>
                      <Form.Item label="时间快照数" name="num_snapshots" tooltip="每个 UE 的时间采样点数">
                        <InputNumber min={1} max={100} style={{ width: '100%' }} />
                      </Form.Item>
                    </Col>
                  </Row>
                  <Row gutter={16}>
                    <Col span={8}>
                      <Form.Item label="系统带宽" name="bandwidth_hz" rules={[{ required: true }]}>
                        <Select options={BANDWIDTH_OPTIONS} />
                      </Form.Item>
                    </Col>
                    <Col span={8}>
                      <Form.Item label="子载波间隔" name="subcarrier_spacing" rules={[{ required: true }]}>
                        <Select options={SCS_OPTIONS} />
                      </Form.Item>
                    </Col>
                    <Col span={8}>
                      <Form.Item label="RB 数（自动）" tooltip="根据带宽+SCS 查 3GPP TS 38.101 标准表自动计算">
                        <InputNumber
                          value={nrRbLookup(qrConfig.bandwidth_hz, qrConfig.subcarrier_spacing)}
                          disabled
                          style={{ width: '100%' }}
                        />
                      </Form.Item>
                    </Col>
                  </Row>
                </Card>

                <Card title="天线阵列" size="small">
                  <Row gutter={16}>
                    <Col span={6}>
                      <Form.Item label="BS 垂直" name="bs_ant_v" tooltip="基站天线垂直方向单元数">
                        <InputNumber min={1} max={16} style={{ width: '100%' }} />
                      </Form.Item>
                    </Col>
                    <Col span={6}>
                      <Form.Item label="BS 水平" name="bs_ant_h" tooltip="基站天线水平方向单元数">
                        <InputNumber min={1} max={16} style={{ width: '100%' }} />
                      </Form.Item>
                    </Col>
                    <Col span={6}>
                      <Form.Item label="UE 垂直" name="ue_ant_v" tooltip="终端天线垂直方向单元数">
                        <InputNumber min={1} max={8} style={{ width: '100%' }} />
                      </Form.Item>
                    </Col>
                    <Col span={6}>
                      <Form.Item label="UE 水平" name="ue_ant_h" tooltip="终端天线水平方向单元数">
                        <InputNumber min={1} max={8} style={{ width: '100%' }} />
                      </Form.Item>
                    </Col>
                  </Row>
                  <Paragraph type="secondary" style={{ marginBottom: 0 }}>
                    总天线数：BS {qrConfig.bs_ant_v * qrConfig.bs_ant_h} 端口，UE {qrConfig.ue_ant_v * qrConfig.ue_ant_h} 端口
                  </Paragraph>
                </Card>

                <Card title="终端配置" size="small">
                  <Row gutter={16}>
                    <Col span={8}>
                      <Form.Item
                        label="用户数"
                        name="num_ues"
                        rules={[{ required: true, message: '请输入用户数' }]}
                      >
                        <InputNumber min={1} max={10000} style={{ width: '100%' }} />
                      </Form.Item>
                    </Col>
                    <Col span={8}>
                      <Form.Item
                        label="用户分布"
                        name="ue_distribution"
                        rules={[{ required: true }]}
                      >
                        <Select options={UE_DISTRIBUTION_OPTIONS} />
                      </Form.Item>
                    </Col>
                    <Col span={8}>
                      <Form.Item label="移动速度 (km/h)" name="ue_speed_kmh" rules={[{ required: true }]}>
                        <InputNumber min={0} max={500} style={{ width: '100%' }} />
                      </Form.Item>
                    </Col>
                  </Row>
                  <Row gutter={16}>
                    <Col span={12}>
                      <Form.Item
                        label="运动模式"
                        name="mobility_mode"
                        tooltip="QuaDRiGa 原生支持轨迹建模，此参数传递给 MATLAB 端"
                        rules={[{ required: true }]}
                      >
                        <Select options={MOBILITY_MODE_OPTIONS} />
                      </Form.Item>
                    </Col>
                  </Row>
                </Card>

                <Card title="MATLAB 生成控制" size="small">
                  <Row gutter={16}>
                    <Col span={12}>
                      <Form.Item label="每分片 UE 数" name="ues_per_shard" tooltip="每次 MATLAB 调用生成的 UE 数量，影响内存占用">
                        <InputNumber min={10} max={500} style={{ width: '100%' }} />
                      </Form.Item>
                    </Col>
                    <Col span={12}>
                      <Form.Item label="跳过生成" name="skip_generation" tooltip="仅读取已有 .mat 文件，不启动 MATLAB">
                        <Radio.Group>
                          <Radio.Button value={false}>生成新数据</Radio.Button>
                          <Radio.Button value={true}>仅读取已有</Radio.Button>
                        </Radio.Group>
                      </Form.Item>
                    </Col>
                  </Row>
                </Card>
              </Space>
            </Form>
          </Col>

          {/* RIGHT: topology preview */}
          <Col xs={24} lg={14}>
            <Card
              title="组网拓扑预览"
              loading={previewLoading}
              styles={{ body: { height: 480, padding: 0 } }}
            >
              <TopologyCanvas
                sites={previewData?.sites ?? []}
                ues={previewData?.ues ?? []}
                bounds={previewData?.bounds ?? { min_x: -500, max_x: 500, min_y: -500, max_y: 500 }}
                cellRadius={previewData?.cell_radius_m ?? 250}
                onSiteDrag={handleSiteDrag}
                onUEDrag={handleUEDrag}
              />
            </Card>
          </Col>
        </Row>
      )}

      {/* ====== Step 2: 组网与设备配置 (sionna_rt / internal_sim) ====== */}
      {currentStep === 1 && source !== 'quadriga_real' && (
        <Row gutter={24}>
          {/* LEFT: config forms */}
          <Col xs={24} lg={10}>
            <Space direction="vertical" size="middle" style={{ width: '100%' }}>
              {/* Topology */}
              <Card title="拓扑配置" size="small">
                <Form
                  form={topoForm}
                  layout="vertical"
                  initialValues={topology}
                  onValuesChange={(_changed, all) => {
                    const vals = all as TopologyValues;
                    setTopology(vals);
                    schedulePreview(vals, ueConfig);
                  }}
                >
                  <Row gutter={16}>
                    <Col span={12}>
                      <Form.Item
                        label="小区站点数"
                        name="num_sites"
                        rules={[{ required: true, message: '请选择站点数' }]}
                      >
                        <Select
                          options={SITE_OPTIONS.map((v) => ({ value: v, label: `${v}` }))}
                        />
                      </Form.Item>
                    </Col>
                    <Col span={12}>
                      <Form.Item
                        label={
                          <Tooltip title="站间距离，单位米">
                            站间距 (m)
                          </Tooltip>
                        }
                        name="isd_m"
                        rules={[{ required: true, message: '请输入站间距' }]}
                      >
                        <InputNumber min={50} max={5000} style={{ width: '100%' }} />
                      </Form.Item>
                    </Col>
                  </Row>
                  <Row gutter={16}>
                    <Col span={12}>
                      <Form.Item
                        label="扇区数/站"
                        name="sectors_per_site"
                        rules={[{ required: true, message: '请选择扇区数' }]}
                      >
                        <Select
                          options={SECTOR_OPTIONS.map((v) => ({ value: v, label: `${v}` }))}
                        />
                      </Form.Item>
                    </Col>
                    <Col span={12}>
                      <Form.Item
                        label="基站高度 (m)"
                        name="tx_height_m"
                        rules={[{ required: true, message: '请输入基站高度' }]}
                      >
                        <InputNumber min={1} max={200} style={{ width: '100%' }} />
                      </Form.Item>
                    </Col>
                  </Row>
                </Form>
              </Card>

              {/* BS Config */}
              <Card title="基站配置" size="small">
                <Form
                  form={bsForm}
                  layout="vertical"
                  initialValues={bsConfig}
                  onValuesChange={(_changed, all) => {
                    setBsConfig(all as BSConfigValues);
                  }}
                >
                  <Row gutter={16}>
                    <Col span={12}>
                      <Form.Item
                        label="发射天线数 (TX)"
                        name="num_bs_tx_ant"
                        rules={[{ required: true, message: '请选择天线数' }]}
                      >
                        <Select
                          options={BS_ANTENNA_OPTIONS.map((v) => ({ value: v, label: `${v}T` }))}
                        />
                      </Form.Item>
                    </Col>
                    <Col span={12}>
                      <Form.Item
                        label="接收天线数 (RX)"
                        name="num_bs_rx_ant"
                        rules={[{ required: true, message: '请选择天线数' }]}
                      >
                        <Select
                          options={BS_ANTENNA_OPTIONS.map((v) => ({ value: v, label: `${v}R` }))}
                        />
                      </Form.Item>
                    </Col>
                  </Row>
                  <Row gutter={16}>
                    <Col span={12}>
                      <Form.Item
                        label="载波频率"
                        name="carrier_freq_hz"
                        rules={[{ required: true, message: '请选择载波频率' }]}
                      >
                        <Select options={CARRIER_FREQ_OPTIONS} />
                      </Form.Item>
                    </Col>
                    <Col span={12}>
                      <Form.Item
                        label="带宽"
                        name="bandwidth_hz"
                        rules={[{ required: true, message: '请选择带宽' }]}
                      >
                        <Select options={BANDWIDTH_OPTIONS} />
                      </Form.Item>
                    </Col>
                  </Row>
                  <Row gutter={16}>
                    <Col span={8}>
                      <Form.Item
                        label="子载波间隔"
                        name="subcarrier_spacing"
                        tooltip="NR 子载波间隔 (μ=0→15kHz, μ=1→30kHz, μ=2→60kHz, μ=3→120kHz)"
                        rules={[{ required: true, message: '请选择子载波间隔' }]}
                      >
                        <Select options={SCS_OPTIONS} />
                      </Form.Item>
                    </Col>
                    <Col span={8}>
                      <Form.Item label="RB 数 (自动)">
                        <InputNumber
                          value={nrRbLookup(bsConfig.bandwidth_hz, bsConfig.subcarrier_spacing)}
                          disabled
                          style={{ width: '100%' }}
                          placeholder={
                            nrRbLookup(bsConfig.bandwidth_hz, bsConfig.subcarrier_spacing)
                              ? undefined
                              : '该组合非标准'
                          }
                        />
                      </Form.Item>
                    </Col>
                    <Col span={8}>
                      <Form.Item
                        label="发射功率 (dBm)"
                        name="tx_power_dbm"
                        rules={[{ required: true, message: '请输入发射功率' }]}
                      >
                        <InputNumber min={0} max={60} style={{ width: '100%' }} />
                      </Form.Item>
                    </Col>
                  </Row>
                </Form>
              </Card>

              {/* UE Config */}
              <Card title="终端配置" size="small">
                <Form
                  form={ueForm}
                  layout="vertical"
                  initialValues={ueConfig}
                  onValuesChange={(_changed, all) => {
                    const vals = all as UEConfigValues;
                    setUeConfig(vals);
                    schedulePreview(topology, vals);
                  }}
                >
                  <Row gutter={16}>
                    <Col span={12}>
                      <Form.Item
                        label="用户数"
                        name="num_ues"
                        rules={[{ required: true, message: '请输入用户数' }]}
                      >
                        <InputNumber min={1} max={10000} style={{ width: '100%' }} />
                      </Form.Item>
                    </Col>
                    <Col span={12}>
                      <Form.Item
                        label="用户分布"
                        name="ue_distribution"
                        rules={[{ required: true, message: '请选择用户分布' }]}
                      >
                        <Select options={UE_DISTRIBUTION_OPTIONS} />
                      </Form.Item>
                    </Col>
                  </Row>
                  <Row gutter={16}>
                    <Col span={12}>
                      <Form.Item
                        label="发射天线数 (TX)"
                        name="num_ue_tx_ant"
                        rules={[{ required: true, message: '请选择天线数' }]}
                      >
                        <Select
                          options={UE_ANTENNA_OPTIONS.map((v) => ({ value: v, label: `${v}T` }))}
                        />
                      </Form.Item>
                    </Col>
                    <Col span={12}>
                      <Form.Item
                        label="接收天线数 (RX)"
                        name="num_ue_rx_ant"
                        rules={[{ required: true, message: '请选择天线数' }]}
                      >
                        <Select
                          options={UE_ANTENNA_OPTIONS.map((v) => ({ value: v, label: `${v}R` }))}
                        />
                      </Form.Item>
                    </Col>
                  </Row>
                  <Row gutter={16}>
                    <Col span={12}>
                      <Form.Item
                        label="运动模式"
                        name="mobility_mode"
                        tooltip="静止=固定位置；匀速直线=恒速恒向；随机游走=每步随机转向（步行/车辆）；随机航路点=随机选目标点→匀速移动→到达后再选"
                        rules={[{ required: true, message: '请选择运动模式' }]}
                      >
                        <Select options={MOBILITY_MODE_OPTIONS} />
                      </Form.Item>
                    </Col>
                    <Col span={12}>
                      <Form.Item
                        label="移动速度 (km/h)"
                        name="ue_speed_kmh"
                        rules={[{ required: true, message: '请输入移动速度' }]}
                      >
                        <InputNumber min={0} max={500} style={{ width: '100%' }} />
                      </Form.Item>
                    </Col>
                  </Row>
                  {ueConfig.mobility_mode !== 'static' && (
                    <Row gutter={16}>
                      <Col span={12}>
                        <Form.Item
                          label="采样间隔 (秒)"
                          name="sample_interval_s"
                          tooltip="相邻两个样本之间的时间间隔。0.5ms = 1 slot @ 30kHz SCS；1ms = 1 子帧；10ms = 1 帧"
                        >
                          <Select
                            options={[
                              { label: '0.5 ms (1 slot)', value: 0.0005 },
                              { label: '1 ms (1 子帧)', value: 0.001 },
                              { label: '5 ms', value: 0.005 },
                              { label: '10 ms (1 帧)', value: 0.01 },
                              { label: '100 ms', value: 0.1 },
                              { label: '1 s', value: 1.0 },
                            ]}
                          />
                        </Form.Item>
                      </Col>
                    </Row>
                  )}
                </Form>
              </Card>
            </Space>
          </Col>

          {/* RIGHT: topology preview */}
          <Col xs={24} lg={14}>
            <Card
              title="组网拓扑预览"
              loading={previewLoading}
              styles={{ body: { height: 480, padding: 0 } }}
            >
              <TopologyCanvas
                sites={previewData?.sites ?? []}
                ues={previewData?.ues ?? []}
                bounds={previewData?.bounds ?? { min_x: -500, max_x: 500, min_y: -500, max_y: 500 }}
                cellRadius={previewData?.cell_radius_m ?? 250}
                onSiteDrag={handleSiteDrag}
                onUEDrag={handleUEDrag}
              />
            </Card>
          </Col>
        </Row>
      )}

      {/* ====== Step 3: 信道配置 ====== */}
      {currentStep === 2 && (
        <Card style={{ maxWidth: 720 }}>
          <Form
            form={channelForm}
            layout="vertical"
            initialValues={channelConfig}
            onValuesChange={(_changed, all) => {
              const vals = all as ChannelConfigValues;
              setChannelConfig(vals);
            }}
          >
            <Form.Item
              label="链路方向"
              name="link"
              rules={[{ required: true, message: '请选择链路方向' }]}
            >
              <Radio.Group
                onChange={(e) => handleLinkChange(e.target.value)}
              >
                <Radio.Button value="UL">上行</Radio.Button>
                <Radio.Button value="DL">下行</Radio.Button>
                <Radio.Button value="both">双向</Radio.Button>
              </Radio.Group>
            </Form.Item>

            <Form.Item
              label="信道估计模式"
              name="channel_est_mode"
              rules={[{ required: true, message: '请选择信道估计模式' }]}
            >
              <Select
                options={[
                  { label: '理想', value: 'ideal' },
                  { label: 'LS 线性插值', value: 'ls_linear' },
                  { label: 'LS+MMSE', value: 'ls_mmse' },
                ]}
              />
            </Form.Item>

            {channelConfig.link === 'both' ? (
              <>
                <Row gutter={16}>
                  <Col span={12}>
                    <Form.Item
                      label="下行导频"
                      name="pilot_type_dl"
                      rules={[{ required: true, message: '请选择下行导频' }]}
                    >
                      <Select options={DL_PILOT_OPTIONS} />
                    </Form.Item>
                  </Col>
                  <Col span={12}>
                    <Form.Item
                      label="上行导频"
                      name="pilot_type_ul"
                      rules={[{ required: true, message: '请选择上行导频' }]}
                    >
                      <Select options={UL_PILOT_OPTIONS} />
                    </Form.Item>
                  </Col>
                </Row>
                <Paragraph type="secondary" style={{ marginTop: -8 }}>
                  双向模式下，下行使用 CSI-RS 信道状态信息参考信号，上行使用 SRS 探测参考信号。
                </Paragraph>
              </>
            ) : channelConfig.link === 'DL' ? (
              <Form.Item
                label="下行导频类型"
                name="pilot_type_dl"
                rules={[{ required: true, message: '请选择导频类型' }]}
              >
                <Select options={DL_PILOT_OPTIONS} />
              </Form.Item>
            ) : (
              <Form.Item
                label="上行导频类型"
                name="pilot_type_ul"
                rules={[{ required: true, message: '请选择导频类型' }]}
              >
                <Select options={UL_PILOT_OPTIONS} />
              </Form.Item>
            )}

            <Divider orientation="left" plain>信道模型与帧结构</Divider>

            <Row gutter={16}>
              <Col span={12}>
                <Form.Item
                  label="信道模型"
                  name="channel_model"
                  tooltip="3GPP 38.901 TDL 信道模型。NLOS(A/B/C)为瑞利衰落，LOS(D/E)含莱斯分量"
                  rules={[{ required: true, message: '请选择信道模型' }]}
                >
                  <Select options={CHANNEL_MODEL_OPTIONS} />
                </Form.Item>
              </Col>
              <Col span={12}>
                <Form.Item
                  label="TDD 时隙配比"
                  name="tdd_pattern"
                  tooltip="D=下行, U=上行, S=特殊时隙(10D+2G+2U symbols)"
                  rules={[{ required: true, message: '请选择 TDD 配比' }]}
                >
                  <Select options={TDD_PATTERN_OPTIONS} />
                </Form.Item>
              </Col>
            </Row>

            <Row gutter={16}>
              <Col span={8}>
                <Form.Item
                  label="SSB 波束数"
                  name="num_ssb_beams"
                  tooltip="SSB 波束扫描使用的 DFT 波束数量"
                >
                  <Select
                    options={[
                      { label: '4', value: 4 },
                      { label: '8', value: 8 },
                      { label: '16', value: 16 },
                    ]}
                  />
                </Form.Item>
              </Col>
              <Col span={8}>
                <Form.Item
                  label="SRS 组跳频"
                  name="srs_group_hopping"
                  valuePropName="checked"
                  tooltip="启用 SRS 组跳频 (38.211 §6.4.1.4)"
                >
                  <Radio.Group>
                    <Radio.Button value={false}>关闭</Radio.Button>
                    <Radio.Button value={true}>开启</Radio.Button>
                  </Radio.Group>
                </Form.Item>
              </Col>
              <Col span={8}>
                <Form.Item
                  label="SRS 序列跳频"
                  name="srs_sequence_hopping"
                  valuePropName="checked"
                  tooltip="启用 SRS 序列跳频 (38.211 §6.4.1.4)"
                >
                  <Radio.Group>
                    <Radio.Button value={false}>关闭</Radio.Button>
                    <Radio.Button value={true}>开启</Radio.Button>
                  </Radio.Group>
                </Form.Item>
              </Col>
            </Row>
            <Row gutter={16}>
              <Col span={8}>
                <Form.Item
                  label="SRS 周期 (T_SRS)"
                  name="srs_periodicity"
                  tooltip="SRS 发送周期 (38.211 Table 6.4.1.4.4-1)，单位：时隙"
                >
                  <Select options={SRS_PERIODICITY_OPTIONS} />
                </Form.Item>
              </Col>
              <Col span={8}>
                <Form.Item
                  label="SRS 频域跳频 (b_hop)"
                  name="srs_b_hop"
                  tooltip="频域跳频参数 (38.211 §6.4.1.4.3)，b_hop < B_SRS 时启用频域跳频；b_hop ≥ B_SRS 时不跳频"
                >
                  <Select options={SRS_B_HOP_OPTIONS} />
                </Form.Item>
              </Col>
              <Col span={8}>
                <Form.Item
                  label="SRS 传输梳齿 (K_TC)"
                  name="srs_comb"
                  tooltip="SRS 传输梳齿 K_TC (38.211 §6.4.1.4.2)，决定 SRS 的频域密度"
                >
                  <Select
                    options={[
                      { label: 'K_TC = 2', value: 2 },
                      { label: 'K_TC = 4', value: 4 },
                      { label: 'K_TC = 8', value: 8 },
                    ]}
                  />
                </Form.Item>
              </Col>
            </Row>
            <Row gutter={16}>
              <Col span={8}>
                <Form.Item
                  label="SRS 带宽配置 (C_SRS)"
                  name="srs_c_srs"
                  tooltip="SRS 带宽配置索引 (38.211 Table 6.4.1.4.3-1)，决定 SRS 总带宽和跳频树结构"
                >
                  <Select options={SRS_C_SRS_OPTIONS} />
                </Form.Item>
              </Col>
              <Col span={8}>
                <Form.Item
                  label="SRS 带宽层级 (B_SRS)"
                  name="srs_b_srs"
                  tooltip="带宽树层级 (38.211 §6.4.1.4.3)，决定每次 SRS 发送覆盖的 RB 数。B_SRS 越大单次覆盖越窄，跳频倍数越高"
                >
                  <Select options={SRS_B_SRS_OPTIONS} />
                </Form.Item>
              </Col>
              <Col span={8}>
                <Form.Item
                  label="SRS 频域起始位置 (n_RRC)"
                  name="srs_n_rrc"
                  tooltip="RRC 配置的 SRS 频域起始 RB 位置 (38.211 §6.4.1.4.3)"
                >
                  <InputNumber min={0} max={270} style={{ width: '100%' }} />
                </Form.Item>
              </Col>
            </Row>

            <Divider orientation="left" plain>干扰与采样配置</Divider>

            <Row gutter={16}>
              <Col span={12}>
                <Form.Item
                  label="邻区干扰 UE 数（上限）"
                  name="num_interfering_ues"
                  tooltip="上行 SRS 干扰：每个邻区随机 0~N 个 UE 同时发送 SRS。下行 CSI-RS 干扰：随机 0~K-1 个邻区发送 CSI-RS。设为 0 则无干扰注入"
                >
                  <InputNumber min={0} max={20} style={{ width: '100%' }} />
                </Form.Item>
              </Col>
              <Col span={12}>
                <Form.Item
                  label="采样数量"
                  name="num_samples"
                  rules={[{ required: true, message: '请输入采样数量' }]}
                >
                  <InputNumber min={1} max={100000} style={{ width: '100%' }} />
                </Form.Item>
              </Col>
            </Row>
            {channelConfig.link === 'both' && (
              <Paragraph type="secondary" style={{ marginTop: -8 }}>
                双向配对模式：每个采样点同时生成 UL（含邻区 SRS 干扰）和 DL（含邻区 CSI-RS 干扰）的理想/估计信道对，干扰 UE 数和邻区数逐样本随机。
              </Paragraph>
            )}

            {source === 'sionna_rt' && (
              <Form.Item
                label="场景"
                name="scenario"
                rules={[{ required: true, message: '请选择场景' }]}
              >
                <Select
                  options={[
                    { label: 'Munich', value: 'munich' },
                    { label: 'Etoile', value: 'etoile' },
                    { label: '自定义 OSM', value: 'custom_osm' },
                  ]}
                />
              </Form.Item>
            )}
            {source === 'sionna_rt' && channelConfig.scenario === 'custom_osm' && (
              <Form.Item
                label="OSM 文件路径"
                name="osm_path"
                rules={[{ required: true, message: '自定义场景需要指定 OSM 文件路径' }]}
              >
                <Input placeholder="例如：D:\maps\custom_area.osm" />
              </Form.Item>
            )}
          </Form>
        </Card>
      )}

      {/* ====== Step 4: 确认提交 ====== */}
      {currentStep === 3 && (
        <Row gutter={24}>
          <Col xs={24} lg={12}>
            <Card title={source === 'quadriga_real' ? '数据源配置' : '拓扑与设备配置'} size="small" style={{ marginBottom: 16 }}>
              <Descriptions column={1} size="small" bordered>
                <Descriptions.Item label="数据来源">
                  {sourceLabel(source)}
                </Descriptions.Item>
              </Descriptions>

              {source === 'quadriga_real' ? (
                <>
                  <Divider orientation="left" orientationMargin={0} style={{ margin: '12px 0' }}>
                    MATLAB 生成配置
                  </Divider>
                  <Descriptions column={2} size="small" bordered>
                    <Descriptions.Item label="3GPP 场景">{qrConfig.scenario}</Descriptions.Item>
                    <Descriptions.Item label="小区数">{qrConfig.num_cells}</Descriptions.Item>
                    <Descriptions.Item label="站间距">{qrConfig.isd_m} m</Descriptions.Item>
                    <Descriptions.Item label="基站高度">{qrConfig.tx_height_m} m</Descriptions.Item>
                    <Descriptions.Item label="载波频率">{freqLabel(qrConfig.carrier_freq_hz)}</Descriptions.Item>
                    <Descriptions.Item label="带宽">{bwLabel(qrConfig.bandwidth_hz)}</Descriptions.Item>
                    <Descriptions.Item label="子载波间隔">{Math.round(qrConfig.subcarrier_spacing / 1000)} kHz</Descriptions.Item>
                    <Descriptions.Item label="RB 数">{nrRbLookup(qrConfig.bandwidth_hz, qrConfig.subcarrier_spacing) ?? '—'}</Descriptions.Item>
                    <Descriptions.Item label="运动模式">{mobilityLabel(qrConfig.mobility_mode)}</Descriptions.Item>
                    <Descriptions.Item label="UE 速度">{qrConfig.ue_speed_kmh} km/h</Descriptions.Item>
                    <Descriptions.Item label="BS 天线">{qrConfig.bs_ant_v}V x {qrConfig.bs_ant_h}H = {qrConfig.bs_ant_v * qrConfig.bs_ant_h}</Descriptions.Item>
                    <Descriptions.Item label="UE 天线">{qrConfig.ue_ant_v}V x {qrConfig.ue_ant_h}H = {qrConfig.ue_ant_v * qrConfig.ue_ant_h}</Descriptions.Item>
                    <Descriptions.Item label="时间快照">{qrConfig.num_snapshots}</Descriptions.Item>
                    <Descriptions.Item label="用户数">{qrConfig.num_ues}</Descriptions.Item>
                    <Descriptions.Item label="用户分布">{distLabel(qrConfig.ue_distribution)}</Descriptions.Item>
                    <Descriptions.Item label="每分片 UE">{qrConfig.ues_per_shard}</Descriptions.Item>
                    <Descriptions.Item label="跳过生成">{qrConfig.skip_generation ? '是' : '否'}</Descriptions.Item>
                  </Descriptions>
                </>
              ) : (
                <>
                  <Divider orientation="left" orientationMargin={0} style={{ margin: '12px 0' }}>
                    拓扑配置
                  </Divider>
                  <Descriptions column={2} size="small" bordered>
                    <Descriptions.Item label="站点数">{topology.num_sites}</Descriptions.Item>
                    <Descriptions.Item label="站间距">{topology.isd_m} m</Descriptions.Item>
                    <Descriptions.Item label="扇区数/站">{topology.sectors_per_site}</Descriptions.Item>
                    <Descriptions.Item label="基站高度">{topology.tx_height_m} m</Descriptions.Item>
                  </Descriptions>

                  <Divider orientation="left" orientationMargin={0} style={{ margin: '12px 0' }}>
                    基站配置
                  </Divider>
                  <Descriptions column={2} size="small" bordered>
                    <Descriptions.Item label="天线配置">{bsConfig.num_bs_tx_ant}T{bsConfig.num_bs_rx_ant}R</Descriptions.Item>
                    <Descriptions.Item label="载波频率">
                      {freqLabel(bsConfig.carrier_freq_hz)}
                    </Descriptions.Item>
                    <Descriptions.Item label="带宽">
                      {bwLabel(bsConfig.bandwidth_hz)}
                    </Descriptions.Item>
                    <Descriptions.Item label="子载波间隔">
                      {bsConfig.subcarrier_spacing / 1000} kHz
                    </Descriptions.Item>
                    <Descriptions.Item label="RB 数">
                      {nrRbLookup(bsConfig.bandwidth_hz, bsConfig.subcarrier_spacing) ?? '非标准组合'}
                    </Descriptions.Item>
                    <Descriptions.Item label="发射功率">{bsConfig.tx_power_dbm} dBm</Descriptions.Item>
                  </Descriptions>

                  <Divider orientation="left" orientationMargin={0} style={{ margin: '12px 0' }}>
                    终端配置
                  </Divider>
                  <Descriptions column={2} size="small" bordered>
                    <Descriptions.Item label="用户数">{ueConfig.num_ues}</Descriptions.Item>
                    <Descriptions.Item label="天线配置">{ueConfig.num_ue_tx_ant}T{ueConfig.num_ue_rx_ant}R</Descriptions.Item>
                    <Descriptions.Item label="运动模式">
                      {mobilityLabel(ueConfig.mobility_mode)}
                    </Descriptions.Item>
                    <Descriptions.Item label="移动速度">
                      {ueConfig.ue_speed_kmh} km/h
                    </Descriptions.Item>
                    {ueConfig.mobility_mode !== 'static' && (
                      <Descriptions.Item label="采样间隔">
                        {ueConfig.sample_interval_s >= 0.001
                          ? `${ueConfig.sample_interval_s * 1000} ms`
                          : `${ueConfig.sample_interval_s * 1e6} μs`}
                      </Descriptions.Item>
                    )}
                    <Descriptions.Item label="用户分布">
                      {distLabel(ueConfig.ue_distribution)}
                    </Descriptions.Item>
                  </Descriptions>
                  {hasCustomPositions && (
                    <Paragraph type="warning" style={{ marginTop: 8, marginBottom: 0 }}>
                      已手动调整站点/终端位置，仿真将使用自定义坐标。
                    </Paragraph>
                  )}
                </>
              )}
            </Card>
          </Col>

          <Col xs={24} lg={12}>
            <Card title="信道配置" size="small" style={{ marginBottom: 16 }}>
              <Descriptions column={1} size="small" bordered>
                <Descriptions.Item label="链路方向">
                  {linkLabel(channelConfig.link)}
                </Descriptions.Item>
                <Descriptions.Item label="信道估计模式">
                  {estLabel(channelConfig.channel_est_mode)}
                </Descriptions.Item>
                {channelConfig.link === 'both' ? (
                  <>
                    <Descriptions.Item label="下行导频">
                      {pilotLabel(channelConfig.pilot_type_dl)}
                    </Descriptions.Item>
                    <Descriptions.Item label="上行导频">
                      {pilotLabel(channelConfig.pilot_type_ul)}
                    </Descriptions.Item>
                  </>
                ) : channelConfig.link === 'DL' ? (
                  <Descriptions.Item label="导频类型">
                    {pilotLabel(channelConfig.pilot_type_dl)}
                  </Descriptions.Item>
                ) : (
                  <Descriptions.Item label="导频类型">
                    {pilotLabel(channelConfig.pilot_type_ul)}
                  </Descriptions.Item>
                )}
                <Descriptions.Item label="信道模型">
                  {channelConfig.channel_model}
                </Descriptions.Item>
                <Descriptions.Item label="TDD 配比">
                  {channelConfig.tdd_pattern}
                </Descriptions.Item>
                <Descriptions.Item label="SSB 波束数">
                  {channelConfig.num_ssb_beams}
                </Descriptions.Item>
                {(channelConfig.srs_group_hopping || channelConfig.srs_sequence_hopping) && (
                  <Descriptions.Item label="SRS 跳频">
                    {[
                      channelConfig.srs_group_hopping && '组跳频',
                      channelConfig.srs_sequence_hopping && '序列跳频',
                    ].filter(Boolean).join(' + ')}
                  </Descriptions.Item>
                )}
                <Descriptions.Item label="SRS 周期">
                  {channelConfig.srs_periodicity} slots
                </Descriptions.Item>
                <Descriptions.Item label="SRS 频域跳频">
                  {channelConfig.srs_b_hop >= channelConfig.srs_b_srs ? '不跳频' : `b_hop=${channelConfig.srs_b_hop}, 跳频开启`}
                </Descriptions.Item>
                <Descriptions.Item label="SRS 带宽配置">
                  C_SRS={channelConfig.srs_c_srs}, B_SRS={channelConfig.srs_b_srs}, n_RRC={channelConfig.srs_n_rrc}
                </Descriptions.Item>
                <Descriptions.Item label="SRS 梳齿">
                  K_TC = {channelConfig.srs_comb}
                </Descriptions.Item>
                <Descriptions.Item label="邻区干扰 UE 上限">
                  {channelConfig.num_interfering_ues}
                </Descriptions.Item>
                <Descriptions.Item label="采样数量">
                  {channelConfig.num_samples}
                </Descriptions.Item>
                {channelConfig.link === 'both' && (
                  <Descriptions.Item label="配对模式">
                    UL+DL 配对（干扰随机化）
                  </Descriptions.Item>
                )}
                {source === 'sionna_rt' && channelConfig.scenario && (
                  <Descriptions.Item label="场景">
                    {scenarioLabel(channelConfig.scenario)}
                  </Descriptions.Item>
                )}
                {source === 'sionna_rt' && channelConfig.scenario === 'custom_osm' && channelConfig.osm_path && (
                  <Descriptions.Item label="OSM 文件">
                    {channelConfig.osm_path}
                  </Descriptions.Item>
                )}
              </Descriptions>
            </Card>

            <Card title="拓扑预览" size="small">
                <div style={{ height: 280 }}>
                  <TopologyCanvas
                    sites={previewData?.sites ?? []}
                    ues={previewData?.ues ?? []}
                    bounds={previewData?.bounds ?? { min_x: -500, max_x: 500, min_y: -500, max_y: 500 }}
                    cellRadius={previewData?.cell_radius_m ?? 250}
                    readOnly
                  />
                </div>
              </Card>
          </Col>
        </Row>
      )}

      {/* ====== Bottom navigation bar ====== */}
      <div
        style={{
          marginTop: 32,
          paddingTop: 16,
          borderTop: '1px solid #f0f0f0',
          display: 'flex',
          justifyContent: 'space-between',
        }}
      >
        <div>
          {currentStep > 0 && (
            <Button onClick={goPrev}>上一步</Button>
          )}
        </div>
        <div>
          {currentStep < 3 && (
            <Button type="primary" onClick={goNext}>
              下一步
            </Button>
          )}
          {currentStep === 3 && (
            <Button
              type="primary"
              loading={collect.isPending}
              onClick={handleSubmit}
            >
              提交采集任务
            </Button>
          )}
        </div>
      </div>
    </div>
  );
}
