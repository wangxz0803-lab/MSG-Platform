import { useCallback, useMemo, useState } from 'react';
import {
  Alert,
  Breadcrumb,
  Button,
  Card,
  Checkbox,
  Col,
  Collapse,
  Form,
  Input,
  InputNumber,
  Row,
  Select,
  Space,
  Steps,
  Table,
  Tag,
  Typography,
  message,
} from 'antd';
import type { ColumnsType } from 'antd/es/table';
import {
  ArrowRightOutlined,
  ToolOutlined,
  ThunderboltOutlined,
} from '@ant-design/icons';
import { Link, useNavigate } from 'react-router-dom';
import { useDatasets, useCreateJob } from '@/api/queries';

const { Title, Text, Paragraph } = Typography;

// ---------------------------------------------------------------------------
// Token layout — matches FeatureExtractor's 16 tokens + 8 gates exactly
// ---------------------------------------------------------------------------

interface TokenRow {
  key: string;
  index: string;
  field: string;
  shape: string;
  dtype: string;
  source: string;
  description: string;
}

const TOKEN_ROWS: TokenRow[] = [
  { key: 'pdp', index: '0', field: 'pdp_crop', shape: '[1, 64]', dtype: 'float32', source: 'IFFT(H)', description: '功率延迟谱 (PDP)，归一化到 [0, 1]' },
  { key: 'srs1', index: '1', field: 'srs1', shape: '[1, 64]', dtype: 'complex64', source: 'SVD(H_avg)', description: '左奇异向量 U[:,0] — 第1全带鲁棒权值' },
  { key: 'srs2', index: '2', field: 'srs2', shape: '[1, 64]', dtype: 'complex64', source: 'SVD(H_avg)', description: '左奇异向量 U[:,1] — 第2全带鲁棒权值' },
  { key: 'srs3', index: '3', field: 'srs3', shape: '[1, 64]', dtype: 'complex64', source: 'SVD(H_avg)', description: '左奇异向量 U[:,2] — 第3全带鲁棒权值' },
  { key: 'srs4', index: '4', field: 'srs4', shape: '[1, 64]', dtype: 'complex64', source: 'SVD(H_avg)', description: '左奇异向量 U[:,3] — 第4全带鲁棒权值' },
  { key: 'pmi1', index: '5', field: 'pmi1', shape: '[1, 64]', dtype: 'complex64', source: 'SVD Vh / 码本', description: '右奇异向量 Vh[0,:] — 第1 PMI 权值' },
  { key: 'pmi2', index: '6', field: 'pmi2', shape: '[1, 64]', dtype: 'complex64', source: 'SVD Vh / 码本', description: '右奇异向量 Vh[1,:] — 第2 PMI 权值' },
  { key: 'pmi3', index: '7', field: 'pmi3', shape: '[1, 64]', dtype: 'complex64', source: 'SVD Vh / 码本', description: '右奇异向量 Vh[2,:] — 第3 PMI 权值' },
  { key: 'pmi4', index: '8', field: 'pmi4', shape: '[1, 64]', dtype: 'complex64', source: 'SVD Vh / 码本', description: '右奇异向量 Vh[3,:] — 第4 PMI 权值' },
  { key: 'dft1', index: '9', field: 'dft1', shape: '[1, 64]', dtype: 'complex64', source: 'DFT(H_avg)', description: 'Top-1 能量 DFT 波束' },
  { key: 'dft2', index: '10', field: 'dft2', shape: '[1, 64]', dtype: 'complex64', source: 'DFT(H_avg)', description: 'Top-2 能量 DFT 波束' },
  { key: 'dft3', index: '11', field: 'dft3', shape: '[1, 64]', dtype: 'complex64', source: 'DFT(H_avg)', description: 'Top-3 能量 DFT 波束' },
  { key: 'dft4', index: '12', field: 'dft4', shape: '[1, 64]', dtype: 'complex64', source: 'DFT(H_avg)', description: 'Top-4 能量 DFT 波束' },
  { key: 'rsrp_srs', index: '13', field: 'rsrp_srs', shape: '[1, 64]', dtype: 'float32', source: '天线功率', description: '每天线 RSRP (dBm)' },
  { key: 'rsrp_cb', index: '14', field: 'rsrp_cb', shape: '[1, 64]', dtype: 'float32', source: '波束功率', description: '波束域 RSRP (dBm)' },
  { key: 'cell_rsrp', index: '15', field: 'cell_rsrp', shape: '[1, 16]', dtype: 'float32', source: 'SSB 测量', description: '小区 RSRP (服务+邻区，降序排列)' },
];

const TOKEN_COLUMNS: ColumnsType<TokenRow> = [
  {
    title: '#',
    dataIndex: 'index',
    key: 'index',
    width: 50,
    align: 'center',
    render: (v: string) => <Tag color="blue">{v}</Tag>,
  },
  { title: '字段名', dataIndex: 'field', key: 'field', width: 100, render: (v: string) => <Text code>{v}</Text> },
  { title: '维度', dataIndex: 'shape', key: 'shape', width: 80, render: (v: string) => <Text type="secondary">{v}</Text> },
  { title: '类型', dataIndex: 'dtype', key: 'dtype', width: 90, render: (v: string) => <Tag>{v}</Tag> },
  { title: '来源', dataIndex: 'source', key: 'source', width: 120 },
  { title: '说明', dataIndex: 'description', key: 'description' },
];

// Gate inputs
interface GateRow {
  key: string;
  index: string;
  field: string;
  dtype: string;
  description: string;
}

const GATE_ROWS: GateRow[] = [
  { key: 'srs_w1', index: '1', field: 'srs_w1', dtype: 'float32', description: 'SVD 奇异值归一化权重 sigma_1 (4个 sum=1.0)' },
  { key: 'srs_w2', index: '2', field: 'srs_w2', dtype: 'float32', description: 'SVD 奇异值归一化权重 sigma_2' },
  { key: 'srs_w3', index: '3', field: 'srs_w3', dtype: 'float32', description: 'SVD 奇异值归一化权重 sigma_3' },
  { key: 'srs_w4', index: '4', field: 'srs_w4', dtype: 'float32', description: 'SVD 奇异值归一化权重 sigma_4' },
  { key: 'srs_sinr', index: '5', field: 'srs_sinr', dtype: 'float32', description: 'SRS SINR (dB，裁剪到 [-20, 20])' },
  { key: 'srs_cb_sinr', index: '6', field: 'srs_cb_sinr', dtype: 'float32', description: '码本 SINR (dB)' },
  { key: 'cqi', index: '7', field: 'cqi', dtype: 'int64', description: '信道质量指标 CQI (0-15, Shannon SE 映射)' },
];

const GATE_COLUMNS: ColumnsType<GateRow> = [
  {
    title: '#',
    dataIndex: 'index',
    key: 'index',
    width: 50,
    align: 'center',
    render: (v: string) => <Tag color="orange">{v}</Tag>,
  },
  { title: '字段名', dataIndex: 'field', key: 'field', width: 120, render: (v: string) => <Text code>{v}</Text> },
  { title: '类型', dataIndex: 'dtype', key: 'dtype', width: 90, render: (v: string) => <Tag>{v}</Tag> },
  { title: '说明', dataIndex: 'description', key: 'description' },
];

// Pipeline steps
const PIPELINE_STEPS = [
  { title: 'ChannelSample', description: '(H_est, H_true, meta)' },
  { title: 'Bridge 特征提取', description: '(SVD, PMI, DFT, PDP)' },
  { title: '16 Tokens + 8 Gates', description: '(固定 23 字段)' },
  { title: '模型输入', description: '(bridge_out/*.pt)' },
];

// Processing logic
const PROCESSING_NOTES = [
  {
    key: '1',
    label: '1. 时频平均',
    children: <Paragraph style={{ marginBottom: 0 }}>H_avg = mean(H_est, axis=(T, RB)) → [BS, UE]，作为后续 SVD/DFT 的输入</Paragraph>,
  },
  {
    key: '2',
    label: '2. SVD 分解 → SRS tokens (1-4) + SRS weights',
    children: <Paragraph style={{ marginBottom: 0 }}>U, sigma, Vh = SVD(H_avg)。左奇异向量 U 的前 4 列作为 SRS token，归一化奇异值 sigma/sum(sigma) 作为 gate 权重 (srs_w1~w4, sum=1)</Paragraph>,
  },
  {
    key: '3',
    label: '3. PMI 提取 → PMI tokens (5-8)',
    children: <Paragraph style={{ marginBottom: 0 }}>右奇异向量 Vh 的前 4 行作为预编码权值 (PMI)。走 SVD fallback 或 legacy CsiChanProcFunc 码本路径</Paragraph>,
  },
  {
    key: '4',
    label: '4. DFT 波束 → DFT tokens (9-12)',
    children: <Paragraph style={{ marginBottom: 0 }}>标准 DFT 矩阵 F = fft(eye(N))/sqrt(N)，计算 beam_response = F @ H_avg，按能量排序取 Top-4 波束行</Paragraph>,
  },
  {
    key: '5',
    label: '5. PDP → Token 0',
    children: <Paragraph style={{ marginBottom: 0 }}>time 维平均 → first UE ant → IFFT(RB) → |·|² → BS avg → normalize → crop 64 taps</Paragraph>,
  },
  {
    key: '6',
    label: '6. RSRP → Tokens 13-14',
    children: <Paragraph style={{ marginBottom: 0 }}>rsrp_srs: 10*log10(mean(|H|², axis=(T,RB,UE))) + offset → per-antenna dBm。rsrp_cb: 波束域功率 → dBm</Paragraph>,
  },
  {
    key: '7',
    label: '7. Cell RSRP → Token 15',
    children: <Paragraph style={{ marginBottom: 0 }}>SSB RSRP 降序排列，填充 16 个小区槽位，默认 -160 dBm</Paragraph>,
  },
  {
    key: '8',
    label: '8. CQI 映射 → Gate',
    children: <Paragraph style={{ marginBottom: 0 }}>SINR → Shannon SE = log2(1 + 10^(SINR/10)) → CQI = round(SE * 15 / 7.4)，裁剪到 [0, 15]</Paragraph>,
  },
];

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function DataProcess() {
  const navigate = useNavigate();
  const createJob = useCreateJob();
  const { data: datasetsData, isLoading: datasetsLoading } = useDatasets();

  const [selectedSources, setSelectedSources] = useState<string[]>([]);
  const [outputDir, setOutputDir] = useState('bridge_out');
  const [numWorkers, setNumWorkers] = useState(4);
  const [skipProcessed, setSkipProcessed] = useState(true);
  const [useLegacyPmi, setUseLegacyPmi] = useState(false);

  const datasetOptions = useMemo(() => {
    if (!datasetsData?.items) return [];
    return datasetsData.items.map((ds) => ({
      label: `${ds.source} (${ds.count} 样本)`,
      value: ds.source,
    }));
  }, [datasetsData]);

  const handleSubmit = useCallback(async () => {
    if (selectedSources.length === 0) {
      message.warning('请选择至少一个源数据集');
      return;
    }

    try {
      const res = await createJob.mutateAsync({
        type: 'bridge',
        display_name: `数据处理 - ${selectedSources.join(', ')}`,
        config_overrides: {
          sources: selectedSources,
          output_dir: outputDir,
          num_workers: numWorkers,
          skip_processed: skipProcessed,
          use_legacy_pmi: useLegacyPmi,
        },
      });
      message.success(`处理任务 ${res.job_id} 已提交`);
      navigate(`/jobs/${res.job_id}`);
    } catch (e) {
      message.error((e as Error).message ?? '提交失败');
    }
  }, [selectedSources, outputDir, numWorkers, skipProcessed, useLegacyPmi, createJob, navigate]);

  return (
    <div className="msg-page">
      <Breadcrumb
        className="msg-breadcrumb"
        items={[
          { title: <Link to="/">首页</Link> },
          { title: '数据处理' },
        ]}
      />

      <Title level={3} style={{ marginBottom: 4 }}>数据处理</Title>
      <Paragraph type="secondary" style={{ marginBottom: 24 }}>
        将原始采集的信道数据 (ChannelSample) 经过 Bridge 特征提取，转换为模型可直接消费的 16 Tokens + 8 Gates 输入张量。
      </Paragraph>

      {/* Pipeline Visualization */}
      <Card title="处理流水线" size="small" style={{ marginBottom: 24 }}>
        <Steps
          current={-1}
          items={PIPELINE_STEPS.map((step) => ({
            title: step.title,
            description: <Text type="secondary" style={{ fontSize: 12 }}>{step.description}</Text>,
          }))}
          style={{ padding: '8px 0' }}
        />
        <div style={{ textAlign: 'center', marginTop: 8 }}>
          <Space>
            <Tag color="green">输入：ChannelSample (H_est + meta)</Tag>
            <ArrowRightOutlined />
            <Tag color="blue">输出：16 Tokens + 8 Gates (.pt)</Tag>
          </Space>
        </div>
      </Card>

      {/* Token Layout */}
      <Card title="Token 布局 (16 个)" size="small" style={{ marginBottom: 24 }}>
        <Table<TokenRow>
          columns={TOKEN_COLUMNS}
          dataSource={TOKEN_ROWS}
          pagination={false}
          size="small"
          bordered
          rowKey="key"
        />
      </Card>

      {/* Gate Layout */}
      <Card title="Gate 值 (8 个门控输入)" size="small" style={{ marginBottom: 24 }}>
        <Table<GateRow>
          columns={GATE_COLUMNS}
          dataSource={GATE_ROWS}
          pagination={false}
          size="small"
          bordered
          rowKey="key"
        />
        <div style={{ marginTop: 8 }}>
          <Text type="secondary">
            Gate 机制：quality_gate (SINR→sigmoid→[0.7,1.0])、energy_gates (srs_w→sigmoid→[0.7,1.0])、pmi_gate (CQI→sigmoid→[0.7,1.0])
          </Text>
        </div>
      </Card>

      {/* Processing Config */}
      <Card
        title={<Space><ToolOutlined /><span>处理配置</span></Space>}
        style={{ marginBottom: 24 }}
      >
        <Form layout="vertical" style={{ maxWidth: 720 }}>
          <Form.Item label="源数据集" required>
            <Select
              mode="multiple"
              placeholder="请选择要处理的数据集"
              value={selectedSources}
              onChange={setSelectedSources}
              options={datasetOptions}
              loading={datasetsLoading}
              notFoundContent={datasetsLoading ? '加载中...' : '暂无可用数据集'}
              style={{ width: '100%' }}
            />
            {selectedSources.length === 0 && (
              <Text type="secondary" style={{ fontSize: 12 }}>
                请先在「数据采集」页面生成原始信道数据
              </Text>
            )}
          </Form.Item>

          <Row gutter={16}>
            <Col xs={24} sm={12}>
              <Form.Item label="输出目录名">
                <Input
                  value={outputDir}
                  onChange={(e) => setOutputDir(e.target.value)}
                  placeholder="bridge_out"
                  addonBefore="data/"
                />
              </Form.Item>
            </Col>
            <Col xs={24} sm={12}>
              <Form.Item label="并行 Worker 数">
                <InputNumber
                  value={numWorkers}
                  onChange={(v) => setNumWorkers(v ?? 4)}
                  min={1}
                  max={32}
                  style={{ width: '100%' }}
                />
              </Form.Item>
            </Col>
          </Row>

          <Form.Item>
            <Space direction="vertical">
              <Checkbox
                checked={skipProcessed}
                onChange={(e) => setSkipProcessed(e.target.checked)}
              >
                跳过已处理样本（增量处理模式）
              </Checkbox>
              <Checkbox
                checked={useLegacyPmi}
                onChange={(e) => setUseLegacyPmi(e.target.checked)}
              >
                使用 Legacy PMI 码本路径 (CsiChanProcFunc)，否则使用 SVD Vh 回退
              </Checkbox>
            </Space>
          </Form.Item>

          <Alert
            type="info"
            showIcon
            style={{ marginBottom: 16 }}
            message={
              <span>
                将对{' '}
                <Text strong>{selectedSources.length > 0 ? selectedSources.join('、') : '(未选择)'}</Text>
                {' '}数据集执行 Bridge 特征提取，生成{' '}
                <Text strong>16 Tokens + 8 Gates</Text> 固定格式张量，输出到{' '}
                <Text code>data/{outputDir}</Text>，使用{' '}
                <Text strong>{numWorkers}</Text> 个并行 Worker。
              </span>
            }
          />

          <Form.Item>
            <Button
              type="primary"
              size="large"
              icon={<ThunderboltOutlined />}
              loading={createJob.isPending}
              onClick={handleSubmit}
              disabled={selectedSources.length === 0}
            >
              开始处理
            </Button>
          </Form.Item>
        </Form>
      </Card>

      {/* Processing Logic */}
      <Card size="small" style={{ marginBottom: 24 }}>
        <Collapse
          items={[{
            key: 'notes',
            label: '处理逻辑说明（Bridge 算法细节）',
            children: <Collapse size="small" items={PROCESSING_NOTES} ghost />,
          }]}
        />
      </Card>
    </div>
  );
}
