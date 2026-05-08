import { useEffect, useMemo, useState } from 'react';
import {
  Button,
  Card,
  Descriptions,
  Form,
  Input,
  InputNumber,
  Modal,
  Select,
  Space,
  Table,
  Tabs,
  Tag,
  Typography,
  Upload,
  message,
} from 'antd';
import {
  InboxOutlined,
  PlayCircleOutlined,
  RocketOutlined,
  TrophyOutlined,
  UploadOutlined,
} from '@ant-design/icons';
import type { ColumnsType } from 'antd/es/table';
import { Link, useNavigate } from 'react-router-dom';
import { useModels } from '@/api/queries';
import {
  evaluateModel,
  getLeaderboard,
  inferModel,
  uploadModel,
} from '@/api/endpoints';
import type {
  LeaderboardEntry,
  ModelArtifact,
  ModelFilters,
  ModelFormat,
  ModelUploadResponse,
} from '@/api/types';
import { formatBytes, formatDateTime, shortSha } from '@/utils/format';

const { Title, Text } = Typography;
const { Dragger } = Upload;

const FORMATS: ModelFormat[] = ['pt', 'onnx', 'torchscript'];

export default function Models() {
  const navigate = useNavigate();
  const [filters, setFilters] = useState<ModelFilters>({});
  const { data, isLoading, refetch } = useModels(filters);

  // --- Upload state ---
  const [uploadModalOpen, setUploadModalOpen] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [uploadRunId, setUploadRunId] = useState('');
  const [uploadTags, setUploadTags] = useState('');
  const [uploadResult, setUploadResult] = useState<ModelUploadResponse | null>(null);

  // --- Inference state ---
  const [inferModalOpen, setInferModalOpen] = useState(false);
  const [inferRunId, setInferRunId] = useState('');
  const [inferSplit, setInferSplit] = useState('test');
  const [inferBatchSize, setInferBatchSize] = useState(32);

  // --- Leaderboard state ---
  const [leaderboard, setLeaderboard] = useState<LeaderboardEntry[]>([]);
  const [lbLoading, setLbLoading] = useState(false);

  const loadLeaderboard = () => {
    setLbLoading(true);
    getLeaderboard()
      .then((r) => setLeaderboard(r.entries))
      .catch(() => {})
      .finally(() => setLbLoading(false));
  };

  useEffect(() => { loadLeaderboard(); }, []);

  const handleInfer = async () => {
    try {
      const res = await inferModel(inferRunId, {
        split: inferSplit,
        batch_size: inferBatchSize,
      });
      message.success('推理任务已创建');
      setInferModalOpen(false);
      navigate(`/jobs/${res.job_id}`);
    } catch (e: any) {
      message.error(e?.response?.data?.detail || e.message);
    }
  };

  const handleUpload = async () => {
    if (!uploadFile) {
      message.warning('请选择模型文件');
      return;
    }
    setUploading(true);
    try {
      const res = await uploadModel(uploadFile, uploadRunId || undefined, uploadTags || undefined);
      setUploadResult(res);
      message.success(`模型上传成功: ${res.run_id}`);
      refetch();
    } catch (e: any) {
      message.error(e?.response?.data?.detail || e.message);
    } finally {
      setUploading(false);
    }
  };

  const handleEvaluate = async (runId: string) => {
    try {
      const res = await evaluateModel(runId, { test_split: 'test' });
      message.success(`评估任务已创建`);
      navigate(`/jobs/${res.job_id}`);
    } catch (e: any) {
      message.error(e?.response?.data?.detail || e.message);
    }
  };

  const columns: ColumnsType<ModelArtifact> = useMemo(
    () => [
      {
        title: '训练记录',
        dataIndex: 'run_id',
        key: 'run_id',
        render: (id: string) => <Link to={`/runs/${id}`}>{shortSha(id)}</Link>,
      },
      { title: '格式', dataIndex: 'format', key: 'format' },
      { title: '路径', dataIndex: 'path', key: 'path', ellipsis: true },
      {
        title: '大小',
        dataIndex: 'size_bytes',
        key: 'size_bytes',
        align: 'right',
        render: (v: number) => formatBytes(v),
      },
      {
        title: '创建时间',
        dataIndex: 'created_at',
        key: 'created_at',
        render: (v: string) => formatDateTime(v),
      },
      {
        title: '下载',
        key: 'download',
        render: (_v, r) =>
          r.download_url ? (
            <a href={r.download_url} target="_blank" rel="noreferrer">
              下载
            </a>
          ) : (
            '-'
          ),
      },
      {
        title: '操作',
        key: 'actions',
        render: (_v, r) =>
          r.format === 'pt' ? (
            <Space size="small">
              <Button
                size="small"
                icon={<PlayCircleOutlined />}
                onClick={() => handleEvaluate(r.run_id)}
              >
                评估
              </Button>
              <Button
                size="small"
                icon={<RocketOutlined />}
                onClick={() => { setInferRunId(r.run_id); setInferModalOpen(true); }}
              >
                推理
              </Button>
            </Space>
          ) : null,
      },
    ],
    [],
  );

  const lbColumns: ColumnsType<LeaderboardEntry> = [
    {
      title: '排名', key: 'rank', width: 60,
      render: (_v, _r, i) => <Tag color={i === 0 ? 'gold' : i === 1 ? 'silver' : i === 2 ? '#cd7f32' : undefined}>{i + 1}</Tag>,
    },
    {
      title: '运行 ID', dataIndex: 'run_id', key: 'run_id',
      render: (id: string) => <Link to={`/runs/${id}`}>{shortSha(id)}</Link>,
    },
    { title: '标签', dataIndex: 'tags', key: 'tags', render: (v: string | null) => v || '-' },
    {
      title: '兼容性', dataIndex: 'compatible', key: 'compatible',
      render: (v: boolean) => <Tag color={v ? 'green' : 'orange'}>{v ? '兼容' : '未确认'}</Tag>,
    },
    {
      title: 'KNN Acc', key: 'knn_acc',
      render: (_v, r) => r.metrics.knn_acc != null ? `${(r.metrics.knn_acc * 100).toFixed(2)}%` : '-',
      sorter: (a, b) => (a.metrics.knn_acc ?? 0) - (b.metrics.knn_acc ?? 0),
    },
    {
      title: 'NMSE (dB)', key: 'nmse_dB',
      render: (_v, r) => r.metrics.nmse_dB != null ? r.metrics.nmse_dB.toFixed(2) : '-',
      sorter: (a, b) => (a.metrics.nmse_dB ?? 0) - (b.metrics.nmse_dB ?? 0),
    },
    {
      title: 'CT', key: 'ct',
      render: (_v, r) => r.metrics.ct != null ? r.metrics.ct.toFixed(4) : '-',
    },
    {
      title: 'TW', key: 'tw',
      render: (_v, r) => r.metrics.tw != null ? r.metrics.tw.toFixed(4) : '-',
    },
    {
      title: '评估时间', dataIndex: 'evaluated_at', key: 'evaluated_at',
      render: (v: string | null) => v ? formatDateTime(v) : '-',
    },
  ];

  return (
    <div className="msg-page">
      <Space direction="vertical" size="large" style={{ width: '100%' }}>
        <Space style={{ justifyContent: 'space-between', width: '100%' }}>
          <Title level={3} style={{ margin: 0 }}>
            模型仓库
          </Title>
          <Button type="primary" icon={<UploadOutlined />} onClick={() => setUploadModalOpen(true)}>
            上传模型
          </Button>
        </Space>

        <Tabs defaultActiveKey="artifacts" items={[
          {
            key: 'artifacts',
            label: '模型列表',
            children: (
              <>
                <Card style={{ marginBottom: 16 }}>
                  <Form
                    layout="inline"
                    initialValues={filters}
                    onValuesChange={(_cv, av) => setFilters((f) => ({ ...f, ...av }))}
                  >
                    <Form.Item label="记录 ID" name="run_id">
                      <Input style={{ width: 240 }} allowClear placeholder="按记录 ID 筛选" />
                    </Form.Item>
                    <Form.Item label="格式" name="format">
                      <Select style={{ width: 160 }} allowClear options={FORMATS.map((f) => ({ value: f }))} />
                    </Form.Item>
                  </Form>
                </Card>
                <Card>
                  <Table<ModelArtifact>
                    columns={columns}
                    dataSource={data?.items ?? []}
                    rowKey="id"
                    loading={isLoading}
                    pagination={{ pageSize: 20 }}
                  />
                </Card>
              </>
            ),
          },
          {
            key: 'leaderboard',
            label: <><TrophyOutlined /> 排行榜</>,
            children: (
              <Card>
                <Table<LeaderboardEntry>
                  columns={lbColumns}
                  dataSource={leaderboard}
                  rowKey="run_id"
                  loading={lbLoading}
                  pagination={{ pageSize: 20 }}
                />
              </Card>
            ),
          },
        ]} />

        {/* --- Upload modal --- */}
        <Modal
          title="上传外部训练模型"
          open={uploadModalOpen}
          onCancel={() => {
            setUploadModalOpen(false);
            setUploadResult(null);
            setUploadFile(null);
          }}
          footer={
            uploadResult
              ? [
                  <Button key="close" onClick={() => { setUploadModalOpen(false); setUploadResult(null); setUploadFile(null); }}>
                    关闭
                  </Button>,
                  <Button key="eval" type="primary" icon={<PlayCircleOutlined />}
                    onClick={() => { handleEvaluate(uploadResult.run_id); setUploadModalOpen(false); }}>
                    立即评估
                  </Button>,
                ]
              : undefined
          }
          onOk={handleUpload}
          confirmLoading={uploading}
          okText="上传"
          width={560}
        >
          {!uploadResult ? (
            <Space direction="vertical" style={{ width: '100%' }} size="middle">
              <Dragger
                accept=".pt,.pth,.ckpt"
                maxCount={1}
                beforeUpload={(file) => { setUploadFile(file); return false; }}
                onRemove={() => setUploadFile(null)}
                fileList={uploadFile ? [uploadFile as any] : []}
              >
                <p className="ant-upload-drag-icon"><InboxOutlined /></p>
                <p className="ant-upload-text">点击或拖拽上传模型 Checkpoint</p>
                <p className="ant-upload-hint">支持 .pt / .pth / .ckpt 格式</p>
              </Dragger>
              <Form layout="vertical">
                <Form.Item label="Run ID (可选，留空自动生成)">
                  <Input value={uploadRunId} onChange={(e) => setUploadRunId(e.target.value)}
                    placeholder="upload-20260501-123456-abc123" />
                </Form.Item>
                <Form.Item label="标签 (逗号分隔)">
                  <Input value={uploadTags} onChange={(e) => setUploadTags(e.target.value)}
                    placeholder="uploaded, v2, mae-vicreg" />
                </Form.Item>
              </Form>
            </Space>
          ) : (
            <Space direction="vertical" style={{ width: '100%' }} size="middle">
              <Tag color={uploadResult.compatible ? 'green' : 'orange'}>
                {uploadResult.compatible ? '兼容 ChannelMAE' : '兼容性未确认'}
              </Tag>
              <Text type="secondary">{uploadResult.compatibility_detail}</Text>
              <table style={{ width: '100%' }}>
                <tbody>
                  <tr><td>Run ID</td><td><Text code>{uploadResult.run_id}</Text></td></tr>
                  <tr><td>路径</td><td><Text ellipsis>{uploadResult.path}</Text></td></tr>
                  <tr><td>大小</td><td>{formatBytes(uploadResult.size_bytes)}</td></tr>
                </tbody>
              </table>
            </Space>
          )}
        </Modal>
        {/* --- Inference modal --- */}
        <Modal
          title="模型推理"
          open={inferModalOpen}
          onOk={handleInfer}
          onCancel={() => setInferModalOpen(false)}
          okText="提交推理"
          width={480}
        >
          <Space direction="vertical" style={{ width: '100%' }} size="middle">
            <Descriptions column={1} size="small">
              <Descriptions.Item label="模型">{shortSha(inferRunId)}</Descriptions.Item>
            </Descriptions>
            <Form layout="vertical">
              <Form.Item label="数据划分">
                <Select value={inferSplit} onChange={setInferSplit} style={{ width: '100%' }}>
                  <Select.Option value="test">测试集</Select.Option>
                  <Select.Option value="val">验证集</Select.Option>
                  <Select.Option value="train">训练集</Select.Option>
                </Select>
              </Form.Item>
              <Form.Item label="Batch Size">
                <InputNumber value={inferBatchSize} onChange={(v) => setInferBatchSize(v ?? 32)} min={1} max={512} style={{ width: '100%' }} />
              </Form.Item>
            </Form>
          </Space>
        </Modal>
      </Space>
    </div>
  );
}
