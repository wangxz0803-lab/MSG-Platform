import { useState } from 'react';
import {
  Alert,
  Breadcrumb,
  Button,
  Card,
  Descriptions,
  Space,
  Table,
  Tabs,
  Tag,
  Typography,
  message,
} from 'antd';
import type { ColumnsType } from 'antd/es/table';
import { Link, useParams } from 'react-router-dom';
import { useExportModel, useModels, useRun } from '@/api/queries';
import type { ModelArtifact } from '@/api/types';
import { formatBytes, formatDateTime, shortSha } from '@/utils/format';
import LoadingBox from '@/components/Common/LoadingBox';
import MetricsTable from '@/components/Metrics/MetricsTable';
import LossChart from '@/components/Plots/LossChart';
import UMAPPlot, { type UMAPPoint } from '@/components/Plots/UMAPPlot';
import EmptyState from '@/components/Common/EmptyState';

const { Title, Text } = Typography;

function toYaml(obj: unknown, indent = 0): string {
  // Minimal YAML-ish pretty-printer for display (not round-trip).
  const pad = (n: number) => '  '.repeat(n);
  if (obj === null || obj === undefined) return 'null';
  if (typeof obj !== 'object') return JSON.stringify(obj);
  if (Array.isArray(obj)) {
    if (obj.length === 0) return '[]';
    return obj.map((v) => `${pad(indent)}- ${toYaml(v, indent + 1).trimStart()}`).join('\n');
  }
  const entries = Object.entries(obj as Record<string, unknown>);
  if (entries.length === 0) return '{}';
  return entries
    .map(([k, v]) => {
      if (v !== null && typeof v === 'object') {
        return `${pad(indent)}${k}:\n${toYaml(v, indent + 1)}`;
      }
      return `${pad(indent)}${k}: ${JSON.stringify(v)}`;
    })
    .join('\n');
}

export default function RunDetail() {
  const { runId = '' } = useParams();
  const { data: run, isLoading, error } = useRun(runId);
  const models = useModels({ run_id: runId });
  const exportModel = useExportModel();
  const [exportingFmt, setExportingFmt] = useState<'onnx' | 'torchscript' | null>(null);

  const doExport = async (format: 'onnx' | 'torchscript') => {
    setExportingFmt(format);
    try {
      const res = await exportModel.mutateAsync({ runId, req: { format } });
      message.success(`导出任务 ${res.job_id} 已排队`);
    } catch (e) {
      message.error((e as Error).message);
    } finally {
      setExportingFmt(null);
    }
  };

  if (isLoading) {
    return (
      <div className="msg-page">
        <LoadingBox tip="加载训练记录中..." />
      </div>
    );
  }
  if (error || !run) {
    return (
      <div className="msg-page">
        <Alert type="error" message="加载训练记录失败" description={String(error)} />
      </div>
    );
  }

  const umapPoints: UMAPPoint[] =
    (run as unknown as { umap_points?: UMAPPoint[] }).umap_points ?? [];

  const modelCols: ColumnsType<ModelArtifact> = [
    { title: '格式', dataIndex: 'format', key: 'format' },
    { title: '路径', dataIndex: 'path', key: 'path', ellipsis: true },
    {
      title: '大小',
      dataIndex: 'size_bytes',
      key: 'size_bytes',
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
  ];

  return (
    <div className="msg-page">
      <Breadcrumb
        className="msg-breadcrumb"
        items={[{ title: <Link to="/runs">训练记录</Link> }, { title: shortSha(run.run_id) }]}
      />

      <Space direction="vertical" size="large" style={{ width: '100%' }}>
        <Space align="center" size="middle" wrap>
          <Title level={3} style={{ margin: 0 }}>
            训练记录 {shortSha(run.run_id)}
          </Title>
          {run.tags?.map((t) => <Tag key={t}>{t}</Tag>)}
        </Space>

        <Card>
          <Descriptions column={3} size="small" bordered>
            <Descriptions.Item label="记录 ID">{run.run_id}</Descriptions.Item>
            <Descriptions.Item label="Git SHA">{shortSha(run.git_sha)}</Descriptions.Item>
            <Descriptions.Item label="创建时间">{formatDateTime(run.created_at)}</Descriptions.Item>
            <Descriptions.Item label="最佳检查点">{run.ckpt_best ?? run.ckpt_path ?? '-'}</Descriptions.Item>
            <Descriptions.Item label="最新检查点">{run.ckpt_last ?? '-'}</Descriptions.Item>
          </Descriptions>
        </Card>

        <Tabs
          defaultActiveKey="overview"
          items={[
            {
              key: 'overview',
              label: '概览',
              children: (
                <Space direction="vertical" style={{ width: '100%' }} size="large">
                  <Card title="关键指标">
                    <MetricsTable metrics={run.metrics ?? {}} />
                  </Card>
                </Space>
              ),
            },
            {
              key: 'metrics',
              label: '指标',
              children: (
                <Card>
                  <MetricsTable metrics={run.metrics ?? {}} />
                </Card>
              ),
            },
            {
              key: 'curves',
              label: '训练曲线',
              children: (
                <Card>
                  {run.tb_scalars && run.tb_scalars.length > 0 ? (
                    <LossChart series={run.tb_scalars} />
                  ) : (
                    <EmptyState description="当前记录中无标量序列数据。" />
                  )}
                </Card>
              ),
            },
            {
              key: 'latent',
              label: '隐空间',
              children: (
                <Card>
                  {umapPoints.length > 0 ? (
                    <UMAPPlot points={umapPoints} />
                  ) : (
                    <EmptyState description="暂无 UMAP 嵌入点。请先运行 eval 或 infer 任务来生成。" />
                  )}
                </Card>
              ),
            },
            {
              key: 'models',
              label: '模型',
              children: (
                <Card
                  title="产物"
                  extra={
                    <Space>
                      <Button
                        loading={exportingFmt === 'onnx'}
                        onClick={() => doExport('onnx')}
                      >
                        导出 ONNX
                      </Button>
                      <Button
                        loading={exportingFmt === 'torchscript'}
                        onClick={() => doExport('torchscript')}
                      >
                        导出 TorchScript
                      </Button>
                    </Space>
                  }
                >
                  <Table<ModelArtifact>
                    columns={modelCols}
                    dataSource={models.data?.items ?? run.artifacts ?? []}
                    rowKey="id"
                    loading={models.isLoading}
                    pagination={false}
                    size="small"
                  />
                </Card>
              ),
            },
            {
              key: 'raw',
              label: '原始配置',
              children: (
                <Card>
                  <Text type="secondary">此训练记录的最终 Hydra 配置。</Text>
                  <pre className="msg-yaml-block" style={{ marginTop: 12 }}>
                    {toYaml(run.config ?? {})}
                  </pre>
                </Card>
              ),
            },
          ]}
        />
      </Space>
    </div>
  );
}
