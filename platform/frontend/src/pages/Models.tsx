import { useMemo, useState } from 'react';
import { Card, Form, Input, Select, Space, Table, Typography } from 'antd';
import type { ColumnsType } from 'antd/es/table';
import { Link } from 'react-router-dom';
import { useModels } from '@/api/queries';
import type { ModelArtifact, ModelFilters, ModelFormat } from '@/api/types';
import { formatBytes, formatDateTime, shortSha } from '@/utils/format';

const { Title } = Typography;

const FORMATS: ModelFormat[] = ['pt', 'onnx', 'torchscript'];

export default function Models() {
  const [filters, setFilters] = useState<ModelFilters>({});
  const { data, isLoading } = useModels(filters);

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
    ],
    [],
  );

  return (
    <div className="msg-page">
      <Space direction="vertical" size="large" style={{ width: '100%' }}>
        <Title level={3} style={{ margin: 0 }}>
          模型仓库
        </Title>

        <Card>
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
      </Space>
    </div>
  );
}
