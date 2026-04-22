import { useMemo, useState } from 'react';
import {
  Button,
  Card,
  Form,
  Input,
  InputNumber,
  Popconfirm,
  Select,
  Space,
  Table,
  Tag,
  Typography,
  message,
} from 'antd';
import { DeleteOutlined } from '@ant-design/icons';
import type { ColumnsType } from 'antd/es/table';
import type { TableRowSelection } from 'antd/es/table/interface';
import { Link, useNavigate } from 'react-router-dom';
import { useDatasets, useDeleteDataset } from '@/api/queries';
import type { DatasetFilters, DatasetSummary, LinkType } from '@/api/types';
import { formatNumber } from '@/utils/format';
import DatasetDistributionCharts from '@/components/Charts/DatasetDistributionCharts';

const { Title } = Typography;

export default function Datasets() {
  const navigate = useNavigate();
  const [filters, setFilters] = useState<DatasetFilters>({ limit: 50, offset: 0 });
  const [selectedRowKeys, setSelectedRowKeys] = useState<string[]>([]);
  const { data, isLoading } = useDatasets(filters);
  const deleteMutation = useDeleteDataset();

  const handleDelete = async (source: string) => {
    try {
      await deleteMutation.mutateAsync(source);
      message.success(`已删除数据源 ${source}`);
      setSelectedRowKeys((keys) => keys.filter((k) => k !== source));
    } catch (e) {
      message.error((e as Error).message);
    }
  };

  const handleBatchDelete = () => {
    if (selectedRowKeys.length === 0) return;
    const count = selectedRowKeys.length;
    Promise.all(selectedRowKeys.map((source) => deleteMutation.mutateAsync(source)))
      .then(() => {
        message.success(`已删除 ${count} 个数据源`);
        setSelectedRowKeys([]);
      })
      .catch((e) => {
        message.error((e as Error).message);
      });
  };

  const rowSelection: TableRowSelection<DatasetSummary> = {
    selectedRowKeys,
    onChange: (keys) => setSelectedRowKeys(keys.map(String)),
  };

  const columns: ColumnsType<DatasetSummary> = useMemo(
    () => [
      {
        title: '来源',
        dataIndex: 'source',
        key: 'source',
        render: (v: string) => <Link to={`/datasets/${encodeURIComponent(v)}`}>{v}</Link>,
      },
      { title: '样本数', dataIndex: 'count', key: 'count', align: 'right' },
      {
        title: 'SNR mean +/- std',
        key: 'snr',
        align: 'right',
        render: (_v, r) => `${formatNumber(r.snr_mean, 2)} +/- ${formatNumber(r.snr_std, 2)}`,
      },
      {
        title: 'SIR mean',
        dataIndex: 'sir_mean',
        key: 'sir_mean',
        align: 'right',
        render: (v: number | null) => formatNumber(v, 2),
      },
      {
        title: 'SINR mean',
        dataIndex: 'sinr_mean',
        key: 'sinr_mean',
        align: 'right',
        render: (v: number | null) => formatNumber(v, 2),
      },
      {
        title: '链路',
        dataIndex: 'links',
        key: 'links',
        render: (links: LinkType[]) => (
          <Space wrap>
            {links?.map((l) => <Tag key={l}>{l}</Tag>)}
          </Space>
        ),
      },
      {
        title: '操作',
        key: 'action',
        render: (_, record) => (
          <Popconfirm
            title={`确定要删除数据源 "${record.source}" 吗？`}
            description={`将删除该来源的全部 ${record.count} 条数据，此操作不可恢复。`}
            onConfirm={() => handleDelete(record.source)}
            okText="删除"
            cancelText="取消"
            okButtonProps={{ danger: true }}
          >
            <Button size="small" danger icon={<DeleteOutlined />}>
              删除
            </Button>
          </Popconfirm>
        ),
      },
    ],
    [deleteMutation],
  );

  return (
    <div className="msg-page">
      <Space direction="vertical" size="large" style={{ width: '100%' }}>
        <Space align="center" style={{ justifyContent: 'space-between', width: '100%' }}>
          <Title level={3} style={{ margin: 0 }}>
            数据集
          </Title>
          <Space>
            <Popconfirm
              title={`确定要批量删除 ${selectedRowKeys.length} 个数据源吗？`}
              description="将删除选中来源的全部数据，此操作不可恢复。"
              onConfirm={handleBatchDelete}
              okText="删除"
              cancelText="取消"
              okButtonProps={{ danger: true }}
              disabled={selectedRowKeys.length === 0}
            >
              <Button danger disabled={selectedRowKeys.length === 0}>
                批量删除 ({selectedRowKeys.length})
              </Button>
            </Popconfirm>
            <Button type="primary" onClick={() => navigate('/collect')}>
              触发采集
            </Button>
          </Space>
        </Space>

        <Card>
          <Form
            layout="inline"
            onValuesChange={(_cv, av) =>
              setFilters((f) => ({ ...f, ...av, offset: 0 }))
            }
            initialValues={filters}
          >
            <Form.Item label="Source" name="source">
              <Input placeholder="substring filter" allowClear />
            </Form.Item>
            <Form.Item label="Link" name="link">
              <Select style={{ width: 120 }} allowClear options={[{ value: 'UL' }, { value: 'DL' }]} />
            </Form.Item>
            <Form.Item label="Min SNR" name="min_snr">
              <InputNumber style={{ width: 120 }} />
            </Form.Item>
            <Form.Item label="Max SNR" name="max_snr">
              <InputNumber style={{ width: 120 }} />
            </Form.Item>
            <Form.Item label="Limit" name="limit">
              <InputNumber min={1} max={1000} style={{ width: 100 }} />
            </Form.Item>
          </Form>
        </Card>

        <Card>
          <Table<DatasetSummary>
            rowSelection={rowSelection}
            columns={columns}
            dataSource={data?.items ?? []}
            rowKey="source"
            loading={isLoading}
            pagination={{
              total: data?.total ?? 0,
              pageSize: filters.limit,
              current: Math.floor((filters.offset ?? 0) / (filters.limit ?? 50)) + 1,
              onChange: (page, pageSize) =>
                setFilters((f) => ({ ...f, limit: pageSize, offset: (page - 1) * pageSize })),
            }}
          />
        </Card>

        {data?.items && data.items.length > 0 && (
          <DatasetDistributionCharts datasets={data.items} />
        )}
      </Space>
    </div>
  );
}
