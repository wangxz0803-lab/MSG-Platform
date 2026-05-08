import { useEffect, useMemo, useState } from 'react';
import {
  Alert,
  Badge,
  Button,
  Card,
  Col,
  Descriptions,
  Divider,
  Form,
  Input,
  InputNumber,
  Modal,
  Popconfirm,
  Radio,
  Row,
  Select,
  Space,
  Statistic,
  Switch,
  Table,
  Tag,
  Typography,
  message,
} from 'antd';
import {
  DeleteOutlined,
  ExportOutlined,
  LockOutlined,
  UnlockOutlined,
} from '@ant-design/icons';
import type { ColumnsType } from 'antd/es/table';
import type { TableRowSelection } from 'antd/es/table/interface';
import { Link, useNavigate } from 'react-router-dom';
import { useDatasets, useDeleteDataset } from '@/api/queries';
import {
  computeAndLockSplit,
  exportDataset,
  getSplitStatus,
  listExports,
  unlockSplit,
} from '@/api/endpoints';
import type {
  DatasetExportRequest,
  DatasetFilters,
  DatasetSummary,
  ExportInfo,
  LinkType,
  SplitInfoResponse,
} from '@/api/types';
import { formatNumber } from '@/utils/format';
import DatasetDistributionCharts from '@/components/Charts/DatasetDistributionCharts';

const { Title } = Typography;

export default function Datasets() {
  const navigate = useNavigate();
  const [filters, setFilters] = useState<DatasetFilters>({ limit: 50, offset: 0 });
  const [selectedRowKeys, setSelectedRowKeys] = useState<string[]>([]);
  const { data, isLoading } = useDatasets(filters);
  const deleteMutation = useDeleteDataset();

  // --- Split management state ---
  const [splitInfo, setSplitInfo] = useState<SplitInfoResponse | null>(null);
  const [splitLoading, setSplitLoading] = useState(false);
  const [splitModalOpen, setSplitModalOpen] = useState(false);
  const [splitStrategy, setSplitStrategy] = useState<'random' | 'by_position' | 'by_beam'>('by_position');
  const [splitSeed, setSplitSeed] = useState(0);
  const [splitRatios, setSplitRatios] = useState<[number, number, number]>([0.8, 0.1, 0.1]);
  const [splitLock, setSplitLock] = useState(true);

  // --- Export state ---
  const [exportModalOpen, setExportModalOpen] = useState(false);
  const [exportFormat, setExportFormat] = useState<'hdf5' | 'webdataset' | 'pt_dir'>('hdf5');
  const [exportSplit, setExportSplit] = useState<string>('train');
  const [exportInclIntf, setExportInclIntf] = useState(false);
  const [exports, setExports] = useState<ExportInfo[]>([]);

  useEffect(() => {
    getSplitStatus().then(setSplitInfo).catch(() => {});
    listExports().then((r) => setExports(r.exports)).catch(() => {});
  }, []);

  const refreshSplitInfo = () => {
    getSplitStatus().then(setSplitInfo).catch(() => {});
  };

  const handleComputeSplit = async () => {
    setSplitLoading(true);
    try {
      const info = await computeAndLockSplit({
        strategy: splitStrategy,
        seed: splitSeed,
        ratios: splitRatios,
        lock: splitLock,
      });
      setSplitInfo(info);
      setSplitModalOpen(false);
      message.success(splitLock ? '数据集已划分并锁定' : '数据集已划分');
    } catch (e: any) {
      message.error(e?.response?.data?.detail || e.message);
    } finally {
      setSplitLoading(false);
    }
  };

  const handleUnlockSplit = async () => {
    try {
      await unlockSplit();
      refreshSplitInfo();
      message.success('测试集已解锁');
    } catch (e: any) {
      message.error(e?.response?.data?.detail || e.message);
    }
  };

  const handleExport = async () => {
    try {
      const req: DatasetExportRequest = {
        format: exportFormat,
        split: exportSplit || null,
        include_interferers: exportInclIntf,
      };
      const res = await exportDataset(req);
      setExportModalOpen(false);
      message.success(`导出任务已创建: ${res.job_id}`);
      navigate(`/jobs/${res.job_id}`);
    } catch (e: any) {
      message.error(e?.response?.data?.detail || e.message);
    }
  };

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
        title: 'UL SIR',
        dataIndex: 'ul_sir_mean',
        key: 'ul_sir_mean',
        align: 'right',
        render: (v: number | null) => (v != null ? formatNumber(v, 2) : '-'),
      },
      {
        title: 'DL SIR',
        dataIndex: 'dl_sir_mean',
        key: 'dl_sir_mean',
        align: 'right',
        render: (v: number | null) => (v != null ? formatNumber(v, 2) : '-'),
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
        render: (links: LinkType[], record) => (
          <Space wrap>
            {links?.map((l) => <Tag key={l}>{l}</Tag>)}
            {record.has_paired && <Tag color="blue">配对</Tag>}
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

        {/* --- Split management card --- */}
        <Card
          title="数据集划分与导出"
          extra={
            <Space>
              {splitInfo?.locked ? (
                <Badge status="success" text={`已锁定 (v${splitInfo.version})`} />
              ) : (
                <Badge status="warning" text="未锁定" />
              )}
            </Space>
          }
        >
          <Row gutter={[24, 16]}>
            <Col span={4}>
              <Statistic title="训练集" value={splitInfo?.counts?.train ?? '-'} />
            </Col>
            <Col span={4}>
              <Statistic title="验证集" value={splitInfo?.counts?.val ?? '-'} />
            </Col>
            <Col span={4}>
              <Statistic title="测试集" value={splitInfo?.counts?.test ?? '-'} />
            </Col>
            <Col span={4}>
              <Statistic title="未分配" value={splitInfo?.counts?.unassigned ?? '-'} />
            </Col>
            <Col span={8} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <Button
                type="primary"
                icon={<LockOutlined />}
                onClick={() => setSplitModalOpen(true)}
                disabled={splitInfo?.locked === true}
              >
                划分数据集
              </Button>
              {splitInfo?.locked && (
                <Popconfirm
                  title="确定要解锁测试集吗？"
                  description="解锁后可重新划分，但会丢失当前的锁定状态。"
                  onConfirm={handleUnlockSplit}
                  okText="解锁"
                  cancelText="取消"
                >
                  <Button icon={<UnlockOutlined />}>解锁</Button>
                </Popconfirm>
              )}
              <Button
                icon={<ExportOutlined />}
                onClick={() => setExportModalOpen(true)}
              >
                导出数据
              </Button>
            </Col>
          </Row>

          {splitInfo?.locked && (
            <>
              <Divider />
              <Descriptions size="small" column={4}>
                <Descriptions.Item label="策略">{splitInfo.strategy ?? '-'}</Descriptions.Item>
                <Descriptions.Item label="Seed">{splitInfo.seed ?? '-'}</Descriptions.Item>
                <Descriptions.Item label="比例">
                  {splitInfo.ratios ? splitInfo.ratios.join(' / ') : '-'}
                </Descriptions.Item>
                <Descriptions.Item label="锁定时间">
                  {splitInfo.locked_at ? new Date(splitInfo.locked_at).toLocaleString() : '-'}
                </Descriptions.Item>
              </Descriptions>
            </>
          )}

          {exports.length > 0 && (
            <>
              <Divider orientation="left">已导出包</Divider>
              <Table
                size="small"
                dataSource={exports}
                rowKey="name"
                pagination={false}
                columns={[
                  { title: '名称', dataIndex: 'name', key: 'name' },
                  { title: '格式', dataIndex: 'format', key: 'format', render: (v: string) => <Tag>{v}</Tag> },
                  { title: '样本数', dataIndex: 'num_samples', key: 'num_samples', align: 'right' as const },
                  { title: 'Split', dataIndex: 'split', key: 'split' },
                  { title: 'Split版本', dataIndex: 'split_version', key: 'split_version', align: 'right' as const },
                  {
                    title: '大小',
                    dataIndex: 'total_bytes',
                    key: 'total_bytes',
                    align: 'right' as const,
                    render: (v: number) => `${(v / 1024 / 1024).toFixed(1)} MB`,
                  },
                  {
                    title: '下载',
                    key: 'download',
                    render: (_v: unknown, r: ExportInfo) =>
                      r.download_url ? (
                        <a href={r.download_url} target="_blank" rel="noreferrer">
                          下载
                        </a>
                      ) : '-',
                  },
                ]}
              />
            </>
          )}
        </Card>

        {/* --- Split compute modal --- */}
        <Modal
          title="划分数据集"
          open={splitModalOpen}
          onOk={handleComputeSplit}
          onCancel={() => setSplitModalOpen(false)}
          confirmLoading={splitLoading}
          okText="划分"
        >
          <Space direction="vertical" style={{ width: '100%' }} size="middle">
            <Alert
              type="info"
              showIcon
              message="划分后测试集固定不变，后续新增数据自动归入训练集。"
            />
            <Form layout="vertical">
              <Form.Item label="划分策略">
                <Radio.Group value={splitStrategy} onChange={(e) => setSplitStrategy(e.target.value)}>
                  <Radio.Button value="by_position">按位置</Radio.Button>
                  <Radio.Button value="by_beam">按波束</Radio.Button>
                  <Radio.Button value="random">随机</Radio.Button>
                </Radio.Group>
              </Form.Item>
              <Form.Item label="随机种子">
                <InputNumber value={splitSeed} onChange={(v) => setSplitSeed(v ?? 0)} />
              </Form.Item>
              <Form.Item label="比例 (train / val / test)">
                <Space>
                  <InputNumber
                    min={0} max={1} step={0.05} value={splitRatios[0]}
                    onChange={(v) => setSplitRatios([v ?? 0.8, splitRatios[1], splitRatios[2]])}
                  />
                  <InputNumber
                    min={0} max={1} step={0.05} value={splitRatios[1]}
                    onChange={(v) => setSplitRatios([splitRatios[0], v ?? 0.1, splitRatios[2]])}
                  />
                  <InputNumber
                    min={0} max={1} step={0.05} value={splitRatios[2]}
                    onChange={(v) => setSplitRatios([splitRatios[0], splitRatios[1], v ?? 0.1])}
                  />
                </Space>
              </Form.Item>
              <Form.Item label="锁定测试集">
                <Switch checked={splitLock} onChange={setSplitLock} />
                <Typography.Text type="secondary" style={{ marginLeft: 8 }}>
                  锁定后新数据自动进训练集，测试集不变
                </Typography.Text>
              </Form.Item>
            </Form>
          </Space>
        </Modal>

        {/* --- Export modal --- */}
        <Modal
          title="导出数据"
          open={exportModalOpen}
          onOk={handleExport}
          onCancel={() => setExportModalOpen(false)}
          okText="开始导出"
        >
          <Form layout="vertical">
            <Form.Item label="导出格式">
              <Radio.Group value={exportFormat} onChange={(e) => setExportFormat(e.target.value)}>
                <Radio.Button value="hdf5">HDF5 (.h5)</Radio.Button>
                <Radio.Button value="webdataset">WebDataset (.tar)</Radio.Button>
                <Radio.Button value="pt_dir">原始 .pt 目录</Radio.Button>
              </Radio.Group>
            </Form.Item>
            <Form.Item label="导出 Split">
              <Select value={exportSplit} onChange={setExportSplit} style={{ width: 200 }}>
                <Select.Option value="train">训练集</Select.Option>
                <Select.Option value="val">验证集</Select.Option>
                <Select.Option value="test">测试集</Select.Option>
                <Select.Option value="">全部</Select.Option>
              </Select>
            </Form.Item>
            <Form.Item label="包含干扰信道">
              <Switch checked={exportInclIntf} onChange={setExportInclIntf} />
              <Typography.Text type="secondary" style={{ marginLeft: 8 }}>
                开启后导出包体积显著增大
              </Typography.Text>
            </Form.Item>
          </Form>
        </Modal>

        {data?.items && data.items.length > 0 && (
          <DatasetDistributionCharts datasets={data.items} />
        )}
      </Space>
    </div>
  );
}
