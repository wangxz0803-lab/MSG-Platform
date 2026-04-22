import { useMemo, useState } from 'react';
import {
  Button,
  Card,
  Form,
  Popconfirm,
  Progress,
  Select,
  Space,
  Table,
  Tag,
  Tooltip,
  Typography,
  message,
} from 'antd';
import { DeleteOutlined } from '@ant-design/icons';
import type { ColumnsType } from 'antd/es/table';
import { Link, useNavigate } from 'react-router-dom';
import { useDeleteJob, useJobs } from '@/api/queries';
import type { Job, JobFilters, JobStatus, JobType } from '@/api/types';
import { durationBetween, formatDateTime } from '@/utils/format';
import { isActiveStatus, statusToColor, statusToLabel } from '@/utils/jobStatus';

const { Title, Text } = Typography;

const JOB_STATUSES: JobStatus[] = ['queued', 'running', 'completed', 'failed', 'cancelled'];
const JOB_TYPES: JobType[] = [
  'convert',
  'bridge',
  'train',
  'eval',
  'infer',
  'export',
  'report',
  'simulate',
];

export default function Jobs() {
  const navigate = useNavigate();
  const [filters, setFilters] = useState<JobFilters>({ limit: 20, offset: 0 });
  const { data, isLoading } = useJobs(filters);
  const deleteMutation = useDeleteJob();

  const handleDelete = async (jobId: string) => {
    try {
      await deleteMutation.mutateAsync(jobId);
      message.success('任务已删除');
    } catch (e) {
      message.error((e as Error).message);
    }
  };

  const columns: ColumnsType<Job> = useMemo(
    () => [
      {
        title: '任务',
        dataIndex: 'job_id',
        key: 'job_id',
        ellipsis: true,
        render: (id: string, r) => <Link to={`/jobs/${id}`}>{r.display_name ?? id}</Link>,
      },
      { title: '类型', dataIndex: 'type', key: 'type' },
      {
        title: '状态',
        dataIndex: 'status',
        key: 'status',
        render: (s: JobStatus) => <Tag color={statusToColor(s)}>{statusToLabel(s)}</Tag>,
      },
      {
        title: 'Run',
        dataIndex: 'run_id',
        key: 'run_id',
        render: (rid: string | null) =>
          rid ? (
            <Link to={`/runs/${rid}`}>{rid.slice(0, 12)}...</Link>
          ) : (
            <Text type="secondary">-</Text>
          ),
      },
      {
        title: '进度',
        key: 'progress',
        render: (_v, r) =>
          r.status === 'running' || r.status === 'queued' ? (
            <Progress
              percent={Math.round(r.progress_pct ?? 0)}
              size="small"
              status={r.status === 'running' ? 'active' : 'normal'}
            />
          ) : (
            <Progress
              percent={Math.round(r.progress_pct ?? 0)}
              size="small"
              status={r.status === 'failed' ? 'exception' : 'success'}
              showInfo={false}
            />
          ),
      },
      {
        title: '创建时间',
        dataIndex: 'created_at',
        key: 'created_at',
        render: (v: string) => formatDateTime(v),
      },
      {
        title: '耗时',
        key: 'duration',
        render: (_v, r) => durationBetween(r.started_at, r.finished_at),
      },
      {
        title: '操作',
        key: 'action',
        render: (_, record) => {
          const active = isActiveStatus(record.status);
          if (active) {
            return (
              <Tooltip title="运行中的任务无法删除">
                <Button size="small" danger disabled icon={<DeleteOutlined />}>
                  删除
                </Button>
              </Tooltip>
            );
          }
          return (
            <Popconfirm
              title={`确定要删除任务 ${record.display_name ?? record.job_id} 吗？`}
              description="此操作不可恢复。"
              onConfirm={() => handleDelete(record.job_id)}
              okText="删除"
              cancelText="取消"
              okButtonProps={{ danger: true }}
            >
              <Button size="small" danger icon={<DeleteOutlined />}>
                删除
              </Button>
            </Popconfirm>
          );
        },
      },
    ],
    [deleteMutation],
  );

  return (
    <div className="msg-page">
      <Space direction="vertical" size="large" style={{ width: '100%' }}>
        <Space align="center" style={{ justifyContent: 'space-between', width: '100%' }}>
          <Title level={3} style={{ margin: 0 }}>
            任务
          </Title>
          <Button type="primary" onClick={() => navigate('/jobs/new')}>
            新建任务
          </Button>
        </Space>

        <Card>
          <Form
            layout="inline"
            initialValues={filters}
            onValuesChange={(_cv, av) => setFilters((f) => ({ ...f, ...av, offset: 0 }))}
          >
            <Form.Item label="状态" name="status">
              <Select
                style={{ width: 140 }}
                allowClear
                options={JOB_STATUSES.map((s) => ({ value: s }))}
              />
            </Form.Item>
            <Form.Item label="类型" name="type">
              <Select
                style={{ width: 140 }}
                allowClear
                options={JOB_TYPES.map((t) => ({ value: t }))}
              />
            </Form.Item>
          </Form>
        </Card>

        <Card>
          <Table<Job>
            columns={columns}
            dataSource={data?.items ?? []}
            rowKey="job_id"
            loading={isLoading}
            pagination={{
              total: data?.total ?? 0,
              pageSize: filters.limit,
              current: Math.floor((filters.offset ?? 0) / (filters.limit ?? 20)) + 1,
              onChange: (page, pageSize) =>
                setFilters((f) => ({ ...f, limit: pageSize, offset: (page - 1) * pageSize })),
            }}
          />
        </Card>
      </Space>
    </div>
  );
}
