import { Card, Col, Row, Space, Statistic, Table, Tag, Typography } from 'antd';
import type { ColumnsType } from 'antd/es/table';
import { Link } from 'react-router-dom';
import { useDatasets, useHealth, useJobs, useRuns } from '@/api/queries';
import type { Job, Run } from '@/api/types';
import { formatDateTime, formatNumber, shortSha } from '@/utils/format';
import { statusToColor, statusToLabel } from '@/utils/jobStatus';

const { Title } = Typography;

export default function Dashboard() {
  const health = useHealth();
  const jobs = useJobs({ limit: 5 });
  const runs = useRuns({ limit: 5 });
  const datasets = useDatasets({ limit: 1000 });

  const activeJobs =
    jobs.data?.items.filter((j) => j.status === 'running' || j.status === 'queued').length ?? 0;
  const totalRuns = runs.data?.total ?? 0;
  const totalSamples = datasets.data?.items.reduce((sum, d) => sum + (d.count ?? 0), 0) ?? 0;

  const jobCols: ColumnsType<Job> = [
    {
      title: '任务',
      dataIndex: 'job_id',
      key: 'job_id',
      render: (id: string, r) => <Link to={`/jobs/${id}`}>{r.display_name ?? id}</Link>,
      ellipsis: true,
    },
    { title: '类型', dataIndex: 'type', key: 'type' },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (s: Job['status']) => <Tag color={statusToColor(s)}>{statusToLabel(s)}</Tag>,
    },
    {
      title: '创建时间',
      dataIndex: 'created_at',
      key: 'created_at',
      render: (v: string) => formatDateTime(v),
    },
  ];

  const runCols: ColumnsType<Run> = [
    {
      title: '训练记录',
      dataIndex: 'run_id',
      key: 'run_id',
      render: (id: string) => <Link to={`/runs/${id}`}>{shortSha(id)}</Link>,
    },
    {
      title: '创建时间',
      dataIndex: 'created_at',
      key: 'created_at',
      render: (v: string) => formatDateTime(v),
    },
    {
      title: 'CT',
      key: 'ct',
      render: (_v, r) => formatNumber(r.metrics?.ct, 4),
    },
    {
      title: 'NMSE (dB)',
      key: 'nmse',
      render: (_v, r) => formatNumber(r.metrics?.nmse_dB, 2),
    },
  ];

  const healthColor =
    health.isError || !health.data
      ? 'red'
      : health.data.status === 'ok'
        ? 'green'
        : health.data.status === 'degraded'
          ? 'orange'
          : 'red';

  return (
    <div className="msg-page">
      <Space direction="vertical" size="large" style={{ width: '100%' }}>
        <Space align="center" size="middle">
          <Title level={3} style={{ margin: 0 }}>
            仪表盘
          </Title>
          <Tag color={healthColor}>API {health.data?.status ?? '...'}</Tag>
          {health.data?.version && <Tag>v{health.data.version}</Tag>}
        </Space>

        <Row gutter={16} className="msg-card-row">
          <Col span={6}>
            <Card>
              <Statistic title="进行中的任务" value={activeJobs} />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic title="训练记录" value={totalRuns} />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic title="样本总数" value={totalSamples} />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="数据来源"
                value={datasets.data?.total ?? datasets.data?.items.length ?? 0}
              />
            </Card>
          </Col>
        </Row>

        <Row gutter={16}>
          <Col span={12}>
            <Card title="最近任务" extra={<Link to="/jobs">查看全部</Link>}>
              <Table<Job>
                columns={jobCols}
                dataSource={jobs.data?.items ?? []}
                rowKey="job_id"
                loading={jobs.isLoading}
                pagination={false}
                size="small"
              />
            </Card>
          </Col>
          <Col span={12}>
            <Card title="最近训练" extra={<Link to="/runs">查看全部</Link>}>
              <Table<Run>
                columns={runCols}
                dataSource={runs.data?.items ?? []}
                rowKey="run_id"
                loading={runs.isLoading}
                pagination={false}
                size="small"
              />
            </Card>
          </Col>
        </Row>
      </Space>
    </div>
  );
}
