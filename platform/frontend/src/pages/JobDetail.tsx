import { Breadcrumb, Button, Card, Descriptions, Modal, Space, Typography, message } from 'antd';
import { Link, useParams } from 'react-router-dom';
import { useCancelJob, useJob, useJobLogs } from '@/api/queries';
import JobProgressCard from '@/components/JobProgress/JobProgressCard';
import JobLogViewer from '@/components/JobProgress/JobLogViewer';
import LoadingBox from '@/components/Common/LoadingBox';
import { formatDateTime } from '@/utils/format';
import { isActiveStatus } from '@/utils/jobStatus';

const { Title } = Typography;

export default function JobDetail() {
  const { jobId = '' } = useParams();
  const { data: job, isLoading } = useJob(jobId);
  const { data: logs } = useJobLogs(jobId, 500, Boolean(job));
  const cancel = useCancelJob();

  const handleCancel = () => {
    Modal.confirm({
      title: '确定要取消此任务?',
      content: '正在运行的任务将收到停止信号。',
      okText: '取消任务',
      okButtonProps: { danger: true },
      cancelText: '继续运行',
      onOk: async () => {
        try {
          await cancel.mutateAsync(jobId);
          message.success('已请求取消');
        } catch (e) {
          message.error((e as Error).message);
        }
      },
    });
  };

  if (isLoading || !job) {
    return (
      <div className="msg-page">
        <LoadingBox tip="加载任务中..." />
      </div>
    );
  }

  const canCancel = isActiveStatus(job.status);

  return (
    <div className="msg-page">
      <Breadcrumb
        className="msg-breadcrumb"
        items={[{ title: <Link to="/jobs">任务</Link> }, { title: job.display_name ?? job.job_id }]}
      />

      <Space direction="vertical" size="large" style={{ width: '100%' }}>
        <Space align="center" style={{ justifyContent: 'space-between', width: '100%' }}>
          <Title level={3} style={{ margin: 0 }}>
            任务详情
          </Title>
          <Space>
            {job.run_id && job.status === 'completed' && (
              <Link to={`/runs/${job.run_id}`}>
                <Button>查看生成的训练记录</Button>
              </Link>
            )}
            <Button danger disabled={!canCancel} loading={cancel.isPending} onClick={handleCancel}>
              取消任务
            </Button>
          </Space>
        </Space>

        <JobProgressCard job={job} />

        <Card title="元数据">
          <Descriptions column={2} size="small" bordered>
            <Descriptions.Item label="任务 ID">{job.job_id}</Descriptions.Item>
            <Descriptions.Item label="类型">{job.type}</Descriptions.Item>
            <Descriptions.Item label="创建时间">{formatDateTime(job.created_at)}</Descriptions.Item>
            <Descriptions.Item label="开始时间">{formatDateTime(job.started_at)}</Descriptions.Item>
            <Descriptions.Item label="结束时间">{formatDateTime(job.finished_at)}</Descriptions.Item>
            <Descriptions.Item label="记录 ID">
              {job.run_id ? <Link to={`/runs/${job.run_id}`}>{job.run_id}</Link> : '-'}
            </Descriptions.Item>
            <Descriptions.Item label="预计剩余">
              {job.eta_seconds != null ? `${job.eta_seconds}s` : '-'}
            </Descriptions.Item>
          </Descriptions>
        </Card>

        <JobLogViewer lines={logs?.lines ?? []} />
      </Space>
    </div>
  );
}
