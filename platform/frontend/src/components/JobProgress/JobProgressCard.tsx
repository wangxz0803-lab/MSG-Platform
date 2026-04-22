import { Card, Progress, Space, Statistic, Tag, Typography } from 'antd';
import type { Job } from '@/api/types';
import { statusToColor, statusToLabel } from '@/utils/jobStatus';
import { durationBetween, formatDateTime, formatDuration } from '@/utils/format';

const { Text } = Typography;

interface Props {
  job: Job;
}

export default function JobProgressCard({ job }: Props) {
  const pct = Math.max(0, Math.min(100, Math.round(job.progress_pct ?? 0)));
  const running = job.status === 'running';
  const statusColor = statusToColor(job.status);
  const progressStatus: 'success' | 'active' | 'exception' | 'normal' =
    job.status === 'completed'
      ? 'success'
      : job.status === 'failed' || job.status === 'cancelled'
        ? 'exception'
        : running
          ? 'active'
          : 'normal';

  return (
    <Card>
      <Space direction="vertical" style={{ width: '100%' }} size="middle">
        <Space align="center" size="middle" wrap>
          <Text strong style={{ fontSize: 16 }}>
            {job.display_name ?? job.job_id}
          </Text>
          <Tag color={statusColor}>{statusToLabel(job.status)}</Tag>
          <Tag>{job.type}</Tag>
        </Space>
        <Progress percent={pct} status={progressStatus} />
        <Space size="large" wrap>
          <Statistic title="当前步骤" value={job.current_step ?? '-'} />
          <Statistic
            title="预计剩余"
            value={job.eta_seconds != null ? formatDuration(job.eta_seconds) : '-'}
          />
          <Statistic title="开始时间" value={formatDateTime(job.started_at)} />
          <Statistic title="耗时" value={durationBetween(job.started_at, job.finished_at)} />
        </Space>
        {job.error && (
          <Text type="danger" style={{ whiteSpace: 'pre-wrap' }}>
            {job.error}
          </Text>
        )}
      </Space>
    </Card>
  );
}
