import type { JobStatus } from '@/api/types';

export type BadgeStatus = 'default' | 'processing' | 'success' | 'error' | 'warning';

const STATUS_TO_BADGE: Record<JobStatus, BadgeStatus> = {
  queued: 'default',
  running: 'processing',
  completed: 'success',
  failed: 'error',
  cancelled: 'warning',
};

const STATUS_TO_COLOR: Record<JobStatus, string> = {
  queued: '#8c8c8c',
  running: '#1677ff',
  completed: '#52c41a',
  failed: '#ff4d4f',
  cancelled: '#faad14',
};

const STATUS_TO_LABEL: Record<JobStatus, string> = {
  queued: '排队中',
  running: '运行中',
  completed: '已完成',
  failed: '失败',
  cancelled: '已取消',
};

export function statusToBadge(status: JobStatus): BadgeStatus {
  return STATUS_TO_BADGE[status] ?? 'default';
}

export function statusToColor(status: JobStatus): string {
  return STATUS_TO_COLOR[status] ?? '#8c8c8c';
}

export function statusToLabel(status: JobStatus): string {
  return STATUS_TO_LABEL[status] ?? status;
}

export function isActiveStatus(status: JobStatus): boolean {
  return status === 'queued' || status === 'running';
}
