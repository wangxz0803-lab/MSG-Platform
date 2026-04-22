import { useQuery } from '@tanstack/react-query';
import { getJob } from '@/api/endpoints';
import type { Job } from '@/api/types';

export function useJobPolling(jobId: string | undefined, intervalMs = 2000) {
  return useQuery<Job>({
    queryKey: ['job-polling', jobId],
    queryFn: () => getJob(jobId as string),
    enabled: Boolean(jobId),
    refetchInterval: (query) => {
      const data = query.state.data;
      if (!data) return intervalMs;
      return data.status === 'running' || data.status === 'queued' ? intervalMs : false;
    },
  });
}
