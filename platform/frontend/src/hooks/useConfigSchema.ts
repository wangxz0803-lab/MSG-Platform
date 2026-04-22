import { useQuery } from '@tanstack/react-query';
import { getConfigSchema } from '@/api/endpoints';

export function useConfigSchema() {
  return useQuery({
    queryKey: ['config-schema'],
    queryFn: getConfigSchema,
    staleTime: Infinity,
    gcTime: Infinity,
  });
}
