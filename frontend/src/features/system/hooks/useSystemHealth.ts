import { useQuery } from '@tanstack/react-query'
import { checkLiveness, checkReadiness, getMetrics } from '@/lib/api/health'

export function useSystemHealth(options?: { enableMetrics?: boolean }) {
  const liveness = useQuery({
    queryKey: ['health', 'liveness'],
    queryFn: checkLiveness,
    staleTime: 30_000,
    refetchInterval: 30_000,
  })

  const readiness = useQuery({
    queryKey: ['health', 'readiness'],
    queryFn: checkReadiness,
    staleTime: 5_000,
    refetchInterval: 10_000,
    retry: 3,
  })

  const metrics = useQuery({
    queryKey: ['health', 'metrics'],
    queryFn: getMetrics,
    enabled: options?.enableMetrics ?? false,
    staleTime: 30_000,
    refetchInterval: 30_000,
  })

  return { liveness, readiness, metrics }
}
