import { apiClient } from './client'
import {
  livenessResponseSchema,
  metricsResponseSchema,
  readinessResponseSchema,
  serviceInfoSchema,
  type MetricsResponse,
  type ReadinessResponse,
  type ServiceInfo,
} from '@/lib/schemas'

export async function checkLiveness(): Promise<boolean> {
  try {
    const data = await apiClient<unknown>('/api/v1/health')
    const parsed = livenessResponseSchema.parse(data)
    return parsed.status === 'ok'
  } catch {
    return false
  }
}

export async function checkReadiness(): Promise<ReadinessResponse> {
  console.log(import.meta.env.VITE_API_BASE)
  const base = import.meta.env.VITE_API_BASE ?? ''
  const url = new URL('/ready', base || window.location.origin)
  const response = await fetch(url.toString())
  const data = await response.json()
  return readinessResponseSchema.parse(data)
}

export async function getMetrics(): Promise<MetricsResponse> {
  const data = await apiClient<unknown>('/metrics')
  return metricsResponseSchema.parse(data)
}

export async function getServiceInfo(): Promise<ServiceInfo> {
  try {
    const data = await apiClient<unknown>('/api/v1/info')
    return serviceInfoSchema.parse(data)
  } catch {
    const root = await apiClient<unknown>('/')
    return serviceInfoSchema.parse(root)
  }
}

export function getDocsUrl(): string {
  const base = import.meta.env.VITE_API_BASE ?? ''
  if (base) {
    return `${base.replace(/\/$/, '')}/docs`
  }
  return '/docs'
}
