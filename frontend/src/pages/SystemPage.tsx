import { motion } from 'framer-motion'
import { ExternalLink, RefreshCw } from 'lucide-react'
import { useQuery } from '@tanstack/react-query'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Separator } from '@/components/ui/separator'
import { HealthBadge } from '@/features/system/components/HealthBadge'
import { ModelBadge } from '@/features/system/components/ModelBadge'
import { useSystemHealth } from '@/features/system/hooks/useSystemHealth'
import { getDocsUrl, getServiceInfo } from '@/lib/api/health'
import { APP_NAME, APP_TAGLINE } from '@/lib/constants'
import { fadeIn } from '@/lib/motion'

function formatUptime(seconds: number): string {
  const h = Math.floor(seconds / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  const s = Math.floor(seconds % 60)
  if (h > 0) return `${h}h ${m}m ${s}s`
  if (m > 0) return `${m}m ${s}s`
  return `${s}s`
}

export function SystemPage() {
  const { liveness, readiness, metrics } = useSystemHealth({ enableMetrics: true })

  const serviceInfo = useQuery({
    queryKey: ['service', 'info'],
    queryFn: getServiceInfo,
    staleTime: 60_000,
  })

  const checks = readiness.data?.checks
  const ready = readiness.data?.status === 'ready'

  return (
    <motion.div {...fadeIn} className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">System Information</h1>
        <p className="text-sm text-muted-foreground">
          Backend status, model availability, and project details
        </p>
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Backend Health</CardTitle>
            <CardDescription>Live status from the inference server</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex flex-wrap items-center gap-3">
              <HealthBadge ready={!!ready} checks={checks} />
              <ModelBadge
                staticReady={checks?.static_model ?? false}
                dynamicReady={checks?.dynamic_predictor ?? false}
              />
            </div>

            <Separator />

            <dl className="space-y-2 text-sm">
              <div className="flex justify-between">
                <dt className="text-muted-foreground">Liveness</dt>
                <dd>{liveness.data ? 'OK' : 'Unreachable'}</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-muted-foreground">Readiness</dt>
                <dd>{ready ? 'Ready' : 'Not ready'}</dd>
              </div>
              {metrics.data && (
                <>
                  <div className="flex justify-between">
                    <dt className="text-muted-foreground">Uptime</dt>
                    <dd className="font-mono">{formatUptime(metrics.data.uptime_seconds)}</dd>
                  </div>
                  <div className="flex justify-between">
                    <dt className="text-muted-foreground">Total predictions</dt>
                    <dd className="font-mono">{metrics.data.total_predictions}</dd>
                  </div>
                  <div className="flex justify-between">
                    <dt className="text-muted-foreground">Active sessions</dt>
                    <dd className="font-mono">{metrics.data.active_sessions}</dd>
                  </div>
                </>
              )}
            </dl>

            <Button
              variant="outline"
              size="sm"
              onClick={() => {
                void readiness.refetch()
                void metrics.refetch()
              }}
            >
              <RefreshCw className="h-4 w-4" />
              Refresh
            </Button>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">Service Details</CardTitle>
            <CardDescription>API metadata and documentation</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {serviceInfo.data && (
              <dl className="space-y-2 text-sm">
                <div className="flex justify-between gap-4">
                  <dt className="text-muted-foreground">Service</dt>
                  <dd>{serviceInfo.data.service}</dd>
                </div>
                <div className="flex justify-between gap-4">
                  <dt className="text-muted-foreground">Version</dt>
                  <dd className="font-mono">{serviceInfo.data.version}</dd>
                </div>
                {serviceInfo.data.api_base && (
                  <div className="flex justify-between gap-4">
                    <dt className="text-muted-foreground">API base</dt>
                    <dd className="font-mono">{serviceInfo.data.api_base}</dd>
                  </div>
                )}
              </dl>
            )}

            <Button variant="outline" size="sm" asChild>
              <a href={getDocsUrl()} target="_blank" rel="noopener noreferrer">
                <ExternalLink className="h-4 w-4" />
                API Documentation
              </a>
            </Button>
          </CardContent>
        </Card>

        <Card className="md:col-span-2">
          <CardHeader>
            <CardTitle className="text-base">About {APP_NAME}</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3 text-sm text-muted-foreground">
            <p>{APP_TAGLINE}</p>
            <p>
              This platform uses MediaPipe hand tracking with XGBoost for static ASL finger-spelling
              (A–Z) and optional dynamic gesture recognition for whole-word signs. Inference runs
              server-side with session-scoped text assembly.
            </p>
            {checks && (
              <>
                <Separator />
                <h3 className="font-medium text-foreground">Model Components</h3>
                <ul className="grid gap-2 sm:grid-cols-2">
                  {Object.entries(checks).map(([key, value]) => (
                    <li key={key} className="flex items-center justify-between rounded-md bg-muted/40 px-3 py-2">
                      <span className="capitalize">{key.replace(/_/g, ' ')}</span>
                      <span className={value ? 'text-accent' : 'text-destructive'}>
                        {value ? 'Loaded' : 'Unavailable'}
                      </span>
                    </li>
                  ))}
                </ul>
              </>
            )}
          </CardContent>
        </Card>
      </div>
    </motion.div>
  )
}
