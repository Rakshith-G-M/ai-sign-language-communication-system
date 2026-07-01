import { memo } from 'react'
import { Clock, Gauge, Hand, Hash } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { useInferenceStore } from '@/features/inference/store/inferenceStore'
import { cn } from '@/lib/cn'

interface MetricsPanelProps {
  compact?: boolean
}

function MetricCell({
  icon: Icon,
  label,
  value,
  variant,
}: {
  icon: React.ComponentType<{ className?: string }>
  label: string
  value: string
  variant?: 'default' | 'success' | 'warning' | 'muted'
}) {
  return (
    <div className="flex flex-col gap-1 rounded-md bg-muted/40 p-3">
      <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
        <Icon className="h-3.5 w-3.5" aria-hidden />
        {label}
      </div>
      <span
        className={cn(
          'font-mono text-lg font-medium',
          variant === 'success' && 'text-accent',
          variant === 'warning' && 'text-warning',
          variant === 'muted' && 'text-muted-foreground',
        )}
      >
        {value}
      </span>
    </div>
  )
}

export const MetricsPanel = memo(function MetricsPanel({ compact }: MetricsPanelProps) {
  const latencyMs = useInferenceStore((s) => s.stats.processingTimeMs)
  const serverLatencyMs = useInferenceStore((s) => s.prediction.serverLatencyMs)
  const fps = useInferenceStore((s) => s.stats.clientFps)
  const handDetected = useInferenceStore((s) => s.prediction.handDetected)
  const frameCount = useInferenceStore((s) => s.stats.frameCount)
  const errorCount = useInferenceStore((s) => s.stats.errorCount)

  const latencyVariant = latencyMs > 200 ? 'warning' : 'default'
  const handVariant = handDetected ? 'success' : 'muted'

  if (compact) {
    return (
      <div className="grid grid-cols-3 gap-2 rounded-lg border border-border/60 bg-card p-3">
        <MetricCell icon={Clock} label="Latency" value={`${latencyMs}ms`} variant={latencyVariant} />
        <MetricCell icon={Gauge} label="FPS" value={String(fps)} />
        <MetricCell
          icon={Hand}
          label="Hand"
          value={handDetected ? 'Yes' : 'No'}
          variant={handVariant}
        />
      </div>
    )
  }

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-base">System Metrics</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 gap-3">
          <MetricCell icon={Clock} label="Client Latency" value={`${latencyMs}ms`} variant={latencyVariant} />
          <MetricCell icon={Clock} label="Server Latency" value={`${Math.round(serverLatencyMs)}ms`} />
          <MetricCell icon={Gauge} label="Client FPS" value={String(fps)} />
          <MetricCell icon={Hash} label="Frames" value={String(frameCount)} />
          <MetricCell
            icon={Hand}
            label="Hand Detected"
            value={handDetected ? 'Yes' : 'No'}
            variant={handVariant}
          />
          {errorCount > 0 && (
            <MetricCell icon={Hash} label="Errors" value={String(errorCount)} variant="warning" />
          )}
        </div>
      </CardContent>
    </Card>
  )
})
