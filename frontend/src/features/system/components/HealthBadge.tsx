import { Badge } from '@/components/ui/badge'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import type { ReadinessResponse } from '@/lib/schemas'
import { cn } from '@/lib/cn'

interface HealthBadgeProps {
  ready: boolean
  checks?: ReadinessResponse['checks']
  compact?: boolean
  className?: string
}

const checkLabels: Record<keyof ReadinessResponse['checks'], string> = {
  static_model: 'Static model',
  mediapipe: 'MediaPipe',
  dynamic_predictor: 'Dynamic predictor',
  prediction_service: 'Prediction service',
}

export function HealthBadge({ ready, checks, compact, className }: HealthBadgeProps) {
  const badge = (
    <Badge
      variant={ready ? 'success' : 'warning'}
      className={cn('cursor-default', className)}
    >
      {ready ? 'Backend ready' : 'Not ready'}
    </Badge>
  )

  if (compact || !checks) {
    return badge
  }

  return (
    <Tooltip>
      <TooltipTrigger asChild>{badge}</TooltipTrigger>
      <TooltipContent side="bottom" className="space-y-1 p-3">
        {Object.entries(checks).map(([key, value]) => (
          <div key={key} className="flex justify-between gap-4 text-xs">
            <span>{checkLabels[key as keyof ReadinessResponse['checks']]}</span>
            <span className={value ? 'text-accent' : 'text-destructive'}>
              {value ? '✓' : '✗'}
            </span>
          </div>
        ))}
      </TooltipContent>
    </Tooltip>
  )
}
