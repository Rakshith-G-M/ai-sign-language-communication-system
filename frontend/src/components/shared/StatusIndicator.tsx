import { Activity } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { cn } from '@/lib/cn'

type Status = 'idle' | 'live' | 'processing' | 'error'

interface StatusIndicatorProps {
  status: Status
  className?: string
  compact?: boolean
}

const labels: Record<Status, string> = {
  idle: 'Idle',
  live: 'Live',
  processing: 'Processing',
  error: 'Error',
}

export function StatusIndicator({ status, className, compact }: StatusIndicatorProps) {
  const variant =
    status === 'live' || status === 'processing'
      ? 'default'
      : status === 'error'
        ? 'destructive'
        : 'secondary'

  return (
    <Badge variant={variant} className={cn('gap-1.5', className)}>
      <span className="relative flex h-2 w-2">
        {status === 'live' && (
          <span className="absolute inline-flex h-full w-full animate-pulseSoft rounded-full bg-primary opacity-75" />
        )}
        <span
          className={cn(
            'relative inline-flex h-2 w-2 rounded-full',
            status === 'live' && 'bg-primary',
            status === 'processing' && 'bg-primary/70',
            status === 'idle' && 'bg-muted-foreground',
            status === 'error' && 'bg-destructive',
          )}
        />
      </span>
      {!compact && (
        <>
          <Activity className="h-3 w-3" aria-hidden />
          {labels[status]}
        </>
      )}
    </Badge>
  )
}
