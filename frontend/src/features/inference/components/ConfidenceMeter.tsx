import { cn } from '@/lib/cn'

interface ConfidenceMeterProps {
  value: number
  size?: 'sm' | 'md' | 'lg'
  showLabel?: boolean
}

export function ConfidenceMeter({ value, size = 'md', showLabel = false }: ConfidenceMeterProps) {
  const percent = Math.round(Math.min(1, Math.max(0, value)) * 100)
  const height = size === 'sm' ? 'h-1.5' : size === 'lg' ? 'h-3' : 'h-2'

  return (
    <div className="w-full" role="meter" aria-valuenow={percent} aria-valuemin={0} aria-valuemax={100}>
      {showLabel && (
        <div className="mb-1.5 flex justify-between text-xs text-muted-foreground">
          <span>Confidence</span>
          <span className="font-mono">{percent}%</span>
        </div>
      )}
      <div className={cn('w-full overflow-hidden rounded-full bg-muted', height)}>
        <div
          className={cn('h-full rounded-full bg-accent transition-[width] duration-200 ease-out', height)}
          style={{ width: `${percent}%` }}
        />
      </div>
    </div>
  )
}
