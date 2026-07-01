import { memo } from 'react'
import { Loader2, Play, RotateCcw, Square } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { cn } from '@/lib/cn'

interface InferenceControlsProps {
  isRunning: boolean
  isResetting: boolean
  disabled?: boolean
  onStartStop: () => void
  onReset: () => void
  className?: string
}

export const InferenceControls = memo(function InferenceControls({
  isRunning,
  isResetting,
  disabled,
  onStartStop,
  onReset,
  className,
}: InferenceControlsProps) {
  return (
    <div
      className={cn(
        'flex items-center justify-center gap-3 rounded-lg border border-border/60 bg-card p-4',
        className,
      )}
    >
      <Button
        size="lg"
        onClick={onStartStop}
        disabled={disabled}
        aria-pressed={isRunning}
        className="min-w-[140px]"
      >
        {isRunning ? (
          <>
            <Square className="h-4 w-4" />
            Stop
          </>
        ) : (
          <>
            <Play className="h-4 w-4" />
            Start
          </>
        )}
      </Button>

      <Button variant="outline" size="lg" onClick={onReset} disabled={isResetting || disabled}>
        {isResetting ? (
          <Loader2 className="h-4 w-4 animate-spin" />
        ) : (
          <RotateCcw className="h-4 w-4" />
        )}
        Reset
      </Button>
    </div>
  )
})
