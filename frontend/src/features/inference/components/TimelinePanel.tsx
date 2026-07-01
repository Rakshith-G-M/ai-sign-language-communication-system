import { memo } from 'react'
import { ChevronDown, ChevronUp, Clock } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { ScrollArea } from '@/components/ui/scroll-area'
import { EmptyState } from '@/components/shared/EmptyState'
import type { TimelineEvent } from '@/features/inference/store/inferenceStore'
import { useInferenceStore } from '@/features/inference/store/inferenceStore'
import { cn } from '@/lib/cn'

const typeLabels: Record<TimelineEvent['type'], string> = {
  letter: 'Letter',
  word: 'Word',
  gesture: 'Gesture',
  sentence: 'Sentence',
  reset: 'Reset',
}

const typeColors: Record<TimelineEvent['type'], string> = {
  letter: 'text-primary',
  word: 'text-foreground',
  gesture: 'text-accent',
  sentence: 'text-accent',
  reset: 'text-muted-foreground',
}

export const TimelinePanel = memo(function TimelinePanel() {
  const events = useInferenceStore((s) => s.timeline)
  const expanded = useInferenceStore((s) => s.timelineExpanded)
  const setTimelineExpanded = useInferenceStore((s) => s.setTimelineExpanded)
  const clearTimeline = useInferenceStore((s) => s.clearTimeline)

  const onToggle = () => setTimelineExpanded(!expanded)

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <button
          type="button"
          onClick={onToggle}
          className="flex items-center gap-2 text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
          aria-expanded={expanded}
        >
          <CardTitle className="text-base">Prediction Timeline</CardTitle>
          {expanded ? (
            <ChevronUp className="h-4 w-4 text-muted-foreground" />
          ) : (
            <ChevronDown className="h-4 w-4 text-muted-foreground" />
          )}
          {!expanded && events.length > 0 && (
            <span className="rounded-sm bg-muted px-1.5 py-0.5 text-xs text-muted-foreground">
              {events.length}
            </span>
          )}
        </button>
        {expanded && events.length > 0 && (
          <Button variant="ghost" size="sm" onClick={clearTimeline}>
            Clear
          </Button>
        )}
      </CardHeader>

      {expanded && (
        <CardContent>
          {events.length === 0 ? (
            <EmptyState
              icon={Clock}
              title="No events yet"
              description="Prediction events will appear here as you sign."
            />
          ) : (
            <ScrollArea className="h-48">
              <ul className="space-y-2">
                {events.map((event) => (
                  <li
                    key={event.id}
                    className="flex items-center justify-between rounded-md bg-muted/40 px-3 py-2 text-sm"
                  >
                    <div className="flex items-center gap-3">
                      <span
                        className={cn(
                          'rounded-sm bg-muted px-1.5 py-0.5 text-xs font-medium',
                          typeColors[event.type],
                        )}
                      >
                        {typeLabels[event.type]}
                      </span>
                      <span className="font-medium">{event.label}</span>
                    </div>
                    <time className="font-mono text-xs text-muted-foreground">
                      {new Date(event.timestamp).toLocaleTimeString()}
                    </time>
                  </li>
                ))}
              </ul>
            </ScrollArea>
          )}
        </CardContent>
      )}
    </Card>
  )
})
