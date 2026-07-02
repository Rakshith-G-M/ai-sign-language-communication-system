import { memo } from 'react'
import { motion } from 'framer-motion'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { ConfidenceMeter } from '@/features/inference/components/ConfidenceMeter'
import { useInferenceStore } from '@/features/inference/store/inferenceStore'
import { cn } from '@/lib/cn'

interface PredictionCardProps {
  compact?: boolean
}

export const PredictionCard = memo(function PredictionCard({ compact }: PredictionCardProps) {
  // Selectors must be called directly at the component top level — never inside
  // useMemo. The previous useMemo(()=>{...},[]) pattern captured initial null/0
  // values once at mount and never re-ran, causing the panel to stay blank.
  const letter = useInferenceStore((s) => s.prediction.letter)
  const confidence = useInferenceStore((s) => s.prediction.confidence)

  if (compact) {
    return (
      <div className="flex items-center justify-between rounded-lg border border-border/60 bg-card px-4 py-3">
        <div className="flex items-baseline gap-3">
          <span className="font-mono text-3xl font-semibold">{letter ?? '—'}</span>
          <ConfidenceMeter value={confidence} size="sm" />
        </div>
      </div>
    )
  }

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-base">Current Prediction</CardTitle>
      </CardHeader>
      <CardContent>
        {/* Fixed height container - prevents layout shift */}
        <div className="h-32 w-full flex flex-col items-center justify-center py-2">
          {/* No skeleton shown - card is always mounted and stable */}
          <motion.div
            initial={false}
            className={cn(
              'font-mono text-6xl font-semibold tracking-tight transition-opacity duration-200',
              !letter && 'text-muted-foreground opacity-50',
              letter && 'opacity-100',
            )}
            aria-live="polite"
            aria-atomic="false"
          >
            {letter ?? '—'}
          </motion.div>
          
          {/* Status message - animates in/out */}
          <motion.p
            initial={false}
            animate={{ opacity: 1 }}
            className="mt-3 text-sm text-muted-foreground h-5"
            aria-live="polite"
          >
            {letter ? 'Stable letter detected' : 'Sign a letter to begin'}
          </motion.p>
          
          {/* Confidence meter - always visible, animates smoothly */}
          <div className="mt-4 w-full">
            <ConfidenceMeter value={confidence} size="md" showLabel />
          </div>
        </div>
      </CardContent>
    </Card>
  )
})