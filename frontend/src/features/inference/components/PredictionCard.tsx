import { memo } from 'react'
import { motion } from 'framer-motion'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Skeleton } from '@/components/ui/skeleton'
import { ConfidenceMeter } from '@/features/inference/components/ConfidenceMeter'
import { useReducedMotion } from '@/hooks/useReducedMotion'
import { useInferenceStore } from '@/features/inference/store/inferenceStore'
import { scaleIn } from '@/lib/motion'
import { cn } from '@/lib/cn'

interface PredictionCardProps {
  compact?: boolean
}

export const PredictionCard = memo(function PredictionCard({ compact }: PredictionCardProps) {
  const reducedMotion = useReducedMotion()
  const letter = useInferenceStore((s) => s.prediction.letter)
  const confidence = useInferenceStore((s) => s.prediction.confidence)
  const isRunning = useInferenceStore((s) => s.isRunning)
  const isLoading = isRunning && !letter

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
        {isLoading ? (
          <Skeleton className="h-20 w-full" />
        ) : (
          <div className="flex flex-col items-center py-2">
            <motion.div
              key={letter ?? 'empty'}
              {...(reducedMotion ? {} : scaleIn)}
              className={cn(
                'font-mono text-6xl font-semibold tracking-tight',
                !letter && 'text-muted-foreground',
              )}
              aria-live="polite"
            >
              {letter ?? '—'}
            </motion.div>
            <p className="mt-1 text-sm text-muted-foreground">
              {letter ? 'Stable letter detected' : 'Sign a letter to begin'}
            </p>
            <div className="mt-4 w-full">
              <ConfidenceMeter value={confidence} size="md" showLabel />
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
})
