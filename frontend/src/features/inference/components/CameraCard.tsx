import { memo } from 'react'
import { motion } from 'framer-motion'
import { Hand, VideoOff } from 'lucide-react'
import { Card } from '@/components/ui/card'
import { cn } from '@/lib/cn'
import { useReducedMotion } from '@/hooks/useReducedMotion'
import { CANVAS_HEIGHT, CANVAS_WIDTH } from '@/lib/constants'
import { useInferenceStore } from '@/features/inference/store/inferenceStore'

interface CameraCardProps {
  videoRef: React.RefObject<HTMLVideoElement | null>
  canvasRef: React.RefObject<HTMLCanvasElement | null>
  mirrored?: boolean
  cameraStatus: 'idle' | 'requesting' | 'active' | 'denied' | 'error'
  cameraError?: string | null
}

export const CameraCard = memo(function CameraCard({
  videoRef,
  canvasRef,
  mirrored = true,
  cameraStatus,
  cameraError,
}: CameraCardProps) {
  const reducedMotion = useReducedMotion()
  const isActive = useInferenceStore((s) => s.isRunning)
  const handDetected = useInferenceStore((s) => s.prediction.handDetected)

  return (
    <Card
      className={cn(
        'relative overflow-hidden border-border shadow-sm',
        isActive && !reducedMotion && 'ring-1 ring-primary/20',
      )}
    >
      <div className="relative aspect-video w-full bg-muted/30">
        <video
          ref={videoRef}
          className={cn(
            'h-full w-full object-cover',
            mirrored && 'scale-x-[-1]',
            !isActive && 'opacity-40',
          )}
          playsInline
          muted
          aria-label="Live webcam feed"
        />

        {!isActive && cameraStatus !== 'active' && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 bg-background/60 text-muted-foreground">
            <VideoOff className="h-8 w-8" aria-hidden />
            <p className="text-sm">
              {cameraStatus === 'denied' || cameraStatus === 'error'
                ? cameraError ?? 'Camera unavailable'
                : 'Start inference to activate camera'}
            </p>
          </div>
        )}

        {isActive && (
          <div
            className={cn(
              'absolute left-3 top-3 flex items-center gap-1.5 rounded-md px-2 py-1 text-xs font-medium backdrop-blur-sm',
              handDetected
                ? 'bg-accent/20 text-accent'
                : 'bg-muted/80 text-muted-foreground',
            )}
          >
            <Hand className="h-3.5 w-3.5" aria-hidden />
            {handDetected ? 'Hand detected' : 'No hand'}
          </div>
        )}

        {isActive && !reducedMotion && (
          <motion.div
            className="pointer-events-none absolute inset-0 border-2 border-primary/10"
            animate={{ opacity: [0.2, 0.35, 0.2] }}
            transition={{ duration: 2.5, repeat: Infinity, ease: 'easeInOut' }}
            aria-hidden
          />
        )}
      </div>

      <canvas
        ref={canvasRef}
        width={CANVAS_WIDTH}
        height={CANVAS_HEIGHT}
        className="hidden"
        aria-hidden
      />
    </Card>
  )
})
