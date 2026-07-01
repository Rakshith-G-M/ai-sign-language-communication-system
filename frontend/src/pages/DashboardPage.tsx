import { useCallback, useEffect } from 'react'
import { motion } from 'framer-motion'
import { CameraCard } from '@/features/inference/components/CameraCard'
import { PredictionCard } from '@/features/inference/components/PredictionCard'
import { SentencePanel } from '@/features/inference/components/SentencePanel'
import { TimelinePanel } from '@/features/inference/components/TimelinePanel'
import { InferenceControls } from '@/features/inference/components/InferenceControls'
import { MetricsPanel } from '@/features/system/components/MetricsPanel'
import { HealthBadge } from '@/features/system/components/HealthBadge'
import { ErrorState } from '@/components/shared/ErrorState'
import { LiveAnnouncement } from '@/components/shared/LiveAnnouncement'
import { useCamera } from '@/features/inference/hooks/useCamera'
import { useTTS } from '@/features/inference/hooks/useTTS'
import { usePredictionLoop } from '@/features/inference/hooks/usePredictionLoop'
import { useInferenceStore } from '@/features/inference/store/inferenceStore'
import { useSettingsStore } from '@/features/settings/store/settingsStore'
import { useSystemHealth } from '@/features/system/hooks/useSystemHealth'
import { useInferenceKeyboardShortcuts } from '@/hooks/useKeyboardShortcuts'
import { useIsMobile } from '@/hooks/useMediaQuery'
import { getSessionState } from '@/lib/api/prediction'
import { getSessionId } from '@/lib/session'
import { fadeIn } from '@/lib/motion'

export function DashboardPage() {
  const isMobile = useIsMobile()
  const cameraMirror = useSettingsStore((s) => s.cameraMirror)
  const showTimelineDefault = useSettingsStore((s) => s.showTimeline)

  const isRunning = useInferenceStore((s) => s.isRunning)
  const error = useInferenceStore((s) => s.error)
  const setRunning = useInferenceStore((s) => s.setRunning)
  const setTimelineExpanded = useInferenceStore((s) => s.setTimelineExpanded)

  const { readiness } = useSystemHealth({ enableMetrics: isRunning })
  const { speak, isSpeaking, error: ttsError, unlockAudio } = useTTS()

  const { videoRef, canvasRef, status, error: cameraError, start, stop, captureBlob } =
    useCamera({ mirrored: cameraMirror })

  const { reset, isResetting } = usePredictionLoop({ captureBlob, speak })

  const handleToggleInference = useCallback(async () => {
    if (isRunning) {
      setRunning(false)
      stop()
    } else {
      unlockAudio()
      const ok = await start()
      if (ok) setRunning(true)
    }
  }, [isRunning, setRunning, start, stop, unlockAudio])

  const handleReset = useCallback(() => {
    reset()
  }, [reset])

  const handleSpeak = useCallback(() => {
    const sentence = useInferenceStore.getState().prediction.sentence
    if (sentence.trim()) {
      speak(sentence)
    }
  }, [speak])

  useEffect(() => {
    setTimelineExpanded(showTimelineDefault)
  }, [showTimelineDefault, setTimelineExpanded])

  useEffect(() => {
    void getSessionState(getSessionId()).then((state) => {
      if (state.word || state.sentence) {
        useInferenceStore.getState().setPrediction({
          word: state.word,
          sentence: state.sentence,
        })
      }
    })
  }, [])

  useInferenceKeyboardShortcuts({
    onToggleInference: () => void handleToggleInference(),
    onReset: handleReset,
    onSpeak: handleSpeak,
  })

  const ready = readiness.data?.status === 'ready'

  return (
    <motion.div {...fadeIn} className="space-y-6">
      <LiveAnnouncement />

      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Live Recognition</h1>
          <p className="text-sm text-muted-foreground">
            Real-time ASL inference from your webcam
          </p>
        </div>
        <HealthBadge ready={ready} checks={readiness.data?.checks} compact={isMobile} />
      </div>

      <div className="grid gap-6 lg:grid-cols-[1fr_320px] xl:grid-cols-[1fr_340px]">
        <div className="space-y-6">
          <CameraCard
            videoRef={videoRef}
            canvasRef={canvasRef}
            mirrored={cameraMirror}
            cameraStatus={status}
            cameraError={cameraError}
          />

          {isMobile && <PredictionCard compact />}

          <TimelinePanel />

          {!isMobile && (
            <InferenceControls
              isRunning={isRunning}
              isResetting={isResetting}
              onStartStop={handleToggleInference}
              onReset={handleReset}
            />
          )}
        </div>

        <div className="space-y-6">
          {!isMobile && <PredictionCard />}

          <SentencePanel onSpeak={handleSpeak} isSpeaking={isSpeaking} />

          <MetricsPanel compact={isMobile} />
        </div>
      </div>

      {isMobile && (
        <InferenceControls
          isRunning={isRunning}
          isResetting={isResetting}
          onStartStop={handleToggleInference}
          onReset={handleReset}
          className="sticky bottom-4 z-30 shadow-md"
        />
      )}

      {error && <ErrorState title="Prediction error" message={error} inline />}
      {ttsError && (
        <ErrorState
          title="Speech playback failed"
          message={ttsError instanceof Error ? ttsError.message : 'Unable to play audio'}
          inline
        />
      )}
    </motion.div>
  )
}
