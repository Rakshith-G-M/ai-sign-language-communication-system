import { useCallback, useEffect, useRef } from 'react'
import { useMutation } from '@tanstack/react-query'
import { predictFrame, resetSession } from '@/lib/api/prediction'
import { getSessionId } from '@/lib/session'
import { useSettingsStore } from '@/features/settings/store/settingsStore'
import { useHistoryStore } from '@/features/history/store/historyStore'
import { useInferenceStore } from '@/features/inference/store/inferenceStore'
import type { useTTS } from './useTTS'

interface UsePredictionLoopOptions {
  captureBlob: () => Promise<Blob | null>
  speak: ReturnType<typeof useTTS>['speak']
}

export function usePredictionLoop({ captureBlob, speak }: UsePredictionLoopOptions) {
  const sessionIdRef = useRef(getSessionId())
  const animationRef = useRef<number | null>(null)
  const lastCallRef = useRef(0)
  const frameTimestampsRef = useRef<number[]>([])
  const lastLetterRef = useRef<string | null>(null)
  const lastWordRef = useRef('')
  const isPendingRef = useRef(false)
  const captureBlobRef = useRef(captureBlob)
  const speakRef = useRef(speak)

  const fpsLimit = useSettingsStore((s) => s.predictionFpsLimit)
  const autoSpeak = useSettingsStore((s) => s.autoSpeakOnFinalize)
  const isRunning = useInferenceStore((s) => s.isRunning)

  captureBlobRef.current = captureBlob
  speakRef.current = speak

  const predictMutation = useMutation({
    mutationFn: (blob: Blob) => predictFrame(blob, sessionIdRef.current),
    onMutate: () => {
      isPendingRef.current = true
    },
    onSettled: () => {
      isPendingRef.current = false
    },
  })

  const predictAsyncRef = useRef(predictMutation.mutateAsync)
  predictAsyncRef.current = predictMutation.mutateAsync

  const resetMutation = useMutation({
    mutationFn: () => resetSession(sessionIdRef.current),
    onSuccess: () => {
      console.debug('[inference] sentence reset')
      useInferenceStore.getState().resetPrediction()
      useInferenceStore.getState().resetStats()
      useInferenceStore.getState().appendTimelineEvent({ type: 'reset', label: 'Session reset' })
      lastLetterRef.current = null
      lastWordRef.current = ''
    },
  })

  const processFrame = useCallback(async () => {
    if (isPendingRef.current) return

    const blob = await captureBlobRef.current()
    if (!blob) return

    const startTime = performance.now()
    const store = useInferenceStore.getState()

    try {
      const data = await predictAsyncRef.current(blob)
      const processingMs = Math.round(performance.now() - startTime)

      console.debug('[inference] prediction received', {
        letter: data.letter,
        confidence: data.confidence,
        word: data.word,
        hand_detected: data.hand_detected,
      })

      const now = performance.now()
      frameTimestampsRef.current.push(now)
      frameTimestampsRef.current = frameTimestampsRef.current.filter((t) => now - t < 1000)

      const currentStats = useInferenceStore.getState().stats
      store.applyFrameResult({
        prediction: {
          letter: data.letter,
          confidence: data.confidence,
          handDetected: data.hand_detected,
          word: data.word,
          suggestions: data.suggestions,
          sentence: data.sentence,
          finalizedSentence: data.finalized_sentence ?? null,
          serverLatencyMs: data.latency,
        },
        stats: {
          processingTimeMs: processingMs,
          frameCount: currentStats.frameCount + 1,
          successCount: currentStats.successCount + 1,
          clientFps: frameTimestampsRef.current.length,
        },
      })

      if (data.letter && data.letter !== lastLetterRef.current && data.confidence > 0) {
        console.debug('[inference] prediction stabilized', { letter: data.letter, confidence: data.confidence })
        store.appendTimelineEvent({ type: 'letter', label: data.letter })
        lastLetterRef.current = data.letter
      }

      if (lastWordRef.current && !data.word) {
        const committed = lastWordRef.current.replace(/_$/, '')
        if (committed) {
          store.appendTimelineEvent({ type: 'word', label: committed })
        }
      }
      lastWordRef.current = data.word

      if (data.finalized_sentence) {
        // Guard: only commit the finalized sentence if inference is still running.
        // An in-flight API response can resolve after the user clicks Stop, which
        // would otherwise silently append a word to a stopped session.
        const stillRunning = useInferenceStore.getState().isRunning
        if (stillRunning) {
          console.debug('[inference] sentence appended', { sentence: data.finalized_sentence })
          store.appendTimelineEvent({ type: 'sentence', label: data.finalized_sentence })
          useHistoryStore.getState().addEntry(data.finalized_sentence)
          if (autoSpeak) {
            speakRef.current(data.finalized_sentence)
          }
        }
      }
    } catch (err) {
      if (err instanceof Error && err.name !== 'AbortError') {
        const currentStats = useInferenceStore.getState().stats
        store.setError(err.message)
        store.setStats({ errorCount: currentStats.errorCount + 1 })
      }
    }
  }, [autoSpeak])

  useEffect(() => {
    if (!isRunning) {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
        animationRef.current = null
      }
      return
    }

    const loop = () => {
      const now = performance.now()
      if (now - lastCallRef.current >= 1000 / fpsLimit) {
        void processFrame().then(() => {
          lastCallRef.current = performance.now()
        })
      }
      animationRef.current = requestAnimationFrame(loop)
    }

    animationRef.current = requestAnimationFrame(loop)

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  }, [isRunning, fpsLimit, processFrame])

  return {
    reset: () => resetMutation.mutate(),
    isResetting: resetMutation.isPending,
  }
}
