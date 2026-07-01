import { create } from 'zustand'

export type TimelineEventType = 'letter' | 'word' | 'gesture' | 'sentence' | 'reset'

export interface TimelineEvent {
  id: string
  type: TimelineEventType
  label: string
  timestamp: number
}

export interface PredictionState {
  letter: string | null
  confidence: number
  handDetected: boolean
  word: string
  suggestions: string[]
  sentence: string
  finalizedSentence: string | null
  serverLatencyMs: number
}

export interface ClientStats {
  processingTimeMs: number
  frameCount: number
  successCount: number
  errorCount: number
  clientFps: number
}

interface FrameResultPayload {
  prediction: Partial<PredictionState>
  stats: Partial<ClientStats>
}

function suggestionsEqual(a: string[], b: string[]): boolean {
  if (a.length !== b.length) return false
  return a.every((value, index) => value === b[index])
}

function predictionFieldsChanged(
  current: PredictionState,
  partial: Partial<PredictionState>,
): boolean {
  if (partial.letter !== undefined && partial.letter !== current.letter) return true
  if (partial.confidence !== undefined && partial.confidence !== current.confidence) return true
  if (partial.handDetected !== undefined && partial.handDetected !== current.handDetected) return true
  if (partial.word !== undefined && partial.word !== current.word) return true
  if (partial.sentence !== undefined && partial.sentence !== current.sentence) return true
  if (
    partial.finalizedSentence !== undefined &&
    partial.finalizedSentence !== current.finalizedSentence
  ) {
    return true
  }
  if (
    partial.serverLatencyMs !== undefined &&
    partial.serverLatencyMs !== current.serverLatencyMs
  ) {
    return true
  }
  if (partial.suggestions !== undefined && !suggestionsEqual(partial.suggestions, current.suggestions)) {
    return true
  }
  return false
}

interface InferenceStore {
  isRunning: boolean
  isProcessing: boolean
  timeline: TimelineEvent[]
  timelineExpanded: boolean
  prediction: PredictionState
  stats: ClientStats
  error: string | null
  setRunning: (running: boolean) => void
  setProcessing: (processing: boolean) => void
  setPrediction: (partial: Partial<PredictionState>) => void
  setStats: (partial: Partial<ClientStats>) => void
  applyFrameResult: (payload: FrameResultPayload) => void
  setError: (error: string | null) => void
  appendTimelineEvent: (event: Omit<TimelineEvent, 'id' | 'timestamp'>) => void
  clearTimeline: () => void
  setTimelineExpanded: (expanded: boolean) => void
  resetPrediction: () => void
  resetStats: () => void
}

const initialPrediction: PredictionState = {
  letter: null,
  confidence: 0,
  handDetected: false,
  word: '',
  suggestions: [],
  sentence: '',
  finalizedSentence: null,
  serverLatencyMs: 0,
}

const initialStats: ClientStats = {
  processingTimeMs: 0,
  frameCount: 0,
  successCount: 0,
  errorCount: 0,
  clientFps: 0,
}

export const useInferenceStore = create<InferenceStore>((set) => ({
  isRunning: false,
  isProcessing: false,
  timeline: [],
  timelineExpanded: false,
  prediction: initialPrediction,
  stats: initialStats,
  error: null,
  setRunning: (running) => set({ isRunning: running }),
  setProcessing: (processing) => set({ isProcessing: processing }),
  setPrediction: (partial) =>
    set((state) => ({ prediction: { ...state.prediction, ...partial } })),
  setStats: (partial) => set((state) => ({ stats: { ...state.stats, ...partial } })),
  applyFrameResult: ({ prediction: predictionPartial, stats: statsPartial }) =>
    set((state) => {
      const predictionChanged = predictionFieldsChanged(state.prediction, predictionPartial)
      const nextPrediction = predictionChanged
        ? { ...state.prediction, ...predictionPartial }
        : state.prediction
      const nextStats = { ...state.stats, ...statsPartial }

      if (!predictionChanged && state.error === null) {
        return { stats: nextStats }
      }

      return {
        prediction: nextPrediction,
        stats: nextStats,
        error: null,
      }
    }),
  setError: (error) => set({ error }),
  appendTimelineEvent: (event) =>
    set((state) => {
      const newEvent: TimelineEvent = {
        ...event,
        id: crypto.randomUUID(),
        timestamp: Date.now(),
      }
      const timeline = [newEvent, ...state.timeline].slice(0, 50)
      return { timeline }
    }),
  clearTimeline: () => set({ timeline: [] }),
  setTimelineExpanded: (expanded) => set({ timelineExpanded: expanded }),
  resetPrediction: () => set({ prediction: initialPrediction }),
  resetStats: () => set({ stats: initialStats }),
}))
