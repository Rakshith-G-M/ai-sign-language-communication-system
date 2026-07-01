import { useShallow } from 'zustand/react/shallow'
import { useInferenceStore } from '@/features/inference/store/inferenceStore'

export function LiveAnnouncement() {
  const { letter, confidence, word } = useInferenceStore(
    useShallow((s) => ({
      letter: s.prediction.letter,
      confidence: s.prediction.confidence,
      word: s.prediction.word,
    })),
  )

  return (
    <div aria-live="polite" aria-atomic="true" className="sr-only">
      {letter && confidence > 0 && `Letter ${letter}`}
      {word && `Building word ${word}`}
    </div>
  )
}
