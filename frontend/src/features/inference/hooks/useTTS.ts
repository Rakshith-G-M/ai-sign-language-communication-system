import { useCallback } from 'react'
import { useMutation } from '@tanstack/react-query'
import { synthesizeSpeech } from '@/lib/api/prediction'
import { playAudioBlob, stopAudio, unlockAudio } from '@/lib/audio'

export function useTTS() {
  const mutation = useMutation({
    mutationFn: async (text: string) => {
      if (!text.trim()) return

      stopAudio()
      const blob = await synthesizeSpeech(text.trim())
      await playAudioBlob(blob)
    },
  })

  const speak = useCallback(
    (text: string) => {
      unlockAudio()
      mutation.mutate(text)
    },
    [mutation.mutate],
  )

  const speakAsync = useCallback(
    async (text: string) => {
      unlockAudio()
      await mutation.mutateAsync(text)
    },
    [mutation.mutateAsync],
  )

  const stop = useCallback(() => {
    mutation.reset()
    stopAudio()
  }, [mutation])

  return {
    speak,
    speakAsync,
    isSpeaking: mutation.isPending,
    error: mutation.error,
    stop,
    unlockAudio,
  }
}
