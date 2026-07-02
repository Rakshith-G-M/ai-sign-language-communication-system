import { useCallback, useRef } from 'react'
import { useMutation } from '@tanstack/react-query'
import { synthesizeSpeech } from '@/lib/api/prediction'
import { playAudioBlob, stopAudio, unlockAudio } from '@/lib/audio'

export function useTTS() {
  const mutationRef = useRef<any>(null)

  const mutation = useMutation({
    mutationFn: async (text: string) => {
      if (!text.trim()) {
        throw new Error('Text cannot be empty')
      }

      stopAudio()

      try {
        const blob = await synthesizeSpeech(text.trim())
        
        // Validate blob before attempting playback
        if (!blob || blob.size === 0) {
          throw new Error('Server returned empty audio')
        }

        await playAudioBlob(blob)
      } catch (error) {
        // Ensure we have a proper error message for the caller
        if (error instanceof Error) {
          throw error
        }
        throw new Error(`TTS failed: ${String(error)}`)
      }
    },
    onError: (error) => {
      // Ensure audio is stopped if playback fails
      stopAudio()
      // Error will be available via mutation.error
      console.error('TTS mutation error:', error)
    },
  })

  mutationRef.current = mutation

  const speak = useCallback(
    (text: string) => {
      try {
        unlockAudio()
        mutation.mutate(text)
      } catch (error) {
        mutation.reset()
        stopAudio()
        throw error
      }
    },
    [mutation],
  )

  const speakAsync = useCallback(
    async (text: string) => {
      try {
        unlockAudio()
        await mutation.mutateAsync(text)
      } catch (error) {
        mutation.reset()
        stopAudio()
        throw error
      }
    },
    [mutation],
  )

  const stop = useCallback(() => {
    mutation.reset()
    stopAudio()
  }, [mutation])

  return {
    speak,
    speakAsync,
    isSpeaking: mutation.isPending,
    error: mutation.error instanceof Error ? mutation.error : null,
    stop,
    unlockAudio,
  }
}