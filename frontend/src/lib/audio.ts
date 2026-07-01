let audioContext: AudioContext | null = null
let activeSource: AudioBufferSourceNode | null = null

export function unlockAudio(): void {
  if (typeof window === 'undefined') return

  if (!audioContext) {
    audioContext = new AudioContext()
  }

  if (audioContext.state === 'suspended') {
    void audioContext.resume()
  }
}

export function stopAudio(): void {
  if (activeSource) {
    try {
      activeSource.stop()
    } catch {
      // Source may already be stopped.
    }
    activeSource.disconnect()
    activeSource = null
  }
}

export async function playAudioBlob(blob: Blob): Promise<void> {
  if (blob.size === 0) {
    throw new Error('Received empty audio response from server')
  }

  unlockAudio()

  if (!audioContext) {
    throw new Error('Audio playback is not available in this browser')
  }

  stopAudio()

  const arrayBuffer = await blob.arrayBuffer()
  const audioBuffer = await audioContext.decodeAudioData(arrayBuffer)

  return new Promise((resolve, reject) => {
    const source = audioContext!.createBufferSource()
    source.buffer = audioBuffer
    source.connect(audioContext!.destination)
    source.onended = () => {
      if (activeSource === source) {
        activeSource = null
      }
      resolve()
    }
    activeSource = source

    try {
      source.start(0)
    } catch (err) {
      activeSource = null
      reject(err instanceof Error ? err : new Error('Failed to start audio playback'))
    }
  })
}
