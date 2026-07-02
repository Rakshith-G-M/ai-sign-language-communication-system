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
  // Validate blob
  if (!blob) {
    throw new Error('No audio blob provided')
  }

  if (blob.size === 0) {
    throw new Error('Received empty audio response from server')
  }

  // Validate blob type
  if (!blob.type.includes('audio')) {
    console.warn(`Expected audio MIME type, got: ${blob.type}`)
  }

  unlockAudio()

  if (!audioContext) {
    throw new Error('Audio playback is not available in this browser')
  }

  stopAudio()

  try {
    const arrayBuffer = await blob.arrayBuffer()
    
    // Validate that we have data
    if (arrayBuffer.byteLength === 0) {
      throw new Error('Audio blob contains no data')
    }

    let audioBuffer: AudioBuffer
    try {
      audioBuffer = await audioContext.decodeAudioData(arrayBuffer)
    } catch (decodeError) {
      const decodeErr = decodeError instanceof Error ? decodeError.message : String(decodeError)
      throw new Error(`Failed to decode audio data: ${decodeErr}`)
    }

    // Validate decoded audio
    if (audioBuffer.length === 0) {
      throw new Error('Decoded audio buffer is empty')
    }

    return new Promise((resolve, reject) => {
      const source = audioContext!.createBufferSource()
      source.buffer = audioBuffer
      source.connect(audioContext!.destination)

      const onEnded = () => {
        if (activeSource === source) {
          activeSource = null
        }
        source.removeEventListener('ended', onEnded)
        resolve()
      }

      source.addEventListener('ended', onEnded)
      activeSource = source

      try {
        source.start(0)
      } catch (err) {
        activeSource = null
        source.removeEventListener('ended', onEnded)
        const errMsg = err instanceof Error ? err.message : String(err)
        reject(new Error(`Failed to start audio playback: ${errMsg}`))
      }
    })
  } catch (err) {
    activeSource = null
    // Re-throw with context if not already an error
    if (err instanceof Error) {
      throw err
    }
    throw new Error(`Audio playback failed: ${String(err)}`)
  }
}