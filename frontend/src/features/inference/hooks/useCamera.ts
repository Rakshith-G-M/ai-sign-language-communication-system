import { useCallback, useEffect, useRef, useState } from 'react'

export type CameraStatus = 'idle' | 'requesting' | 'active' | 'denied' | 'error'

interface UseCameraOptions {
  mirrored?: boolean
}

export function useCamera({ mirrored = true }: UseCameraOptions = {}) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const [status, setStatus] = useState<CameraStatus>('idle')
  const [error, setError] = useState<string | null>(null)

  const stop = useCallback(() => {
    streamRef.current?.getTracks().forEach((track) => track.stop())
    streamRef.current = null
    if (videoRef.current) {
      videoRef.current.srcObject = null
    }
    setStatus('idle')
  }, [])

  const start = useCallback(async () => {
    setStatus('requesting')
    setError(null)

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
        audio: false,
      })

      streamRef.current = stream
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        await videoRef.current.play()
      }
      setStatus('active')
      return true
    } catch (err) {
      const message =
        err instanceof DOMException && err.name === 'NotAllowedError'
          ? 'Camera permission was denied. Please allow camera access in your browser settings.'
          : 'Unable to access the camera. Check that a camera is connected and not in use.'
      setError(message)
      setStatus(err instanceof DOMException && err.name === 'NotAllowedError' ? 'denied' : 'error')
      return false
    }
  }, [])

  const captureBlob = useCallback(async (): Promise<Blob | null> => {
    const video = videoRef.current
    const canvas = canvasRef.current
    if (!video || !canvas || video.readyState < video.HAVE_ENOUGH_DATA) {
      return null
    }

    const ctx = canvas.getContext('2d')
    if (!ctx) return null

    ctx.save()
    if (mirrored) {
      ctx.scale(-1, 1)
      ctx.drawImage(video, -canvas.width, 0, canvas.width, canvas.height)
    } else {
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height)
    }
    ctx.restore()

    return new Promise((resolve) => {
      canvas.toBlob((blob) => resolve(blob), 'image/jpeg', 0.85)
    })
  }, [mirrored])

  useEffect(() => () => stop(), [stop])

  return {
    videoRef,
    canvasRef,
    status,
    error,
    start,
    stop,
    captureBlob,
    stream: streamRef.current,
  }
}

/** Request camera permission without keeping stream — used in onboarding */
export async function requestCameraPermission(): Promise<{
  granted: boolean
  error?: string
}> {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false })
    stream.getTracks().forEach((track) => track.stop())
    return { granted: true }
  } catch (err) {
    const message =
      err instanceof DOMException && err.name === 'NotAllowedError'
        ? 'Camera permission was denied.'
        : 'Unable to access the camera.'
    return { granted: false, error: message }
  }
}
