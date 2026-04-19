import React, { useEffect, useState } from 'react'
import { motion } from 'framer-motion'

function WebcamFeed({ videoRef, isActive, handDetected }) {
  const [loaded, setLoaded] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    let stream = null

    const startCamera = async () => {
      try {
        const video = videoRef.current
        if (!video) return

        stream = await navigator.mediaDevices.getUserMedia({
          video: true,
          audio: false,
        })

        video.srcObject = stream

        video.onloadedmetadata = () => {
          setLoaded(true)
        }
      } catch (err) {
        console.error('Camera error:', err)
        setError('Camera access denied')
      }
    }

    if (isActive) {
      startCamera()
    } else {
      const video = videoRef.current
      if (video && video.srcObject) {
        video.srcObject.getTracks().forEach(track => track.stop())
        video.srcObject = null
      }
      setLoaded(false)
    }

    return () => {
      if (stream) {
        stream.getTracks().forEach(track => track.stop())
      }
    }
  }, [isActive])

  return (
    <motion.div className="webcam-feed">
      {error ? (
        <div className="webcam-placeholder">{error}</div>
      ) : (
        <>
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            style={{
              width: '100%',
              height: '100%',
              objectFit: 'cover',
              transform: 'scaleX(-1)'
            }}
          />

          {!loaded && (
            <div className="webcam-placeholder">
              Initializing camera...
            </div>
          )}

          <div className={`hand-indicator ${handDetected ? 'detected' : 'missing'}`}>
            {handDetected ? '✓ HAND DETECTED' : '✗ NO HAND'}
          </div>
        </>
      )}
    </motion.div>
  )
}

export default WebcamFeed