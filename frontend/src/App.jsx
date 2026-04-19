import React, { useState, useEffect, useRef } from 'react'
import { motion } from 'framer-motion'

import WebcamFeed from './components/WebcamFeed'
import PredictionPanel from './components/PredictionPanel'
import WordBuilder from './components/WordBuilder'
import ControlBar from './components/ControlBar'
import useASLPredictor from './hooks/useASLPredictor'

import './App.css'

function App() {
  const [isRunning, setIsRunning] = useState(false)

  const webcamRef = useRef(null)
  const canvasRef = useRef(null)
  const animationRef = useRef(null)
  const lastCallRef = useRef(0)
  const isProcessingRef = useRef(false)

  const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000'

  // 🔥 Use hook
  const {
    prediction,
    stats,
    predictFromBlob,
    speak,
    resetState,
    error,
  } = useASLPredictor(API_BASE)

  const FPS_LIMIT = 10 // ✅ Prevent backend overload

  // ─────────────────────────────────────────────────────────────
  // Prediction Loop
  // ─────────────────────────────────────────────────────────────
  useEffect(() => {
    if (!isRunning) return

    const canvas = canvasRef.current
    const video = webcamRef.current

    if (!canvas || !video) return

    const ctx = canvas.getContext('2d')

    const loop = async () => {
      try {
        const now = performance.now()

        // ✅ Throttle requests
        if (now - lastCallRef.current < 1000 / FPS_LIMIT) {
          animationRef.current = requestAnimationFrame(loop)
          return
        }

        // ✅ Prevent overlapping requests
        if (isProcessingRef.current) {
          animationRef.current = requestAnimationFrame(loop)
          return
        }

        if (video.readyState === video.HAVE_ENOUGH_DATA) {
          ctx.save()
          ctx.scale(-1, 1)
          ctx.drawImage(video, -canvas.width, 0, canvas.width, canvas.height)
          ctx.restore()

          isProcessingRef.current = true

          canvas.toBlob(async (blob) => {
            if (!blob) return

            await predictFromBlob(blob)

            isProcessingRef.current = false
            lastCallRef.current = performance.now()
          }, 'image/jpeg', 0.85)
        }
      } catch (err) {
        console.error('Loop error:', err)
      }

      animationRef.current = requestAnimationFrame(loop)
    }

    animationRef.current = requestAnimationFrame(loop)

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  }, [isRunning, predictFromBlob])

  // ─────────────────────────────────────────────────────────────
  // Controls
  // ─────────────────────────────────────────────────────────────
  const handleStartStop = () => {
    setIsRunning((prev) => !prev)
  }

  const handleReset = async () => {
    await resetState()
  }

  // ─────────────────────────────────────────────────────────────
  // UI
  // ─────────────────────────────────────────────────────────────
  return (
    <div className="asl-app">
      <div className="background-grid" />
      <div className="background-gradient" />

      <motion.div
        className="app-container"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
      >
        {/* Header */}
        <div className="header">
          <div className="logo">
            <span className="logo-symbol">◆</span>
            <span className="logo-text">ASL Recognition</span>
          </div>

          <div className="status-indicator">
            <div className={`status-dot ${isRunning ? 'active' : ''}`} />
            <span>{isRunning ? 'LIVE' : 'IDLE'}</span>
          </div>
        </div>

        {/* Main Layout */}
        <div className="main-content">
          {/* Webcam */}
          <div className="webcam-section">
            <WebcamFeed
              videoRef={webcamRef}
              isActive={isRunning}
              handDetected={prediction.handDetected}
            />
            <canvas
              ref={canvasRef}
              style={{ display: 'none' }}
              width={640}
              height={480}
            />
          </div>

          {/* Prediction */}
          <div className="prediction-section">
            <PredictionPanel
              letter={prediction.letter}
              confidence={prediction.confidence}
            />

            <WordBuilder
              word={prediction.word}
              sentence={prediction.sentence}
              suggestions={prediction.suggestions}
              onSpeak={() => speak(prediction.sentence)}
            />

            {/* Metrics */}
            <div className="metrics-panel">
              <div className="metric">
                <span className="metric-label">Latency</span>
                <span className="metric-value">
                  {stats.processingTime}ms
                </span>
              </div>

              <div className="metric">
                <span className="metric-label">Frames</span>
                <span className="metric-value">
                  {stats.frameCount}
                </span>
              </div>

              <div className="metric">
                <span className="metric-label">Hand</span>
                <span className={`metric-value ${prediction.handDetected ? 'detected' : 'missing'}`}>
                  {prediction.handDetected ? '✓ Yes' : '✗ No'}
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Controls */}
        <ControlBar
          isRunning={isRunning}
          onStartStop={handleStartStop}
          onReset={handleReset}
        />

        {/* Error display */}
        {error && (
          <div style={{ color: 'red', textAlign: 'center' }}>
            {error}
          </div>
        )}
      </motion.div>
    </div>
  )
}

export default App