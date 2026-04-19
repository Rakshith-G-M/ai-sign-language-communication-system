import { useState, useCallback, useRef } from 'react';

/**
 * useASLPredictor - Custom React hook for ASL prediction API
 * 
 * Handles frame-to-backend communication with error handling and state management
 */
function useASLPredictor(apiBase = 'http://localhost:8000') {
  const [prediction, setPrediction] = useState({
    letter: null,
    confidence: 0,
    handDetected: false,
    word: '',
    suggestions: [], // ✨ New suggestions list
    sentence: '',
  });

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [stats, setStats] = useState({
    processingTime: 0,
    frameCount: 0,
    successCount: 0,
    errorCount: 0,
  });

  const abortControllerRef = useRef(null);
  const audioRef = useRef(null);

  /**
   * Play TTS audio from text
   */
  const speak = useCallback(async (text) => {
    if (!text || !text.trim()) return;
    
    try {
      const response = await fetch(`${apiBase}/api/v1/tts`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      });

      if (!response.ok) throw new Error('TTS fetch failed');

      const blob = await response.blob();
      const audioUrl = URL.createObjectURL(blob);
      
      if (audioRef.current) {
        audioRef.current.pause();
      }
      
      audioRef.current = new Audio(audioUrl);
      audioRef.current.play();
      
      // Cleanup URL after playback starts/ends
      audioRef.current.onended = () => URL.revokeObjectURL(audioUrl);
    } catch (err) {
      console.error('TTS Playback Error:', err);
    }
  }, [apiBase]);

  /**
   * Predict from image blob (file upload or canvas capture)
   */
  const predictFromBlob = useCallback(
    async (blob) => {
      setLoading(true);
      setError(null);

      try {
        const startTime = performance.now();

        const formData = new FormData();
        formData.append('file', blob, 'frame.jpg');

        abortControllerRef.current = new AbortController();

        const response = await fetch(`${apiBase}/api/v1/predict`, {
          method: 'POST',
          body: formData,
          signal: abortControllerRef.current.signal,
        });

        if (!response.ok) {
          throw new Error(`API error: ${response.status}`);
        }

        const data = await response.json();
        const processingMs = Math.round(performance.now() - startTime);

        setPrediction({
          letter: data.letter,
          confidence: data.confidence,
          handDetected: data.hand_detected,
          word: data.word,
          suggestions: data.suggestions || [],
          sentence: data.sentence,
        });

        // Manual speak triggered via UI

        setStats((prev) => ({
          ...prev,
          processingTime: processingMs,
          frameCount: prev.frameCount + 1,
          successCount: prev.successCount + 1,
        }));

        return data;
      } catch (err) {
        if (err.name !== 'AbortError') {
          setError(err.message);
          setStats((prev) => ({
            ...prev,
            errorCount: prev.errorCount + 1,
          }));
          console.error('Prediction error:', err);
        }
      } finally {
        setLoading(false);
      }
    },
    [apiBase]
  );

  /**
   * Predict from base64-encoded image
   */
  const predictFromBase64 = useCallback(
    async (base64String) => {
      setLoading(true);
      setError(null);

      try {
        const startTime = performance.now();

        const response = await fetch(`${apiBase}/api/v1/predict-base64`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            image: base64String,
          }),
          signal: abortControllerRef.current?.signal,
        });

        if (!response.ok) {
          throw new Error(`API error: ${response.status}`);
        }

        const data = await response.json();
        const processingMs = Math.round(performance.now() - startTime);

        setPrediction({
          letter: data.letter,
          confidence: data.confidence,
          handDetected: data.hand_detected,
          word: data.word,
          suggestions: data.suggestions || [],
          sentence: data.sentence,
        });

        // Manual speak triggered via UI

        setStats((prev) => ({
          ...prev,
          processingTime: processingMs,
          frameCount: prev.frameCount + 1,
          successCount: prev.successCount + 1,
        }));

        return data;
      } catch (err) {
        if (err.name !== 'AbortError') {
          setError(err.message);
          setStats((prev) => ({
            ...prev,
            errorCount: prev.errorCount + 1,
          }));
        }
      } finally {
        setLoading(false);
      }
    },
    [apiBase]
  );

  /**
   * Reset text builder state on backend
   */
  const resetState = useCallback(async () => {
    try {
      const response = await fetch(`${apiBase}/api/v1/reset`, {
        method: 'POST',
      });

      if (!response.ok) {
        throw new Error(`Reset failed: ${response.status}`);
      }

      setPrediction((prev) => ({
        ...prev,
        word: '',
        suggestions: [],
        sentence: '',
      }));

      setStats((prev) => ({
        ...prev,
        frameCount: 0,
        successCount: 0,
        errorCount: 0,
      }));

      return true;
    } catch (err) {
      setError(err.message);
      return false;
    }
  }, [apiBase]);

  /**
   * Get current state without making a prediction
   */
  const getState = useCallback(async () => {
    try {
      const response = await fetch(`${apiBase}/api/v1/state`);

      if (!response.ok) {
        throw new Error(`State fetch failed: ${response.status}`);
      }

      const data = await response.json();
      return data;
    } catch (err) {
      setError(err.message);
      return null;
    }
  }, [apiBase]);

  /**
   * Check backend health
   */
  const checkHealth = useCallback(async () => {
    try {
      const response = await fetch(`${apiBase}/api/v1/health`);
      return response.ok;
    } catch {
      return false;
    }
  }, [apiBase]);

  /**
   * Cancel ongoing request
   */
  const cancel = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
  }, []);

  return {
    // State
    prediction,
    loading,
    error,
    stats,

    // Methods
    predictFromBlob,
    predictFromBase64,
    speak,
    resetState,
    getState,
    checkHealth,
    cancel,

    // Utilities
    clearError: () => setError(null),
    resetStats: () =>
      setStats({
        processingTime: 0,
        frameCount: 0,
        successCount: 0,
        errorCount: 0,
      }),
  };
}

export default useASLPredictor;