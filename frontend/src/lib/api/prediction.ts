import { apiBlob, apiClient } from './client'
import {
  predictionResponseSchema,
  resetResponseSchema,
  stateResponseSchema,
  type PredictionResponse,
  type StateResponse,
} from '@/lib/schemas'

export async function predictFrame(
  blob: Blob,
  sessionId: string,
): Promise<PredictionResponse> {
  const formData = new FormData()
  formData.append('file', blob, 'frame.jpg')

  const data = await apiClient<unknown>('/api/v1/predict', {
    method: 'POST',
    params: { session_id: sessionId },
    body: formData,
  })

  return predictionResponseSchema.parse(data)
}

export async function resetSession(sessionId: string): Promise<void> {
  const data = await apiClient<unknown>('/api/v1/reset', {
    method: 'POST',
    params: { session_id: sessionId },
  })
  resetResponseSchema.parse(data)
}

export async function getSessionState(sessionId: string): Promise<StateResponse> {
  const data = await apiClient<unknown>('/api/v1/state', {
    params: { session_id: sessionId },
  })
  return stateResponseSchema.parse(data)
}

export async function synthesizeSpeech(text: string): Promise<Blob> {
  return apiBlob('/api/v1/tts', {
    method: 'POST',
    body: { text },
  })
}
