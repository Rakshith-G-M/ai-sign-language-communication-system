import { z } from 'zod'

export const predictionResponseSchema = z.object({
  letter: z.string().nullable(),
  confidence: z.number(),
  word: z.string(),
  sentence: z.string(),
  suggestions: z.array(z.string()).default([]),
  finalized_sentence: z.string().nullable().optional(),
  hand_detected: z.boolean(),
  latency: z.number(),
})

export type PredictionResponse = z.infer<typeof predictionResponseSchema>

export const stateResponseSchema = z.object({
  word: z.string(),
  sentence: z.string(),
})

export type StateResponse = z.infer<typeof stateResponseSchema>

export const resetResponseSchema = z.object({
  status: z.string(),
})

export const readinessCheckSchema = z.object({
  static_model: z.boolean(),
  mediapipe: z.boolean(),
  dynamic_predictor: z.boolean(),
  prediction_service: z.boolean(),
})

export const readinessResponseSchema = z.object({
  status: z.enum(['ready', 'not_ready']),
  checks: readinessCheckSchema,
})

export type ReadinessResponse = z.infer<typeof readinessResponseSchema>

export const livenessResponseSchema = z.object({
  status: z.string(),
})

export const metricsResponseSchema = z.object({
  uptime_seconds: z.number(),
  active_sessions: z.number(),
  total_predictions: z.number(),
  static_predictions: z.number(),
  dynamic_predictions: z.number(),
})

export type MetricsResponse = z.infer<typeof metricsResponseSchema>

export const serviceInfoSchema = z.object({
  service: z.string(),
  version: z.string(),
  status: z.string().optional(),
  docs: z.string().optional(),
  api_base: z.string().optional(),
  endpoints: z.array(z.string()).optional(),
})

export type ServiceInfo = z.infer<typeof serviceInfoSchema>

export const settingsSchema = z.object({
  theme: z.enum(['dark', 'light', 'system']),
  autoSpeakOnFinalize: z.boolean(),
  predictionFpsLimit: z.union([z.literal(5), z.literal(10), z.literal(15)]),
  showTimeline: z.boolean(),
  cameraMirror: z.boolean(),
})

export type Settings = z.infer<typeof settingsSchema>

export const defaultSettings: Settings = {
  theme: 'dark',
  autoSpeakOnFinalize: false,
  predictionFpsLimit: 10,
  showTimeline: false,
  cameraMirror: true,
}
