import { API_BASE } from '@/lib/constants'

export class ApiError extends Error {
  status: number
  requestId?: string
  detail?: unknown

  constructor(message: string, status: number, detail?: unknown, requestId?: string) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.detail = detail
    this.requestId = requestId
  }
}

type RequestOptions = Omit<RequestInit, 'body'> & {
  body?: BodyInit | Record<string, unknown> | null
  params?: Record<string, string | number | boolean | undefined>
}

function buildUrl(path: string, params?: RequestOptions['params']): string {
  const base = API_BASE || window.location.origin
  const url = new URL(path, base)

  if (params) {
    for (const [key, value] of Object.entries(params)) {
      if (value !== undefined) {
        url.searchParams.set(key, String(value))
      }
    }
  }

  return url.toString()
}

export async function apiClient<T>(
  path: string,
  options: RequestOptions = {},
): Promise<T> {
  const { body, params, headers, ...rest } = options
  const isJsonBody = body !== null && body !== undefined && !(body instanceof FormData) && !(body instanceof Blob)

  const response = await fetch(buildUrl(path, params), {
    ...rest,
    headers: {
      ...(isJsonBody ? { 'Content-Type': 'application/json' } : {}),
      ...headers,
    },
    body: isJsonBody ? JSON.stringify(body) : (body as BodyInit | undefined),
  })

  const requestId = response.headers.get('X-Request-ID') ?? undefined

  if (!response.ok) {
    let detail: unknown
    try {
      detail = await response.json()
    } catch {
      detail = await response.text()
    }
    throw new ApiError(
      `Request failed: ${response.status}`,
      response.status,
      detail,
      requestId,
    )
  }

  if (response.status === 204) {
    return undefined as T
  }

  const contentType = response.headers.get('Content-Type') ?? ''
  if (contentType.includes('application/json')) {
    return response.json() as Promise<T>
  }

  return response as unknown as T
}

export async function apiBlob(path: string, options: RequestOptions = {}): Promise<Blob> {
  const { body, params, headers, ...rest } = options
  const isJsonBody = body !== null && body !== undefined && !(body instanceof FormData)

  const response = await fetch(buildUrl(path, params), {
    ...rest,
    method: rest.method ?? 'POST',
    headers: {
      ...(isJsonBody ? { 'Content-Type': 'application/json' } : {}),
      ...headers,
    },
    body: isJsonBody ? JSON.stringify(body) : (body as BodyInit | undefined),
  })

  if (!response.ok) {
    throw new ApiError(`Request failed: ${response.status}`, response.status)
  }

  return response.blob()
}
