import { SESSION_STORAGE_KEY } from './constants'

const SESSION_ID_PATTERN = /^[a-zA-Z0-9_-]{1,128}$/

export function getSessionId(): string {
  const stored = sessionStorage.getItem(SESSION_STORAGE_KEY)
  if (stored && SESSION_ID_PATTERN.test(stored)) {
    return stored
  }

  let id: string
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    id = crypto.randomUUID()
  } else {
    id = `${Math.random().toString(36).slice(2, 11)}_${Date.now()}`
  }
  sessionStorage.setItem(SESSION_STORAGE_KEY, id)
  return id
}

export function resetSessionId(): string {
  sessionStorage.removeItem(SESSION_STORAGE_KEY)
  return getSessionId()
}
