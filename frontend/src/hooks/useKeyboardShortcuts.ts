import { useCallback, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { useCommandPalette } from '@/app/providers/CommandPaletteProvider'

interface UseGlobalKeyboardShortcutsOptions {
  onOpenSettings: () => void
  onOpenHistory: () => void
  onOpenShortcuts: () => void
}

export function useGlobalKeyboardShortcuts({
  onOpenSettings,
  onOpenHistory,
  onOpenShortcuts,
}: UseGlobalKeyboardShortcutsOptions) {
  const navigate = useNavigate()
  const { registerCommand, unregisterCommand } = useCommandPalette()

  const handleKeyDown = useCallback(
    (event: KeyboardEvent) => {
      const target = event.target as HTMLElement
      if (
        target.tagName === 'INPUT' ||
        target.tagName === 'TEXTAREA' ||
        target.tagName === 'SELECT' ||
        target.isContentEditable
      ) {
        return
      }

      const isMod = event.metaKey || event.ctrlKey

      if (event.key === '?' && !isMod) {
        event.preventDefault()
        onOpenShortcuts()
        return
      }

      if (isMod && event.key.toLowerCase() === 'k') {
        event.preventDefault()
        return
      }

      if (isMod && event.key === ',') {
        event.preventDefault()
        onOpenSettings()
        return
      }

      if (isMod && event.key.toLowerCase() === 'h') {
        event.preventDefault()
        onOpenHistory()
        return
      }

      if (isMod && event.key.toLowerCase() === 'd') {
        event.preventDefault()
        navigate('/dashboard')
        return
      }

      if (isMod && event.key.toLowerCase() === 'i') {
        event.preventDefault()
        navigate('/system')
      }
    },
    [navigate, onOpenHistory, onOpenSettings, onOpenShortcuts],
  )

  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [handleKeyDown])

  useEffect(() => {
    registerCommand('open-settings', onOpenSettings)
    registerCommand('open-history', onOpenHistory)
    registerCommand('go-dashboard', () => navigate('/dashboard'))
    registerCommand('go-system', () => navigate('/system'))

    return () => {
      unregisterCommand('open-settings')
      unregisterCommand('open-history')
      unregisterCommand('go-dashboard')
      unregisterCommand('go-system')
    }
  }, [navigate, onOpenHistory, onOpenSettings, registerCommand, unregisterCommand])
}

export function useInferenceKeyboardShortcuts(options: {
  onSpeak: () => void
  onReset: () => void
  onToggleInference: () => void
  enabled?: boolean
}) {
  const { registerCommand, unregisterCommand } = useCommandPalette()
  const { onSpeak, onReset, onToggleInference, enabled = true } = options

  const handleKeyDown = useCallback(
    (event: KeyboardEvent) => {
      if (!enabled) return

      const target = event.target as HTMLElement
      if (
        target.tagName === 'INPUT' ||
        target.tagName === 'TEXTAREA' ||
        target.tagName === 'SELECT' ||
        target.isContentEditable
      ) {
        return
      }

      const isMod = event.metaKey || event.ctrlKey
      if (isMod) return

      if (event.key === ' ') {
        event.preventDefault()
        onToggleInference()
        return
      }

      if (event.key.toLowerCase() === 'r') {
        event.preventDefault()
        onReset()
        return
      }

      if (event.key.toLowerCase() === 's') {
        event.preventDefault()
        onSpeak()
      }
    },
    [enabled, onReset, onSpeak, onToggleInference],
  )

  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [handleKeyDown])

  useEffect(() => {
    registerCommand('toggle-inference', onToggleInference)
    registerCommand('reset-session', onReset)
    registerCommand('speak-sentence', onSpeak)

    return () => {
      unregisterCommand('toggle-inference')
      unregisterCommand('reset-session')
      unregisterCommand('speak-sentence')
    }
  }, [onReset, onSpeak, onToggleInference, registerCommand, unregisterCommand])
}
