export type ShortcutCategory = 'navigation' | 'inference' | 'general'

export interface ShortcutDefinition {
  id: string
  keys: string[]
  label: string
  category: ShortcutCategory
  /** When true, reserved for future command palette registration */
  commandPalette?: boolean
  handler?: () => void
}

export const SHORTCUT_REGISTRY: ShortcutDefinition[] = [
  {
    id: 'toggle-inference',
    keys: ['Space'],
    label: 'Start / stop inference',
    category: 'inference',
    commandPalette: true,
  },
  {
    id: 'reset-session',
    keys: ['R'],
    label: 'Reset session',
    category: 'inference',
    commandPalette: true,
  },
  {
    id: 'speak-sentence',
    keys: ['S'],
    label: 'Speak sentence',
    category: 'inference',
    commandPalette: true,
  },
  {
    id: 'open-settings',
    keys: ['Mod+,'],
    label: 'Open settings',
    category: 'general',
    commandPalette: true,
  },
  {
    id: 'open-history',
    keys: ['Mod+H'],
    label: 'Open history',
    category: 'navigation',
    commandPalette: true,
  },
  {
    id: 'open-shortcuts',
    keys: ['?'],
    label: 'Show keyboard shortcuts',
    category: 'general',
  },
  {
    id: 'open-command-palette',
    keys: ['Mod+K'],
    label: 'Open command palette (coming soon)',
    category: 'general',
    commandPalette: true,
  },
  {
    id: 'go-dashboard',
    keys: ['G', 'D'],
    label: 'Go to dashboard',
    category: 'navigation',
    commandPalette: true,
  },
  {
    id: 'go-system',
    keys: ['Mod', 'I'],
    label: 'Go to system information',
    category: 'navigation',
    commandPalette: true,
  },
]

export function formatShortcutKeys(keys: string[]): string {
  const isMac = typeof navigator !== 'undefined' && /Mac/.test(navigator.platform)
  return keys
    .map((key) => {
      if (key === 'Mod') return isMac ? '⌘' : 'Ctrl'
      if (key === 'Space') return 'Space'
      return key
    })
    .join(isMac && keys.length === 1 && keys[0].length === 1 ? '' : ' + ')
}

export function matchesShortcut(event: KeyboardEvent, keys: string[]): boolean {
  const isMac = /Mac/.test(navigator.platform)
  const modKey = isMac ? event.metaKey : event.ctrlKey

  for (const combo of keys) {
    if (combo === 'Mod') {
      if (!modKey) return false
      continue
    }
    if (combo === 'Space') {
      if (event.key !== ' ') return false
      continue
    }
    if (event.key.toLowerCase() !== combo.toLowerCase()) {
      return false
    }
  }

  return true
}

/** Registry hook point for future command palette — maps action IDs to handlers */
export type CommandHandler = () => void
export type CommandRegistry = Map<string, CommandHandler>

export function createCommandRegistry(): CommandRegistry {
  return new Map()
}

export function getCommandPaletteShortcuts(): ShortcutDefinition[] {
  return SHORTCUT_REGISTRY.filter((s) => s.commandPalette)
}
