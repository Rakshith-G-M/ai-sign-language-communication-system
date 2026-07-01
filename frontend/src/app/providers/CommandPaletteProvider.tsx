import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useRef,
  type ReactNode,
} from 'react'
import {
  createCommandRegistry,
  type CommandHandler,
  type CommandRegistry,
  getCommandPaletteShortcuts,
} from '@/lib/shortcuts'

interface CommandPaletteContextValue {
  /** Register a command handler — used by future ⌘K palette */
  registerCommand: (id: string, handler: CommandHandler) => void
  unregisterCommand: (id: string) => void
  executeCommand: (id: string) => void
  registry: CommandRegistry
  /** Shortcut definitions reserved for command palette */
  paletteCommands: ReturnType<typeof getCommandPaletteShortcuts>
}

const CommandPaletteContext = createContext<CommandPaletteContextValue | null>(null)

export function CommandPaletteProvider({ children }: { children: ReactNode }) {
  const registryRef = useRef<CommandRegistry>(createCommandRegistry())

  const registerCommand = useCallback((id: string, handler: CommandHandler) => {
    registryRef.current.set(id, handler)
  }, [])

  const unregisterCommand = useCallback((id: string) => {
    registryRef.current.delete(id)
  }, [])

  const executeCommand = useCallback((id: string) => {
    registryRef.current.get(id)?.()
  }, [])

  const value = useMemo<CommandPaletteContextValue>(
    () => ({
      registerCommand,
      unregisterCommand,
      executeCommand,
      registry: registryRef.current,
      paletteCommands: getCommandPaletteShortcuts(),
    }),
    [registerCommand, unregisterCommand, executeCommand],
  )

  return (
    <CommandPaletteContext.Provider value={value}>{children}</CommandPaletteContext.Provider>
  )
}

export function useCommandPalette() {
  const ctx = useContext(CommandPaletteContext)
  if (!ctx) {
    throw new Error('useCommandPalette must be used within CommandPaletteProvider')
  }
  return ctx
}
