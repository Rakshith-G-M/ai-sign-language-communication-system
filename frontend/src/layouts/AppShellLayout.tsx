import { Outlet } from 'react-router-dom'
import { useState } from 'react'
import { TopNav } from '@/components/shared/TopNav'
import { SettingsDialog } from '@/features/settings/components/SettingsDialog'
import { HistoryDrawer } from '@/features/history/components/HistoryDrawer'
import { ShortcutsDialog } from '@/components/shared/ShortcutsDialog'
import { useGlobalKeyboardShortcuts } from '@/hooks/useKeyboardShortcuts'
import { useTTS } from '@/features/inference/hooks/useTTS'
import { useHistoryStore } from '@/features/history/store/historyStore'

export function AppShellLayout() {
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [shortcutsOpen, setShortcutsOpen] = useState(false)
  const historyOpen = useHistoryStore((s) => s.isOpen)
  const setHistoryOpen = useHistoryStore((s) => s.setOpen)
  const { speak } = useTTS()

  useGlobalKeyboardShortcuts({
    onOpenSettings: () => setSettingsOpen(true),
    onOpenHistory: () => setHistoryOpen(true),
    onOpenShortcuts: () => setShortcutsOpen(true),
  })

  return (
    <div className="flex min-h-screen flex-col">
      <TopNav
        onOpenSettings={() => setSettingsOpen(true)}
        onOpenHistory={() => setHistoryOpen(true)}
      />
      <main className="mx-auto w-full max-w-7xl flex-1 px-4 py-6 sm:px-6 lg:px-8">
        <Outlet />
      </main>

      <SettingsDialog open={settingsOpen} onOpenChange={setSettingsOpen} />
      <HistoryDrawer open={historyOpen} onOpenChange={setHistoryOpen} onSpeak={speak} />
      <ShortcutsDialog open={shortcutsOpen} onOpenChange={setShortcutsOpen} />
    </div>
  )
}
