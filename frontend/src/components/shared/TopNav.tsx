import { Link, useLocation } from 'react-router-dom'
import { History, Info, Settings } from 'lucide-react'
import { AppLogo } from '@/components/shared/AppLogo'
import { StatusIndicator } from '@/components/shared/StatusIndicator'
import { Button } from '@/components/ui/button'
import { cn } from '@/lib/cn'
import { useInferenceStore } from '@/features/inference/store/inferenceStore'

interface TopNavProps {
  onOpenSettings: () => void
  onOpenHistory: () => void
}

export function TopNav({ onOpenSettings, onOpenHistory }: TopNavProps) {
  const location = useLocation()
  const isRunning = useInferenceStore((s) => s.isRunning)

  const status = isRunning ? 'live' : 'idle'

  const navLinkClass = (path: string) =>
    cn(
      'rounded-md px-3 py-1.5 text-sm font-medium transition-colors',
      location.pathname === path
        ? 'bg-muted text-foreground'
        : 'text-muted-foreground hover:bg-muted/50 hover:text-foreground',
    )

  return (
    <header className="sticky top-0 z-40 border-b border-border/60 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/80">
      <div className="mx-auto flex h-14 max-w-7xl items-center justify-between gap-4 px-4 sm:px-6 lg:px-8">
        <div className="flex items-center gap-6">
          <Link to="/dashboard" className="shrink-0">
            <AppLogo size="sm" />
          </Link>

          <nav className="hidden items-center gap-1 sm:flex" aria-label="Main navigation">
            <Link to="/dashboard" className={navLinkClass('/dashboard')}>
              Dashboard
            </Link>
            <Link to="/system" className={navLinkClass('/system')}>
              System
            </Link>
          </nav>
        </div>

        <div className="flex items-center gap-2">
          {location.pathname === '/dashboard' && (
            <StatusIndicator status={status} className="hidden sm:inline-flex" />
          )}

          <Button variant="ghost" size="icon" onClick={onOpenHistory} aria-label="Open history">
            <History className="h-4 w-4" />
          </Button>

          <Button variant="ghost" size="icon" onClick={onOpenSettings} aria-label="Open settings">
            <Settings className="h-4 w-4" />
          </Button>

          <Link to="/system" className="sm:hidden">
            <Button variant="ghost" size="icon" aria-label="System information">
              <Info className="h-4 w-4" />
            </Button>
          </Link>
        </div>
      </div>
    </header>
  )
}
