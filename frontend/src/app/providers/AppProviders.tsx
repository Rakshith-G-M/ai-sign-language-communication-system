import { TooltipProvider } from '@/components/ui/tooltip'
import { QueryProvider } from '@/app/providers/QueryProvider'
import { CommandPaletteProvider } from '@/app/providers/CommandPaletteProvider'
import type { ReactNode } from 'react'

export function AppProviders({ children }: { children: ReactNode }) {
  return (
    <QueryProvider>
      <CommandPaletteProvider>
        <TooltipProvider delayDuration={300}>{children}</TooltipProvider>
      </CommandPaletteProvider>
    </QueryProvider>
  )
}
