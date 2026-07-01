import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { Separator } from '@/components/ui/separator'
import { SHORTCUT_REGISTRY, formatShortcutKeys } from '@/lib/shortcuts'

interface ShortcutsDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
}

const categories = [
  { key: 'inference', label: 'Inference' },
  { key: 'navigation', label: 'Navigation' },
  { key: 'general', label: 'General' },
] as const

export function ShortcutsDialog({ open, onOpenChange }: ShortcutsDialogProps) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle>Keyboard Shortcuts</DialogTitle>
          <DialogDescription>
            Press <kbd className="rounded border border-border bg-muted px-1.5 py-0.5 font-mono text-xs">?</kbd>{' '}
            anytime to show this overlay.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4 py-2">
          {categories.map((category) => {
            const shortcuts = SHORTCUT_REGISTRY.filter((s) => s.category === category.key)
            if (shortcuts.length === 0) return null

            return (
              <div key={category.key}>
                <h3 className="mb-2 text-xs font-medium uppercase tracking-wide text-muted-foreground">
                  {category.label}
                </h3>
                <ul className="space-y-2">
                  {shortcuts.map((shortcut) => (
                    <li key={shortcut.id} className="flex items-center justify-between text-sm">
                      <span>{shortcut.label}</span>
                      <kbd className="rounded border border-border bg-muted px-2 py-0.5 font-mono text-xs">
                        {formatShortcutKeys(shortcut.keys)}
                      </kbd>
                    </li>
                  ))}
                </ul>
                <Separator className="mt-4" />
              </div>
            )
          })}
        </div>
      </DialogContent>
    </Dialog>
  )
}
