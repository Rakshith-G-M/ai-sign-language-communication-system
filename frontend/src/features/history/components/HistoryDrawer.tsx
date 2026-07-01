import { Trash2, Volume2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { ScrollArea } from '@/components/ui/scroll-area'
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from '@/components/ui/sheet'
import { EmptyState } from '@/components/shared/EmptyState'
import { useHistoryStore } from '@/features/history/store/historyStore'
import { History } from 'lucide-react'

interface HistoryDrawerProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  onSpeak: (text: string) => void
}

export function HistoryDrawer({ open, onOpenChange, onSpeak }: HistoryDrawerProps) {
  const { entries, removeEntry, clearHistory } = useHistoryStore()

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent side="right" className="flex flex-col">
        <SheetHeader>
          <SheetTitle>History</SheetTitle>
          <SheetDescription>Past finalized sentences from this session.</SheetDescription>
        </SheetHeader>

        {entries.length === 0 ? (
          <EmptyState
            icon={History}
            title="No history yet"
            description="Finalized sentences will appear here automatically."
          />
        ) : (
          <>
            <div className="flex justify-end">
              <Button variant="ghost" size="sm" onClick={clearHistory}>
                Clear all
              </Button>
            </div>
            <ScrollArea className="flex-1 pr-4">
              <ul className="space-y-3">
                {entries.map((entry) => (
                  <li
                    key={entry.id}
                    className="rounded-lg border border-border/60 bg-muted/30 p-3"
                  >
                    <p className="text-sm leading-relaxed">{entry.text}</p>
                    <div className="mt-2 flex items-center justify-between">
                      <time className="text-xs text-muted-foreground">
                        {new Date(entry.timestamp).toLocaleString()} · {entry.wordCount} words
                      </time>
                      <div className="flex gap-1">
                        <Button
                          variant="ghost"
                          size="icon"
                          className="h-7 w-7"
                          onClick={() => onSpeak(entry.text)}
                          aria-label="Speak entry"
                        >
                          <Volume2 className="h-3.5 w-3.5" />
                        </Button>
                        <Button
                          variant="ghost"
                          size="icon"
                          className="h-7 w-7"
                          onClick={() => removeEntry(entry.id)}
                          aria-label="Delete entry"
                        >
                          <Trash2 className="h-3.5 w-3.5" />
                        </Button>
                      </div>
                    </div>
                  </li>
                ))}
              </ul>
            </ScrollArea>
          </>
        )}
      </SheetContent>
    </Sheet>
  )
}
