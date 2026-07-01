import { create } from 'zustand'
import { persist } from 'zustand/middleware'

export interface HistoryEntry {
  id: string
  text: string
  timestamp: number
  wordCount: number
}

interface HistoryStore {
  entries: HistoryEntry[]
  isOpen: boolean
  addEntry: (text: string) => void
  removeEntry: (id: string) => void
  clearHistory: () => void
  setOpen: (open: boolean) => void
}

export const useHistoryStore = create<HistoryStore>()(
  persist(
    (set) => ({
      entries: [],
      isOpen: false,
      addEntry: (text) =>
        set((state) => {
          if (!text.trim()) return state
          const entry: HistoryEntry = {
            id: crypto.randomUUID(),
            text: text.trim(),
            timestamp: Date.now(),
            wordCount: text.trim().split(/\s+/).length,
          }
          return { entries: [entry, ...state.entries].slice(0, 100) }
        }),
      removeEntry: (id) =>
        set((state) => ({ entries: state.entries.filter((e) => e.id !== id) })),
      clearHistory: () => set({ entries: [] }),
      setOpen: (open) => set({ isOpen: open }),
    }),
    { name: 'signflow-history' },
  ),
)
