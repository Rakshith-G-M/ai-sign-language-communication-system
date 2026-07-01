import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import { defaultSettings, type Settings } from '@/lib/schemas'

interface SettingsStore extends Settings {
  setSettings: (partial: Partial<Settings>) => void
  resetSettings: () => void
}

export const useSettingsStore = create<SettingsStore>()(
  persist(
    (set) => ({
      ...defaultSettings,
      setSettings: (partial) => set((state) => ({ ...state, ...partial })),
      resetSettings: () => set(defaultSettings),
    }),
    { name: 'signflow-settings' },
  ),
)
