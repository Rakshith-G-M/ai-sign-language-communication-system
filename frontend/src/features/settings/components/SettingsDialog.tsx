import { useEffect } from 'react'
import { useForm } from 'react-hook-form'
import { zodResolver } from '@hookform/resolvers/zod'

import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'

import { Label } from '@/components/ui/label'
import { Switch } from '@/components/ui/switch'
import { Select } from '@/components/ui/input'
import { Separator } from '@/components/ui/separator'

import { settingsSchema, type Settings } from '@/lib/schemas'
import { useSettingsStore } from '@/features/settings/store/settingsStore'
import { useInferenceStore } from '@/features/inference/store/inferenceStore'

interface SettingsDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
}

export function SettingsDialog({
  open,
  onOpenChange,
}: SettingsDialogProps) {
  const {
    theme,
    predictionFpsLimit,
    autoSpeakOnFinalize,
    cameraMirror,
    showTimeline,
    setSettings,
  } = useSettingsStore()

  const setTimelineExpanded = useInferenceStore(
    (s) => s.setTimelineExpanded,
  )

  const {
    register,
    watch,
    setValue,
    reset,
  } = useForm<Settings>({
    resolver: zodResolver(settingsSchema),
    defaultValues: {
      theme,
      predictionFpsLimit,
      autoSpeakOnFinalize,
      cameraMirror,
      showTimeline,
    },
  })

  useEffect(() => {
    if (!open) return

    reset({
      theme,
      predictionFpsLimit,
      autoSpeakOnFinalize,
      cameraMirror,
      showTimeline,
    })
  }, [
    open,
    reset,
    theme,
    predictionFpsLimit,
    autoSpeakOnFinalize,
    cameraMirror,
    showTimeline,
  ])

  useEffect(() => {
    const subscription = watch((value) => {
      setSettings(value as Partial<Settings>)
    })

    return () => subscription.unsubscribe()
  }, [watch, setSettings])

  const values = watch()

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle>Settings</DialogTitle>
          <DialogDescription>
            Configure your SignFlow experience.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-6 py-2">
          <div className="space-y-2">
            <Label htmlFor="theme">Theme</Label>

            <Select id="theme" {...register('theme')}>
              <option value="dark">Dark</option>
              <option value="light">Light</option>
              <option value="system">System</option>
            </Select>
          </div>

          <div className="space-y-2">
            <Label htmlFor="fps">
              Prediction rate (FPS)
            </Label>

            <Select
              id="fps"
              {...register('predictionFpsLimit', {
                valueAsNumber: true,
              })}
            >
              <option value={5}>5 FPS</option>
              <option value={10}>10 FPS</option>
              <option value={15}>15 FPS</option>
            </Select>
          </div>

          <Separator />

          <div className="flex items-center justify-between">
            <div className="space-y-0.5">
              <Label htmlFor="auto-speak">
                Auto-speak on finalize
              </Label>

              <p className="text-xs text-muted-foreground">
                Speak automatically when a sentence completes.
              </p>
            </div>

            <Switch
              id="auto-speak"
              checked={values.autoSpeakOnFinalize}
              onCheckedChange={(checked) =>
                setValue('autoSpeakOnFinalize', checked)
              }
            />
          </div>

          <div className="flex items-center justify-between">
            <div className="space-y-0.5">
              <Label htmlFor="mirror">
                Mirror camera
              </Label>

              <p className="text-xs text-muted-foreground">
                Flip webcam horizontally.
              </p>
            </div>

            <Switch
              id="mirror"
              checked={values.cameraMirror}
              onCheckedChange={(checked) =>
                setValue('cameraMirror', checked)
              }
            />
          </div>

          <div className="flex items-center justify-between">
            <div className="space-y-0.5">
              <Label htmlFor="timeline">
                Show timeline expanded
              </Label>

              <p className="text-xs text-muted-foreground">
                Expand prediction timeline by default.
              </p>
            </div>

            <Switch
              id="timeline"
              checked={values.showTimeline}
              onCheckedChange={(checked) => {
                setValue('showTimeline', checked)
                setTimelineExpanded(checked)
              }}
            />
          </div>
        </div>
      </DialogContent>
    </Dialog>
  )
}