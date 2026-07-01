import { useCallback, useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { Camera, CheckCircle2, Server, Sparkles } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { LoadingOverlay, type CheckItem } from '@/components/shared/LoadingOverlay'
import { ErrorState } from '@/components/shared/ErrorState'
import { AppLogo } from '@/components/shared/AppLogo'
import { checkReadiness } from '@/lib/api/health'
import { requestCameraPermission } from '@/features/inference/hooks/useCamera'
import { ONBOARDING_STORAGE_KEY, APP_TAGLINE } from '@/lib/constants'
import { slideUp } from '@/lib/motion'

type Step = 'welcome' | 'backend' | 'camera' | 'ready'

const STEPS: Step[] = ['welcome', 'backend', 'camera', 'ready']

export function OnboardingPage() {
  const navigate = useNavigate()
  const [step, setStep] = useState<Step>('welcome')
  const [checks, setChecks] = useState<CheckItem[]>([])
  const [backendError, setBackendError] = useState<string | null>(null)
  const [cameraError, setCameraError] = useState<string | null>(null)
  const [isCheckingBackend, setIsCheckingBackend] = useState(false)

  const runBackendCheck = useCallback(async () => {
    setIsCheckingBackend(true)
    setBackendError(null)
    setChecks([
      { label: 'Backend connection', status: 'loading' },
      { label: 'Static model', status: 'pending' },
      { label: 'MediaPipe', status: 'pending' },
      { label: 'Dynamic predictor', status: 'pending' },
      { label: 'Prediction service', status: 'pending' },
    ])

    try {
      const result = await checkReadiness()
      const ready = result.status === 'ready'

      setChecks([
        { label: 'Backend connection', status: 'success' },
        {
          label: 'Static model',
          status: result.checks.static_model ? 'success' : 'error',
        },
        {
          label: 'MediaPipe',
          status: result.checks.mediapipe ? 'success' : 'error',
        },
        {
          label: 'Dynamic predictor',
          status: result.checks.dynamic_predictor ? 'success' : 'error',
        },
        {
          label: 'Prediction service',
          status: result.checks.prediction_service ? 'success' : 'error',
        },
      ])

      if (!ready) {
        setBackendError('Some backend components are not ready. You can retry or continue anyway.')
      }

      setIsCheckingBackend(false)
      return ready
    } catch {
      setChecks((prev) =>
        prev.map((c, i) => (i === 0 ? { ...c, status: 'error' } : c)),
      )
      setBackendError('Unable to connect to the backend. Make sure the server is running.')
      setIsCheckingBackend(false)
      return false
    }
  }, [])

  const runCameraCheck = useCallback(async () => {
    setCameraError(null)
    const result = await requestCameraPermission()
    if (!result.granted) {
      setCameraError(result.error ?? 'Camera permission denied.')
      return false
    }
    return true
  }, [])

  const completeOnboarding = useCallback(() => {
    localStorage.setItem(ONBOARDING_STORAGE_KEY, 'true')
    navigate('/dashboard', { replace: true })
  }, [navigate])

  const stepIndex = STEPS.indexOf(step)

  const handleNext = async () => {
    if (step === 'welcome') {
      setStep('backend')
      await runBackendCheck()
      return
    }

    if (step === 'backend') {
      setStep('camera')
      await runCameraCheck()
      return
    }

    if (step === 'camera') {
      if (!cameraError) {
        setStep('ready')
      } else {
        const ok = await runCameraCheck()
        if (ok) setStep('ready')
      }
      return
    }

    completeOnboarding()
  }

  const handleSkipCamera = () => {
    setStep('ready')
  }

  useEffect(() => {
    if (step === 'backend' && checks.length === 0) {
      void runBackendCheck()
    }
  }, [step, checks.length, runBackendCheck])

  if (isCheckingBackend && step === 'backend' && checks.every((c) => c.status === 'loading' || c.status === 'pending')) {
    return (
      <LoadingOverlay
        message="Checking backend readiness"
        submessage="Verifying models and services…"
        checks={checks}
      />
    )
  }

  return (
    <div className="flex min-h-screen items-center justify-center p-4">
      <div className="w-full max-w-lg">
        <div className="mb-8 flex justify-center">
          <AppLogo />
        </div>

        <div className="mb-6 flex justify-center gap-2">
          {STEPS.map((s, i) => (
            <div
              key={s}
              className={`h-1.5 w-12 rounded-full transition-colors ${
                i <= stepIndex ? 'bg-primary' : 'bg-muted'
              }`}
              aria-hidden
            />
          ))}
        </div>

        <AnimatePresence mode="wait">
          <motion.div key={step} {...slideUp} className="rounded-lg border border-border bg-card p-8">
            {step === 'welcome' && (
              <>
                <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-lg bg-primary/10">
                  <Sparkles className="h-6 w-6 text-primary" />
                </div>
                <h1 className="text-xl font-semibold">Welcome to SignFlow</h1>
                <p className="mt-2 text-sm text-muted-foreground">{APP_TAGLINE}</p>
                <p className="mt-4 text-sm text-muted-foreground">
                  This quick setup checks your backend connection, AI models, and camera before you
                  start signing.
                </p>
              </>
            )}

            {step === 'backend' && (
              <>
                <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-lg bg-primary/10">
                  <Server className="h-6 w-6 text-primary" />
                </div>
                <h1 className="text-xl font-semibold">Backend & Models</h1>
                <ul className="mt-4 space-y-2">
                  {checks.map((check) => (
                    <li
                      key={check.label}
                      className="flex items-center justify-between rounded-md bg-muted/40 px-3 py-2 text-sm"
                    >
                      <span>{check.label}</span>
                      <span
                        className={
                          check.status === 'success'
                            ? 'text-accent'
                            : check.status === 'error'
                              ? 'text-destructive'
                              : 'text-muted-foreground'
                        }
                      >
                        {check.status === 'success' && 'Ready'}
                        {check.status === 'error' && 'Failed'}
                        {check.status === 'loading' && 'Checking…'}
                        {check.status === 'pending' && 'Pending'}
                      </span>
                    </li>
                  ))}
                </ul>
                {backendError && (
                  <div className="mt-4">
                    <ErrorState title="Backend check" message={backendError} inline onRetry={runBackendCheck} />
                  </div>
                )}
              </>
            )}

            {step === 'camera' && (
              <>
                <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-lg bg-primary/10">
                  <Camera className="h-6 w-6 text-primary" />
                </div>
                <h1 className="text-xl font-semibold">Camera Access</h1>
                <p className="mt-2 text-sm text-muted-foreground">
                  SignFlow needs your webcam to capture hand signs for real-time recognition.
                </p>
                {cameraError && (
                  <div className="mt-4">
                    <ErrorState title="Camera access" message={cameraError} inline />
                  </div>
                )}
              </>
            )}

            {step === 'ready' && (
              <>
                <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-lg bg-accent/10">
                  <CheckCircle2 className="h-6 w-6 text-accent" />
                </div>
                <h1 className="text-xl font-semibold">You&apos;re all set</h1>
                <p className="mt-2 text-sm text-muted-foreground">
                  Head to the dashboard to start live ASL recognition. Press{' '}
                  <kbd className="rounded border border-border bg-muted px-1 font-mono text-xs">?</kbd>{' '}
                  anytime for keyboard shortcuts.
                </p>
              </>
            )}

            <div className="mt-8 flex gap-3">
              {step === 'camera' && (
                <Button variant="outline" onClick={handleSkipCamera}>
                  Skip for now
                </Button>
              )}
              <Button className="flex-1" onClick={() => void handleNext()}>
                {step === 'ready' ? 'Enter Dashboard' : 'Continue'}
              </Button>
            </div>
          </motion.div>
        </AnimatePresence>
      </div>
    </div>
  )
}
