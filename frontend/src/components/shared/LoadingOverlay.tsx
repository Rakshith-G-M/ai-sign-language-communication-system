import { Loader2 } from 'lucide-react'
import { cn } from '@/lib/cn'

export interface CheckItem {
  label: string
  status: 'pending' | 'loading' | 'success' | 'error'
}

interface LoadingOverlayProps {
  message: string
  submessage?: string
  checks?: CheckItem[]
  className?: string
}

export function LoadingOverlay({ message, submessage, checks, className }: LoadingOverlayProps) {
  return (
    <div
      className={cn(
        'fixed inset-0 z-50 flex items-center justify-center bg-background/80 backdrop-blur-sm',
        className,
      )}
      role="status"
      aria-live="polite"
      aria-busy="true"
    >
      <div className="w-full max-w-md space-y-6 rounded-lg border border-border bg-card p-8 shadow-md">
        <div className="flex flex-col items-center text-center">
          <Loader2 className="h-8 w-8 animate-spin text-primary" aria-hidden />
          <h2 className="mt-4 text-lg font-semibold">{message}</h2>
          {submessage && <p className="mt-1 text-sm text-muted-foreground">{submessage}</p>}
        </div>

        {checks && checks.length > 0 && (
          <ul className="space-y-2" aria-label="System checks">
            {checks.map((check) => (
              <li
                key={check.label}
                className="flex items-center justify-between rounded-md bg-muted/50 px-3 py-2 text-sm"
              >
                <span>{check.label}</span>
                <span
                  className={cn(
                    'text-xs font-medium',
                    check.status === 'success' && 'text-accent',
                    check.status === 'error' && 'text-destructive',
                    check.status === 'loading' && 'text-primary',
                    check.status === 'pending' && 'text-muted-foreground',
                  )}
                >
                  {check.status === 'loading' && 'Checking…'}
                  {check.status === 'success' && 'Ready'}
                  {check.status === 'error' && 'Failed'}
                  {check.status === 'pending' && 'Pending'}
                </span>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  )
}
