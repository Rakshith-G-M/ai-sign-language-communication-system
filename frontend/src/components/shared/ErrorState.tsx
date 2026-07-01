import { AlertCircle } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent } from '@/components/ui/card'

interface ErrorStateProps {
  title: string
  message: string
  requestId?: string
  onRetry?: () => void
  inline?: boolean
}

export function ErrorState({ title, message, requestId, onRetry, inline }: ErrorStateProps) {
  const content = (
    <>
      <div className="flex items-start gap-3">
        <AlertCircle className="mt-0.5 h-5 w-5 shrink-0 text-destructive" aria-hidden />
        <div className="space-y-1">
          <h3 className="font-medium">{title}</h3>
          <p className="text-sm text-muted-foreground">{message}</p>
          {requestId && (
            <p className="font-mono text-xs text-muted-foreground">Request ID: {requestId}</p>
          )}
        </div>
      </div>
      {onRetry && (
        <Button variant="outline" size="sm" className="mt-4" onClick={onRetry}>
          Try again
        </Button>
      )}
    </>
  )

  if (inline) {
    return <div className="rounded-lg border border-destructive/20 bg-destructive/5 p-4">{content}</div>
  }

  return (
    <Card className="border-destructive/20">
      <CardContent className="pt-6">{content}</CardContent>
    </Card>
  )
}
