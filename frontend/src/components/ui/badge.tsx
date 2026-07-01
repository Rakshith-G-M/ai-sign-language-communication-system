import * as React from 'react'
import { cn } from '@/lib/cn'

export interface BadgeProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: 'default' | 'secondary' | 'success' | 'warning' | 'destructive' | 'outline'
}

export function Badge({ className, variant = 'default', ...props }: BadgeProps) {
  return (
    <div
      className={cn(
        'inline-flex items-center rounded-sm border px-2 py-0.5 text-xs font-medium transition-colors',
        variant === 'default' && 'border-primary/20 bg-primary/10 text-primary',
        variant === 'secondary' && 'border-border bg-muted text-muted-foreground',
        variant === 'success' && 'border-accent/20 bg-accent/10 text-accent',
        variant === 'warning' && 'border-warning/20 bg-warning/10 text-warning',
        variant === 'destructive' && 'border-destructive/20 bg-destructive/10 text-destructive',
        variant === 'outline' && 'border-border bg-transparent text-foreground',
        className,
      )}
      {...props}
    />
  )
}
