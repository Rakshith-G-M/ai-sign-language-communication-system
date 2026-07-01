import { APP_NAME } from '@/lib/constants'
import { cn } from '@/lib/cn'

interface AppLogoProps {
  size?: 'sm' | 'md'
  showText?: boolean
  className?: string
}

export function AppLogo({ size = 'md', showText = true, className }: AppLogoProps) {
  const iconSize = size === 'sm' ? 'h-7 w-7 text-xs' : 'h-8 w-8 text-sm'

  return (
    <div className={cn('flex items-center gap-2.5', className)}>
      <div
        className={cn(
          'flex items-center justify-center rounded-lg bg-primary/10 font-semibold text-primary',
          iconSize,
        )}
        aria-hidden
      >
        SF
      </div>
      {showText && (
        <span className={cn('font-semibold tracking-tight', size === 'sm' ? 'text-sm' : 'text-base')}>
          {APP_NAME}
        </span>
      )}
    </div>
  )
}
