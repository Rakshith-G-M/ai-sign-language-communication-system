import { Badge } from '@/components/ui/badge'

interface ModelBadgeProps {
  staticReady: boolean
  dynamicReady: boolean
}

export function ModelBadge({ staticReady, dynamicReady }: ModelBadgeProps) {
  if (staticReady && dynamicReady) {
    return <Badge variant="success">Static + Dynamic</Badge>
  }
  if (staticReady) {
    return <Badge variant="default">Static (A–Z)</Badge>
  }
  return <Badge variant="warning">Models loading</Badge>
}
