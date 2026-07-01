import { memo } from 'react'
import { Loader2, Volume2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { cn } from '@/lib/cn'
import { useInferenceStore } from '@/features/inference/store/inferenceStore'

interface SentencePanelProps {
  onSpeak: () => void
  isSpeaking?: boolean
  onSelectSuggestion?: (word: string) => void
}

export const SentencePanel = memo(function SentencePanel({
  onSpeak,
  isSpeaking,
  onSelectSuggestion,
}: SentencePanelProps) {
  const word = useInferenceStore((s) => s.prediction.word)
  const sentence = useInferenceStore((s) => s.prediction.sentence)
  const suggestions = useInferenceStore((s) => s.prediction.suggestions)

  const displayText = sentence || 'Your sentence will appear here'
  const hasContent = Boolean(sentence.trim() || word.trim())

  return (
    <Card className="flex flex-col">
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-base">Sentence Builder</CardTitle>
        <Button
          variant="outline"
          size="sm"
          onClick={onSpeak}
          disabled={!sentence.trim() || isSpeaking}
          aria-label="Speak sentence"
        >
          {isSpeaking ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : (
            <Volume2 className="h-4 w-4" />
          )}
          <span className="hidden sm:inline">Speak</span>
        </Button>
      </CardHeader>
      <CardContent className="space-y-4">
        {word && (
          <div>
            <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
              Building
            </p>
            <p className="font-mono text-2xl font-medium" aria-live="polite">
              {word}
              <span className="animate-pulse text-primary">_</span>
            </p>
          </div>
        )}

        <div>
          <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
            Sentence
          </p>
          <p
            className={cn(
              'mt-1 text-base leading-relaxed',
              !hasContent && 'text-muted-foreground',
            )}
            aria-live="polite"
          >
            {displayText}
          </p>
        </div>

        {suggestions.length > 0 && (
          <div>
            <p className="mb-2 text-xs font-medium uppercase tracking-wide text-muted-foreground">
              Suggestions
            </p>
            <div className="flex flex-wrap gap-2">
              {suggestions.map((suggestion) => (
                <button
                  key={suggestion}
                  type="button"
                  onClick={() => onSelectSuggestion?.(suggestion)}
                  className="rounded-md border border-border bg-muted/50 px-2.5 py-1 text-sm transition-colors hover:bg-muted"
                >
                  {suggestion}
                </button>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
})
