'use client';

interface SourceItem {
  id: string;
  text: string;
  relevance: number;
  type?: string;
}

/** Renders source attribution list from a Jenny SOURCES panel */
export function JennySourcesContent({ sources }: { sources: SourceItem[] }) {
  return (
    <div className="space-y-2">
      {sources.map((source) => (
        <div
          key={source.id}
          className="flex items-start gap-3 p-3 rounded-lg bg-bg-secondary/50"
        >
          {/* Relevance bar */}
          <div className="flex-shrink-0 w-1 h-full rounded-full self-stretch"
            style={{
              backgroundColor: source.relevance > 0.8
                ? 'var(--color-confidence-high)'
                : source.relevance > 0.5
                ? 'var(--color-confidence-medium)'
                : 'var(--color-confidence-low)',
            }}
          />
          <div className="flex-1 min-w-0">
            <p className="text-sm text-fg-primary line-clamp-2">{source.text}</p>
            <div className="flex items-center gap-2 mt-1">
              <span className="text-xs text-fg-tertiary">
                {(source.relevance * 100).toFixed(0)}% relevant
              </span>
              {source.type && (
                <span className="text-xs text-fg-tertiary px-1.5 py-0.5 bg-bg-tertiary rounded">
                  {source.type}
                </span>
              )}
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}
