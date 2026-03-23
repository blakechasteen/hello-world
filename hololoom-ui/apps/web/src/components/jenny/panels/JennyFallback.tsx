'use client';

/** Generic fallback renderer for unknown Jenny panel types — never crash */
export function JennyFallback({ content }: { content: Record<string, unknown> }) {
  return (
    <div className="bg-bg-tertiary rounded-lg p-4">
      <pre className="text-xs text-fg-tertiary font-mono whitespace-pre-wrap overflow-x-auto">
        {JSON.stringify(content, null, 2)}
      </pre>
    </div>
  );
}
