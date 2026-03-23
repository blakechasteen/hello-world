'use client';

/** Renders code with syntax highlighting from a Jenny CODE panel */
export function JennyCodeContent({ code, language }: { code: string; language?: string }) {
  return (
    <div className="relative">
      {language && (
        <span className="absolute top-2 right-2 text-xs text-fg-tertiary bg-bg-tertiary px-2 py-0.5 rounded">
          {language}
        </span>
      )}
      <pre className="bg-bg-tertiary rounded-lg p-4 overflow-x-auto">
        <code className="text-sm text-fg-primary font-mono">{code}</code>
      </pre>
    </div>
  );
}
