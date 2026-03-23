'use client';

/** Renders markdown-like text content from a Jenny TEXT panel */
export function JennyTextContent({ text }: { text: string }) {
  return (
    <div className="prose prose-sm prose-invert max-w-none">
      <p className="text-sm text-fg-primary whitespace-pre-wrap">{text}</p>
    </div>
  );
}
