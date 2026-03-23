'use client';

import { useState } from 'react';

interface ReasoningStep {
  label: string;
  content: string;
  confidence?: number;
}

/** Renders step-by-step reasoning chain from a Jenny REASONING panel */
export function JennyReasoningContent({ steps }: { steps: ReasoningStep[] }) {
  const [expanded, setExpanded] = useState<Set<number>>(new Set());

  const toggle = (index: number) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      next.has(index) ? next.delete(index) : next.add(index);
      return next;
    });
  };

  return (
    <div className="space-y-2">
      {steps.map((step, i) => (
        <div key={i} className="border border-border-primary rounded-lg">
          <button
            onClick={() => toggle(i)}
            className="w-full flex items-center justify-between p-3 text-left hover:bg-bg-secondary/50 transition-colors"
          >
            <div className="flex items-center gap-2">
              <span className="text-xs text-fg-tertiary font-mono w-6">{i + 1}.</span>
              <span className="text-sm font-medium text-fg-primary">{step.label}</span>
            </div>
            {step.confidence !== undefined && (
              <span className="text-xs text-fg-tertiary">
                {(step.confidence * 100).toFixed(0)}%
              </span>
            )}
          </button>
          {expanded.has(i) && (
            <div className="px-3 pb-3 pl-11">
              <p className="text-sm text-fg-secondary whitespace-pre-wrap">{step.content}</p>
            </div>
          )}
        </div>
      ))}
    </div>
  );
}
