'use client';

import type { JennySpecDTO } from '@hololoom/api-client';
import { JennyPanelShell } from './JennyPanelShell';
import { registerPanel, getRenderer } from './panel-registry';

// Content components
import { JennyTextContent } from './panels/JennyTextContent';
import { JennyCodeContent } from './panels/JennyCodeContent';
import { JennyReasoningContent } from './panels/JennyReasoningContent';
import { JennySourcesContent } from './panels/JennySourcesContent';
import { JennyFallback } from './panels/JennyFallback';

// ---------------------------------------------------------------------------
// Register panel types → React components
// Adding a new panel type = one registerPanel() call + one component. No switch.
// ---------------------------------------------------------------------------

registerPanel('_fallback', {
  component: JennyFallback,
  adapt: (spec) => ({ content: spec.content }),
});

registerPanel('text', {
  component: JennyTextContent,
  adapt: (spec) => ({ text: String(spec.content.text ?? spec.content.value ?? '') }),
});

registerPanel('code', {
  component: JennyCodeContent,
  adapt: (spec) => ({
    code: String(spec.content.code ?? ''),
    language: spec.content.language as string | undefined,
  }),
});

registerPanel('reasoning', {
  component: JennyReasoningContent,
  adapt: (spec) => ({
    steps: Array.isArray(spec.content.steps) ? spec.content.steps : [],
  }),
});

registerPanel('sources', {
  component: JennySourcesContent,
  adapt: (spec) => ({
    sources: Array.isArray(spec.content.sources) ? spec.content.sources : [],
  }),
});

// Metric, confidence, graph, timeline — these will be wired to existing
// analytics components via adapters as those components are made compatible.
// For now they fall through to the generic fallback.

// ---------------------------------------------------------------------------
// JennyPanel — the public component
// ---------------------------------------------------------------------------

interface JennyPanelProps {
  spec: JennySpecDTO;
  onAction?: (action: string) => void;
}

/**
 * Renders a Jenny visualization panel.
 *
 * Jenny computes WHAT to show (spec). This component resolves HOW via the
 * panel registry — looking up the panel_type, adapting the spec content to
 * component props, and wrapping everything in JennyPanelShell for lifecycle,
 * actions, and accessibility.
 */
export function JennyPanel({ spec, onAction }: JennyPanelProps) {
  const renderer = getRenderer(spec.panel_type);
  const props = renderer.adapt(spec);

  return (
    <JennyPanelShell spec={spec} onAction={onAction}>
      <renderer.component {...props} />
    </JennyPanelShell>
  );
}
