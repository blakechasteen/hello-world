import type { PanelType, JennySpecDTO } from '@hololoom/api-client';
import type { ComponentType } from 'react';

/**
 * Panel renderer protocol — each entry maps a Jenny PanelType to a React component
 * and an adapter function that transforms the JennySpec content into component props.
 *
 * The registry is extensible: add a new panel type = one set() call + one component.
 * Unknown panel types fall back to JennyFallback — never crash.
 */
export interface PanelRenderer<P = Record<string, unknown>> {
  component: ComponentType<P>;
  adapt: (spec: JennySpecDTO) => P;
}

// The registry — populated by JennyPanel.tsx which has access to React components
export const panelRegistry = new Map<PanelType | '_fallback', PanelRenderer<any>>();

/** Register a panel renderer for a given panel type */
export function registerPanel<P>(
  panelType: PanelType | '_fallback',
  renderer: PanelRenderer<P>
): void {
  panelRegistry.set(panelType, renderer);
}

/** Get the renderer for a panel type, falling back to the generic renderer */
export function getRenderer(panelType: PanelType): PanelRenderer {
  return panelRegistry.get(panelType) ?? panelRegistry.get('_fallback')!;
}
