'use client';

import { useState, useEffect, useCallback } from 'react';
import { Card, CardHeader, CardBody, Button, Badge } from '@hololoom/design-system';
import { useHoloLoom } from '@hololoom/api-client';
import type { JennySpecDTO, LifecycleStage } from '@hololoom/api-client';

interface JennyPanelShellProps {
  spec: JennySpecDTO;
  children: React.ReactNode;
  onAction?: (action: string) => void;
}

const LIFECYCLE_CLASSES: Record<LifecycleStage, string> = {
  nascent: 'animate-[fadeIn_300ms_ease-out]',
  stable: '',
  dissolving: 'animate-[fadeOut_300ms_ease-in] opacity-0',
  archived: 'hidden',
  system: '',
};

const SIZE_CLASSES: Record<NonNullable<JennySpecDTO['size']>, string> = {
  compact: 'max-w-sm',
  normal: '',
  expanded: 'col-span-2',
  full: 'col-span-full',
};

/**
 * Universal wrapper for any Jenny panel. Handles lifecycle animations,
 * action toolbar, streaming indicator, and accessibility attributes.
 *
 * Jenny decides WHAT to show (spec). This shell handles the HOW of
 * the surrounding chrome. The actual content is rendered by children
 * (dispatched from the panel registry).
 */
export function JennyPanelShell({ spec, children, onAction }: JennyPanelShellProps) {
  const { client } = useHoloLoom();
  const [visible, setVisible] = useState(true);

  // Handle dissolving lifecycle — hide after animation completes
  useEffect(() => {
    if (spec.lifecycle === 'dissolving') {
      const timer = setTimeout(() => setVisible(false), 300);
      return () => clearTimeout(timer);
    }
    if (spec.lifecycle === 'archived') {
      setVisible(false);
    }
    return undefined;
  }, [spec.lifecycle]);

  const handleAction = useCallback(async (actionId: string) => {
    onAction?.(actionId);

    // Fire action to backend
    try {
      await client.jennyAct(spec.spec_id, actionId);
    } catch {
      // Non-critical — action failed, panel remains functional
    }
  }, [client, spec.spec_id, onAction]);

  if (!visible) return null;

  const lifecycleClass = LIFECYCLE_CLASSES[spec.lifecycle] || '';
  const sizeClass = SIZE_CLASSES[spec.size || 'normal'] || '';

  return (
    <Card
      className={`${lifecycleClass} ${sizeClass} relative`}
      role="region"
      aria-label={spec.title}
    >
      <CardHeader className="flex items-center justify-between">
        <div className="flex items-center gap-2 min-w-0">
          <h3 className="text-sm font-semibold text-fg-primary truncate">
            {spec.title}
          </h3>
          {spec.subtitle && (
            <span className="text-xs text-fg-tertiary truncate hidden sm:inline">
              {spec.subtitle}
            </span>
          )}
          {spec.binding_mode === 'streaming' && (
            <span className="flex items-center gap-1">
              <span className="w-1.5 h-1.5 rounded-full bg-confidence-high animate-pulse" />
              <span className="text-xs text-fg-tertiary">live</span>
            </span>
          )}
          {spec.binding_mode === 'reactive' && (
            <Badge variant="secondary" size="sm">auto</Badge>
          )}
        </div>

        {/* Action toolbar */}
        {spec.actions.length > 0 && (
          <div className="flex items-center gap-1 flex-shrink-0">
            {spec.actions.map((action) => (
              <Button
                key={action.action_id}
                variant="ghost"
                size="sm"
                onClick={() => handleAction(action.action_id)}
                title={action.label}
              >
                {getActionIcon(action.type)}
              </Button>
            ))}
          </div>
        )}
      </CardHeader>

      <CardBody>
        {children}
      </CardBody>
    </Card>
  );
}

function getActionIcon(type: string): string {
  switch (type) {
    case 'pin': return '\u{1F4CC}';
    case 'dismiss': return '\u00D7';
    case 'why': return '?';
    case 'expand': return '\u2197';
    case 'copy': return '\u{1F4CB}';
    default: return '\u2022';
  }
}
