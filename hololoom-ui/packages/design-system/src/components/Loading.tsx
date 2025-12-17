/**
 * HoloLoom Design System - Loading Components
 *
 * Various loading indicators for different contexts:
 * - LoadingSpinner: Circular spinner
 * - LoadingDots: Three-dot pulse animation
 * - StreamingCursor: Blinking cursor for streaming text
 */

import React, { forwardRef } from 'react';
import { cn } from '../utils/cn';

// =============================================================================
// LOADING SPINNER
// =============================================================================

export type SpinnerSize = 'xs' | 'sm' | 'md' | 'lg' | 'xl';

export interface LoadingSpinnerProps extends React.HTMLAttributes<HTMLDivElement> {
  /** Size preset */
  size?: SpinnerSize;
  /** Custom color class */
  color?: string;
  /** Show label */
  label?: string;
}

const spinnerSizes: Record<SpinnerSize, string> = {
  xs: 'w-3 h-3 border',
  sm: 'w-4 h-4 border-2',
  md: 'w-6 h-6 border-2',
  lg: 'w-8 h-8 border-2',
  xl: 'w-12 h-12 border-3',
};

export const LoadingSpinner = forwardRef<HTMLDivElement, LoadingSpinnerProps>(
  (
    {
      className,
      size = 'md',
      color = 'border-cosmic-nebula',
      label,
      ...props
    },
    ref
  ) => {
    return (
      <div
        ref={ref}
        className={cn('inline-flex items-center gap-2', className)}
        role="status"
        aria-label={label || 'Loading'}
        {...props}
      >
        <div
          className={cn(
            'rounded-full animate-spin',
            'border-bg-tertiary',
            spinnerSizes[size],
            color
          )}
          style={{
            borderTopColor: 'transparent',
            borderLeftColor: 'transparent',
          }}
        />
        {label && (
          <span className="text-sm text-fg-secondary">{label}</span>
        )}
      </div>
    );
  }
);

LoadingSpinner.displayName = 'LoadingSpinner';

// =============================================================================
// LOADING DOTS
// =============================================================================

export interface LoadingDotsProps extends React.HTMLAttributes<HTMLDivElement> {
  /** Size preset */
  size?: 'sm' | 'md' | 'lg';
  /** Custom color class */
  color?: string;
}

const dotSizes: Record<'sm' | 'md' | 'lg', string> = {
  sm: 'w-1 h-1',
  md: 'w-1.5 h-1.5',
  lg: 'w-2 h-2',
};

const dotGaps: Record<'sm' | 'md' | 'lg', string> = {
  sm: 'gap-1',
  md: 'gap-1.5',
  lg: 'gap-2',
};

export const LoadingDots = forwardRef<HTMLDivElement, LoadingDotsProps>(
  (
    {
      className,
      size = 'md',
      color = 'bg-cosmic-nebula',
      ...props
    },
    ref
  ) => {
    return (
      <div
        ref={ref}
        className={cn('inline-flex items-center', dotGaps[size], className)}
        role="status"
        aria-label="Loading"
        {...props}
      >
        {[0, 1, 2].map((i) => (
          <span
            key={i}
            className={cn(
              'rounded-full',
              dotSizes[size],
              color,
              'animate-[dotPulse_1.4s_ease-in-out_infinite]'
            )}
            style={{ animationDelay: `${i * 0.16}s` }}
          />
        ))}
      </div>
    );
  }
);

LoadingDots.displayName = 'LoadingDots';

// =============================================================================
// STREAMING CURSOR
// =============================================================================

export interface StreamingCursorProps extends React.HTMLAttributes<HTMLSpanElement> {
  /** Is currently streaming */
  streaming?: boolean;
  /** Custom color class */
  color?: string;
}

export const StreamingCursor = forwardRef<HTMLSpanElement, StreamingCursorProps>(
  (
    {
      className,
      streaming = true,
      color = 'bg-fg-primary',
      ...props
    },
    ref
  ) => {
    if (!streaming) return null;

    return (
      <span
        ref={ref}
        className={cn(
          'inline-block w-0.5 h-[1em] ml-0.5',
          color,
          'animate-[cursor_1s_step-end_infinite]',
          className
        )}
        aria-hidden="true"
        {...props}
      />
    );
  }
);

StreamingCursor.displayName = 'StreamingCursor';

// =============================================================================
// WEAVING INDICATOR (HoloLoom-specific)
// =============================================================================

export interface WeavingIndicatorProps extends React.HTMLAttributes<HTMLDivElement> {
  /** Current step in weaving cycle (0-8) */
  step?: number;
  /** Show step labels */
  showLabels?: boolean;
}

const weavingSteps = [
  'Loom Command',
  'Chrono Trigger',
  'Thread Selection',
  'Feature Extraction',
  'Warp Space',
  'Memory Retrieval',
  'Convergence',
  'Tool Execution',
  'Spacetime Fabric',
];

export const WeavingIndicator = forwardRef<HTMLDivElement, WeavingIndicatorProps>(
  (
    {
      className,
      step = 0,
      showLabels = false,
      ...props
    },
    ref
  ) => {
    return (
      <div
        ref={ref}
        className={cn('flex items-center gap-1', className)}
        role="progressbar"
        aria-valuenow={step}
        aria-valuemin={0}
        aria-valuemax={8}
        aria-label={`Weaving: ${weavingSteps[step]}`}
        {...props}
      >
        {weavingSteps.map((label, i) => (
          <div
            key={i}
            className="flex flex-col items-center"
          >
            <div
              className={cn(
                'w-2 h-2 rounded-full transition-all duration-300',
                i < step && 'bg-confidence-high',
                i === step && 'bg-cosmic-nebula animate-pulse scale-125',
                i > step && 'bg-bg-tertiary'
              )}
            />
            {showLabels && (
              <span className={cn(
                'text-2xs mt-1 whitespace-nowrap',
                i === step ? 'text-cosmic-nebula' : 'text-fg-tertiary'
              )}>
                {label}
              </span>
            )}
          </div>
        ))}
      </div>
    );
  }
);

WeavingIndicator.displayName = 'WeavingIndicator';
