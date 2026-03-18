/**
 * HoloLoom Design System - Tooltip Component
 *
 * Simple CSS-only tooltip with positioning variants.
 * For complex tooltips, use Radix UI or similar.
 */

import React, { forwardRef } from 'react';
import { cn } from '../utils/cn';

// =============================================================================
// TYPES
// =============================================================================

export type TooltipPosition = 'top' | 'bottom' | 'left' | 'right';

export interface TooltipProps extends Omit<React.HTMLAttributes<HTMLDivElement>, 'content'> {
  /** Tooltip content */
  content: React.ReactNode;
  /** Position relative to trigger */
  position?: TooltipPosition;
  /** Delay before showing (ms) */
  delay?: number;
  /** Trigger element */
  children: React.ReactNode;
  /** Disable tooltip */
  disabled?: boolean;
}

// =============================================================================
// STYLES
// =============================================================================

const positionStyles: Record<TooltipPosition, { container: string; arrow: string }> = {
  top: {
    container: 'bottom-full left-1/2 -translate-x-1/2 mb-2',
    arrow: 'top-full left-1/2 -translate-x-1/2 border-t-bg-elevated border-x-transparent border-b-transparent',
  },
  bottom: {
    container: 'top-full left-1/2 -translate-x-1/2 mt-2',
    arrow: 'bottom-full left-1/2 -translate-x-1/2 border-b-bg-elevated border-x-transparent border-t-transparent',
  },
  left: {
    container: 'right-full top-1/2 -translate-y-1/2 mr-2',
    arrow: 'left-full top-1/2 -translate-y-1/2 border-l-bg-elevated border-y-transparent border-r-transparent',
  },
  right: {
    container: 'left-full top-1/2 -translate-y-1/2 ml-2',
    arrow: 'right-full top-1/2 -translate-y-1/2 border-r-bg-elevated border-y-transparent border-l-transparent',
  },
};

// =============================================================================
// COMPONENT
// =============================================================================

export const Tooltip = forwardRef<HTMLDivElement, TooltipProps>(
  (
    {
      className,
      children,
      content,
      position = 'top',
      delay = 200,
      disabled = false,
      ...props
    },
    ref
  ) => {
    const posStyles = positionStyles[position];

    if (disabled) {
      return <>{children}</>;
    }

    return (
      <div
        ref={ref}
        className={cn('relative inline-flex group', className)}
        {...props}
      >
        {/* Trigger */}
        {children}

        {/* Tooltip */}
        <div
          className={cn(
            'absolute z-tooltip pointer-events-none',
            'opacity-0 invisible',
            'group-hover:opacity-100 group-hover:visible',
            'group-focus-within:opacity-100 group-focus-within:visible',
            'transition-all duration-fast',
            posStyles.container
          )}
          style={{ transitionDelay: `${delay}ms` }}
          role="tooltip"
        >
          {/* Content */}
          <div className={cn(
            'px-2 py-1 rounded text-xs font-medium',
            'bg-bg-elevated text-fg-primary',
            'shadow-lg border border-border-secondary',
            'whitespace-nowrap'
          )}>
            {content}
          </div>

          {/* Arrow */}
          <div
            className={cn(
              'absolute w-0 h-0 border-4',
              posStyles.arrow
            )}
          />
        </div>
      </div>
    );
  }
);

Tooltip.displayName = 'Tooltip';

// =============================================================================
// INFO TOOLTIP (Common pattern)
// =============================================================================

export interface InfoTooltipProps extends Omit<TooltipProps, 'children'> {
  /** Icon size */
  size?: 'sm' | 'md' | 'lg';
}

const iconSizes = {
  sm: 'w-3 h-3',
  md: 'w-4 h-4',
  lg: 'w-5 h-5',
};

export const InfoTooltip = forwardRef<HTMLDivElement, InfoTooltipProps>(
  (
    {
      content,
      size = 'md',
      ...props
    },
    ref
  ) => {
    return (
      <Tooltip ref={ref} content={content} {...props}>
        <span className="inline-flex items-center justify-center text-fg-tertiary hover:text-fg-secondary cursor-help">
          <svg className={iconSizes[size]} viewBox="0 0 20 20" fill="currentColor">
            <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
          </svg>
        </span>
      </Tooltip>
    );
  }
);

InfoTooltip.displayName = 'InfoTooltip';
