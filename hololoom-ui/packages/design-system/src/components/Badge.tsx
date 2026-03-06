/**
 * HoloLoom Design System - Badge Component
 *
 * Generic badge for labels, tags, and status indicators.
 */

import React, { forwardRef } from 'react';
import { cn } from '../utils/cn';

// =============================================================================
// TYPES
// =============================================================================

export type BadgeVariant =
  | 'default'
  | 'secondary'
  | 'success'
  | 'warning'
  | 'danger'
  | 'info'
  | 'cosmic'
  | 'accent'
  | 'outline';

export type BadgeSize = 'sm' | 'md' | 'lg';

export interface BadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
  /** Badge variant */
  variant?: BadgeVariant;
  /** Size preset */
  size?: BadgeSize;
  /** Rounded pill style */
  pill?: boolean;
  /** Show dot indicator */
  dot?: boolean;
  /** Dot color override */
  dotColor?: string;
  /** Removable (shows X button) */
  removable?: boolean;
  /** Remove handler */
  onRemove?: () => void;
}

// =============================================================================
// STYLES
// =============================================================================

const variantStyles: Record<BadgeVariant, string> = {
  default: 'bg-bg-elevated text-fg-primary border-border-secondary',
  secondary: 'bg-bg-tertiary text-fg-secondary border-transparent',
  success: 'bg-confidence-high/10 text-confidence-high border-confidence-high/20',
  warning: 'bg-safety-warning-bg text-safety-warning border-safety-warning/20',
  danger: 'bg-safety-danger-bg text-safety-danger border-safety-danger/20',
  info: 'bg-cosmic-aurora/10 text-cosmic-aurora border-cosmic-aurora/20',
  cosmic: 'bg-gradient-to-r from-cosmic-nebula/20 to-cosmic-aurora/20 text-cosmic-starlight border-cosmic-nebula/30',
  accent: 'bg-cosmic-nebula/10 text-cosmic-nebula border-cosmic-nebula/20',
  outline: 'bg-transparent text-fg-primary border-border-primary',
};

const sizeStyles: Record<BadgeSize, string> = {
  sm: 'text-2xs px-1.5 py-0.5 gap-1',
  md: 'text-xs px-2 py-0.5 gap-1.5',
  lg: 'text-sm px-2.5 py-1 gap-2',
};

const dotSizes: Record<BadgeSize, string> = {
  sm: 'w-1 h-1',
  md: 'w-1.5 h-1.5',
  lg: 'w-2 h-2',
};

// =============================================================================
// COMPONENT
// =============================================================================

export const Badge = forwardRef<HTMLSpanElement, BadgeProps>(
  (
    {
      className,
      children,
      variant = 'default',
      size = 'md',
      pill = false,
      dot = false,
      dotColor,
      removable = false,
      onRemove,
      ...props
    },
    ref
  ) => {
    return (
      <span
        ref={ref}
        className={cn(
          'inline-flex items-center font-medium border',
          pill ? 'rounded-full' : 'rounded',
          variantStyles[variant],
          sizeStyles[size],
          className
        )}
        {...props}
      >
        {dot && (
          <span
            className={cn(
              'rounded-full flex-shrink-0',
              dotSizes[size],
              dotColor || 'bg-current'
            )}
          />
        )}
        {children}
        {removable && (
          <button
            type="button"
            className={cn(
              'flex-shrink-0 rounded-full p-0.5 -mr-0.5',
              'hover:bg-black/10 dark:hover:bg-white/10',
              'transition-colors duration-fast'
            )}
            onClick={(e) => {
              e.stopPropagation();
              onRemove?.();
            }}
            aria-label="Remove"
          >
            <svg
              className={cn(
                size === 'sm' && 'w-2.5 h-2.5',
                size === 'md' && 'w-3 h-3',
                size === 'lg' && 'w-3.5 h-3.5'
              )}
              viewBox="0 0 20 20"
              fill="currentColor"
            >
              <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
            </svg>
          </button>
        )}
      </span>
    );
  }
);

Badge.displayName = 'Badge';
