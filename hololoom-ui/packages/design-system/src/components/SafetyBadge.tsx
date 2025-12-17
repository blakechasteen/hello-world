/**
 * HoloLoom Design System - Safety Badge Component
 *
 * Visual indicator for safety/alignment risk levels.
 * Core component for the Alignment Framework UI.
 */

import React, { forwardRef } from 'react';
import { cn } from '../utils/cn';

// =============================================================================
// TYPES
// =============================================================================

export type SafetyLevel = 'safe' | 'caution' | 'warning' | 'danger' | 'critical';
export type SafetySize = 'sm' | 'md' | 'lg';

export interface SafetyBadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
  /** Safety level */
  level: SafetyLevel;
  /** Size preset */
  size?: SafetySize;
  /** Show icon */
  showIcon?: boolean;
  /** Show label */
  showLabel?: boolean;
  /** Custom label */
  label?: string;
  /** Pulsing animation for critical levels */
  pulse?: boolean;
}

// =============================================================================
// HELPERS
// =============================================================================

export function getSafetyLabel(level: SafetyLevel): string {
  switch (level) {
    case 'safe': return 'Safe';
    case 'caution': return 'Caution';
    case 'warning': return 'Warning';
    case 'danger': return 'Danger';
    case 'critical': return 'Critical';
    default: return 'Unknown';
  }
}

const levelStyles: Record<SafetyLevel, string> = {
  safe: 'bg-safety-safe-bg text-safety-safe border-safety-safe/30',
  caution: 'bg-safety-caution-bg text-safety-caution border-safety-caution/30',
  warning: 'bg-safety-warning-bg text-safety-warning border-safety-warning/30',
  danger: 'bg-safety-danger-bg text-safety-danger border-safety-danger/30',
  critical: 'bg-safety-critical-bg text-safety-critical border-safety-critical/30',
};

const sizeStyles: Record<SafetySize, string> = {
  sm: 'text-xs px-2 py-0.5 gap-1',
  md: 'text-sm px-2.5 py-1 gap-1.5',
  lg: 'text-base px-3 py-1.5 gap-2',
};

const iconSizes: Record<SafetySize, string> = {
  sm: 'w-3 h-3',
  md: 'w-4 h-4',
  lg: 'w-5 h-5',
};

// =============================================================================
// ICONS
// =============================================================================

function SafeIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 20 20" fill="currentColor">
      <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
    </svg>
  );
}

function CautionIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 20 20" fill="currentColor">
      <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
    </svg>
  );
}

function WarningIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 20 20" fill="currentColor">
      <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
    </svg>
  );
}

function DangerIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 20 20" fill="currentColor">
      <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
    </svg>
  );
}

function CriticalIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 20 20" fill="currentColor">
      <path fillRule="evenodd" d="M13.477 14.89A6 6 0 015.11 6.524l8.367 8.368zm1.414-1.414L6.524 5.11a6 6 0 008.367 8.367zM18 10a8 8 0 11-16 0 8 8 0 0116 0z" clipRule="evenodd" />
    </svg>
  );
}

const levelIcons: Record<SafetyLevel, React.FC<{ className?: string }>> = {
  safe: SafeIcon,
  caution: CautionIcon,
  warning: WarningIcon,
  danger: DangerIcon,
  critical: CriticalIcon,
};

// =============================================================================
// COMPONENT
// =============================================================================

export const SafetyBadge = forwardRef<HTMLSpanElement, SafetyBadgeProps>(
  (
    {
      className,
      level,
      size = 'md',
      showIcon = true,
      showLabel = true,
      label,
      pulse = false,
      ...props
    },
    ref
  ) => {
    const Icon = levelIcons[level];
    const displayLabel = label || getSafetyLabel(level);
    const shouldPulse = pulse || level === 'critical';

    return (
      <span
        ref={ref}
        className={cn(
          'inline-flex items-center font-medium rounded-full border',
          levelStyles[level],
          sizeStyles[size],
          shouldPulse && 'animate-pulse',
          className
        )}
        role="status"
        aria-label={`Safety level: ${displayLabel}`}
        {...props}
      >
        {showIcon && <Icon className={iconSizes[size]} />}
        {showLabel && <span>{displayLabel}</span>}
      </span>
    );
  }
);

SafetyBadge.displayName = 'SafetyBadge';
