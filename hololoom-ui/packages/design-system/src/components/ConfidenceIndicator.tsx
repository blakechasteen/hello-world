/**
 * HoloLoom Design System - Confidence Indicator Component
 *
 * Visual representation of AI confidence levels.
 * Core component for transparency in AI decision-making.
 */

import React, { forwardRef, useMemo } from 'react';
import { cn } from '../utils/cn';

// =============================================================================
// TYPES
// =============================================================================

export type ConfidenceLevel = 'high' | 'medium' | 'low' | 'unknown';
export type ConfidenceVariant = 'bar' | 'ring' | 'badge' | 'dot';
export type ConfidenceSize = 'sm' | 'md' | 'lg';

export interface ConfidenceIndicatorProps extends React.HTMLAttributes<HTMLDivElement> {
  /** Confidence value 0-1 */
  value: number;
  /** Display variant */
  variant?: ConfidenceVariant;
  /** Size preset */
  size?: ConfidenceSize;
  /** Show numeric value */
  showValue?: boolean;
  /** Show label (High/Medium/Low) */
  showLabel?: boolean;
  /** Animate the indicator */
  animated?: boolean;
  /** Custom label */
  label?: string;
}

// =============================================================================
// HELPERS
// =============================================================================

function getConfidenceLevel(value: number): ConfidenceLevel {
  if (value >= 0.75) return 'high';
  if (value >= 0.5) return 'medium';
  if (value > 0) return 'low';
  return 'unknown';
}

function getConfidenceLabel(level: ConfidenceLevel): string {
  switch (level) {
    case 'high': return 'High Confidence';
    case 'medium': return 'Medium Confidence';
    case 'low': return 'Low Confidence';
    default: return 'Unknown';
  }
}

const levelColors: Record<ConfidenceLevel, string> = {
  high: 'bg-confidence-high',
  medium: 'bg-confidence-medium',
  low: 'bg-confidence-low',
  unknown: 'bg-confidence-unknown',
};

const levelTextColors: Record<ConfidenceLevel, string> = {
  high: 'text-confidence-high',
  medium: 'text-confidence-medium',
  low: 'text-confidence-low',
  unknown: 'text-confidence-unknown',
};

const levelBorderColors: Record<ConfidenceLevel, string> = {
  high: 'border-confidence-high',
  medium: 'border-confidence-medium',
  low: 'border-confidence-low',
  unknown: 'border-confidence-unknown',
};

// =============================================================================
// BAR VARIANT
// =============================================================================

function BarIndicator({
  value,
  level,
  size,
  animated,
  showValue,
}: {
  value: number;
  level: ConfidenceLevel;
  size: ConfidenceSize;
  animated: boolean;
  showValue: boolean;
}) {
  const heights: Record<ConfidenceSize, string> = {
    sm: 'h-1',
    md: 'h-2',
    lg: 'h-3',
  };

  return (
    <div className="flex items-center gap-2 w-full">
      <div className={cn(
        'flex-1 bg-bg-tertiary rounded-full overflow-hidden',
        heights[size]
      )}>
        <div
          className={cn(
            levelColors[level],
            'h-full rounded-full',
            animated && 'transition-all duration-slow ease-weave-in'
          )}
          style={{ width: `${Math.max(0, Math.min(100, value * 100))}%` }}
        />
      </div>
      {showValue && (
        <span className={cn(
          'text-sm font-mono font-medium tabular-nums',
          levelTextColors[level]
        )}>
          {Math.round(value * 100)}%
        </span>
      )}
    </div>
  );
}

// =============================================================================
// RING VARIANT
// =============================================================================

function RingIndicator({
  value,
  level,
  size,
  animated,
  showValue,
}: {
  value: number;
  level: ConfidenceLevel;
  size: ConfidenceSize;
  animated: boolean;
  showValue: boolean;
}) {
  const dimensions: Record<ConfidenceSize, { size: number; stroke: number }> = {
    sm: { size: 32, stroke: 3 },
    md: { size: 48, stroke: 4 },
    lg: { size: 64, stroke: 5 },
  };

  const { size: d, stroke } = dimensions[size];
  const radius = (d - stroke) / 2;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (value * circumference);

  const strokeColors: Record<ConfidenceLevel, string> = {
    high: 'stroke-confidence-high',
    medium: 'stroke-confidence-medium',
    low: 'stroke-confidence-low',
    unknown: 'stroke-confidence-unknown',
  };

  return (
    <div className="relative inline-flex items-center justify-center">
      <svg
        width={d}
        height={d}
        className="transform -rotate-90"
      >
        {/* Background circle */}
        <circle
          cx={d / 2}
          cy={d / 2}
          r={radius}
          strokeWidth={stroke}
          fill="none"
          className="stroke-bg-tertiary"
        />
        {/* Progress circle */}
        <circle
          cx={d / 2}
          cy={d / 2}
          r={radius}
          strokeWidth={stroke}
          fill="none"
          strokeLinecap="round"
          className={cn(
            strokeColors[level],
            animated && 'transition-all duration-slow ease-weave-in'
          )}
          style={{
            strokeDasharray: circumference,
            strokeDashoffset: offset,
          }}
        />
      </svg>
      {showValue && (
        <span className={cn(
          'absolute text-xs font-mono font-medium tabular-nums',
          levelTextColors[level]
        )}>
          {Math.round(value * 100)}
        </span>
      )}
    </div>
  );
}

// =============================================================================
// BADGE VARIANT
// =============================================================================

function BadgeIndicator({
  value,
  level,
  size,
  showValue,
  label,
}: {
  value: number;
  level: ConfidenceLevel;
  size: ConfidenceSize;
  showValue: boolean;
  label?: string;
}) {
  const sizeStyles: Record<ConfidenceSize, string> = {
    sm: 'text-xs px-2 py-0.5',
    md: 'text-sm px-2.5 py-1',
    lg: 'text-base px-3 py-1.5',
  };

  const levelBgColors: Record<ConfidenceLevel, string> = {
    high: 'bg-confidence-high/10',
    medium: 'bg-confidence-medium/10',
    low: 'bg-confidence-low/10',
    unknown: 'bg-confidence-unknown/10',
  };

  return (
    <span className={cn(
      'inline-flex items-center gap-1.5 rounded-full font-medium',
      sizeStyles[size],
      levelBgColors[level],
      levelTextColors[level]
    )}>
      <span className={cn(
        'w-1.5 h-1.5 rounded-full',
        levelColors[level]
      )} />
      {label || getConfidenceLabel(level)}
      {showValue && (
        <span className="font-mono tabular-nums">
          ({Math.round(value * 100)}%)
        </span>
      )}
    </span>
  );
}

// =============================================================================
// DOT VARIANT
// =============================================================================

function DotIndicator({
  level,
  size,
  animated,
}: {
  level: ConfidenceLevel;
  size: ConfidenceSize;
  animated: boolean;
}) {
  const sizeStyles: Record<ConfidenceSize, string> = {
    sm: 'w-2 h-2',
    md: 'w-3 h-3',
    lg: 'w-4 h-4',
  };

  return (
    <span
      className={cn(
        'inline-block rounded-full',
        sizeStyles[size],
        levelColors[level],
        animated && level !== 'unknown' && 'animate-pulse'
      )}
    />
  );
}

// =============================================================================
// COMPONENT
// =============================================================================

export const ConfidenceIndicator = forwardRef<HTMLDivElement, ConfidenceIndicatorProps>(
  (
    {
      className,
      value,
      variant = 'bar',
      size = 'md',
      showValue = false,
      showLabel = false,
      animated = true,
      label,
      ...props
    },
    ref
  ) => {
    const level = useMemo(() => getConfidenceLevel(value), [value]);
    const displayLabel = label || getConfidenceLabel(level);

    return (
      <div
        ref={ref}
        className={cn(
          'inline-flex items-center gap-2',
          className
        )}
        role="meter"
        aria-valuenow={Math.round(value * 100)}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-label={`Confidence: ${displayLabel} (${Math.round(value * 100)}%)`}
        {...props}
      >
        {variant === 'bar' && (
          <BarIndicator
            value={value}
            level={level}
            size={size}
            animated={animated}
            showValue={showValue}
          />
        )}
        {variant === 'ring' && (
          <RingIndicator
            value={value}
            level={level}
            size={size}
            animated={animated}
            showValue={showValue}
          />
        )}
        {variant === 'badge' && (
          <BadgeIndicator
            value={value}
            level={level}
            size={size}
            showValue={showValue}
            label={label}
          />
        )}
        {variant === 'dot' && (
          <DotIndicator
            level={level}
            size={size}
            animated={animated}
          />
        )}
        {showLabel && variant !== 'badge' && (
          <span className={cn(
            'text-sm font-medium',
            levelTextColors[level]
          )}>
            {displayLabel}
          </span>
        )}
      </div>
    );
  }
);

ConfidenceIndicator.displayName = 'ConfidenceIndicator';

// Export types for external use
export { getConfidenceLevel, getConfidenceLabel };
