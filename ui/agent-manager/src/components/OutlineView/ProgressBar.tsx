/**
 * ProgressBar Component
 * Reusable single progress bar with smooth animations and overflow handling
 *
 * Features:
 * - Smooth CSS transitions for value changes
 * - Color-coded bars (blue, amber, purple, green, red)
 * - Overflow detection (shows red when value > max)
 * - Optional shimmer animation for "in progress" states
 * - Graceful handling of undefined max values
 * - Configurable size (sm=2px, md=4px, lg=8px)
 */

import React, { useMemo } from 'react';

export type ProgressColor = 'blue' | 'amber' | 'purple' | 'green' | 'red';
export type ProgressSize = 'sm' | 'md' | 'lg';

export interface ProgressBarProps {
  /** Current value */
  value: number;

  /** Maximum value (default: 100 for percentage mode) */
  max?: number;

  /** Color theme */
  color: ProgressColor;

  /** Size of the progress bar */
  size?: ProgressSize;

  /** Show percentage label to the right of bar */
  showLabel?: boolean;

  /** Custom label to show */
  label?: string;

  /** Custom formatter for display value */
  formatValue?: (value: number, max: number | undefined) => string;

  /** Enable shimmer animation for "in progress" state */
  animated?: boolean;

  /** CSS class name for the container */
  className?: string;
}

/**
 * Get color classes based on color prop and overflow state
 */
const getColorClasses = (
  color: ProgressColor,
  isOverflow: boolean
): {
  background: string;
  fill: string;
} => {
  if (isOverflow) {
    return {
      background: 'bg-slate-700',
      fill: 'bg-gradient-to-r from-red-500 to-red-400',
    };
  }

  switch (color) {
    case 'blue':
      return {
        background: 'bg-slate-700',
        fill: 'bg-gradient-to-r from-blue-500 to-cyan-500',
      };

    case 'amber':
      return {
        background: 'bg-slate-700',
        fill: 'bg-gradient-to-r from-amber-500 to-orange-500',
      };

    case 'purple':
      return {
        background: 'bg-slate-700',
        fill: 'bg-gradient-to-r from-purple-500 to-pink-500',
      };

    case 'green':
      return {
        background: 'bg-slate-700',
        fill: 'bg-gradient-to-r from-emerald-500 to-teal-500',
      };

    case 'red':
      return {
        background: 'bg-slate-700',
        fill: 'bg-gradient-to-r from-red-500 to-rose-500',
      };
  }
};

/**
 * Get size classes based on size prop
 */
const getSizeClasses = (size: ProgressSize): string => {
  switch (size) {
    case 'sm':
      return 'h-1';
    case 'md':
      return 'h-1.5';
    case 'lg':
      return 'h-2';
  }
};

/**
 * Format a number as a percentage
 */
const formatPercentage = (value: number, max: number | undefined): string => {
  if (!max) return `${Math.round(value)}`;
  const pct = Math.round((value / max) * 100);
  return `${Math.min(pct, 100)}%`;
};

/**
 * ProgressBar Component
 */
export const ProgressBar: React.FC<ProgressBarProps> = ({
  value,
  max = 100,
  color,
  size = 'md',
  showLabel = false,
  label,
  formatValue,
  animated = false,
  className = '',
}) => {
  // Calculate percentage clamped to 0-100 for display
  const percentage = useMemo(() => {
    if (!max) return 0;
    return Math.min((value / max) * 100, 100);
  }, [value, max]);

  // Detect overflow (value > max)
  const isOverflow = useMemo(() => {
    if (!max) return false;
    return value > max;
  }, [value, max]);

  // Generate display label
  const displayLabel = useMemo(() => {
    if (label) return label;
    if (!showLabel) return undefined;
    return formatValue ? formatValue(value, max) : formatPercentage(value, max);
  }, [label, showLabel, value, max, formatValue]);

  // Get color classes
  const colors = getColorClasses(color, isOverflow);
  const sizeClass = getSizeClasses(size);

  return (
    <div className={`flex items-center gap-2 ${className}`}>
      {/* Progress bar container */}
      <div className={`flex-1 min-w-0 ${sizeClass} ${colors.background} rounded-full overflow-hidden`}>
        {/* Progress fill */}
        <div
          className={`
            ${sizeClass} ${colors.fill}
            transition-all duration-300 ease-out
            ${animated ? 'animate-pulse' : ''}
          `}
          style={{
            width: `${percentage}%`,
          }}
          role="progressbar"
          aria-valuenow={Math.round(percentage)}
          aria-valuemin={0}
          aria-valuemax={100}
        />
      </div>

      {/* Label (optional) */}
      {displayLabel && (
        <span className="flex-shrink-0 text-xs font-mono text-slate-400 whitespace-nowrap">
          {displayLabel}
        </span>
      )}
    </div>
  );
};

ProgressBar.displayName = 'ProgressBar';

export default ProgressBar;
