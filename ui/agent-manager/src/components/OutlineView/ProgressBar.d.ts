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
import React from 'react';
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
 * ProgressBar Component
 */
export declare const ProgressBar: React.FC<ProgressBarProps>;
export default ProgressBar;
//# sourceMappingURL=ProgressBar.d.ts.map