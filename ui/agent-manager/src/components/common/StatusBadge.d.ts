import React from 'react';
/**
 * StatusBadge Component
 * Small badge showing thread status with color-coded styling:
 * - idle: gray
 * - running: blue with pulse animation
 * - paused: amber
 * - completed: green
 * - failed: red
 * - cancelled: gray with strikethrough
 */
interface StatusBadgeProps {
    status: 'idle' | 'running' | 'paused' | 'completed' | 'failed' | 'cancelled';
    size?: 'sm' | 'md' | 'lg';
    showLabel?: boolean;
    className?: string;
}
export declare const StatusBadge: React.FC<StatusBadgeProps>;
/**
 * Compact Status Indicator (dot only)
 * Useful for inline status indicators in tables/lists
 */
interface StatusIndicatorProps {
    status: 'idle' | 'running' | 'paused' | 'completed' | 'failed' | 'cancelled';
    size?: 'sm' | 'md' | 'lg';
    className?: string;
}
export declare const StatusIndicator: React.FC<StatusIndicatorProps>;
/**
 * Status Badge Grid (for overview pages)
 * Shows multiple status counts in a grid
 */
interface StatusGridProps {
    idle?: number;
    running?: number;
    paused?: number;
    completed?: number;
    failed?: number;
    cancelled?: number;
    className?: string;
}
export declare const StatusGrid: React.FC<StatusGridProps>;
export default StatusBadge;
//# sourceMappingURL=StatusBadge.d.ts.map