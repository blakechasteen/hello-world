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

export const StatusBadge: React.FC<StatusBadgeProps> = ({
  status,
  size = 'md',
  showLabel = true,
  className = '',
}) => {
  // Size classes
  const sizeClasses = {
    sm: 'px-2 py-1 text-xs',
    md: 'px-2.5 py-1.5 text-sm',
    lg: 'px-3 py-2 text-base',
  };

  // Status configuration: color, icon, label, animation
  const statusConfig = {
    idle: {
      bgColor: 'bg-slate-700',
      textColor: 'text-slate-200',
      icon: '○',
      label: 'Idle',
      animate: false,
    },
    running: {
      bgColor: 'bg-blue-600',
      textColor: 'text-white',
      icon: '▶',
      label: 'Running',
      animate: true,
    },
    paused: {
      bgColor: 'bg-amber-600',
      textColor: 'text-white',
      icon: '⏸',
      label: 'Paused',
      animate: false,
    },
    completed: {
      bgColor: 'bg-emerald-600',
      textColor: 'text-white',
      icon: '✓',
      label: 'Completed',
      animate: false,
    },
    failed: {
      bgColor: 'bg-red-600',
      textColor: 'text-white',
      icon: '✕',
      label: 'Failed',
      animate: false,
    },
    cancelled: {
      bgColor: 'bg-slate-700',
      textColor: 'text-slate-300 line-through',
      icon: '×',
      label: 'Cancelled',
      animate: false,
    },
  };

  const config = statusConfig[status];

  return (
    <span
      className={`
        inline-flex items-center gap-1.5 rounded-full font-semibold
        ${sizeClasses[size]}
        ${config.bgColor}
        ${config.textColor}
        ${config.animate ? 'animate-pulse' : ''}
        ${className}
      `}
    >
      <span className={`inline-block ${config.animate ? 'animate-bounce' : ''}`}>
        {config.icon}
      </span>
      {showLabel && <span>{config.label}</span>}
    </span>
  );
};

/**
 * Compact Status Indicator (dot only)
 * Useful for inline status indicators in tables/lists
 */
interface StatusIndicatorProps {
  status: 'idle' | 'running' | 'paused' | 'completed' | 'failed' | 'cancelled';
  size?: 'sm' | 'md' | 'lg';
  className?: string;
}

export const StatusIndicator: React.FC<StatusIndicatorProps> = ({
  status,
  size = 'md',
  className = '',
}) => {
  const sizeClasses = {
    sm: 'w-2 h-2',
    md: 'w-2.5 h-2.5',
    lg: 'w-3 h-3',
  };

  const colorClasses = {
    idle: 'bg-slate-500',
    running: 'bg-blue-500 animate-pulse',
    paused: 'bg-amber-500',
    completed: 'bg-emerald-500',
    failed: 'bg-red-500',
    cancelled: 'bg-slate-400',
  };

  return (
    <div
      className={`
        inline-flex rounded-full
        ${sizeClasses[size]}
        ${colorClasses[status]}
        ${className}
      `}
      title={status}
    />
  );
};

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

export const StatusGrid: React.FC<StatusGridProps> = ({
  idle = 0,
  running = 0,
  paused = 0,
  completed = 0,
  failed = 0,
  cancelled = 0,
  className = '',
}) => {
  const statuses = [
    { key: 'idle', value: idle, config: { bgColor: 'bg-slate-700', label: 'Idle' } },
    { key: 'running', value: running, config: { bgColor: 'bg-blue-600', label: 'Running' } },
    { key: 'paused', value: paused, config: { bgColor: 'bg-amber-600', label: 'Paused' } },
    { key: 'completed', value: completed, config: { bgColor: 'bg-emerald-600', label: 'Completed' } },
    { key: 'failed', value: failed, config: { bgColor: 'bg-red-600', label: 'Failed' } },
    { key: 'cancelled', value: cancelled, config: { bgColor: 'bg-slate-700', label: 'Cancelled' } },
  ];

  return (
    <div className={`grid grid-cols-3 gap-2 ${className}`}>
      {statuses.map(
        (status) =>
          status.value > 0 && (
            <div
              key={status.key}
              className={`${status.config.bgColor} rounded-lg px-3 py-2 text-center text-white`}
            >
              <div className="text-lg font-bold">{status.value}</div>
              <div className="text-xs opacity-90">{status.config.label}</div>
            </div>
          )
      )}
    </div>
  );
};

export default StatusBadge;
