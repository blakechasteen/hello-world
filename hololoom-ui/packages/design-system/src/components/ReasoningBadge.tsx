/**
 * HoloLoom Design System - Reasoning Badge Component
 *
 * Indicates the reasoning mode used by the agentic system.
 * Maps to the 4 reasoning modes: DIRECT, VERIFY, RESEARCH, PLAN_EXECUTE.
 */

import React, { forwardRef } from 'react';
import { cn } from '../utils/cn';

// =============================================================================
// TYPES
// =============================================================================

export type ReasoningMode = 'direct' | 'verify' | 'research' | 'plan_execute';

export type ReasoningBadgeSize = 'sm' | 'md' | 'lg';

export interface ReasoningBadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
  /** Reasoning mode */
  mode: ReasoningMode;
  /** Size variant */
  size?: ReasoningBadgeSize;
  /** Show icon */
  showIcon?: boolean;
  /** Show description */
  showDescription?: boolean;
  /** Compact mode (icon only) */
  compact?: boolean;
}

// =============================================================================
// CONFIG
// =============================================================================

const modeConfig: Record<ReasoningMode, {
  label: string;
  description: string;
  color: string;
  bgColor: string;
  icon: React.ReactNode;
}> = {
  direct: {
    label: 'Direct',
    description: 'Single-pass answer',
    color: 'text-reasoning-direct',
    bgColor: 'bg-reasoning-direct/10',
    icon: (
      <svg className="w-4 h-4" viewBox="0 0 20 20" fill="currentColor">
        <path fillRule="evenodd" d="M10.293 3.293a1 1 0 011.414 0l6 6a1 1 0 010 1.414l-6 6a1 1 0 01-1.414-1.414L14.586 11H3a1 1 0 110-2h11.586l-4.293-4.293a1 1 0 010-1.414z" clipRule="evenodd" />
      </svg>
    ),
  },
  verify: {
    label: 'Verify',
    description: 'Answer + verification',
    color: 'text-reasoning-verify',
    bgColor: 'bg-reasoning-verify/10',
    icon: (
      <svg className="w-4 h-4" viewBox="0 0 20 20" fill="currentColor">
        <path fillRule="evenodd" d="M6.267 3.455a3.066 3.066 0 001.745-.723 3.066 3.066 0 013.976 0 3.066 3.066 0 001.745.723 3.066 3.066 0 012.812 2.812c.051.643.304 1.254.723 1.745a3.066 3.066 0 010 3.976 3.066 3.066 0 00-.723 1.745 3.066 3.066 0 01-2.812 2.812 3.066 3.066 0 00-1.745.723 3.066 3.066 0 01-3.976 0 3.066 3.066 0 00-1.745-.723 3.066 3.066 0 01-2.812-2.812 3.066 3.066 0 00-.723-1.745 3.066 3.066 0 010-3.976 3.066 3.066 0 00.723-1.745 3.066 3.066 0 012.812-2.812zm7.44 5.252a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
      </svg>
    ),
  },
  research: {
    label: 'Research',
    description: 'Multi-query exploration',
    color: 'text-reasoning-research',
    bgColor: 'bg-reasoning-research/10',
    icon: (
      <svg className="w-4 h-4" viewBox="0 0 20 20" fill="currentColor">
        <path d="M9 4.804A7.968 7.968 0 005.5 4c-1.255 0-2.443.29-3.5.804v10A7.969 7.969 0 015.5 14c1.669 0 3.218.51 4.5 1.385A7.962 7.962 0 0114.5 14c1.255 0 2.443.29 3.5.804v-10A7.968 7.968 0 0014.5 4c-1.255 0-2.443.29-3.5.804V12a1 1 0 11-2 0V4.804z" />
      </svg>
    ),
  },
  plan_execute: {
    label: 'Plan & Execute',
    description: 'Goal decomposition',
    color: 'text-reasoning-plan-execute',
    bgColor: 'bg-reasoning-plan-execute/10',
    icon: (
      <svg className="w-4 h-4" viewBox="0 0 20 20" fill="currentColor">
        <path d="M5 3a2 2 0 00-2 2v2a2 2 0 002 2h2a2 2 0 002-2V5a2 2 0 00-2-2H5zM5 11a2 2 0 00-2 2v2a2 2 0 002 2h2a2 2 0 002-2v-2a2 2 0 00-2-2H5zM11 5a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V5zM14 11a1 1 0 011 1v1h1a1 1 0 110 2h-1v1a1 1 0 11-2 0v-1h-1a1 1 0 110-2h1v-1a1 1 0 011-1z" />
      </svg>
    ),
  },
};

// =============================================================================
// COMPONENT
// =============================================================================

export const ReasoningBadge = forwardRef<HTMLSpanElement, ReasoningBadgeProps>(
  (
    {
      className,
      mode,
      size = 'md',
      showIcon = true,
      showDescription = false,
      compact = false,
      ...props
    },
    ref
  ) => {
    const config = modeConfig[mode];

    const sizeClasses = {
      sm: 'px-1.5 py-0.5 text-xs',
      md: 'px-2.5 py-1 text-sm',
      lg: 'px-3 py-1.5 text-base',
    };

    if (compact) {
      return (
        <span
          ref={ref}
          className={cn(
            'inline-flex items-center justify-center w-6 h-6 rounded-full',
            config.bgColor,
            config.color,
            className
          )}
          title={`${config.label}: ${config.description}`}
          {...props}
        >
          {config.icon}
        </span>
      );
    }

    return (
      <span
        ref={ref}
        className={cn(
          'inline-flex items-center gap-1.5 rounded-full font-medium',
          sizeClasses[size],
          config.bgColor,
          config.color,
          className
        )}
        {...props}
      >
        {showIcon && config.icon}
        <span>{config.label}</span>
        {showDescription && (
          <span className="text-xs opacity-70">• {config.description}</span>
        )}
      </span>
    );
  }
);

ReasoningBadge.displayName = 'ReasoningBadge';
