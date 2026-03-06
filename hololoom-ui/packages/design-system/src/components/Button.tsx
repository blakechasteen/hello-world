/**
 * HoloLoom Design System - Button Component
 *
 * Primary interactive element with multiple variants, sizes, and states.
 * Supports cosmic glow effects and loading states.
 */

import React, { forwardRef } from 'react';
import { cn } from '../utils/cn';

// =============================================================================
// TYPES
// =============================================================================

export type ButtonVariant = 'primary' | 'secondary' | 'ghost' | 'danger' | 'cosmic' | 'default';
export type ButtonSize = 'xs' | 'sm' | 'md' | 'lg' | 'xl';

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  /** Visual variant */
  variant?: ButtonVariant;
  /** Size preset */
  size?: ButtonSize;
  /** Full width button */
  fullWidth?: boolean;
  /** Loading state */
  loading?: boolean;
  /** Icon to display before text */
  leftIcon?: React.ReactNode;
  /** Icon to display after text */
  rightIcon?: React.ReactNode;
  /** Render as child component (for composition) */
  asChild?: boolean;
}

// =============================================================================
// STYLES
// =============================================================================

const baseStyles = `
  inline-flex items-center justify-center gap-2
  font-medium rounded-lg
  transition-all duration-fast ease-out
  focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2
  disabled:opacity-50 disabled:pointer-events-none
  select-none
`;

const variantStyles: Record<ButtonVariant, string> = {
  primary: `
    bg-interactive-primary text-fg-inverse
    hover:bg-interactive-primary-hover
    focus-visible:ring-cosmic-nebula
    active:scale-[0.98]
  `,
  secondary: `
    bg-interactive-secondary text-fg-primary
    hover:bg-interactive-secondary-hover
    border border-border-primary
    focus-visible:ring-cosmic-nebula
    active:scale-[0.98]
  `,
  ghost: `
    bg-transparent text-fg-primary
    hover:bg-bg-secondary
    focus-visible:ring-cosmic-nebula
    active:scale-[0.98]
  `,
  danger: `
    bg-safety-danger text-white
    hover:bg-safety-critical
    focus-visible:ring-safety-danger
    active:scale-[0.98]
  `,
  cosmic: `
    bg-gradient-to-r from-cosmic-nebula via-cosmic-aurora to-cosmic-nebula
    bg-[length:200%_100%]
    text-white font-semibold
    hover:animate-aurora
    shadow-glow-nebula hover:shadow-glow-nebula-intense
    focus-visible:ring-cosmic-aurora
    active:scale-[0.98]
  `,
  default: `
    bg-interactive-primary text-fg-inverse
    hover:bg-interactive-primary-hover
    focus-visible:ring-cosmic-nebula
    active:scale-[0.98]
  `,
};

const sizeStyles: Record<ButtonSize, string> = {
  xs: 'h-7 px-2 text-xs gap-1',
  sm: 'h-8 px-3 text-sm gap-1.5',
  md: 'h-10 px-4 text-sm gap-2',
  lg: 'h-11 px-6 text-base gap-2',
  xl: 'h-12 px-8 text-lg gap-3',
};

// =============================================================================
// LOADING SPINNER
// =============================================================================

function LoadingSpinner({ size }: { size: ButtonSize }) {
  const spinnerSize = {
    xs: 'w-3 h-3',
    sm: 'w-3.5 h-3.5',
    md: 'w-4 h-4',
    lg: 'w-5 h-5',
    xl: 'w-6 h-6',
  }[size];

  return (
    <svg
      className={cn(spinnerSize, 'animate-spin')}
      xmlns="http://www.w3.org/2000/svg"
      fill="none"
      viewBox="0 0 24 24"
    >
      <circle
        className="opacity-25"
        cx="12"
        cy="12"
        r="10"
        stroke="currentColor"
        strokeWidth="4"
      />
      <path
        className="opacity-75"
        fill="currentColor"
        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
      />
    </svg>
  );
}

// =============================================================================
// COMPONENT
// =============================================================================

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  (
    {
      className,
      variant = 'primary',
      size = 'md',
      fullWidth = false,
      loading = false,
      leftIcon,
      rightIcon,
      disabled,
      children,
      ...props
    },
    ref
  ) => {
    return (
      <button
        ref={ref}
        className={cn(
          baseStyles,
          variantStyles[variant],
          sizeStyles[size],
          fullWidth && 'w-full',
          loading && 'cursor-wait',
          className
        )}
        disabled={disabled || loading}
        {...props}
      >
        {loading ? (
          <LoadingSpinner size={size} />
        ) : leftIcon ? (
          <span className="shrink-0">{leftIcon}</span>
        ) : null}
        <span className={cn(loading && 'opacity-0')}>{children}</span>
        {!loading && rightIcon && (
          <span className="shrink-0">{rightIcon}</span>
        )}
      </button>
    );
  }
);

Button.displayName = 'Button';
