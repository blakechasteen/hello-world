/**
 * HoloLoom Design System - Input Component
 *
 * Text input with variants for different states and contexts.
 * Supports icons, loading states, and validation.
 */

import React, { forwardRef } from 'react';
import { cn } from '../utils/cn';

// =============================================================================
// TYPES
// =============================================================================

export type InputSize = 'sm' | 'md' | 'lg';
export type InputVariant = 'default' | 'filled' | 'ghost';

export interface InputProps extends Omit<React.InputHTMLAttributes<HTMLInputElement>, 'size'> {
  /** Size preset */
  size?: InputSize;
  /** Visual variant */
  variant?: InputVariant;
  /** Error state */
  error?: boolean;
  /** Success state */
  success?: boolean;
  /** Loading state */
  loading?: boolean;
  /** Icon to display at start */
  startIcon?: React.ReactNode;
  /** Icon to display at end */
  endIcon?: React.ReactNode;
  /** Full width */
  fullWidth?: boolean;
}

// =============================================================================
// STYLES
// =============================================================================

const baseStyles = `
  w-full rounded-lg
  font-normal
  transition-all duration-fast ease-in-out
  placeholder:text-fg-tertiary
  focus:outline-none
  disabled:opacity-50 disabled:cursor-not-allowed
`;

const variantStyles: Record<InputVariant, string> = {
  default: `
    bg-bg-primary
    border border-border-primary
    hover:border-border-focus
    focus:border-border-focus focus:ring-2 focus:ring-border-focus/20
  `,
  filled: `
    bg-bg-secondary
    border border-transparent
    hover:bg-bg-tertiary
    focus:bg-bg-primary focus:border-border-focus focus:ring-2 focus:ring-border-focus/20
  `,
  ghost: `
    bg-transparent
    border border-transparent
    hover:bg-bg-secondary
    focus:bg-bg-secondary focus:border-border-focus
  `,
};

const sizeStyles: Record<InputSize, string> = {
  sm: 'h-8 px-3 text-sm',
  md: 'h-10 px-4 text-sm',
  lg: 'h-12 px-5 text-base',
};

const stateStyles = {
  error: `
    border-safety-danger
    focus:border-safety-danger focus:ring-safety-danger/20
    text-fg-primary
  `,
  success: `
    border-safety-safe
    focus:border-safety-safe focus:ring-safety-safe/20
    text-fg-primary
  `,
};

// =============================================================================
// LOADING SPINNER
// =============================================================================

function LoadingSpinner() {
  return (
    <svg
      className="w-4 h-4 animate-spin text-fg-tertiary"
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

export const Input = forwardRef<HTMLInputElement, InputProps>(
  (
    {
      className,
      size = 'md',
      variant = 'default',
      error = false,
      success = false,
      loading = false,
      startIcon,
      endIcon,
      fullWidth = true,
      disabled,
      ...props
    },
    ref
  ) => {
    const hasStartIcon = !!startIcon;
    const hasEndIcon = !!endIcon || loading;

    return (
      <div className={cn('relative', fullWidth ? 'w-full' : 'w-auto')}>
        {hasStartIcon && (
          <div className="absolute left-3 top-1/2 -translate-y-1/2 text-fg-tertiary pointer-events-none">
            {startIcon}
          </div>
        )}
        <input
          ref={ref}
          className={cn(
            baseStyles,
            variantStyles[variant],
            sizeStyles[size],
            error && stateStyles.error,
            success && !error && stateStyles.success,
            hasStartIcon && 'pl-10',
            hasEndIcon && 'pr-10',
            className
          )}
          disabled={disabled || loading}
          {...props}
        />
        {hasEndIcon && (
          <div className="absolute right-3 top-1/2 -translate-y-1/2 text-fg-tertiary">
            {loading ? <LoadingSpinner /> : endIcon}
          </div>
        )}
      </div>
    );
  }
);

Input.displayName = 'Input';

// =============================================================================
// TEXTAREA VARIANT
// =============================================================================

export interface TextareaProps extends React.TextareaHTMLAttributes<HTMLTextAreaElement> {
  /** Visual variant */
  variant?: InputVariant;
  /** Error state */
  error?: boolean;
  /** Success state */
  success?: boolean;
  /** Full width */
  fullWidth?: boolean;
}

export const Textarea = forwardRef<HTMLTextAreaElement, TextareaProps>(
  (
    {
      className,
      variant = 'default',
      error = false,
      success = false,
      fullWidth = true,
      ...props
    },
    ref
  ) => {
    return (
      <textarea
        ref={ref}
        className={cn(
          baseStyles,
          variantStyles[variant],
          'min-h-[80px] py-3 px-4 text-sm resize-y',
          error && stateStyles.error,
          success && !error && stateStyles.success,
          fullWidth ? 'w-full' : 'w-auto',
          className
        )}
        {...props}
      />
    );
  }
);

Textarea.displayName = 'Textarea';
