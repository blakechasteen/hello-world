/**
 * HoloLoom Design System - Card Component
 *
 * Container component for grouping related content.
 * Supports different elevation levels and interactive states.
 */

import React, { forwardRef } from 'react';
import { cn } from '../utils/cn';

// =============================================================================
// TYPES
// =============================================================================

export type CardVariant = 'default' | 'outlined' | 'elevated' | 'cosmic' | 'interactive';
export type CardPadding = 'none' | 'sm' | 'md' | 'lg';

export interface CardProps extends React.HTMLAttributes<HTMLDivElement> {
  /** Visual variant */
  variant?: CardVariant;
  /** Padding preset */
  padding?: CardPadding;
  /** Interactive hover state */
  hoverable?: boolean;
  /** Clickable card */
  clickable?: boolean;
  /** As a link */
  as?: 'div' | 'article' | 'section';
}

// =============================================================================
// STYLES
// =============================================================================

const baseStyles = `
  rounded-xl
  transition-all duration-normal ease-weave-in
`;

const variantStyles: Record<CardVariant, string> = {
  default: `
    bg-bg-elevated
    border border-border-secondary
  `,
  outlined: `
    bg-transparent
    border border-border-primary
  `,
  elevated: `
    bg-bg-elevated
    shadow-lg
    border border-transparent
  `,
  cosmic: `
    bg-gradient-to-br from-cosmic-void-light to-cosmic-void
    border border-cosmic-nebula/20
    shadow-glow-nebula
  `,
  interactive: `
    bg-bg-elevated
    border border-border-secondary
    hover:border-cosmic-nebula/30
    hover:shadow-md
    cursor-pointer
  `,
};

const paddingStyles: Record<CardPadding, string> = {
  none: '',
  sm: 'p-3',
  md: 'p-4',
  lg: 'p-6',
};

const interactiveStyles = {
  hoverable: `
    hover:border-border-focus
    hover:shadow-md
  `,
  clickable: `
    cursor-pointer
    hover:border-border-focus
    hover:shadow-md
    active:scale-[0.99]
    focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cosmic-nebula
  `,
};

// =============================================================================
// COMPONENT
// =============================================================================

export const Card = forwardRef<HTMLDivElement, CardProps>(
  (
    {
      className,
      variant = 'default',
      padding = 'md',
      hoverable = false,
      clickable = false,
      as: Component = 'div',
      children,
      ...props
    },
    ref
  ) => {
    return (
      <Component
        ref={ref as any}
        className={cn(
          baseStyles,
          variantStyles[variant],
          paddingStyles[padding],
          hoverable && interactiveStyles.hoverable,
          clickable && interactiveStyles.clickable,
          className
        )}
        tabIndex={clickable ? 0 : undefined}
        role={clickable ? 'button' : undefined}
        {...props}
      >
        {children}
      </Component>
    );
  }
);

Card.displayName = 'Card';

// =============================================================================
// SUBCOMPONENTS
// =============================================================================

export interface CardHeaderProps extends Omit<React.HTMLAttributes<HTMLDivElement>, 'title'> {
  /** Title text */
  title?: React.ReactNode;
  /** Subtitle text */
  subtitle?: React.ReactNode;
  /** Action element (button, icon, etc.) */
  action?: React.ReactNode;
}

export const CardHeader = forwardRef<HTMLDivElement, CardHeaderProps>(
  ({ className, title, subtitle, action, children, ...props }, ref) => {
    return (
      <div
        ref={ref}
        className={cn(
          'flex items-start justify-between gap-4',
          'pb-4 border-b border-border-secondary',
          className
        )}
        {...props}
      >
        {(title || subtitle) ? (
          <div className="flex-1 min-w-0">
            {title && (
              <h3 className="text-lg font-semibold text-fg-primary truncate">
                {title}
              </h3>
            )}
            {subtitle && (
              <p className="mt-1 text-sm text-fg-secondary">
                {subtitle}
              </p>
            )}
          </div>
        ) : (
          children
        )}
        {action && (
          <div className="shrink-0">
            {action}
          </div>
        )}
      </div>
    );
  }
);

CardHeader.displayName = 'CardHeader';

export const CardBody = forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, children, ...props }, ref) => {
  return (
    <div
      ref={ref}
      className={cn('py-4', className)}
      {...props}
    >
      {children}
    </div>
  );
});

CardBody.displayName = 'CardBody';

export const CardFooter = forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, children, ...props }, ref) => {
  return (
    <div
      ref={ref}
      className={cn(
        'flex items-center justify-end gap-3',
        'pt-4 border-t border-border-secondary',
        className
      )}
      {...props}
    >
      {children}
    </div>
  );
});

CardFooter.displayName = 'CardFooter';
