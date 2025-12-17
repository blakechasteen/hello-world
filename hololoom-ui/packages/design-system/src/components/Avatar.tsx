/**
 * HoloLoom Design System - Avatar Component
 *
 * User and AI avatar representations.
 * Supports images, initials, and HoloLoom AI avatar.
 */

import React, { forwardRef, useState } from 'react';
import { cn } from '../utils/cn';

// =============================================================================
// TYPES
// =============================================================================

export type AvatarSize = 'xs' | 'sm' | 'md' | 'lg' | 'xl' | '2xl';
export type AvatarVariant = 'user' | 'ai' | 'system';

export interface AvatarProps extends React.HTMLAttributes<HTMLDivElement> {
  /** Image source */
  src?: string;
  /** Alt text for image */
  alt?: string;
  /** Name for initials fallback */
  name?: string;
  /** Size preset */
  size?: AvatarSize;
  /** Avatar type */
  variant?: AvatarVariant;
  /** Show online indicator */
  showStatus?: boolean;
  /** Online status */
  online?: boolean;
  /** Thinking/processing state (for AI) */
  thinking?: boolean;
}

// =============================================================================
// HELPERS
// =============================================================================

function getInitials(name: string): string {
  const words = name.trim().split(/\s+/);
  if (words.length === 1) {
    return words[0].substring(0, 2).toUpperCase();
  }
  return (words[0][0] + words[words.length - 1][0]).toUpperCase();
}

const sizeStyles: Record<AvatarSize, { container: string; text: string; status: string }> = {
  xs: { container: 'w-6 h-6', text: 'text-2xs', status: 'w-1.5 h-1.5 ring-1' },
  sm: { container: 'w-8 h-8', text: 'text-xs', status: 'w-2 h-2 ring-1' },
  md: { container: 'w-10 h-10', text: 'text-sm', status: 'w-2.5 h-2.5 ring-2' },
  lg: { container: 'w-12 h-12', text: 'text-base', status: 'w-3 h-3 ring-2' },
  xl: { container: 'w-16 h-16', text: 'text-lg', status: 'w-3.5 h-3.5 ring-2' },
  '2xl': { container: 'w-20 h-20', text: 'text-xl', status: 'w-4 h-4 ring-2' },
};

const variantStyles: Record<AvatarVariant, string> = {
  user: 'bg-cosmic-nebula/20 text-cosmic-nebula',
  ai: 'bg-gradient-to-br from-cosmic-nebula via-cosmic-aurora to-cosmic-sun text-white',
  system: 'bg-bg-tertiary text-fg-secondary',
};

// =============================================================================
// AI LOGO SVG
// =============================================================================

function HoloLoomLogo({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 32 32" fill="none">
      {/* Weaving threads pattern */}
      <path
        d="M8 8 L24 24 M8 24 L24 8"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        opacity="0.5"
      />
      <path
        d="M16 4 L16 28 M4 16 L28 16"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        opacity="0.7"
      />
      {/* Center nebula */}
      <circle cx="16" cy="16" r="4" fill="currentColor" />
      {/* Outer ring */}
      <circle
        cx="16"
        cy="16"
        r="10"
        stroke="currentColor"
        strokeWidth="1.5"
        fill="none"
        opacity="0.8"
      />
    </svg>
  );
}

// =============================================================================
// COMPONENT
// =============================================================================

export const Avatar = forwardRef<HTMLDivElement, AvatarProps>(
  (
    {
      className,
      src,
      alt,
      name,
      size = 'md',
      variant = 'user',
      showStatus = false,
      online = false,
      thinking = false,
      ...props
    },
    ref
  ) => {
    const [imageError, setImageError] = useState(false);
    const sizes = sizeStyles[size];
    const showImage = src && !imageError;
    const showInitials = !showImage && name;
    const showAiLogo = variant === 'ai' && !showImage && !showInitials;

    return (
      <div
        ref={ref}
        className={cn(
          'relative inline-flex items-center justify-center rounded-full overflow-hidden',
          'font-medium select-none flex-shrink-0',
          sizes.container,
          !showImage && variantStyles[variant],
          thinking && 'animate-pulse',
          className
        )}
        {...props}
      >
        {showImage && (
          <img
            src={src}
            alt={alt || name || 'Avatar'}
            className="w-full h-full object-cover"
            onError={() => setImageError(true)}
          />
        )}

        {showInitials && (
          <span className={sizes.text}>
            {getInitials(name)}
          </span>
        )}

        {showAiLogo && (
          <HoloLoomLogo className="w-2/3 h-2/3" />
        )}

        {variant === 'system' && !showImage && !showInitials && (
          <svg className="w-1/2 h-1/2" viewBox="0 0 20 20" fill="currentColor">
            <path fillRule="evenodd" d="M11.49 3.17c-.38-1.56-2.6-1.56-2.98 0a1.532 1.532 0 01-2.286.948c-1.372-.836-2.942.734-2.106 2.106.54.886.061 2.042-.947 2.287-1.561.379-1.561 2.6 0 2.978a1.532 1.532 0 01.947 2.287c-.836 1.372.734 2.942 2.106 2.106a1.532 1.532 0 012.287.947c.379 1.561 2.6 1.561 2.978 0a1.533 1.533 0 012.287-.947c1.372.836 2.942-.734 2.106-2.106a1.533 1.533 0 01.947-2.287c1.561-.379 1.561-2.6 0-2.978a1.532 1.532 0 01-.947-2.287c.836-1.372-.734-2.942-2.106-2.106a1.532 1.532 0 01-2.287-.947zM10 13a3 3 0 100-6 3 3 0 000 6z" clipRule="evenodd" />
          </svg>
        )}

        {/* Status indicator */}
        {showStatus && (
          <span
            className={cn(
              'absolute bottom-0 right-0 rounded-full ring-bg-primary',
              sizes.status,
              online ? 'bg-confidence-high' : 'bg-fg-tertiary'
            )}
          />
        )}

        {/* Thinking indicator */}
        {thinking && variant === 'ai' && (
          <span className="absolute inset-0 rounded-full border-2 border-cosmic-aurora/50 animate-ping" />
        )}
      </div>
    );
  }
);

Avatar.displayName = 'Avatar';

// =============================================================================
// AVATAR GROUP
// =============================================================================

export interface AvatarGroupProps extends React.HTMLAttributes<HTMLDivElement> {
  /** Maximum avatars to show */
  max?: number;
  /** Size for all avatars */
  size?: AvatarSize;
  /** Children (Avatar components) */
  children: React.ReactNode;
}

export const AvatarGroup = forwardRef<HTMLDivElement, AvatarGroupProps>(
  (
    {
      className,
      children,
      max = 5,
      size = 'md',
      ...props
    },
    ref
  ) => {
    const avatars = React.Children.toArray(children);
    const visibleAvatars = avatars.slice(0, max);
    const remainingCount = avatars.length - max;

    return (
      <div
        ref={ref}
        className={cn('flex -space-x-2', className)}
        {...props}
      >
        {visibleAvatars.map((avatar, i) => (
          <div key={i} className="ring-2 ring-bg-primary rounded-full">
            {React.isValidElement(avatar)
              ? React.cloneElement(avatar as React.ReactElement<AvatarProps>, { size })
              : avatar}
          </div>
        ))}
        {remainingCount > 0 && (
          <Avatar
            size={size}
            variant="system"
            name={`+${remainingCount}`}
          />
        )}
      </div>
    );
  }
);

AvatarGroup.displayName = 'AvatarGroup';
