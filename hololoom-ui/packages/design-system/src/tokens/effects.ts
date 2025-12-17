/**
 * HoloLoom Design System - Effects Tokens
 *
 * Shadows, borders, animations, and transitions
 */

// =============================================================================
// BORDER RADIUS
// =============================================================================

export const borderRadius = {
  none: '0',
  sm: '0.25rem',       // 4px
  md: '0.375rem',      // 6px
  lg: '0.5rem',        // 8px
  xl: '0.75rem',       // 12px
  '2xl': '1rem',       // 16px
  '3xl': '1.5rem',     // 24px
  full: '9999px',      // Pill/circle
} as const;

// =============================================================================
// BORDER WIDTH
// =============================================================================

export const borderWidth = {
  '0': '0',
  '1': '1px',
  '2': '2px',
  '4': '4px',
  '8': '8px',
} as const;

// =============================================================================
// SHADOWS
// =============================================================================

export const shadows = {
  none: 'none',
  xs: '0 1px 2px 0 rgba(0, 0, 0, 0.05)',
  sm: '0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px -1px rgba(0, 0, 0, 0.1)',
  md: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -2px rgba(0, 0, 0, 0.1)',
  lg: '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -4px rgba(0, 0, 0, 0.1)',
  xl: '0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1)',
  '2xl': '0 25px 50px -12px rgba(0, 0, 0, 0.25)',
  inner: 'inset 0 2px 4px 0 rgba(0, 0, 0, 0.05)',

  // Cosmic glow effects
  glow: {
    nebula: '0 0 20px rgba(99, 102, 241, 0.3)',
    nebulaIntense: '0 0 30px rgba(99, 102, 241, 0.5)',
    aurora: '0 0 20px rgba(34, 211, 238, 0.3)',
    auroraIntense: '0 0 30px rgba(34, 211, 238, 0.5)',
    sun: '0 0 20px rgba(251, 191, 36, 0.3)',
    sunIntense: '0 0 30px rgba(251, 191, 36, 0.5)',
  },

  // Focus rings
  focus: {
    default: '0 0 0 2px rgba(99, 102, 241, 0.5)',
    error: '0 0 0 2px rgba(239, 68, 68, 0.5)',
    success: '0 0 0 2px rgba(34, 197, 94, 0.5)',
  },
} as const;

// =============================================================================
// TRANSITIONS
// =============================================================================

export const duration = {
  instant: '0ms',
  fast: '100ms',
  normal: '200ms',
  slow: '300ms',
  slower: '500ms',
  slowest: '700ms',
} as const;

export const easing = {
  // Standard easings
  linear: 'linear',
  easeIn: 'cubic-bezier(0.4, 0, 1, 1)',
  easeOut: 'cubic-bezier(0, 0, 0.2, 1)',
  easeInOut: 'cubic-bezier(0.4, 0, 0.2, 1)',

  // Custom HoloLoom easings
  weaveIn: 'cubic-bezier(0.22, 1, 0.36, 1)',     // Smooth arrival
  weaveOut: 'cubic-bezier(0.64, 0, 0.78, 0)',    // Smooth departure
  spring: 'cubic-bezier(0.34, 1.56, 0.64, 1)',   // Bouncy
  cosmic: 'cubic-bezier(0.16, 1, 0.3, 1)',       // Ethereal, slow start
} as const;

export const transition = {
  none: 'none',
  all: `all ${duration.normal} ${easing.easeInOut}`,
  colors: `color ${duration.fast} ${easing.easeInOut}, background-color ${duration.fast} ${easing.easeInOut}, border-color ${duration.fast} ${easing.easeInOut}`,
  opacity: `opacity ${duration.fast} ${easing.easeInOut}`,
  transform: `transform ${duration.normal} ${easing.easeInOut}`,
  shadow: `box-shadow ${duration.normal} ${easing.easeInOut}`,

  // Component-specific
  button: `all ${duration.fast} ${easing.easeOut}`,
  input: `border-color ${duration.fast} ${easing.easeInOut}, box-shadow ${duration.fast} ${easing.easeInOut}`,
  card: `transform ${duration.normal} ${easing.weaveIn}, box-shadow ${duration.normal} ${easing.easeInOut}`,
  modal: `opacity ${duration.slow} ${easing.cosmic}, transform ${duration.slow} ${easing.weaveIn}`,
  toast: `all ${duration.normal} ${easing.spring}`,
} as const;

// =============================================================================
// KEYFRAME ANIMATIONS
// =============================================================================

export const keyframes = {
  fadeIn: {
    from: { opacity: '0' },
    to: { opacity: '1' },
  },
  fadeOut: {
    from: { opacity: '1' },
    to: { opacity: '0' },
  },
  slideUp: {
    from: { transform: 'translateY(10px)', opacity: '0' },
    to: { transform: 'translateY(0)', opacity: '1' },
  },
  slideDown: {
    from: { transform: 'translateY(-10px)', opacity: '0' },
    to: { transform: 'translateY(0)', opacity: '1' },
  },
  slideLeft: {
    from: { transform: 'translateX(10px)', opacity: '0' },
    to: { transform: 'translateX(0)', opacity: '1' },
  },
  slideRight: {
    from: { transform: 'translateX(-10px)', opacity: '0' },
    to: { transform: 'translateX(0)', opacity: '1' },
  },
  scaleIn: {
    from: { transform: 'scale(0.95)', opacity: '0' },
    to: { transform: 'scale(1)', opacity: '1' },
  },
  scaleOut: {
    from: { transform: 'scale(1)', opacity: '1' },
    to: { transform: 'scale(0.95)', opacity: '0' },
  },
  spin: {
    from: { transform: 'rotate(0deg)' },
    to: { transform: 'rotate(360deg)' },
  },
  pulse: {
    '0%, 100%': { opacity: '1' },
    '50%': { opacity: '0.5' },
  },
  bounce: {
    '0%, 100%': { transform: 'translateY(-5%)', animationTimingFunction: 'cubic-bezier(0.8, 0, 1, 1)' },
    '50%': { transform: 'translateY(0)', animationTimingFunction: 'cubic-bezier(0, 0, 0.2, 1)' },
  },

  // Cosmic animations
  shimmer: {
    '0%': { backgroundPosition: '-200% 0' },
    '100%': { backgroundPosition: '200% 0' },
  },
  glow: {
    '0%, 100%': { boxShadow: '0 0 5px rgba(99, 102, 241, 0.2)' },
    '50%': { boxShadow: '0 0 20px rgba(99, 102, 241, 0.4)' },
  },
  float: {
    '0%, 100%': { transform: 'translateY(0)' },
    '50%': { transform: 'translateY(-5px)' },
  },
  weave: {
    '0%': { transform: 'translateX(0) scaleX(1)' },
    '50%': { transform: 'translateX(5px) scaleX(1.02)' },
    '100%': { transform: 'translateX(0) scaleX(1)' },
  },
  aurora: {
    '0%': { backgroundPosition: '0% 50%' },
    '50%': { backgroundPosition: '100% 50%' },
    '100%': { backgroundPosition: '0% 50%' },
  },

  // Loading animations
  dotPulse: {
    '0%, 80%, 100%': { transform: 'scale(0)' },
    '40%': { transform: 'scale(1)' },
  },
  progressIndeterminate: {
    '0%': { transform: 'translateX(-100%)' },
    '100%': { transform: 'translateX(200%)' },
  },

  // Streaming text animation
  textStream: {
    from: { width: '0', opacity: '0' },
    to: { width: '100%', opacity: '1' },
  },
  cursor: {
    '0%, 100%': { opacity: '1' },
    '50%': { opacity: '0' },
  },
} as const;

export const animation = {
  none: 'none',
  fadeIn: `fadeIn ${duration.normal} ${easing.easeOut}`,
  fadeOut: `fadeOut ${duration.normal} ${easing.easeIn}`,
  slideUp: `slideUp ${duration.normal} ${easing.weaveIn}`,
  slideDown: `slideDown ${duration.normal} ${easing.weaveIn}`,
  slideLeft: `slideLeft ${duration.normal} ${easing.weaveIn}`,
  slideRight: `slideRight ${duration.normal} ${easing.weaveIn}`,
  scaleIn: `scaleIn ${duration.normal} ${easing.weaveIn}`,
  scaleOut: `scaleOut ${duration.fast} ${easing.easeIn}`,
  spin: `spin 1s ${easing.linear} infinite`,
  pulse: `pulse 2s ${easing.easeInOut} infinite`,
  bounce: `bounce 1s infinite`,
  shimmer: `shimmer 2s ${easing.linear} infinite`,
  glow: `glow 2s ${easing.easeInOut} infinite`,
  float: `float 3s ${easing.easeInOut} infinite`,
  weave: `weave 2s ${easing.cosmic} infinite`,
  aurora: `aurora 6s ${easing.easeInOut} infinite`,
} as const;

// =============================================================================
// CSS CUSTOM PROPERTIES EXPORT
// =============================================================================

export const effectsCSSVariables = {
  '--radius-none': borderRadius.none,
  '--radius-sm': borderRadius.sm,
  '--radius-md': borderRadius.md,
  '--radius-lg': borderRadius.lg,
  '--radius-xl': borderRadius.xl,
  '--radius-2xl': borderRadius['2xl'],
  '--radius-full': borderRadius.full,

  '--shadow-xs': shadows.xs,
  '--shadow-sm': shadows.sm,
  '--shadow-md': shadows.md,
  '--shadow-lg': shadows.lg,
  '--shadow-xl': shadows.xl,
  '--shadow-2xl': shadows['2xl'],
  '--shadow-inner': shadows.inner,
  '--shadow-glow-nebula': shadows.glow.nebula,
  '--shadow-glow-aurora': shadows.glow.aurora,
  '--shadow-focus': shadows.focus.default,

  '--duration-fast': duration.fast,
  '--duration-normal': duration.normal,
  '--duration-slow': duration.slow,

  '--ease-in': easing.easeIn,
  '--ease-out': easing.easeOut,
  '--ease-in-out': easing.easeInOut,
  '--ease-weave-in': easing.weaveIn,
  '--ease-weave-out': easing.weaveOut,
  '--ease-spring': easing.spring,
  '--ease-cosmic': easing.cosmic,

  '--transition-all': transition.all,
  '--transition-colors': transition.colors,
  '--transition-opacity': transition.opacity,
  '--transition-transform': transition.transform,
} as const;
