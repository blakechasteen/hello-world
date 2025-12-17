/**
 * HoloLoom Design System - Typography Tokens
 *
 * Type scale based on 1.25 ratio (Major Third)
 * Base size: 16px
 *
 * Font families:
 * - Sans: Inter for UI text
 * - Mono: JetBrains Mono for code
 * - Cosmic: Cinzel for brand/mythological moments
 */

// =============================================================================
// FONT FAMILIES
// =============================================================================

export const fontFamily = {
  sans: "'Inter', 'SF Pro Display', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
  mono: "'JetBrains Mono', 'Fira Code', 'SF Mono', Monaco, 'Cascadia Code', monospace",
  cosmic: "'Cinzel', 'Cormorant Garamond', Georgia, 'Times New Roman', serif",
} as const;

// =============================================================================
// FONT SIZES (1.25 ratio - Major Third)
// =============================================================================

export const fontSize = {
  '2xs': '0.64rem',    // 10.24px - tiny labels
  xs: '0.8rem',        // 12.8px - captions
  sm: '0.875rem',      // 14px - small text
  base: '1rem',        // 16px - body text
  lg: '1.125rem',      // 18px - large body
  xl: '1.25rem',       // 20px - small headings
  '2xl': '1.5rem',     // 24px - h4
  '3xl': '1.875rem',   // 30px - h3
  '4xl': '2.25rem',    // 36px - h2
  '5xl': '3rem',       // 48px - h1
  '6xl': '3.75rem',    // 60px - display
  '7xl': '4.5rem',     // 72px - hero
} as const;

// =============================================================================
// LINE HEIGHTS
// =============================================================================

export const lineHeight = {
  none: '1',
  tight: '1.25',       // Headings
  snug: '1.375',       // Subheadings
  normal: '1.5',       // Body text
  relaxed: '1.625',    // Long-form reading
  loose: '2',          // Extra space
} as const;

// =============================================================================
// FONT WEIGHTS
// =============================================================================

export const fontWeight = {
  thin: '100',
  extralight: '200',
  light: '300',
  normal: '400',
  medium: '500',
  semibold: '600',
  bold: '700',
  extrabold: '800',
  black: '900',
} as const;

// =============================================================================
// LETTER SPACING
// =============================================================================

export const letterSpacing = {
  tighter: '-0.05em',
  tight: '-0.025em',
  normal: '0em',
  wide: '0.025em',
  wider: '0.05em',
  widest: '0.1em',
} as const;

// =============================================================================
// TEXT STYLES (Composite)
// =============================================================================

export const textStyles = {
  // Display styles (for hero sections, cosmic moments)
  displayLarge: {
    fontFamily: fontFamily.cosmic,
    fontSize: fontSize['7xl'],
    lineHeight: lineHeight.tight,
    fontWeight: fontWeight.bold,
    letterSpacing: letterSpacing.tight,
  },
  displayMedium: {
    fontFamily: fontFamily.cosmic,
    fontSize: fontSize['6xl'],
    lineHeight: lineHeight.tight,
    fontWeight: fontWeight.bold,
    letterSpacing: letterSpacing.tight,
  },
  displaySmall: {
    fontFamily: fontFamily.cosmic,
    fontSize: fontSize['5xl'],
    lineHeight: lineHeight.tight,
    fontWeight: fontWeight.semibold,
    letterSpacing: letterSpacing.tight,
  },

  // Heading styles
  h1: {
    fontFamily: fontFamily.sans,
    fontSize: fontSize['5xl'],
    lineHeight: lineHeight.tight,
    fontWeight: fontWeight.bold,
    letterSpacing: letterSpacing.tight,
  },
  h2: {
    fontFamily: fontFamily.sans,
    fontSize: fontSize['4xl'],
    lineHeight: lineHeight.tight,
    fontWeight: fontWeight.semibold,
    letterSpacing: letterSpacing.tight,
  },
  h3: {
    fontFamily: fontFamily.sans,
    fontSize: fontSize['3xl'],
    lineHeight: lineHeight.snug,
    fontWeight: fontWeight.semibold,
    letterSpacing: letterSpacing.normal,
  },
  h4: {
    fontFamily: fontFamily.sans,
    fontSize: fontSize['2xl'],
    lineHeight: lineHeight.snug,
    fontWeight: fontWeight.medium,
    letterSpacing: letterSpacing.normal,
  },
  h5: {
    fontFamily: fontFamily.sans,
    fontSize: fontSize.xl,
    lineHeight: lineHeight.snug,
    fontWeight: fontWeight.medium,
    letterSpacing: letterSpacing.normal,
  },
  h6: {
    fontFamily: fontFamily.sans,
    fontSize: fontSize.lg,
    lineHeight: lineHeight.normal,
    fontWeight: fontWeight.medium,
    letterSpacing: letterSpacing.normal,
  },

  // Body styles
  bodyLarge: {
    fontFamily: fontFamily.sans,
    fontSize: fontSize.lg,
    lineHeight: lineHeight.relaxed,
    fontWeight: fontWeight.normal,
    letterSpacing: letterSpacing.normal,
  },
  body: {
    fontFamily: fontFamily.sans,
    fontSize: fontSize.base,
    lineHeight: lineHeight.normal,
    fontWeight: fontWeight.normal,
    letterSpacing: letterSpacing.normal,
  },
  bodySmall: {
    fontFamily: fontFamily.sans,
    fontSize: fontSize.sm,
    lineHeight: lineHeight.normal,
    fontWeight: fontWeight.normal,
    letterSpacing: letterSpacing.normal,
  },

  // UI styles
  label: {
    fontFamily: fontFamily.sans,
    fontSize: fontSize.sm,
    lineHeight: lineHeight.normal,
    fontWeight: fontWeight.medium,
    letterSpacing: letterSpacing.wide,
  },
  labelSmall: {
    fontFamily: fontFamily.sans,
    fontSize: fontSize.xs,
    lineHeight: lineHeight.normal,
    fontWeight: fontWeight.medium,
    letterSpacing: letterSpacing.wider,
  },
  caption: {
    fontFamily: fontFamily.sans,
    fontSize: fontSize.xs,
    lineHeight: lineHeight.normal,
    fontWeight: fontWeight.normal,
    letterSpacing: letterSpacing.normal,
  },
  overline: {
    fontFamily: fontFamily.sans,
    fontSize: fontSize['2xs'],
    lineHeight: lineHeight.normal,
    fontWeight: fontWeight.semibold,
    letterSpacing: letterSpacing.widest,
    textTransform: 'uppercase' as const,
  },

  // Code styles
  code: {
    fontFamily: fontFamily.mono,
    fontSize: fontSize.sm,
    lineHeight: lineHeight.normal,
    fontWeight: fontWeight.normal,
    letterSpacing: letterSpacing.normal,
  },
  codeSmall: {
    fontFamily: fontFamily.mono,
    fontSize: fontSize.xs,
    lineHeight: lineHeight.normal,
    fontWeight: fontWeight.normal,
    letterSpacing: letterSpacing.normal,
  },
  codeBlock: {
    fontFamily: fontFamily.mono,
    fontSize: fontSize.sm,
    lineHeight: lineHeight.relaxed,
    fontWeight: fontWeight.normal,
    letterSpacing: letterSpacing.normal,
  },

  // Data/metrics styles (for dashboards)
  metric: {
    fontFamily: fontFamily.mono,
    fontSize: fontSize['3xl'],
    lineHeight: lineHeight.none,
    fontWeight: fontWeight.semibold,
    letterSpacing: letterSpacing.tight,
  },
  metricSmall: {
    fontFamily: fontFamily.mono,
    fontSize: fontSize.xl,
    lineHeight: lineHeight.none,
    fontWeight: fontWeight.medium,
    letterSpacing: letterSpacing.tight,
  },
  metricLabel: {
    fontFamily: fontFamily.sans,
    fontSize: fontSize['2xs'],
    lineHeight: lineHeight.normal,
    fontWeight: fontWeight.medium,
    letterSpacing: letterSpacing.wider,
    textTransform: 'uppercase' as const,
  },
} as const;

// =============================================================================
// CSS CUSTOM PROPERTIES EXPORT
// =============================================================================

export const typographyCSSVariables = {
  '--font-sans': fontFamily.sans,
  '--font-mono': fontFamily.mono,
  '--font-cosmic': fontFamily.cosmic,

  '--text-2xs': fontSize['2xs'],
  '--text-xs': fontSize.xs,
  '--text-sm': fontSize.sm,
  '--text-base': fontSize.base,
  '--text-lg': fontSize.lg,
  '--text-xl': fontSize.xl,
  '--text-2xl': fontSize['2xl'],
  '--text-3xl': fontSize['3xl'],
  '--text-4xl': fontSize['4xl'],
  '--text-5xl': fontSize['5xl'],
  '--text-6xl': fontSize['6xl'],
  '--text-7xl': fontSize['7xl'],

  '--leading-none': lineHeight.none,
  '--leading-tight': lineHeight.tight,
  '--leading-snug': lineHeight.snug,
  '--leading-normal': lineHeight.normal,
  '--leading-relaxed': lineHeight.relaxed,
  '--leading-loose': lineHeight.loose,

  '--tracking-tighter': letterSpacing.tighter,
  '--tracking-tight': letterSpacing.tight,
  '--tracking-normal': letterSpacing.normal,
  '--tracking-wide': letterSpacing.wide,
  '--tracking-wider': letterSpacing.wider,
  '--tracking-widest': letterSpacing.widest,
} as const;
