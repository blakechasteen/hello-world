/**
 * HoloLoom Design System - Color Tokens
 *
 * Cosmic-infused color system with three switchable themes:
 * - Tufte: Data-dense, minimal decoration
 * - Modern: Clean SaaS aesthetic
 * - Terminal: Hacker/developer vibe
 *
 * All themes share the cosmic foundation overlay.
 */

// =============================================================================
// COSMIC FOUNDATION (Always Present)
// =============================================================================

export const cosmic = {
  void: '#0a0a0f',           // The deep background
  voidLight: '#12121a',      // Slightly lifted void
  nebula: '#6366f1',         // Indigo - the "thread" color
  nebulaLight: '#818cf8',    // Lighter nebula
  nebulaDark: '#4f46e5',     // Darker nebula
  aurora: '#22d3ee',         // Cyan - activation/energy
  auroraLight: '#67e8f9',    // Lighter aurora
  auroraDark: '#06b6d4',     // Darker aurora
  sun: '#fbbf24',            // Warmth, success, confidence
  sunLight: '#fcd34d',       // Lighter sun
  sunDark: '#f59e0b',        // Darker sun
  starlight: '#f8fafc',      // Pure white/light
  starlightDim: '#e2e8f0',   // Dimmed starlight
  constellation: '#94a3b8',  // Muted, guides
} as const;

// =============================================================================
// SEMANTIC COLORS
// =============================================================================

export const confidence = {
  high: '#22c55e',       // Green - confident
  highLight: '#4ade80',
  medium: '#eab308',     // Yellow - uncertain
  mediumLight: '#facc15',
  low: '#ef4444',        // Red - low confidence
  lowLight: '#f87171',
  unknown: '#6b7280',    // Gray - no data
} as const;

export const safety = {
  safe: '#22c55e',       // Green
  safeLight: '#4ade80',
  safeBg: 'rgba(34, 197, 94, 0.1)',
  caution: '#eab308',    // Yellow
  cautionLight: '#facc15',
  cautionBg: 'rgba(234, 179, 8, 0.1)',
  warning: '#f97316',    // Orange
  warningLight: '#fb923c',
  warningBg: 'rgba(249, 115, 22, 0.1)',
  danger: '#ef4444',     // Red
  dangerLight: '#f87171',
  dangerBg: 'rgba(239, 68, 68, 0.1)',
  critical: '#dc2626',   // Dark red
  criticalLight: '#ef4444',
  criticalBg: 'rgba(220, 38, 38, 0.1)',
} as const;

export const memory = {
  active: '#22d3ee',     // Cyan - currently active
  activeBg: 'rgba(34, 211, 238, 0.1)',
  recent: '#a78bfa',     // Purple - recently accessed
  recentBg: 'rgba(167, 139, 250, 0.1)',
  dormant: '#6b7280',    // Gray - inactive
  dormantBg: 'rgba(107, 114, 128, 0.1)',
  archived: '#374151',   // Dark gray - archived
  archivedBg: 'rgba(55, 65, 81, 0.1)',
  hot: '#f97316',        // Orange - hot pattern
  hotBg: 'rgba(249, 115, 22, 0.1)',
} as const;

export const edgeTypes = {
  isA: '#3b82f6',        // Blue - taxonomy
  uses: '#22c55e',       // Green - functional
  mentions: '#6b7280',   // Gray - reference
  leadsTo: '#f97316',    // Orange - causal
  partOf: '#a855f7',     // Purple - composition
  inTime: '#06b6d4',     // Cyan - temporal
  occurredAt: '#14b8a6', // Teal - event
} as const;

export const reasoning = {
  direct: '#3b82f6',     // Blue - direct mode
  verify: '#22c55e',     // Green - verify mode
  research: '#a855f7',   // Purple - research mode
  planExecute: '#f97316', // Orange - plan mode
} as const;

// =============================================================================
// THEME: TUFTE (Data-Dense, Minimal)
// =============================================================================

export const tufte = {
  // Backgrounds
  bg: {
    primary: '#fafaf9',      // Warm white
    secondary: '#f5f5f4',    // Stone 100
    tertiary: '#e7e5e4',     // Stone 200
    elevated: '#ffffff',     // Pure white for cards
    overlay: 'rgba(0, 0, 0, 0.4)',
  },
  // Foregrounds
  fg: {
    primary: '#1c1917',      // Stone 900 - main text
    secondary: '#44403c',    // Stone 700 - secondary text
    tertiary: '#78716c',     // Stone 500 - muted text
    disabled: '#a8a29e',     // Stone 400
    inverse: '#fafaf9',      // For dark backgrounds
  },
  // Borders
  border: {
    primary: '#d6d3d1',      // Stone 300
    secondary: '#e7e5e4',    // Stone 200
    focus: cosmic.nebula,
    error: safety.danger,
  },
  // Interactive
  interactive: {
    primary: '#292524',      // Stone 800
    primaryHover: '#1c1917', // Stone 900
    secondary: '#78716c',    // Stone 500
    secondaryHover: '#57534e', // Stone 600
  },
} as const;

// =============================================================================
// THEME: MODERN (Clean SaaS)
// =============================================================================

export const modern = {
  // Backgrounds
  bg: {
    primary: '#ffffff',      // Pure white
    secondary: '#f8fafc',    // Slate 50
    tertiary: '#f1f5f9',     // Slate 100
    elevated: '#ffffff',
    overlay: 'rgba(0, 0, 0, 0.5)',
  },
  // Foregrounds
  fg: {
    primary: '#0f172a',      // Slate 900
    secondary: '#334155',    // Slate 700
    tertiary: '#64748b',     // Slate 500
    disabled: '#94a3b8',     // Slate 400
    inverse: '#ffffff',
  },
  // Borders
  border: {
    primary: '#e2e8f0',      // Slate 200
    secondary: '#f1f5f9',    // Slate 100
    focus: cosmic.nebula,
    error: safety.danger,
  },
  // Interactive
  interactive: {
    primary: cosmic.nebula,
    primaryHover: cosmic.nebulaDark,
    secondary: '#e2e8f0',
    secondaryHover: '#cbd5e1',
  },
} as const;

// =============================================================================
// THEME: TERMINAL (Hacker/Developer)
// =============================================================================

export const terminal = {
  // Backgrounds
  bg: {
    primary: '#0a0a0f',      // Cosmic void
    secondary: '#12121a',    // Void light
    tertiary: '#1a1a24',     // Slightly lighter
    elevated: '#1e1e28',     // Cards
    overlay: 'rgba(0, 0, 0, 0.7)',
  },
  // Foregrounds
  fg: {
    primary: '#22d3ee',      // Cyan - primary text
    secondary: '#94a3b8',    // Muted
    tertiary: '#64748b',     // More muted
    disabled: '#475569',     // Slate 600
    inverse: '#0a0a0f',
  },
  // Borders
  border: {
    primary: '#2d2d3a',
    secondary: '#1e1e28',
    focus: cosmic.aurora,
    error: safety.danger,
  },
  // Interactive
  interactive: {
    primary: cosmic.aurora,
    primaryHover: cosmic.auroraLight,
    secondary: '#2d2d3a',
    secondaryHover: '#3d3d4a',
  },
  // Terminal-specific
  syntax: {
    keyword: '#c792ea',      // Purple
    string: '#c3e88d',       // Green
    number: '#f78c6c',       // Orange
    function: '#82aaff',     // Blue
    comment: '#546e7a',      // Gray
    operator: '#89ddff',     // Cyan
  },
} as const;

// =============================================================================
// THEME EXPORTS
// =============================================================================

export type ThemeName = 'tufte' | 'modern' | 'terminal';

export const themes = {
  tufte,
  modern,
  terminal,
} as const;

export type Theme = typeof tufte | typeof modern | typeof terminal;

// =============================================================================
// CSS CUSTOM PROPERTIES EXPORT
// =============================================================================

export function getThemeCSSVariables(theme: ThemeName): Record<string, string> {
  const t = themes[theme];
  return {
    // Cosmic (always present)
    '--cosmic-void': cosmic.void,
    '--cosmic-void-light': cosmic.voidLight,
    '--cosmic-nebula': cosmic.nebula,
    '--cosmic-nebula-light': cosmic.nebulaLight,
    '--cosmic-nebula-dark': cosmic.nebulaDark,
    '--cosmic-aurora': cosmic.aurora,
    '--cosmic-aurora-light': cosmic.auroraLight,
    '--cosmic-aurora-dark': cosmic.auroraDark,
    '--cosmic-sun': cosmic.sun,
    '--cosmic-sun-light': cosmic.sunLight,
    '--cosmic-sun-dark': cosmic.sunDark,
    '--cosmic-starlight': cosmic.starlight,
    '--cosmic-starlight-dim': cosmic.starlightDim,
    '--cosmic-constellation': cosmic.constellation,

    // Theme backgrounds
    '--bg-primary': t.bg.primary,
    '--bg-secondary': t.bg.secondary,
    '--bg-tertiary': t.bg.tertiary,
    '--bg-elevated': t.bg.elevated,
    '--bg-overlay': t.bg.overlay,

    // Theme foregrounds
    '--fg-primary': t.fg.primary,
    '--fg-secondary': t.fg.secondary,
    '--fg-tertiary': t.fg.tertiary,
    '--fg-disabled': t.fg.disabled,
    '--fg-inverse': t.fg.inverse,

    // Theme borders
    '--border-primary': t.border.primary,
    '--border-secondary': t.border.secondary,
    '--border-focus': t.border.focus,
    '--border-error': t.border.error,

    // Theme interactive
    '--interactive-primary': t.interactive.primary,
    '--interactive-primary-hover': t.interactive.primaryHover,
    '--interactive-secondary': t.interactive.secondary,
    '--interactive-secondary-hover': t.interactive.secondaryHover,

    // Semantic - confidence
    '--confidence-high': confidence.high,
    '--confidence-medium': confidence.medium,
    '--confidence-low': confidence.low,
    '--confidence-unknown': confidence.unknown,

    // Semantic - safety
    '--safety-safe': safety.safe,
    '--safety-safe-bg': safety.safeBg,
    '--safety-caution': safety.caution,
    '--safety-caution-bg': safety.cautionBg,
    '--safety-warning': safety.warning,
    '--safety-warning-bg': safety.warningBg,
    '--safety-danger': safety.danger,
    '--safety-danger-bg': safety.dangerBg,
    '--safety-critical': safety.critical,
    '--safety-critical-bg': safety.criticalBg,

    // Semantic - memory
    '--memory-active': memory.active,
    '--memory-active-bg': memory.activeBg,
    '--memory-recent': memory.recent,
    '--memory-recent-bg': memory.recentBg,
    '--memory-dormant': memory.dormant,
    '--memory-hot': memory.hot,
    '--memory-hot-bg': memory.hotBg,

    // Semantic - edges
    '--edge-is-a': edgeTypes.isA,
    '--edge-uses': edgeTypes.uses,
    '--edge-mentions': edgeTypes.mentions,
    '--edge-leads-to': edgeTypes.leadsTo,
    '--edge-part-of': edgeTypes.partOf,
    '--edge-in-time': edgeTypes.inTime,
    '--edge-occurred-at': edgeTypes.occurredAt,

    // Semantic - reasoning modes
    '--reasoning-direct': reasoning.direct,
    '--reasoning-verify': reasoning.verify,
    '--reasoning-research': reasoning.research,
    '--reasoning-plan-execute': reasoning.planExecute,
  };
}
