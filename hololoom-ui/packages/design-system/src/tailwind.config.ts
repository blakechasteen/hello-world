/**
 * HoloLoom Design System - Tailwind CSS Preset
 *
 * This preset extends Tailwind with HoloLoom's design tokens.
 * Apps should extend this preset rather than defining their own config.
 *
 * Usage in app's tailwind.config.ts:
 * ```ts
 * import hololoomPreset from '@hololoom/design-system/tailwind';
 *
 * export default {
 *   presets: [hololoomPreset],
 *   content: ['./src/** /*.{ts,tsx}'],
 * }
 * ```
 */

import type { Config } from 'tailwindcss';
import { cosmic, confidence, safety, memory, edgeTypes, reasoning } from './tokens/colors';
import { fontFamily, fontSize, lineHeight, fontWeight, letterSpacing } from './tokens/typography';
import { spacing, zIndex, layout } from './tokens/spacing';
import { borderRadius, shadows, duration, easing, animation, keyframes } from './tokens/effects';

const hololoomPreset: Partial<Config> = {
  theme: {
    extend: {
      // =======================================================================
      // COLORS
      // =======================================================================
      colors: {
        // Cosmic palette (always available)
        cosmic: {
          void: cosmic.void,
          'void-light': cosmic.voidLight,
          nebula: cosmic.nebula,
          'nebula-light': cosmic.nebulaLight,
          'nebula-dark': cosmic.nebulaDark,
          aurora: cosmic.aurora,
          'aurora-light': cosmic.auroraLight,
          'aurora-dark': cosmic.auroraDark,
          sun: cosmic.sun,
          'sun-light': cosmic.sunLight,
          'sun-dark': cosmic.sunDark,
          starlight: cosmic.starlight,
          'starlight-dim': cosmic.starlightDim,
          constellation: cosmic.constellation,
        },

        // Semantic colors
        confidence: {
          high: confidence.high,
          'high-light': confidence.highLight,
          medium: confidence.medium,
          'medium-light': confidence.mediumLight,
          low: confidence.low,
          'low-light': confidence.lowLight,
          unknown: confidence.unknown,
        },

        safety: {
          safe: safety.safe,
          'safe-bg': safety.safeBg,
          caution: safety.caution,
          'caution-bg': safety.cautionBg,
          warning: safety.warning,
          'warning-bg': safety.warningBg,
          danger: safety.danger,
          'danger-bg': safety.dangerBg,
          critical: safety.critical,
          'critical-bg': safety.criticalBg,
        },

        memory: {
          active: memory.active,
          'active-bg': memory.activeBg,
          recent: memory.recent,
          'recent-bg': memory.recentBg,
          dormant: memory.dormant,
          'dormant-bg': memory.dormantBg,
          hot: memory.hot,
          'hot-bg': memory.hotBg,
        },

        edge: {
          'is-a': edgeTypes.isA,
          uses: edgeTypes.uses,
          mentions: edgeTypes.mentions,
          'leads-to': edgeTypes.leadsTo,
          'part-of': edgeTypes.partOf,
          'in-time': edgeTypes.inTime,
          'occurred-at': edgeTypes.occurredAt,
        },

        reasoning: {
          direct: reasoning.direct,
          verify: reasoning.verify,
          research: reasoning.research,
          'plan-execute': reasoning.planExecute,
        },

        // Theme-aware colors (use CSS variables)
        bg: {
          primary: 'var(--bg-primary)',
          secondary: 'var(--bg-secondary)',
          tertiary: 'var(--bg-tertiary)',
          elevated: 'var(--bg-elevated)',
          overlay: 'var(--bg-overlay)',
        },

        fg: {
          primary: 'var(--fg-primary)',
          secondary: 'var(--fg-secondary)',
          tertiary: 'var(--fg-tertiary)',
          disabled: 'var(--fg-disabled)',
          inverse: 'var(--fg-inverse)',
        },

        border: {
          primary: 'var(--border-primary)',
          secondary: 'var(--border-secondary)',
          focus: 'var(--border-focus)',
          error: 'var(--border-error)',
        },

        interactive: {
          primary: 'var(--interactive-primary)',
          'primary-hover': 'var(--interactive-primary-hover)',
          secondary: 'var(--interactive-secondary)',
          'secondary-hover': 'var(--interactive-secondary-hover)',
        },
      },

      // =======================================================================
      // TYPOGRAPHY
      // =======================================================================
      fontFamily: {
        sans: [fontFamily.sans],
        mono: [fontFamily.mono],
        cosmic: [fontFamily.cosmic],
      },

      fontSize: {
        '2xs': fontSize['2xs'],
        xs: fontSize.xs,
        sm: fontSize.sm,
        base: fontSize.base,
        lg: fontSize.lg,
        xl: fontSize.xl,
        '2xl': fontSize['2xl'],
        '3xl': fontSize['3xl'],
        '4xl': fontSize['4xl'],
        '5xl': fontSize['5xl'],
        '6xl': fontSize['6xl'],
        '7xl': fontSize['7xl'],
      },

      lineHeight: {
        none: lineHeight.none,
        tight: lineHeight.tight,
        snug: lineHeight.snug,
        normal: lineHeight.normal,
        relaxed: lineHeight.relaxed,
        loose: lineHeight.loose,
      },

      fontWeight: {
        thin: fontWeight.thin,
        extralight: fontWeight.extralight,
        light: fontWeight.light,
        normal: fontWeight.normal,
        medium: fontWeight.medium,
        semibold: fontWeight.semibold,
        bold: fontWeight.bold,
        extrabold: fontWeight.extrabold,
        black: fontWeight.black,
      },

      letterSpacing: {
        tighter: letterSpacing.tighter,
        tight: letterSpacing.tight,
        normal: letterSpacing.normal,
        wide: letterSpacing.wide,
        wider: letterSpacing.wider,
        widest: letterSpacing.widest,
      },

      // =======================================================================
      // SPACING
      // =======================================================================
      spacing: {
        ...spacing,
        'sidebar-collapsed': layout.sidebarCollapsed,
        'sidebar-narrow': layout.sidebarNarrow,
        'sidebar-wide': layout.sidebarWide,
        'panel-sm': layout.panelSm,
        'panel-md': layout.panelMd,
        'panel-lg': layout.panelLg,
        'header-sm': layout.headerSm,
        'header-md': layout.headerMd,
        'header-lg': layout.headerLg,
      },

      maxWidth: {
        'container-sm': layout.containerSm,
        'container-md': layout.containerMd,
        'container-lg': layout.containerLg,
        'container-xl': layout.containerXl,
        'container-2xl': layout.container2xl,
        'modal-sm': layout.modalSm,
        'modal-md': layout.modalMd,
        'modal-lg': layout.modalLg,
        'modal-xl': layout.modalXl,
      },

      zIndex: {
        hide: String(zIndex.hide),
        base: String(zIndex.base),
        docked: String(zIndex.docked),
        dropdown: String(zIndex.dropdown),
        sticky: String(zIndex.sticky),
        banner: String(zIndex.banner),
        overlay: String(zIndex.overlay),
        modal: String(zIndex.modal),
        popover: String(zIndex.popover),
        skipLink: String(zIndex.skipLink),
        toast: String(zIndex.toast),
        tooltip: String(zIndex.tooltip),
      },

      // =======================================================================
      // EFFECTS
      // =======================================================================
      borderRadius: {
        none: borderRadius.none,
        sm: borderRadius.sm,
        md: borderRadius.md,
        DEFAULT: borderRadius.md,
        lg: borderRadius.lg,
        xl: borderRadius.xl,
        '2xl': borderRadius['2xl'],
        '3xl': borderRadius['3xl'],
        full: borderRadius.full,
      },

      boxShadow: {
        none: shadows.none,
        xs: shadows.xs,
        sm: shadows.sm,
        md: shadows.md,
        DEFAULT: shadows.md,
        lg: shadows.lg,
        xl: shadows.xl,
        '2xl': shadows['2xl'],
        inner: shadows.inner,
        // Glow effects
        'glow-nebula': shadows.glow.nebula,
        'glow-nebula-intense': shadows.glow.nebulaIntense,
        'glow-aurora': shadows.glow.aurora,
        'glow-aurora-intense': shadows.glow.auroraIntense,
        'glow-sun': shadows.glow.sun,
        'glow-sun-intense': shadows.glow.sunIntense,
        // Focus rings
        focus: shadows.focus.default,
        'focus-error': shadows.focus.error,
        'focus-success': shadows.focus.success,
      },

      transitionDuration: {
        instant: duration.instant,
        fast: duration.fast,
        normal: duration.normal,
        DEFAULT: duration.normal,
        slow: duration.slow,
        slower: duration.slower,
        slowest: duration.slowest,
      },

      transitionTimingFunction: {
        linear: easing.linear,
        'ease-in': easing.easeIn,
        'ease-out': easing.easeOut,
        'ease-in-out': easing.easeInOut,
        'weave-in': easing.weaveIn,
        'weave-out': easing.weaveOut,
        spring: easing.spring,
        cosmic: easing.cosmic,
      },

      keyframes: {
        'fade-in': keyframes.fadeIn,
        'fade-out': keyframes.fadeOut,
        'slide-up': keyframes.slideUp,
        'slide-down': keyframes.slideDown,
        'slide-left': keyframes.slideLeft,
        'slide-right': keyframes.slideRight,
        'scale-in': keyframes.scaleIn,
        'scale-out': keyframes.scaleOut,
        spin: keyframes.spin,
        pulse: keyframes.pulse,
        bounce: keyframes.bounce,
        shimmer: keyframes.shimmer,
        glow: keyframes.glow,
        float: keyframes.float,
        weave: keyframes.weave,
        aurora: keyframes.aurora,
        'dot-pulse': keyframes.dotPulse,
        'progress-indeterminate': keyframes.progressIndeterminate,
        'text-stream': keyframes.textStream,
        cursor: keyframes.cursor,
      },

      animation: {
        none: animation.none,
        'fade-in': animation.fadeIn,
        'fade-out': animation.fadeOut,
        'slide-up': animation.slideUp,
        'slide-down': animation.slideDown,
        'slide-left': animation.slideLeft,
        'slide-right': animation.slideRight,
        'scale-in': animation.scaleIn,
        'scale-out': animation.scaleOut,
        spin: animation.spin,
        pulse: animation.pulse,
        bounce: animation.bounce,
        shimmer: animation.shimmer,
        glow: animation.glow,
        float: animation.float,
        weave: animation.weave,
        aurora: animation.aurora,
      },
    },
  },

  plugins: [
    // Custom plugin for HoloLoom-specific utilities
    function({ addUtilities, addComponents, theme }: any) {
      // Text glow effects
      addUtilities({
        '.text-glow-nebula': {
          textShadow: '0 0 10px rgba(99, 102, 241, 0.5)',
        },
        '.text-glow-aurora': {
          textShadow: '0 0 10px rgba(34, 211, 238, 0.5)',
        },
        '.text-glow-sun': {
          textShadow: '0 0 10px rgba(251, 191, 36, 0.5)',
        },
      });

      // Confidence indicator utilities
      addUtilities({
        '.confidence-high': {
          '--confidence-color': theme('colors.confidence.high'),
        },
        '.confidence-medium': {
          '--confidence-color': theme('colors.confidence.medium'),
        },
        '.confidence-low': {
          '--confidence-color': theme('colors.confidence.low'),
        },
      });

      // Safety level utilities
      addUtilities({
        '.safety-safe': {
          '--safety-color': theme('colors.safety.safe'),
          '--safety-bg': theme('colors.safety.safe-bg'),
        },
        '.safety-caution': {
          '--safety-color': theme('colors.safety.caution'),
          '--safety-bg': theme('colors.safety.caution-bg'),
        },
        '.safety-warning': {
          '--safety-color': theme('colors.safety.warning'),
          '--safety-bg': theme('colors.safety.warning-bg'),
        },
        '.safety-danger': {
          '--safety-color': theme('colors.safety.danger'),
          '--safety-bg': theme('colors.safety.danger-bg'),
        },
        '.safety-critical': {
          '--safety-color': theme('colors.safety.critical'),
          '--safety-bg': theme('colors.safety.critical-bg'),
        },
      });

      // Thread component base styles
      addComponents({
        '.thread': {
          display: 'flex',
          alignItems: 'center',
          gap: theme('spacing.2'),
          padding: `${theme('spacing.2')} ${theme('spacing.3')}`,
          borderRadius: theme('borderRadius.lg'),
          backgroundColor: 'var(--bg-secondary)',
          transition: theme('transition.colors'),
          '&:hover': {
            backgroundColor: 'var(--bg-tertiary)',
          },
        },
        '.thread-active': {
          borderLeft: `3px solid ${theme('colors.cosmic.nebula')}`,
          backgroundColor: 'var(--bg-tertiary)',
        },
      });

      // Loom panel base styles
      addComponents({
        '.loom-panel': {
          backgroundColor: 'var(--bg-elevated)',
          borderRadius: theme('borderRadius.xl'),
          boxShadow: theme('boxShadow.lg'),
          overflow: 'hidden',
        },
        '.loom-panel-header': {
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: `${theme('spacing.4')} ${theme('spacing.6')}`,
          borderBottom: '1px solid var(--border-secondary)',
        },
        '.loom-panel-body': {
          padding: theme('spacing.6'),
        },
        '.loom-panel-footer': {
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'flex-end',
          gap: theme('spacing.3'),
          padding: `${theme('spacing.4')} ${theme('spacing.6')}`,
          borderTop: '1px solid var(--border-secondary)',
        },
      });

      // Streaming text cursor
      addUtilities({
        '.streaming-cursor': {
          display: 'inline-block',
          width: '2px',
          height: '1em',
          marginLeft: '2px',
          backgroundColor: 'var(--fg-primary)',
          animation: 'cursor 1s step-end infinite',
        },
      });
    },
  ],
};

export default hololoomPreset;
