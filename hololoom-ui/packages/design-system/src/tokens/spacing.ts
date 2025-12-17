/**
 * HoloLoom Design System - Spacing Tokens
 *
 * 8pt grid system for consistent spacing
 * Base unit: 4px (0.25rem)
 */

// =============================================================================
// SPACING SCALE (Based on 4px unit)
// =============================================================================

export const spacing = {
  '0': '0',
  px: '1px',
  '0.5': '0.125rem',   // 2px
  '1': '0.25rem',      // 4px
  '1.5': '0.375rem',   // 6px
  '2': '0.5rem',       // 8px - base unit
  '2.5': '0.625rem',   // 10px
  '3': '0.75rem',      // 12px
  '3.5': '0.875rem',   // 14px
  '4': '1rem',         // 16px - 2x base
  '5': '1.25rem',      // 20px
  '6': '1.5rem',       // 24px - 3x base
  '7': '1.75rem',      // 28px
  '8': '2rem',         // 32px - 4x base
  '9': '2.25rem',      // 36px
  '10': '2.5rem',      // 40px - 5x base
  '11': '2.75rem',     // 44px
  '12': '3rem',        // 48px - 6x base
  '14': '3.5rem',      // 56px
  '16': '4rem',        // 64px - 8x base
  '20': '5rem',        // 80px
  '24': '6rem',        // 96px
  '28': '7rem',        // 112px
  '32': '8rem',        // 128px
  '36': '9rem',        // 144px
  '40': '10rem',       // 160px
  '44': '11rem',       // 176px
  '48': '12rem',       // 192px
  '52': '13rem',       // 208px
  '56': '14rem',       // 224px
  '60': '15rem',       // 240px
  '64': '16rem',       // 256px
  '72': '18rem',       // 288px
  '80': '20rem',       // 320px
  '96': '24rem',       // 384px
} as const;

// =============================================================================
// SEMANTIC SPACING
// =============================================================================

export const inset = {
  none: spacing['0'],
  xs: spacing['1'],        // 4px
  sm: spacing['2'],        // 8px
  md: spacing['4'],        // 16px
  lg: spacing['6'],        // 24px
  xl: spacing['8'],        // 32px
  '2xl': spacing['12'],    // 48px
} as const;

export const stack = {
  none: spacing['0'],
  xs: spacing['1'],        // 4px
  sm: spacing['2'],        // 8px
  md: spacing['4'],        // 16px
  lg: spacing['6'],        // 24px
  xl: spacing['8'],        // 32px
  '2xl': spacing['12'],    // 48px
  '3xl': spacing['16'],    // 64px
} as const;

export const inline = {
  none: spacing['0'],
  xs: spacing['1'],        // 4px
  sm: spacing['2'],        // 8px
  md: spacing['3'],        // 12px
  lg: spacing['4'],        // 16px
  xl: spacing['6'],        // 24px
} as const;

// =============================================================================
// COMPONENT SPACING
// =============================================================================

export const component = {
  // Button padding
  buttonPaddingXs: { x: spacing['2'], y: spacing['1'] },
  buttonPaddingSm: { x: spacing['3'], y: spacing['1.5'] },
  buttonPaddingMd: { x: spacing['4'], y: spacing['2'] },
  buttonPaddingLg: { x: spacing['6'], y: spacing['3'] },

  // Input padding
  inputPaddingSm: { x: spacing['2'], y: spacing['1.5'] },
  inputPaddingMd: { x: spacing['3'], y: spacing['2'] },
  inputPaddingLg: { x: spacing['4'], y: spacing['3'] },

  // Card padding
  cardPaddingSm: spacing['3'],
  cardPaddingMd: spacing['4'],
  cardPaddingLg: spacing['6'],

  // Panel padding
  panelPaddingSm: spacing['4'],
  panelPaddingMd: spacing['6'],
  panelPaddingLg: spacing['8'],

  // List item spacing
  listItemGap: spacing['2'],
  listItemPadding: { x: spacing['3'], y: spacing['2'] },

  // Icon spacing
  iconGapSm: spacing['1'],
  iconGapMd: spacing['2'],
  iconGapLg: spacing['3'],
} as const;

// =============================================================================
// LAYOUT
// =============================================================================

export const layout = {
  // Container max widths
  containerSm: '640px',
  containerMd: '768px',
  containerLg: '1024px',
  containerXl: '1280px',
  container2xl: '1536px',

  // Sidebar widths
  sidebarCollapsed: '64px',
  sidebarNarrow: '240px',
  sidebarWide: '320px',

  // Panel widths
  panelSm: '320px',
  panelMd: '400px',
  panelLg: '480px',

  // Modal widths
  modalSm: '400px',
  modalMd: '560px',
  modalLg: '720px',
  modalXl: '960px',

  // Header heights
  headerSm: '48px',
  headerMd: '56px',
  headerLg: '64px',

  // Footer heights
  footerSm: '48px',
  footerMd: '64px',
} as const;

// =============================================================================
// Z-INDEX
// =============================================================================

export const zIndex = {
  hide: -1,
  base: 0,
  docked: 10,
  dropdown: 1000,
  sticky: 1100,
  banner: 1200,
  overlay: 1300,
  modal: 1400,
  popover: 1500,
  skipLink: 1600,
  toast: 1700,
  tooltip: 1800,
} as const;

// =============================================================================
// CSS CUSTOM PROPERTIES EXPORT
// =============================================================================

export const spacingCSSVariables = {
  '--space-0': spacing['0'],
  '--space-px': spacing.px,
  '--space-0.5': spacing['0.5'],
  '--space-1': spacing['1'],
  '--space-1.5': spacing['1.5'],
  '--space-2': spacing['2'],
  '--space-3': spacing['3'],
  '--space-4': spacing['4'],
  '--space-5': spacing['5'],
  '--space-6': spacing['6'],
  '--space-8': spacing['8'],
  '--space-10': spacing['10'],
  '--space-12': spacing['12'],
  '--space-16': spacing['16'],
  '--space-20': spacing['20'],
  '--space-24': spacing['24'],

  '--inset-xs': inset.xs,
  '--inset-sm': inset.sm,
  '--inset-md': inset.md,
  '--inset-lg': inset.lg,
  '--inset-xl': inset.xl,

  '--container-sm': layout.containerSm,
  '--container-md': layout.containerMd,
  '--container-lg': layout.containerLg,
  '--container-xl': layout.containerXl,
  '--container-2xl': layout.container2xl,

  '--sidebar-collapsed': layout.sidebarCollapsed,
  '--sidebar-narrow': layout.sidebarNarrow,
  '--sidebar-wide': layout.sidebarWide,

  '--z-dropdown': String(zIndex.dropdown),
  '--z-sticky': String(zIndex.sticky),
  '--z-overlay': String(zIndex.overlay),
  '--z-modal': String(zIndex.modal),
  '--z-popover': String(zIndex.popover),
  '--z-toast': String(zIndex.toast),
  '--z-tooltip': String(zIndex.tooltip),
} as const;
