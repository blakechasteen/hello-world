/**
 * HoloLoom Design System - Design Tokens
 *
 * Complete design token system including:
 * - Colors (cosmic foundation + 3 themes)
 * - Typography (1.25 ratio scale)
 * - Spacing (8pt grid)
 * - Effects (shadows, borders, animations)
 */

// =============================================================================
// RE-EXPORTS
// =============================================================================

export * from './colors';
export * from './typography';
export * from './spacing';
export * from './effects';

// =============================================================================
// COMBINED CSS VARIABLES
// =============================================================================

import {
  getThemeCSSVariables,
  type ThemeName,
} from './colors';
import { typographyCSSVariables } from './typography';
import { spacingCSSVariables } from './spacing';
import { effectsCSSVariables } from './effects';

/**
 * Get all CSS custom properties for a theme
 */
export function getAllCSSVariables(theme: ThemeName = 'modern'): Record<string, string> {
  return {
    ...getThemeCSSVariables(theme),
    ...typographyCSSVariables,
    ...spacingCSSVariables,
    ...effectsCSSVariables,
  };
}

/**
 * Generate CSS string with all variables for a theme
 */
export function generateCSSVariables(theme: ThemeName = 'modern'): string {
  const vars = getAllCSSVariables(theme);
  const lines = Object.entries(vars)
    .map(([key, value]) => `  ${key}: ${value};`)
    .join('\n');

  return `:root {\n${lines}\n}`;
}

/**
 * Generate CSS for all themes with data-theme selectors
 */
export function generateAllThemeCSS(): string {
  const themes: ThemeName[] = ['tufte', 'modern', 'terminal'];

  // Base CSS variables (typography, spacing, effects - same for all themes)
  const baseVars = {
    ...typographyCSSVariables,
    ...spacingCSSVariables,
    ...effectsCSSVariables,
  };

  const baseLines = Object.entries(baseVars)
    .map(([key, value]) => `  ${key}: ${value};`)
    .join('\n');

  // Theme-specific CSS
  const themeCSSBlocks = themes.map(theme => {
    const themeVars = getThemeCSSVariables(theme);
    const lines = Object.entries(themeVars)
      .map(([key, value]) => `  ${key}: ${value};`)
      .join('\n');

    return `[data-theme="${theme}"] {\n${lines}\n}`;
  });

  // Default theme (modern)
  const defaultThemeVars = getThemeCSSVariables('modern');
  const defaultLines = Object.entries(defaultThemeVars)
    .map(([key, value]) => `  ${key}: ${value};`)
    .join('\n');

  return `/* HoloLoom Design System - Auto-generated CSS Variables */
/* Do not edit directly - regenerate with: npm run tokens:build */

:root {
${baseLines}

  /* Default theme: modern */
${defaultLines}
}

${themeCSSBlocks.join('\n\n')}
`;
}

// =============================================================================
// TOKEN TYPES
// =============================================================================

export type { ThemeName, Theme } from './colors';

export interface DesignTokens {
  colors: typeof import('./colors');
  typography: typeof import('./typography');
  spacing: typeof import('./spacing');
  effects: typeof import('./effects');
}

// =============================================================================
// THEME CONTEXT HELPERS
// =============================================================================

/**
 * Check if a theme name is valid
 */
export function isValidTheme(theme: string): theme is ThemeName {
  return ['tufte', 'modern', 'terminal'].includes(theme);
}

/**
 * Get the default theme based on user preferences
 */
export function getDefaultTheme(): ThemeName {
  if (typeof window === 'undefined') return 'modern';

  // Check for stored preference
  const stored = localStorage.getItem('hololoom-theme');
  if (stored && isValidTheme(stored)) return stored;

  // Check system preference
  if (window.matchMedia('(prefers-color-scheme: dark)').matches) {
    return 'terminal';
  }

  return 'modern';
}

/**
 * Apply a theme to the document
 */
export function applyTheme(theme: ThemeName): void {
  if (typeof document === 'undefined') return;

  document.documentElement.setAttribute('data-theme', theme);
  localStorage.setItem('hololoom-theme', theme);
}
