/**
 * HoloLoom Design System - Theme Provider
 *
 * React context provider for theme management.
 * Handles theme switching, persistence, and system preference detection.
 */

import React, {
  createContext,
  useContext,
  useEffect,
  useState,
  useCallback,
  useMemo,
} from 'react';
import type { ThemeName } from '../tokens/colors';

// =============================================================================
// TYPES
// =============================================================================

interface ThemeContextValue {
  /** Current active theme */
  theme: ThemeName;
  /** Set the theme */
  setTheme: (theme: ThemeName) => void;
  /** Toggle between light (modern/tufte) and dark (terminal) */
  toggleDarkMode: () => void;
  /** Whether the current theme is dark */
  isDark: boolean;
  /** Available themes */
  themes: ThemeName[];
  /** System preference for color scheme */
  systemPreference: 'light' | 'dark';
}

interface ThemeProviderProps {
  /** Initial theme (defaults to system preference or 'modern') */
  defaultTheme?: ThemeName;
  /** Persist theme to localStorage */
  persist?: boolean;
  /** Storage key for persistence */
  storageKey?: string;
  /** Force a specific theme (ignores user preference) */
  forcedTheme?: ThemeName;
  /** Children components */
  children: React.ReactNode;
}

// =============================================================================
// CONTEXT
// =============================================================================

const ThemeContext = createContext<ThemeContextValue | undefined>(undefined);

// =============================================================================
// CONSTANTS
// =============================================================================

const THEMES: ThemeName[] = ['tufte', 'modern', 'terminal'];
const DARK_THEMES: ThemeName[] = ['terminal'];
const DEFAULT_STORAGE_KEY = 'hololoom-theme';

// =============================================================================
// HELPERS
// =============================================================================

function getSystemPreference(): 'light' | 'dark' {
  if (typeof window === 'undefined') return 'light';
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

function getStoredTheme(key: string): ThemeName | null {
  if (typeof window === 'undefined') return null;
  const stored = localStorage.getItem(key);
  if (stored && THEMES.includes(stored as ThemeName)) {
    return stored as ThemeName;
  }
  return null;
}

function setStoredTheme(key: string, theme: ThemeName): void {
  if (typeof window === 'undefined') return;
  localStorage.setItem(key, theme);
}

function applyThemeToDocument(theme: ThemeName): void {
  if (typeof document === 'undefined') return;

  // Set data-theme attribute
  document.documentElement.setAttribute('data-theme', theme);

  // Update meta theme-color for mobile browsers
  const isDark = DARK_THEMES.includes(theme);
  const metaThemeColor = document.querySelector('meta[name="theme-color"]');
  if (metaThemeColor) {
    metaThemeColor.setAttribute('content', isDark ? '#0a0a0f' : '#ffffff');
  }
}

// =============================================================================
// PROVIDER COMPONENT
// =============================================================================

export function ThemeProvider({
  defaultTheme,
  persist = true,
  storageKey = DEFAULT_STORAGE_KEY,
  forcedTheme,
  children,
}: ThemeProviderProps): React.ReactElement {
  // Determine initial theme
  const [theme, setThemeState] = useState<ThemeName>(() => {
    if (forcedTheme) return forcedTheme;
    if (typeof window === 'undefined') return defaultTheme || 'modern';

    // Check stored preference
    if (persist) {
      const stored = getStoredTheme(storageKey);
      if (stored) return stored;
    }

    // Check system preference
    const systemPref = getSystemPreference();
    if (defaultTheme) return defaultTheme;
    return systemPref === 'dark' ? 'terminal' : 'modern';
  });

  const [systemPreference, setSystemPreference] = useState<'light' | 'dark'>(
    () => getSystemPreference()
  );

  // Apply theme on mount and changes
  useEffect(() => {
    applyThemeToDocument(forcedTheme || theme);
  }, [theme, forcedTheme]);

  // Persist theme changes
  useEffect(() => {
    if (persist && !forcedTheme) {
      setStoredTheme(storageKey, theme);
    }
  }, [theme, persist, storageKey, forcedTheme]);

  // Listen for system preference changes
  useEffect(() => {
    if (typeof window === 'undefined') return;

    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');

    const handleChange = (e: MediaQueryListEvent) => {
      setSystemPreference(e.matches ? 'dark' : 'light');
    };

    mediaQuery.addEventListener('change', handleChange);
    return () => mediaQuery.removeEventListener('change', handleChange);
  }, []);

  // Theme setters
  const setTheme = useCallback((newTheme: ThemeName) => {
    if (forcedTheme) return; // Ignore if theme is forced
    if (THEMES.includes(newTheme)) {
      setThemeState(newTheme);
    }
  }, [forcedTheme]);

  const toggleDarkMode = useCallback(() => {
    if (forcedTheme) return;
    setThemeState(current => {
      const isDark = DARK_THEMES.includes(current);
      return isDark ? 'modern' : 'terminal';
    });
  }, [forcedTheme]);

  // Memoized context value
  const value = useMemo<ThemeContextValue>(() => ({
    theme: forcedTheme || theme,
    setTheme,
    toggleDarkMode,
    isDark: DARK_THEMES.includes(forcedTheme || theme),
    themes: THEMES,
    systemPreference,
  }), [theme, setTheme, toggleDarkMode, forcedTheme, systemPreference]);

  return (
    <ThemeContext.Provider value={value}>
      {children}
    </ThemeContext.Provider>
  );
}

// =============================================================================
// HOOK
// =============================================================================

/**
 * Hook to access theme context
 */
export function useTheme(): ThemeContextValue {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
}

/**
 * Hook to get CSS variable value (client-side only)
 */
export function useCSSVariable(variableName: string): string {
  const [value, setValue] = useState('');

  useEffect(() => {
    if (typeof window === 'undefined') return;

    const computedStyle = getComputedStyle(document.documentElement);
    setValue(computedStyle.getPropertyValue(variableName).trim());

    // Re-compute on theme change
    const observer = new MutationObserver(() => {
      const newValue = getComputedStyle(document.documentElement)
        .getPropertyValue(variableName)
        .trim();
      setValue(newValue);
    });

    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ['data-theme'],
    });

    return () => observer.disconnect();
  }, [variableName]);

  return value;
}

// =============================================================================
// EXPORTS
// =============================================================================

export type { ThemeName, ThemeContextValue, ThemeProviderProps };
export { THEMES, DARK_THEMES };
