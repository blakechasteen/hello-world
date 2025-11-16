/**
 * Theme Module for HoloLoom Documentation Site
 * Handles dark/light theme toggling with system preference detection and persistence
 *
 * Features:
 * - Dark/light theme toggle
 * - System preference detection (prefers-color-scheme)
 * - localStorage persistence (key: 'hololoom-theme')
 * - Apply theme by setting data-theme attribute on <html>
 * - Smooth CSS transitions between themes
 * - Initialize theme before page renders (no flash)
 * - Keyboard shortcut support (Ctrl+D or Cmd+D)
 * - WCAG AAA compliant color contrasts
 *
 * @module theme
 * @version 1.0.0
 */

(function() {
  'use strict';

  // ============================================
  // CONFIGURATION
  // ============================================

  const CONFIG = {
    // Storage key for localStorage
    storageKey: 'hololoom-theme',

    // Valid theme values
    themes: ['light', 'dark'],
    defaultTheme: 'light',

    // Selectors
    html: 'html',
    themeToggle: '#themeToggle',
    themeButtons: '[data-theme-toggle]',

    // CSS class for transitions
    transitionClass: 'theme-transitioning',
    transitionDuration: 200 // milliseconds
  };

  // ============================================
  // STATE
  // ============================================

  const state = {
    currentTheme: null,
    systemPreference: null,
    isTransitioning: false,
    toggleElements: []
  };

  // ============================================
  // UTILITY FUNCTIONS
  // ============================================

  /**
   * Get elements safely without throwing errors
   * @param {string} selector - CSS selector
   * @returns {NodeList} Matching elements (empty NodeList if none found)
   */
  function safeQuery(selector) {
    try {
      return document.querySelectorAll(selector);
    } catch (e) {
      console.warn(`Invalid selector: ${selector}`);
      return [];
    }
  }

  /**
   * Get single element safely
   * @param {string} selector - CSS selector
   * @returns {Element|null} First matching element or null
   */
  function safeQueryOne(selector) {
    try {
      return document.querySelector(selector);
    } catch (e) {
      console.warn(`Invalid selector: ${selector}`);
      return null;
    }
  }

  /**
   * Check if a value is a valid theme
   * @param {string} theme - Theme to validate
   * @returns {boolean} True if valid theme
   */
  function isValidTheme(theme) {
    return CONFIG.themes.includes(theme);
  }

  /**
   * Get system color scheme preference
   * @returns {string} 'light' or 'dark'
   */
  function getSystemTheme() {
    if (!window.matchMedia) {
      return CONFIG.defaultTheme;
    }

    const darkModeQuery = window.matchMedia('(prefers-color-scheme: dark)');
    state.systemPreference = darkModeQuery.matches ? 'dark' : 'light';
    return state.systemPreference;
  }

  /**
   * Listen for system theme changes
   */
  function listenForSystemThemeChanges() {
    if (!window.matchMedia) return;

    const darkModeQuery = window.matchMedia('(prefers-color-scheme: dark)');

    // Modern browsers
    if (darkModeQuery.addEventListener) {
      darkModeQuery.addEventListener('change', (e) => {
        const newSystemPreference = e.matches ? 'dark' : 'light';

        // Only apply if user hasn't explicitly set a theme
        const storedTheme = localStorage.getItem(CONFIG.storageKey);
        if (!storedTheme) {
          applyTheme(newSystemPreference);
        }

        state.systemPreference = newSystemPreference;
      });
    }
  }

  /**
   * Safely apply theme to document
   * @param {string} theme - Theme name ('light' or 'dark')
   * @param {boolean} animate - Whether to animate transition
   */
  function applyTheme(theme, animate = false) {
    // Validate theme
    if (!isValidTheme(theme)) {
      console.warn(`Invalid theme: ${theme}. Using default: ${CONFIG.defaultTheme}`);
      theme = CONFIG.defaultTheme;
    }

    const html = safeQueryOne(CONFIG.html);
    if (!html) {
      console.error('HTML element not found');
      return;
    }

    // Add transition class if animating
    if (animate && !state.isTransitioning) {
      state.isTransitioning = true;
      html.classList.add(CONFIG.transitionClass);

      // Remove transition class after animation completes
      setTimeout(() => {
        state.isTransitioning = false;
        html.classList.remove(CONFIG.transitionClass);
      }, CONFIG.transitionDuration);
    }

    // Apply theme attribute
    html.setAttribute('data-theme', theme);
    state.currentTheme = theme;

    // Persist to localStorage
    try {
      localStorage.setItem(CONFIG.storageKey, theme);
    } catch (e) {
      console.warn('localStorage not available:', e.message);
    }

    // Update theme toggle buttons
    updateThemeButtons();

    // Dispatch custom event for external listeners
    document.dispatchEvent(new CustomEvent('themechange', {
      detail: { theme, animated: animate }
    }));
  }

  /**
   * Get current theme
   * @returns {string} Current theme name
   */
  function getCurrentTheme() {
    if (state.currentTheme) {
      return state.currentTheme;
    }

    const html = safeQueryOne(CONFIG.html);
    if (!html) return CONFIG.defaultTheme;

    const attrTheme = html.getAttribute('data-theme');
    if (isValidTheme(attrTheme)) {
      state.currentTheme = attrTheme;
      return attrTheme;
    }

    return CONFIG.defaultTheme;
  }

  /**
   * Determine which theme to use
   * 1. Check localStorage
   * 2. Check system preference
   * 3. Use default
   * @returns {string} Theme to use
   */
  function determineInitialTheme() {
    // Check stored preference first
    try {
      const stored = localStorage.getItem(CONFIG.storageKey);
      if (isValidTheme(stored)) {
        return stored;
      }
    } catch (e) {
      console.warn('localStorage not available:', e.message);
    }

    // Fall back to system preference
    return getSystemTheme();
  }

  /**
   * Toggle between light and dark themes
   */
  function toggleTheme() {
    const currentTheme = getCurrentTheme();
    const nextTheme = currentTheme === 'light' ? 'dark' : 'light';
    applyTheme(nextTheme, true); // animate transition
  }

  /**
   * Update all theme toggle button appearances
   */
  function updateThemeButtons() {
    const currentTheme = getCurrentTheme();
    const toggleButtons = safeQuery(CONFIG.themeToggle);
    const otherButtons = safeQuery(CONFIG.themeButtons);

    // Update primary toggle button
    const primaryToggle = safeQueryOne(CONFIG.themeToggle);
    if (primaryToggle) {
      // Update icon/emoji
      const icon = currentTheme === 'dark' ? '☀️' : '🌙';
      primaryToggle.textContent = icon;

      // Update aria-label
      const nextTheme = currentTheme === 'light' ? 'dark' : 'light';
      primaryToggle.setAttribute('aria-label', `Switch to ${nextTheme} mode`);

      // Update title attribute
      primaryToggle.setAttribute('title', `Switch to ${nextTheme} mode (press 'd')`);
    }

    // Update all toggle buttons with data-theme-toggle
    otherButtons.forEach(btn => {
      // Add aria attributes if missing
      if (!btn.hasAttribute('aria-pressed')) {
        btn.setAttribute('aria-pressed', currentTheme === btn.getAttribute('data-theme'));
      } else {
        btn.setAttribute('aria-pressed', currentTheme === btn.getAttribute('data-theme'));
      }

      // Update active state
      if (btn.getAttribute('data-theme') === currentTheme) {
        btn.classList.add('active');
        btn.setAttribute('aria-current', 'true');
      } else {
        btn.classList.remove('active');
        btn.removeAttribute('aria-current');
      }
    });
  }

  // ============================================
  // INITIALIZATION
  // ============================================

  /**
   * Initialize theme system
   * This is called ASAP to prevent theme flash
   */
  function initThemeEarly() {
    // Determine initial theme before DOM is fully loaded
    const theme = determineInitialTheme();

    // Apply to HTML element
    const html = document.documentElement;
    html.setAttribute('data-theme', theme);
    state.currentTheme = theme;
  }

  /**
   * Initialize theme system (after DOM ready)
   */
  function init() {
    // Wait for DOM if needed
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', init);
      return;
    }

    // Get system preference
    getSystemTheme();

    // Listen for system theme changes
    listenForSystemThemeChanges();

    // Set up event listeners on toggle buttons
    const primaryToggle = safeQueryOne(CONFIG.themeToggle);
    if (primaryToggle) {
      primaryToggle.addEventListener('click', toggleTheme);
      updateThemeButtons();
    }

    // Set up event listeners on all theme toggle buttons
    safeQuery(CONFIG.themeButtons).forEach(btn => {
      btn.addEventListener('click', () => {
        const theme = btn.getAttribute('data-theme');
        if (isValidTheme(theme)) {
          applyTheme(theme, true);
        }
      });
    });

    // Set up keyboard shortcut: Ctrl+D or Cmd+D
    document.addEventListener('keydown', (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'd') {
        e.preventDefault();
        toggleTheme();
      }
    });

    // Log initialization
    if (window.location.hash === '#theme-debug') {
      console.log('Theme module initialized', {
        currentTheme: state.currentTheme,
        systemPreference: state.systemPreference,
        stored: localStorage.getItem(CONFIG.storageKey)
      });
    }
  }

  // ============================================
  // PUBLIC API
  // ============================================

  // Expose public methods for external use
  window.HoloLoomTheme = {
    // Get/set theme
    getCurrentTheme,
    applyTheme,
    toggleTheme,

    // Theme information
    getSystemTheme,
    isValidTheme,

    // Configuration
    CONFIG,
    state: () => ({ ...state }) // Return copy of state
  };

  // ============================================
  // EXECUTE EARLY
  // ============================================

  // Apply theme as early as possible to prevent flash
  // This runs before full page load
  if (document.readyState === 'loading') {
    // Run synchronously in <head>
    initThemeEarly();
    // Queue full initialization for after DOM ready
    document.addEventListener('DOMContentLoaded', init);
  } else {
    // DOM already loaded
    initThemeEarly();
    init();
  }

})();
