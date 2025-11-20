/**
 * HoloLoom Modern Dashboard Interactivity
 * ========================================
 * Modern JavaScript features:
 * - View Transitions API (Phase 5)
 * - Theme Management
 * - Accessibility Features (Phase 3)
 * - Progressive Enhancement
 *
 * Author: Claude Code with HoloLoom architecture
 * Date: October 29, 2025
 */

(function() {
  'use strict';

  // ========================================================================
  // Phase 5: Theme Management with View Transitions API
  // ========================================================================

  class ThemeManager {
    constructor() {
      this.root = document.documentElement;
      this.storageKey = 'hololoom-theme';
      this.init();
    }

    init() {
      // Load saved theme or detect system preference
      const savedTheme = localStorage.getItem(this.storageKey);
      const systemPrefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
      const initialTheme = savedTheme || (systemPrefersDark ? 'dark' : 'light');

      this.setTheme(initialTheme, false); // No transition on initial load

      // Listen for system preference changes
      window.matchMedia('(prefers-color-scheme: dark)')
        .addEventListener('change', (e) => {
          if (!localStorage.getItem(this.storageKey)) {
            this.setTheme(e.matches ? 'dark' : 'light');
          }
        });

      // Create theme toggle button
      this.createThemeToggle();
    }

    setTheme(theme, useTransition = true) {
      // Phase 5: Use View Transitions API if available
      if (useTransition && document.startViewTransition) {
        document.startViewTransition(() => {
          this._applyTheme(theme);
        });
      } else {
        this._applyTheme(theme);
      }

      localStorage.setItem(this.storageKey, theme);

      // Dispatch custom event for other components
      window.dispatchEvent(new CustomEvent('themechange', {
        detail: { theme }
      }));
    }

    _applyTheme(theme) {
      this.root.setAttribute('data-theme', theme);
    }

    toggle() {
      const currentTheme = this.root.getAttribute('data-theme') || 'light';
      this.setTheme(currentTheme === 'dark' ? 'light' : 'dark');
    }

    createThemeToggle() {
      const button = document.createElement('button');
      button.className = 'theme-toggle';
      button.setAttribute('aria-label', 'Toggle dark mode');
      button.setAttribute('title', 'Toggle dark mode');
      button.innerHTML = `
        <svg width="20" height="20" viewBox="0 0 20 20" fill="currentColor">
          <path class="sun-icon" d="M10 2a1 1 0 011 1v1a1 1 0 11-2 0V3a1 1 0 011-1zm4 8a4 4 0 11-8 0 4 4 0 018 0zm-.464 4.95l.707.707a1 1 0 001.414-1.414l-.707-.707a1 1 0 00-1.414 1.414zm2.12-10.607a1 1 0 010 1.414l-.706.707a1 1 0 11-1.414-1.414l.707-.707a1 1 0 011.414 0zM17 11a1 1 0 100-2h-1a1 1 0 100 2h1zm-7 4a1 1 0 011 1v1a1 1 0 11-2 0v-1a1 1 0 011-1zM5.05 6.464A1 1 0 106.465 5.05l-.708-.707a1 1 0 00-1.414 1.414l.707.707zm1.414 8.486l-.707.707a1 1 0 01-1.414-1.414l.707-.707a1 1 0 011.414 1.414zM4 11a1 1 0 100-2H3a1 1 0 000 2h1z"/>
          <path class="moon-icon" style="display: none;" d="M17.293 13.293A8 8 0 016.707 2.707a8.001 8.001 0 1010.586 10.586z"/>
        </svg>
      `;

      // Style the button
      Object.assign(button.style, {
        position: 'fixed',
        bottom: '2rem',
        right: '2rem',
        width: '48px',
        height: '48px',
        borderRadius: '50%',
        border: '1px solid var(--color-border-default)',
        background: 'var(--color-accent-primary)',
        color: 'white',
        cursor: 'pointer',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        boxShadow: 'var(--shadow-lg)',
        transition: 'transform var(--transition-fast), box-shadow var(--transition-fast)',
        zIndex: 'var(--z-sticky)'
      });

      button.addEventListener('click', () => this.toggle());

      button.addEventListener('mouseenter', () => {
        button.style.transform = 'scale(1.1)';
        button.style.boxShadow = 'var(--shadow-xl)';
      });

      button.addEventListener('mouseleave', () => {
        button.style.transform = 'scale(1)';
        button.style.boxShadow = 'var(--shadow-lg)';
      });

      // Update icon based on theme
      const updateIcon = () => {
        const theme = this.root.getAttribute('data-theme');
        const sunIcon = button.querySelector('.sun-icon');
        const moonIcon = button.querySelector('.moon-icon');

        if (theme === 'dark') {
          sunIcon.style.display = 'block';
          moonIcon.style.display = 'none';
          button.setAttribute('aria-label', 'Switch to light mode');
        } else {
          sunIcon.style.display = 'none';
          moonIcon.style.display = 'block';
          button.setAttribute('aria-label', 'Switch to dark mode');
        }
      };

      updateIcon();
      window.addEventListener('themechange', updateIcon);

      document.body.appendChild(button);
    }
  }

  // ========================================================================
  // Phase 3: Accessibility - Keyboard Navigation
  // ========================================================================

  class KeyboardNavigationManager {
    constructor() {
      this.focusableSelector = '.panel, button, a, input, select, textarea, [tabindex]:not([tabindex="-1"])';
      this.init();
    }

    init() {
      // Add keyboard shortcuts
      document.addEventListener('keydown', this.handleKeydown.bind(this));

      // Add focus indicators to panels
      this.enhancePanelFocus();
    }

    handleKeydown(e) {
      // T key: Toggle theme
      if (e.key === 't' && !this.isTyping(e)) {
        e.preventDefault();
        window.themeManager?.toggle();
      }

      // Escape: Close expanded panels
      if (e.key === 'Escape') {
        this.closeExpandedPanels();
      }

      // Arrow keys: Navigate between panels
      if (['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(e.key)) {
        if (document.activeElement.classList.contains('panel')) {
          e.preventDefault();
          this.navigatePanels(e.key);
        }
      }
    }

    isTyping(e) {
      const target = e.target;
      return target.tagName === 'INPUT' ||
             target.tagName === 'TEXTAREA' ||
             target.isContentEditable;
    }

    navigatePanels(direction) {
      const panels = Array.from(document.querySelectorAll('.panel'));
      const currentIndex = panels.indexOf(document.activeElement);

      if (currentIndex === -1) return;

      let nextIndex;
      if (direction === 'ArrowRight' || direction === 'ArrowDown') {
        nextIndex = (currentIndex + 1) % panels.length;
      } else {
        nextIndex = (currentIndex - 1 + panels.length) % panels.length;
      }

      panels[nextIndex].focus();
    }

    enhancePanelFocus() {
      document.querySelectorAll('.panel').forEach(panel => {
        // Make panels keyboard focusable
        if (!panel.hasAttribute('tabindex')) {
          panel.setAttribute('tabindex', '0');
        }

        // Add ARIA labels if missing
        if (!panel.getAttribute('aria-label')) {
          const title = panel.querySelector('.panel-title')?.textContent;
          if (title) {
            panel.setAttribute('aria-label', title);
          }
        }
      });
    }

    closeExpandedPanels() {
      document.querySelectorAll('.panel[data-expanded="true"]').forEach(panel => {
        panel.setAttribute('data-expanded', 'false');
      });
    }
  }

  // ========================================================================
  // Panel Interactivity
  // ========================================================================

  class PanelManager {
    constructor() {
      this.init();
    }

    init() {
      document.querySelectorAll('.panel').forEach(panel => {
        this.enhancePanel(panel);
      });
    }

    enhancePanel(panel) {
      // Click to expand (optional)
      const expandable = panel.hasAttribute('data-expandable');

      if (expandable) {
        panel.style.cursor = 'pointer';

        panel.addEventListener('click', (e) => {
          // Don't expand if clicking interactive elements
          if (e.target.closest('button, a, input, select')) return;

          this.togglePanel(panel);
        });

        // Enter key to expand
        panel.addEventListener('keydown', (e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            this.togglePanel(panel);
          }
        });
      }

      // Hover effects are handled by CSS
    }

    togglePanel(panel) {
      const isExpanded = panel.getAttribute('data-expanded') === 'true';
      panel.setAttribute('data-expanded', !isExpanded);

      // Announce to screen readers
      const message = isExpanded ? 'Panel collapsed' : 'Panel expanded';
      this.announceToScreenReader(message);
    }

    announceToScreenReader(message) {
      const announcement = document.createElement('div');
      announcement.className = 'sr-only';
      announcement.setAttribute('role', 'status');
      announcement.setAttribute('aria-live', 'polite');
      announcement.textContent = message;

      document.body.appendChild(announcement);
      setTimeout(() => announcement.remove(), 1000);
    }
  }

  // ========================================================================
  // Performance Monitoring
  // ========================================================================

  class PerformanceMonitor {
    constructor() {
      this.init();
    }

    init() {
      // Monitor paint performance
      if ('PerformanceObserver' in window) {
        try {
          const observer = new PerformanceObserver((list) => {
            for (const entry of list.getEntries()) {
              console.log(`[HoloLoom] ${entry.name}: ${entry.duration.toFixed(2)}ms`);
            }
          });

          observer.observe({ entryTypes: ['measure', 'paint'] });
        } catch (e) {
          // PerformanceObserver not fully supported
        }
      }

      // Log page load metrics
      window.addEventListener('load', () => {
        setTimeout(() => {
          const perfData = performance.getEntriesByType('navigation')[0];
          if (perfData) {
            console.log('[HoloLoom] Page Load Metrics:', {
              domContentLoaded: `${perfData.domContentLoadedEventEnd - perfData.domContentLoadedEventStart}ms`,
              loadComplete: `${perfData.loadEventEnd - perfData.loadEventStart}ms`,
              domInteractive: `${perfData.domInteractive - perfData.fetchStart}ms`
            });
          }
        }, 0);
      });
    }
  }

  // ========================================================================
  // Initialize All Managers
  // ========================================================================

  function init() {
    // Wait for DOM to be ready
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', init);
      return;
    }

    console.log('[HoloLoom] Initializing modern dashboard features...');

    // Initialize managers
    window.themeManager = new ThemeManager();
    window.keyboardNav = new KeyboardNavigationManager();
    window.panelManager = new PanelManager();

    // Performance monitoring (only in development)
    if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
      window.perfMonitor = new PerformanceMonitor();
    }

    // Add keyboard shortcuts help
    createKeyboardShortcutsHelp();

    console.log('[HoloLoom] Dashboard ready! Press "?" for keyboard shortcuts.');
  }

  // ========================================================================
  // Keyboard Shortcuts Help Modal
  // ========================================================================

  function createKeyboardShortcutsHelp() {
    let helpVisible = false;

    document.addEventListener('keydown', (e) => {
      if (e.key === '?' && !window.keyboardNav.isTyping(e)) {
        e.preventDefault();
        toggleHelp();
      }
    });

    function toggleHelp() {
      if (helpVisible) {
        document.getElementById('keyboard-help-modal')?.remove();
        helpVisible = false;
      } else {
        showHelp();
        helpVisible = true;
      }
    }

    function showHelp() {
      const modal = document.createElement('div');
      modal.id = 'keyboard-help-modal';
      modal.setAttribute('role', 'dialog');
      modal.setAttribute('aria-labelledby', 'keyboard-help-title');
      modal.setAttribute('aria-modal', 'true');

      modal.innerHTML = `
        <div style="
          position: fixed;
          top: 0;
          left: 0;
          right: 0;
          bottom: 0;
          background: rgba(0, 0, 0, 0.5);
          display: flex;
          align-items: center;
          justify-content: center;
          z-index: var(--z-modal);
          padding: var(--space-4);
        " id="help-backdrop">
          <div style="
            background: var(--color-bg-elevated);
            border-radius: var(--radius-lg);
            padding: var(--space-8);
            max-width: 500px;
            box-shadow: var(--shadow-xl);
            border: 1px solid var(--color-border-default);
          ">
            <h2 id="keyboard-help-title" style="
              margin: 0 0 var(--space-4) 0;
              font-size: var(--font-size-xl);
              color: var(--color-text-primary);
            ">Keyboard Shortcuts</h2>

            <dl style="margin: 0; display: grid; gap: var(--space-3);">
              <div style="display: flex; justify-content: space-between; gap: var(--space-4);">
                <dt style="font-weight: var(--font-weight-semibold); color: var(--color-text-primary);">
                  <kbd style="
                    background: var(--color-bg-secondary);
                    padding: var(--space-1) var(--space-2);
                    border-radius: var(--radius-sm);
                    font-family: monospace;
                    border: 1px solid var(--color-border-default);
                  ">T</kbd>
                </dt>
                <dd style="margin: 0; color: var(--color-text-secondary);">Toggle dark mode</dd>
              </div>

              <div style="display: flex; justify-content: space-between; gap: var(--space-4);">
                <dt style="font-weight: var(--font-weight-semibold); color: var(--color-text-primary);">
                  <kbd style="
                    background: var(--color-bg-secondary);
                    padding: var(--space-1) var(--space-2);
                    border-radius: var(--radius-sm);
                    font-family: monospace;
                    border: 1px solid var(--color-border-default);
                  ">Arrow Keys</kbd>
                </dt>
                <dd style="margin: 0; color: var(--color-text-secondary);">Navigate panels</dd>
              </div>

              <div style="display: flex; justify-content: space-between; gap: var(--space-4);">
                <dt style="font-weight: var(--font-weight-semibold); color: var(--color-text-primary);">
                  <kbd style="
                    background: var(--color-bg-secondary);
                    padding: var(--space-1) var(--space-2);
                    border-radius: var(--radius-sm);
                    font-family: monospace;
                    border: 1px solid var(--color-border-default);
                  ">Enter / Space</kbd>
                </dt>
                <dd style="margin: 0; color: var(--color-text-secondary);">Expand panel</dd>
              </div>

              <div style="display: flex; justify-content: space-between; gap: var(--space-4);">
                <dt style="font-weight: var(--font-weight-semibold); color: var(--color-text-primary);">
                  <kbd style="
                    background: var(--color-bg-secondary);
                    padding: var(--space-1) var(--space-2);
                    border-radius: var(--radius-sm);
                    font-family: monospace;
                    border: 1px solid var(--color-border-default);
                  ">Esc</kbd>
                </dt>
                <dd style="margin: 0; color: var(--color-text-secondary);">Close expanded panels</dd>
              </div>

              <div style="display: flex; justify-content: space-between; gap: var(--space-4);">
                <dt style="font-weight: var(--font-weight-semibold); color: var(--color-text-primary);">
                  <kbd style="
                    background: var(--color-bg-secondary);
                    padding: var(--space-1) var(--space-2);
                    border-radius: var(--radius-sm);
                    font-family: monospace;
                    border: 1px solid var(--color-border-default);
                  ">?</kbd>
                </dt>
                <dd style="margin: 0; color: var(--color-text-secondary);">Show this help</dd>
              </div>
            </dl>

            <button style="
              margin-top: var(--space-6);
              width: 100%;
              padding: var(--space-3);
              background: var(--color-accent-primary);
              color: white;
              border: none;
              border-radius: var(--radius-md);
              font-weight: var(--font-weight-medium);
              cursor: pointer;
            " id="close-help-btn">Close</button>
          </div>
        </div>
      `;

      document.body.appendChild(modal);

      // Close on button click
      document.getElementById('close-help-btn').addEventListener('click', toggleHelp);

      // Close on backdrop click
      document.getElementById('help-backdrop').addEventListener('click', (e) => {
        if (e.target.id === 'help-backdrop') {
          toggleHelp();
        }
      });

      // Close on Escape
      const escapeHandler = (e) => {
        if (e.key === 'Escape') {
          toggleHelp();
          document.removeEventListener('keydown', escapeHandler);
        }
      };
      document.addEventListener('keydown', escapeHandler);

      // Focus the close button
      setTimeout(() => {
        document.getElementById('close-help-btn').focus();
      }, 100);
    }
  }

  // ========================================================================
  // NEXT LEVEL: Advanced Interactive Effects
  // ========================================================================

  class ParallaxEffect {
    constructor() {
      this.panels = document.querySelectorAll('.panel');
      this.enabled = !window.matchMedia('(prefers-reduced-motion: reduce)').matches;
      if (this.enabled) this.init();
    }

    init() {
      document.addEventListener('mousemove', (e) => {
        const mouseX = e.clientX / window.innerWidth;
        const mouseY = e.clientY / window.innerHeight;

        this.panels.forEach((panel, index) => {
          const speed = (index % 3 + 1) * 0.5; // Different speeds for depth
          const x = (mouseX - 0.5) * speed * 10;
          const y = (mouseY - 0.5) * speed * 10;

          panel.style.transform = `translate(${x}px, ${y}px)`;
        });
      });

      // Reset on mouse leave
      document.addEventListener('mouseleave', () => {
        this.panels.forEach(panel => {
          panel.style.transform = '';
        });
      });
    }
  }

  class CursorTrail {
    constructor() {
      this.trail = [];
      this.maxTrail = 20;
      this.enabled = !window.matchMedia('(prefers-reduced-motion: reduce)').matches;
      if (this.enabled) this.init();
    }

    init() {
      document.addEventListener('mousemove', (e) => {
        this.createParticle(e.clientX, e.clientY);
      });
    }

    createParticle(x, y) {
      const particle = document.createElement('div');
      particle.className = 'cursor-particle';
      particle.style.cssText = `
        position: fixed;
        left: ${x}px;
        top: ${y}px;
        width: 4px;
        height: 4px;
        background: radial-gradient(circle, rgba(99, 102, 241, 0.8), transparent);
        border-radius: 50%;
        pointer-events: none;
        z-index: 9999;
        animation: particle-fade 0.8s ease-out forwards;
      `;
      document.body.appendChild(particle);

      // Add CSS animation if not exists
      if (!document.getElementById('particle-style')) {
        const style = document.createElement('style');
        style.id = 'particle-style';
        style.textContent = `
          @keyframes particle-fade {
            from {
              opacity: 0.8;
              transform: scale(1);
            }
            to {
              opacity: 0;
              transform: scale(0.3);
            }
          }
        `;
        document.head.appendChild(style);
      }

      setTimeout(() => particle.remove(), 800);
    }
  }

  class DataAnimator {
    constructor() {
      this.init();
    }

    init() {
      // Animate metric numbers on scroll into view
      const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
          if (entry.isIntersecting) {
            this.animateNumber(entry.target);
            observer.unobserve(entry.target);
          }
        });
      }, { threshold: 0.5 });

      document.querySelectorAll('.metric-value.numeric').forEach(el => {
        observer.observe(el);
      });
    }

    animateNumber(element) {
      const text = element.textContent.trim();
      const number = parseFloat(text.replace(/[^0-9.]/g, ''));

      if (isNaN(number)) return;

      const suffix = text.replace(/[0-9.,]/g, '');
      const duration = 1000;
      const start = performance.now();

      const animate = (currentTime) => {
        const elapsed = currentTime - start;
        const progress = Math.min(elapsed / duration, 1);

        // Ease out cubic
        const easeProgress = 1 - Math.pow(1 - progress, 3);
        const current = number * easeProgress;

        // Format based on original format
        let formatted;
        if (text.includes('K')) {
          // Value is already in thousands, don't divide again!
          formatted = current.toFixed(1) + 'K';
        } else if (text.includes('%')) {
          formatted = current.toFixed(1) + '%';
        } else if (text.includes('GB')) {
          formatted = current.toFixed(1) + 'GB';
        } else if (text.includes('ms')) {
          formatted = current.toFixed(1) + 'ms';
        } else {
          formatted = Math.floor(current).toString();
        }

        element.textContent = formatted;

        if (progress < 1) {
          requestAnimationFrame(animate);
        }
      };

      requestAnimationFrame(animate);
    }
  }

  class PanelMagnetism {
    constructor() {
      this.panels = document.querySelectorAll('.panel');
      this.enabled = !window.matchMedia('(prefers-reduced-motion: reduce)').matches;
      if (this.enabled) this.init();
    }

    init() {
      this.panels.forEach(panel => {
        panel.addEventListener('mouseenter', (e) => {
          const rect = panel.getBoundingClientRect();
          const centerX = rect.left + rect.width / 2;
          const centerY = rect.top + rect.height / 2;

          panel.addEventListener('mousemove', this.handleMouseMove.bind(this, panel, centerX, centerY));
        });

        panel.addEventListener('mouseleave', () => {
          panel.style.transform = '';
        });
      });
    }

    handleMouseMove(panel, centerX, centerY, e) {
      const mouseX = e.clientX;
      const mouseY = e.clientY;

      const deltaX = (mouseX - centerX) / 20;
      const deltaY = (mouseY - centerY) / 20;

      panel.style.transform = `translate(${deltaX}px, ${deltaY}px) scale(1.02)`;
    }
  }

  class SmoothScroll {
    constructor() {
      this.init();
    }

    init() {
      // Add smooth scroll behavior
      document.documentElement.style.scrollBehavior = 'smooth';

      // Add scroll reveal for panels
      const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
          if (entry.isIntersecting) {
            entry.target.classList.add('revealed');
          }
        });
      }, {
        threshold: 0.1,
        rootMargin: '-50px'
      });

      document.querySelectorAll('.panel').forEach(panel => {
        observer.observe(panel);
      });
    }
  }

  class HoverSound {
    constructor() {
      this.enabled = false; // Disabled by default, can be enabled
      if (this.enabled) this.init();
    }

    init() {
      // Create audio context
      this.audioContext = new (window.AudioContext || window.webkitAudioContext)();

      document.querySelectorAll('.panel').forEach(panel => {
        panel.addEventListener('mouseenter', () => this.playHoverSound());
      });
    }

    playHoverSound() {
      const oscillator = this.audioContext.createOscillator();
      const gainNode = this.audioContext.createGain();

      oscillator.connect(gainNode);
      gainNode.connect(this.audioContext.destination);

      oscillator.frequency.value = 800;
      oscillator.type = 'sine';

      gainNode.gain.setValueAtTime(0.1, this.audioContext.currentTime);
      gainNode.gain.exponentialRampToValueAtTime(0.01, this.audioContext.currentTime + 0.1);

      oscillator.start(this.audioContext.currentTime);
      oscillator.stop(this.audioContext.currentTime + 0.1);
    }
  }

  class DynamicColorShift {
    constructor() {
      this.init();
    }

    init() {
      // Shift panel colors based on time of day
      const hour = new Date().getHours();
      const isDaytime = hour >= 6 && hour < 18;

      if (!isDaytime && document.documentElement.getAttribute('data-theme') === 'light') {
        // Subtle hint that dark mode might be nice
        document.querySelectorAll('.panel').forEach(panel => {
          panel.style.filter = 'brightness(0.95)';
        });
      }
    }
  }

  // Initialize advanced effects
  function initAdvancedEffects() {
    // new ParallaxEffect(); // Uncomment for mouse parallax
    new CursorTrail();
    new DataAnimator();
    // new PanelMagnetism(); // Uncomment for magnetic panels
    new SmoothScroll();
    // new HoverSound(); // Uncomment for sound effects (disabled by default)
    new DynamicColorShift();

    console.log('🎨 HoloLoom Advanced Effects Loaded');
  }

  // Start initialization
  init();

  // Initialize advanced effects after short delay
  setTimeout(initAdvancedEffects, 100);

})();
