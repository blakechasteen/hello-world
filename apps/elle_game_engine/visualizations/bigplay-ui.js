/**
 * BigPlay UI Utilities
 * Version: 1.0.0
 *
 * Shared functionality for all BigPlay visualizations:
 * - Dark mode toggle
 * - Tooltips
 * - Loading states
 * - Analytics
 * - Accessibility enhancements
 */

(function(window) {
    'use strict';

    // ========================================
    // Theme Manager
    // ========================================

    class ThemeManager {
        constructor() {
            this.theme = localStorage.getItem('bigplay-theme') || 'light';
            this.apply();
            this.createToggle();
        }

        apply() {
            if (this.theme === 'dark') {
                document.body.classList.add('dark-mode');
            } else {
                document.body.classList.remove('dark-mode');
            }
        }

        toggle() {
            this.theme = this.theme === 'light' ? 'dark' : 'light';
            localStorage.setItem('bigplay-theme', this.theme);
            this.apply();

            // Animate toggle
            document.body.style.transition = 'background 0.3s ease, color 0.3s ease';

            // Track analytics
            if (window.BigPlayUI.analytics) {
                window.BigPlayUI.analytics.track('theme_toggled', { theme: this.theme });
            }
        }

        createToggle() {
            const button = document.createElement('button');
            button.className = 'theme-toggle';
            button.setAttribute('aria-label', 'Toggle dark mode');
            button.setAttribute('title', 'Toggle dark/light mode');
            button.innerHTML = this.theme === 'dark' ? '☀️' : '🌙';

            button.addEventListener('click', () => {
                this.toggle();
                button.innerHTML = this.theme === 'dark' ? '☀️' : '🌙';

                // Bounce animation
                button.style.animation = 'none';
                setTimeout(() => {
                    button.style.animation = 'bounce 0.5s ease';
                }, 10);
            });

            document.body.appendChild(button);

            // Add bounce animation
            const style = document.createElement('style');
            style.textContent = `
                @keyframes bounce {
                    0%, 100% { transform: scale(1); }
                    50% { transform: scale(1.2); }
                }
            `;
            document.head.appendChild(style);
        }
    }

    // ========================================
    // Tooltip Manager
    // ========================================

    class TooltipManager {
        constructor() {
            this.tooltip = null;
            this.createTooltip();
            this.attachHandlers();
        }

        createTooltip() {
            this.tooltip = document.createElement('div');
            this.tooltip.className = 'tooltip';
            this.tooltip.setAttribute('role', 'tooltip');
            document.body.appendChild(this.tooltip);
        }

        attachHandlers() {
            // Find all elements with data-tooltip attribute
            document.addEventListener('mouseover', (e) => {
                const target = e.target.closest('[data-tooltip]');
                if (target) {
                    this.show(target, target.dataset.tooltip);
                }
            });

            document.addEventListener('mouseout', (e) => {
                const target = e.target.closest('[data-tooltip]');
                if (target) {
                    this.hide();
                }
            });

            // Hide on scroll
            document.addEventListener('scroll', () => this.hide(), { passive: true });
        }

        show(element, content) {
            this.tooltip.textContent = content;
            this.tooltip.classList.add('visible');

            // Position tooltip
            const rect = element.getBoundingClientRect();
            const tooltipRect = this.tooltip.getBoundingClientRect();

            let top = rect.top - tooltipRect.height - 8;
            let left = rect.left + (rect.width / 2) - (tooltipRect.width / 2);

            // Keep tooltip in viewport
            if (top < 0) {
                top = rect.bottom + 8;
            }
            if (left < 0) {
                left = 8;
            }
            if (left + tooltipRect.width > window.innerWidth) {
                left = window.innerWidth - tooltipRect.width - 8;
            }

            this.tooltip.style.top = top + 'px';
            this.tooltip.style.left = left + 'px';
        }

        hide() {
            this.tooltip.classList.remove('visible');
        }
    }

    // ========================================
    // Loading Manager
    // ========================================

    class LoadingManager {
        static showSkeleton(container, count = 3) {
            const skeletons = [];
            for (let i = 0; i < count; i++) {
                const skeleton = document.createElement('div');
                skeleton.className = 'skeleton card';
                skeleton.style.height = '100px';
                skeleton.style.marginBottom = '20px';
                container.appendChild(skeleton);
                skeletons.push(skeleton);
            }
            return skeletons;
        }

        static removeSkeleton(skeletons) {
            skeletons.forEach(skeleton => {
                skeleton.style.transition = 'opacity 0.3s ease';
                skeleton.style.opacity = '0';
                setTimeout(() => skeleton.remove(), 300);
            });
        }

        static showSpinner(container) {
            const spinner = document.createElement('div');
            spinner.className = 'spinner';
            container.appendChild(spinner);
            return spinner;
        }

        static removeSpinner(spinner) {
            spinner.style.transition = 'opacity 0.3s ease';
            spinner.style.opacity = '0';
            setTimeout(() => spinner.remove(), 300);
        }
    }

    // ========================================
    // Analytics (Optional)
    // ========================================

    class Analytics {
        constructor() {
            this.events = [];
            this.sessionStart = Date.now();
        }

        track(event, properties = {}) {
            const eventData = {
                event,
                properties,
                timestamp: Date.now(),
                sessionDuration: Date.now() - this.sessionStart,
                page: window.location.pathname
            };

            this.events.push(eventData);

            // Log to console in development
            if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
                console.log('[Analytics]', event, properties);
            }

            // Send to backend (if configured)
            // this.send(eventData);
        }

        send(eventData) {
            // Example: Send to analytics backend
            // fetch('/api/analytics', {
            //     method: 'POST',
            //     headers: { 'Content-Type': 'application/json' },
            //     body: JSON.stringify(eventData)
            // });
        }

        getSessionStats() {
            return {
                totalEvents: this.events.length,
                sessionDuration: Date.now() - this.sessionStart,
                events: this.events
            };
        }
    }

    // ========================================
    // Accessibility Enhancements
    // ========================================

    class AccessibilityManager {
        static init() {
            // Add skip link
            this.addSkipLink();

            // Enhance keyboard navigation
            this.enhanceKeyboardNav();

            // Add aria-live regions
            this.addLiveRegions();
        }

        static addSkipLink() {
            const skipLink = document.createElement('a');
            skipLink.href = '#main-content';
            skipLink.textContent = 'Skip to main content';
            skipLink.className = 'skip-link';
            skipLink.style.cssText = `
                position: absolute;
                top: -40px;
                left: 0;
                background: var(--color-primary);
                color: white;
                padding: 8px 16px;
                border-radius: 0 0 8px 0;
                text-decoration: none;
                z-index: 9999;
            `;
            skipLink.addEventListener('focus', () => {
                skipLink.style.top = '0';
            });
            skipLink.addEventListener('blur', () => {
                skipLink.style.top = '-40px';
            });
            document.body.insertBefore(skipLink, document.body.firstChild);
        }

        static enhanceKeyboardNav() {
            // Make clickable elements keyboard accessible
            document.querySelectorAll('.component, .card, .viz-card').forEach((el, index) => {
                if (!el.hasAttribute('tabindex')) {
                    el.setAttribute('tabindex', '0');
                }
                if (!el.hasAttribute('role')) {
                    el.setAttribute('role', 'button');
                }

                // Enter/Space to click
                el.addEventListener('keydown', (e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                        e.preventDefault();
                        el.click();
                    }
                });
            });
        }

        static addLiveRegions() {
            if (!document.getElementById('aria-live-region')) {
                const liveRegion = document.createElement('div');
                liveRegion.id = 'aria-live-region';
                liveRegion.setAttribute('aria-live', 'polite');
                liveRegion.setAttribute('aria-atomic', 'true');
                liveRegion.style.cssText = `
                    position: absolute;
                    left: -10000px;
                    width: 1px;
                    height: 1px;
                    overflow: hidden;
                `;
                document.body.appendChild(liveRegion);
            }
        }

        static announce(message) {
            const liveRegion = document.getElementById('aria-live-region');
            if (liveRegion) {
                liveRegion.textContent = message;
            }
        }
    }

    // ========================================
    // Utility Functions
    // ========================================

    const utils = {
        // Debounce function
        debounce(func, wait) {
            let timeout;
            return function executedFunction(...args) {
                const later = () => {
                    clearTimeout(timeout);
                    func(...args);
                };
                clearTimeout(timeout);
                timeout = setTimeout(later, wait);
            };
        },

        // Throttle function
        throttle(func, limit) {
            let inThrottle;
            return function() {
                const args = arguments;
                const context = this;
                if (!inThrottle) {
                    func.apply(context, args);
                    inThrottle = true;
                    setTimeout(() => inThrottle = false, limit);
                }
            };
        },

        // Smooth scroll to element
        scrollTo(element, offset = 0) {
            const top = element.getBoundingClientRect().top + window.pageYOffset - offset;
            window.scrollTo({ top, behavior: 'smooth' });
        },

        // Copy to clipboard
        async copyToClipboard(text) {
            try {
                await navigator.clipboard.writeText(text);
                AccessibilityManager.announce('Copied to clipboard');
                return true;
            } catch (err) {
                console.error('Failed to copy:', err);
                return false;
            }
        },

        // Format number with commas
        formatNumber(num) {
            return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ',');
        },

        // Format bytes
        formatBytes(bytes, decimals = 2) {
            if (bytes === 0) return '0 Bytes';
            const k = 1024;
            const dm = decimals < 0 ? 0 : decimals;
            const sizes = ['Bytes', 'KB', 'MB', 'GB'];
            const i = Math.floor(Math.log(bytes) / Math.log(k));
            return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
        }
    };

    // ========================================
    // Initialize BigPlayUI
    // ========================================

    window.BigPlayUI = {
        theme: new ThemeManager(),
        tooltip: new TooltipManager(),
        loading: LoadingManager,
        analytics: new Analytics(),
        accessibility: AccessibilityManager,
        utils: utils,

        init() {
            console.log('%c🎮 BigPlay UI loaded', 'color: #667eea; font-weight: bold; font-size: 14px;');

            // Initialize accessibility enhancements
            AccessibilityManager.init();

            // Track page view
            this.analytics.track('page_view', {
                title: document.title,
                url: window.location.href
            });

            // Track when user leaves page
            window.addEventListener('beforeunload', () => {
                const stats = this.analytics.getSessionStats();
                console.log('[BigPlay] Session stats:', stats);
            });

            // Log session duration every minute
            setInterval(() => {
                const duration = Math.floor((Date.now() - this.analytics.sessionStart) / 1000);
                console.log(`[BigPlay] Session duration: ${duration}s, Events: ${this.analytics.events.length}`);
            }, 60000);
        }
    };

    // Auto-initialize when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => window.BigPlayUI.init());
    } else {
        window.BigPlayUI.init();
    }

})(window);
