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
    // Touch Enhancements (Mobile)
    // ========================================

    class TouchEnhancements {
        constructor() {
            if (!('ontouchstart' in window)) {
                return; // Desktop - skip touch enhancements
            }

            this.addSwipeGestures();
            this.addBottomSheets();
            this.enlargeTouchTargets();
            this.addPullToRefresh();
        }

        addSwipeGestures() {
            let touchStartX = 0;
            let touchStartY = 0;
            let touchStartTime = 0;

            document.addEventListener('touchstart', (e) => {
                touchStartX = e.touches[0].clientX;
                touchStartY = e.touches[0].clientY;
                touchStartTime = Date.now();
            }, { passive: true });

            document.addEventListener('touchend', (e) => {
                const touchEndX = e.changedTouches[0].clientX;
                const touchEndY = e.changedTouches[0].clientY;
                const touchEndTime = Date.now();

                const diffX = touchStartX - touchEndX;
                const diffY = touchStartY - touchEndY;
                const duration = touchEndTime - touchStartTime;

                // Only trigger if swipe was quick (<300ms)
                if (duration > 300) return;

                // Horizontal swipe (left/right)
                if (Math.abs(diffX) > Math.abs(diffY) && Math.abs(diffX) > 50) {
                    const event = new CustomEvent(diffX > 0 ? 'swipe-left' : 'swipe-right', {
                        detail: { distance: Math.abs(diffX) }
                    });
                    document.dispatchEvent(event);

                    // Navigate in tour if active
                    if (window.GuidedTour && diffX > 0) {
                        // Swipe left = next (if tour exists)
                    }
                }

                // Vertical swipe down to dismiss panels
                if (diffY < -100 && Math.abs(diffY) > Math.abs(diffX)) {
                    this.dismissOpenPanels();
                }
            }, { passive: true });
        }

        addBottomSheets() {
            // Convert side panels to bottom sheets on mobile
            if (window.innerWidth < 768) {
                const panels = document.querySelectorAll('.detail-panel, .sidebar, .controls-panel');
                panels.forEach(panel => {
                    if (!panel.classList.contains('mobile-optimized')) {
                        panel.classList.add('mobile-bottom-sheet');
                        panel.classList.add('mobile-optimized');
                    }
                });

                // Add CSS for bottom sheets
                const style = document.createElement('style');
                style.textContent = `
                    @media (max-width: 768px) {
                        .mobile-bottom-sheet {
                            position: fixed !important;
                            bottom: 0 !important;
                            left: 0 !important;
                            right: 0 !important;
                            top: auto !important;
                            max-height: 70vh !important;
                            overflow-y: auto !important;
                            border-radius: var(--radius-xl) var(--radius-xl) 0 0 !important;
                            transform: translateY(100%);
                            transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                        }

                        .mobile-bottom-sheet.active {
                            transform: translateY(0);
                        }

                        .mobile-bottom-sheet::before {
                            content: '';
                            display: block;
                            width: 40px;
                            height: 4px;
                            background: var(--color-text-tertiary);
                            border-radius: 2px;
                            margin: var(--space-3) auto;
                        }
                    }
                `;
                document.head.appendChild(style);
            }
        }

        dismissOpenPanels() {
            document.querySelectorAll('.mobile-bottom-sheet.active, .detail-panel.active').forEach(panel => {
                panel.classList.remove('active');

                // Announce dismissal
                if (window.BigPlayUI?.accessibility) {
                    window.BigPlayUI.accessibility.announce('Panel dismissed');
                }
            });
        }

        enlargeTouchTargets() {
            // Ensure all interactive elements are at least 44x44px (Apple HIG)
            const selectors = 'button, a, [role="button"], .component, .viz-card, .npc-card, .quest-option';
            document.querySelectorAll(selectors).forEach(el => {
                const rect = el.getBoundingClientRect();
                if (rect.width > 0 && (rect.width < 44 || rect.height < 44)) {
                    const currentPadding = window.getComputedStyle(el).padding;
                    if (currentPadding === '0px' || !currentPadding.includes('px')) {
                        el.style.minWidth = '44px';
                        el.style.minHeight = '44px';
                        el.style.padding = 'var(--space-3) var(--space-4)';
                    }
                }
            });
        }

        addPullToRefresh() {
            // Simple pull-to-refresh for mobile
            let startY = 0;
            let pulling = false;

            document.addEventListener('touchstart', (e) => {
                if (window.scrollY === 0) {
                    startY = e.touches[0].clientY;
                    pulling = true;
                }
            }, { passive: true });

            document.addEventListener('touchmove', (e) => {
                if (pulling) {
                    const currentY = e.touches[0].clientY;
                    const pullDistance = currentY - startY;

                    if (pullDistance > 80) {
                        // Show refresh indicator
                        this.showRefreshIndicator();
                    }
                }
            }, { passive: true });

            document.addEventListener('touchend', (e) => {
                if (pulling) {
                    const currentY = e.changedTouches[0].clientY;
                    const pullDistance = currentY - startY;

                    if (pullDistance > 80) {
                        // Trigger refresh
                        window.location.reload();
                    } else {
                        this.hideRefreshIndicator();
                    }
                }
                pulling = false;
            }, { passive: true });
        }

        showRefreshIndicator() {
            if (!document.getElementById('refresh-indicator')) {
                const indicator = document.createElement('div');
                indicator.id = 'refresh-indicator';
                indicator.innerHTML = '<div class="spinner"></div>';
                indicator.style.cssText = `
                    position: fixed;
                    top: 0;
                    left: 50%;
                    transform: translateX(-50%);
                    padding: var(--space-4);
                    background: var(--color-bg-elevated);
                    border-radius: 0 0 var(--radius-lg) var(--radius-lg);
                    box-shadow: var(--shadow-lg);
                    z-index: 9999;
                `;
                document.body.appendChild(indicator);
            }
        }

        hideRefreshIndicator() {
            const indicator = document.getElementById('refresh-indicator');
            if (indicator) {
                indicator.style.opacity = '0';
                setTimeout(() => indicator.remove(), 300);
            }
        }
    }

    // ========================================
    // Page Transitions & Animations
    // ========================================

    class PageTransitions {
        constructor() {
            this.addPageLoader();
            this.addFadeInAnimations();
            this.addScrollAnimations();
        }

        addPageLoader() {
            // Show loader on page load
            const loader = document.createElement('div');
            loader.id = 'page-loader';
            loader.innerHTML = `
                <div style="text-align: center;">
                    <div class="spinner"></div>
                    <p style="margin-top: var(--space-4); font-size: var(--font-size-lg);">
                        Loading BigPlay...
                    </p>
                </div>
            `;
            loader.style.cssText = `
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: var(--color-bg-primary);
                display: flex;
                align-items: center;
                justify-content: center;
                z-index: 99999;
                transition: opacity 0.3s ease;
            `;
            document.body.appendChild(loader);

            // Hide loader when page is loaded
            window.addEventListener('load', () => {
                setTimeout(() => {
                    loader.style.opacity = '0';
                    setTimeout(() => loader.remove(), 300);
                }, 500); // Minimum 500ms to avoid flash
            });
        }

        addFadeInAnimations() {
            // Fade in elements as they appear
            const elements = document.querySelectorAll('.viz-card, .card, .stat-card, .npc-card, .quest-option');
            elements.forEach((el, index) => {
                el.style.opacity = '0';
                el.style.transform = 'translateY(20px)';

                setTimeout(() => {
                    el.style.transition = 'all 0.6s cubic-bezier(0.4, 0, 0.2, 1)';
                    el.style.opacity = '1';
                    el.style.transform = 'translateY(0)';
                }, index * 50); // Stagger by 50ms
            });
        }

        addScrollAnimations() {
            // Animate elements as they enter viewport
            const observerOptions = {
                threshold: 0.1,
                rootMargin: '0px 0px -50px 0px'
            };

            const observer = new IntersectionObserver((entries) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        entry.target.style.opacity = '1';
                        entry.target.style.transform = 'translateY(0)';
                        observer.unobserve(entry.target);
                    }
                });
            }, observerOptions);

            // Observe elements with .animate-on-scroll class
            document.querySelectorAll('.animate-on-scroll').forEach(el => {
                el.style.opacity = '0';
                el.style.transform = 'translateY(20px)';
                el.style.transition = 'all 0.6s cubic-bezier(0.4, 0, 0.2, 1)';
                observer.observe(el);
            });
        }

        static fadeIn(elements, delay = 100) {
            const els = Array.isArray(elements) ? elements : [elements];
            els.forEach((el, index) => {
                el.style.opacity = '0';
                el.style.transform = 'translateY(20px)';
                setTimeout(() => {
                    el.style.transition = 'all 0.6s ease';
                    el.style.opacity = '1';
                    el.style.transform = 'translateY(0)';
                }, delay * index);
            });
        }

        static fadeOut(elements, callback) {
            const els = Array.isArray(elements) ? elements : [elements];
            els.forEach(el => {
                el.style.transition = 'all 0.3s ease';
                el.style.opacity = '0';
            });
            if (callback) {
                setTimeout(callback, 300);
            }
        }
    }

    // Enhanced LoadingManager methods
    LoadingManager.showPageLoader = function() {
        const loader = document.createElement('div');
        loader.id = 'big-page-loader';
        loader.innerHTML = `
            <div class="loader-content" style="text-align: center;">
                <div class="spinner"></div>
                <p style="margin-top: var(--space-4);">Loading...</p>
            </div>
        `;
        loader.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: var(--color-bg-primary);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 99999;
        `;
        document.body.appendChild(loader);

        return () => {
            loader.style.transition = 'opacity 0.3s ease';
            loader.style.opacity = '0';
            setTimeout(() => loader.remove(), 300);
        };
    };

    LoadingManager.showContentLoader = function(container, message = 'Loading...') {
        const loader = document.createElement('div');
        loader.className = 'content-loader';
        loader.innerHTML = `
            <div class="spinner"></div>
            <p style="margin-top: var(--space-2); color: var(--color-text-secondary);">${message}</p>
        `;
        loader.style.cssText = `
            text-align: center;
            padding: var(--space-8);
        `;
        container.appendChild(loader);
        return loader;
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
        touch: null,          // Initialized if touch device detected
        transitions: null,    // Page transitions and animations

        init() {
            console.log('%c🎮 BigPlay UI loaded', 'color: #667eea; font-weight: bold; font-size: 14px;');

            // Initialize accessibility enhancements
            AccessibilityManager.init();

            // Initialize touch enhancements (mobile only)
            if ('ontouchstart' in window || navigator.maxTouchPoints > 0) {
                this.touch = new TouchEnhancements();
                console.log('%c📱 Touch enhancements enabled', 'color: #4CAF50; font-weight: bold;');
            }

            // Initialize page transitions
            this.transitions = new PageTransitions();
            console.log('%c✨ Page transitions enabled', 'color: #9C27B0; font-weight: bold;');

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
