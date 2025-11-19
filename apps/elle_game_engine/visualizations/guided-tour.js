/**
 * BigPlay Guided Tour System
 * Version: 1.0.0
 *
 * Interactive step-by-step walkthroughs for all visualizations
 * - Highlight specific elements
 * - Show contextual information
 * - Track progress
 * - Keyboard shortcuts
 * - Auto-save state
 */

class GuidedTour {
    constructor(steps, options = {}) {
        this.steps = steps;
        this.currentStep = 0;
        this.options = {
            storageKey: options.storageKey || 'bigplay-tour-progress',
            autoStart: options.autoStart || false,
            showProgress: options.showProgress !== false,
            allowSkip: options.allowSkip !== false,
            ...options
        };

        this.overlay = null;
        this.spotlight = null;
        this.tooltip = null;
        this.controls = null;
        this.isActive = false;

        this.createUI();
        this.attachKeyboardHandlers();

        if (this.options.autoStart && !this.isCompleted()) {
            this.start();
        }
    }

    createUI() {
        // Overlay (dims background)
        this.overlay = document.createElement('div');
        this.overlay.className = 'tour-overlay';
        this.overlay.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.7);
            z-index: var(--z-modal-backdrop);
            opacity: 0;
            transition: opacity 0.3s ease;
            pointer-events: none;
        `;

        // Spotlight (highlights element)
        this.spotlight = document.createElement('div');
        this.spotlight.className = 'tour-spotlight';
        this.spotlight.style.cssText = `
            position: absolute;
            border: 3px solid var(--color-primary);
            border-radius: var(--radius-lg);
            box-shadow: 0 0 0 9999px rgba(0, 0, 0, 0.7),
                        0 0 20px var(--color-primary);
            transition: all 0.5s cubic-bezier(0.4, 0, 0.2, 1);
            pointer-events: none;
            z-index: var(--z-modal);
        `;

        // Tooltip (shows step content)
        this.tooltip = document.createElement('div');
        this.tooltip.className = 'tour-tooltip';
        this.tooltip.style.cssText = `
            position: absolute;
            max-width: 400px;
            background: var(--color-bg-elevated);
            border: 1px solid var(--color-border);
            border-radius: var(--radius-xl);
            padding: var(--space-6);
            box-shadow: var(--shadow-2xl);
            z-index: calc(var(--z-modal) + 1);
            opacity: 0;
            transform: translateY(10px);
            transition: all 0.3s ease;
        `;

        // Controls (progress, next/prev buttons)
        this.controls = document.createElement('div');
        this.controls.className = 'tour-controls';
        this.controls.style.cssText = `
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: var(--space-4);
            gap: var(--space-4);
        `;

        document.body.appendChild(this.overlay);
        document.body.appendChild(this.spotlight);
        document.body.appendChild(this.tooltip);
    }

    start() {
        if (this.isActive) return;

        this.isActive = true;
        this.currentStep = this.getProgress();

        // Show overlay
        this.overlay.style.pointerEvents = 'auto';
        this.overlay.style.opacity = '1';

        // Show first step
        this.showStep(this.currentStep);

        // Track analytics
        if (window.BigPlayUI?.analytics) {
            window.BigPlayUI.analytics.track('tour_started', {
                tourName: this.options.name || 'unnamed'
            });
        }

        // Announce to screen readers
        if (window.BigPlayUI?.accessibility) {
            window.BigPlayUI.accessibility.announce('Tour started. Use arrow keys to navigate, Escape to exit.');
        }
    }

    showStep(stepIndex) {
        if (stepIndex < 0 || stepIndex >= this.steps.length) return;

        const step = this.steps[stepIndex];
        this.currentStep = stepIndex;

        // Find target element
        const target = document.querySelector(step.target);
        if (!target) {
            console.warn(`Tour step ${stepIndex}: Target not found - ${step.target}`);
            return;
        }

        // Execute pre-action (e.g., scroll into view, expand section)
        if (step.before) {
            step.before(target);
        }

        // Highlight target
        this.highlightElement(target);

        // Position and show tooltip
        this.showTooltip(target, step);

        // Save progress
        this.saveProgress(stepIndex);

        // Track analytics
        if (window.BigPlayUI?.analytics) {
            window.BigPlayUI.analytics.track('tour_step_viewed', {
                step: stepIndex,
                title: step.title
            });
        }
    }

    highlightElement(element) {
        const rect = element.getBoundingClientRect();
        const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
        const scrollLeft = window.pageXOffset || document.documentElement.scrollLeft;

        // Position spotlight
        this.spotlight.style.top = (rect.top + scrollTop - 8) + 'px';
        this.spotlight.style.left = (rect.left + scrollLeft - 8) + 'px';
        this.spotlight.style.width = (rect.width + 16) + 'px';
        this.spotlight.style.height = (rect.height + 16) + 'px';

        // Scroll into view if needed
        const buffer = 100;
        if (rect.top < buffer || rect.bottom > window.innerHeight - buffer) {
            element.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }

        // Add pulse animation
        this.spotlight.style.animation = 'tour-pulse 2s ease-in-out infinite';
    }

    showTooltip(target, step) {
        const rect = target.getBoundingClientRect();
        const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
        const scrollLeft = window.pageXOffset || document.documentElement.scrollLeft;

        // Build tooltip content
        this.tooltip.innerHTML = `
            <div class="tour-tooltip-header">
                ${this.options.showProgress ?
                    `<div class="tour-progress" style="font-size: var(--font-size-sm); color: var(--color-text-secondary); margin-bottom: var(--space-2);">
                        Step ${this.currentStep + 1} of ${this.steps.length}
                    </div>` : ''}
                <h3 style="margin: 0 0 var(--space-3) 0; color: var(--color-primary);">
                    ${step.title}
                </h3>
            </div>
            <div class="tour-tooltip-content" style="margin-bottom: var(--space-4); line-height: 1.6;">
                ${step.content}
            </div>
            ${this.controls.outerHTML}
        `;

        // Position tooltip (try bottom, then top, then right, then left)
        const tooltipRect = this.tooltip.getBoundingClientRect();
        const positions = [
            { // Bottom
                top: rect.bottom + scrollTop + 16,
                left: rect.left + scrollLeft + (rect.width / 2) - (tooltipRect.width / 2)
            },
            { // Top
                top: rect.top + scrollTop - tooltipRect.height - 16,
                left: rect.left + scrollLeft + (rect.width / 2) - (tooltipRect.width / 2)
            },
            { // Right
                top: rect.top + scrollTop + (rect.height / 2) - (tooltipRect.height / 2),
                left: rect.right + scrollLeft + 16
            },
            { // Left
                top: rect.top + scrollTop + (rect.height / 2) - (tooltipRect.height / 2),
                left: rect.left + scrollLeft - tooltipRect.width - 16
            }
        ];

        // Find best position (stays in viewport)
        let bestPosition = positions[0];
        for (const pos of positions) {
            if (pos.top >= 0 && pos.top + tooltipRect.height <= document.documentElement.scrollHeight &&
                pos.left >= 0 && pos.left + tooltipRect.width <= document.documentElement.scrollWidth) {
                bestPosition = pos;
                break;
            }
        }

        this.tooltip.style.top = bestPosition.top + 'px';
        this.tooltip.style.left = bestPosition.left + 'px';
        this.tooltip.style.opacity = '1';
        this.tooltip.style.transform = 'translateY(0)';

        // Rebuild controls with event listeners
        this.rebuildControls();
    }

    rebuildControls() {
        const controlsContainer = this.tooltip.querySelector('.tour-controls');
        if (!controlsContainer) return;

        controlsContainer.innerHTML = `
            <div class="tour-controls-left">
                ${this.currentStep > 0 ?
                    '<button class="btn btn-ghost tour-btn-prev">← Previous</button>' :
                    '<div></div>'}
            </div>
            <div class="tour-controls-center">
                <div class="tour-progress-dots" style="display: flex; gap: var(--space-2);"></div>
            </div>
            <div class="tour-controls-right" style="display: flex; gap: var(--space-2);">
                ${this.options.allowSkip ?
                    '<button class="btn btn-ghost tour-btn-skip">Skip Tour</button>' : ''}
                <button class="btn btn-primary tour-btn-next">
                    ${this.currentStep < this.steps.length - 1 ? 'Next →' : 'Finish'}
                </button>
            </div>
        `;

        // Add progress dots
        const dotsContainer = controlsContainer.querySelector('.tour-progress-dots');
        if (dotsContainer && this.options.showProgress) {
            for (let i = 0; i < this.steps.length; i++) {
                const dot = document.createElement('div');
                dot.style.cssText = `
                    width: 8px;
                    height: 8px;
                    border-radius: 50%;
                    background: ${i === this.currentStep ? 'var(--color-primary)' : 'var(--color-bg-tertiary)'};
                    transition: background 0.3s ease;
                    cursor: pointer;
                `;
                dot.addEventListener('click', () => this.showStep(i));
                dotsContainer.appendChild(dot);
            }
        }

        // Attach event listeners
        const prevBtn = controlsContainer.querySelector('.tour-btn-prev');
        const nextBtn = controlsContainer.querySelector('.tour-btn-next');
        const skipBtn = controlsContainer.querySelector('.tour-btn-skip');

        if (prevBtn) {
            prevBtn.addEventListener('click', () => this.previous());
        }
        if (nextBtn) {
            nextBtn.addEventListener('click', () => this.next());
        }
        if (skipBtn) {
            skipBtn.addEventListener('click', () => this.end());
        }
    }

    next() {
        if (this.currentStep < this.steps.length - 1) {
            this.showStep(this.currentStep + 1);
        } else {
            this.complete();
        }
    }

    previous() {
        if (this.currentStep > 0) {
            this.showStep(this.currentStep - 1);
        }
    }

    end() {
        this.isActive = false;

        // Hide UI
        this.overlay.style.opacity = '0';
        this.tooltip.style.opacity = '0';
        this.spotlight.style.opacity = '0';

        setTimeout(() => {
            this.overlay.style.pointerEvents = 'none';
        }, 300);

        // Track analytics
        if (window.BigPlayUI?.analytics) {
            window.BigPlayUI.analytics.track('tour_ended', {
                step: this.currentStep,
                completed: false
            });
        }

        // Announce
        if (window.BigPlayUI?.accessibility) {
            window.BigPlayUI.accessibility.announce('Tour ended.');
        }
    }

    complete() {
        this.markCompleted();

        // Track analytics
        if (window.BigPlayUI?.analytics) {
            window.BigPlayUI.analytics.track('tour_completed', {
                tourName: this.options.name || 'unnamed'
            });
        }

        // Show completion message
        this.showCompletionMessage();

        setTimeout(() => {
            this.end();
        }, 2000);
    }

    showCompletionMessage() {
        this.tooltip.innerHTML = `
            <div style="text-align: center; padding: var(--space-8);">
                <div style="font-size: 3em; margin-bottom: var(--space-4);">🎉</div>
                <h2 style="margin-bottom: var(--space-2); color: var(--color-success);">Tour Complete!</h2>
                <p style="color: var(--color-text-secondary);">
                    You've learned how ${this.options.name || 'this feature'} works.
                </p>
            </div>
        `;
    }

    attachKeyboardHandlers() {
        document.addEventListener('keydown', (e) => {
            if (!this.isActive) return;

            switch (e.key) {
                case 'ArrowRight':
                case 'n':
                case 'N':
                    e.preventDefault();
                    this.next();
                    break;
                case 'ArrowLeft':
                case 'p':
                case 'P':
                    e.preventDefault();
                    this.previous();
                    break;
                case 'Escape':
                    e.preventDefault();
                    this.end();
                    break;
            }
        });
    }

    // Progress management
    saveProgress(stepIndex) {
        localStorage.setItem(this.options.storageKey, stepIndex.toString());
    }

    getProgress() {
        const saved = localStorage.getItem(this.options.storageKey);
        return saved ? parseInt(saved, 10) : 0;
    }

    markCompleted() {
        localStorage.setItem(this.options.storageKey + '-completed', 'true');
    }

    isCompleted() {
        return localStorage.getItem(this.options.storageKey + '-completed') === 'true';
    }

    resetProgress() {
        localStorage.removeItem(this.options.storageKey);
        localStorage.removeItem(this.options.storageKey + '-completed');
    }
}

// Add pulse animation
const style = document.createElement('style');
style.textContent = `
    @keyframes tour-pulse {
        0%, 100% {
            box-shadow: 0 0 0 9999px rgba(0, 0, 0, 0.7),
                        0 0 20px var(--color-primary);
        }
        50% {
            box-shadow: 0 0 0 9999px rgba(0, 0, 0, 0.7),
                        0 0 40px var(--color-primary),
                        0 0 60px var(--color-primary);
        }
    }
`;
document.head.appendChild(style);

// Export
window.GuidedTour = GuidedTour;
