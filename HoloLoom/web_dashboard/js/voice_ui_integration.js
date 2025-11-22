/**
 * Voice UI Integration
 * Milestone 1: Complete voice UX with visual feedback
 * Date: November 2025
 *
 * Integrates:
 * - VoiceOrchestrator (voice commands)
 * - ThreadListView (visual thread management)
 * - AnalyticsMonitor (performance monitoring)
 * - Phase 3.11 battery/network/memory constraints
 */

class VoiceUIIntegration {
    constructor(config = {}) {
        // Configuration
        this.config = {
            containerSelector: config.containerSelector || '#thread-list-container',
            enableVoiceButton: config.enableVoiceButton !== false,
            enableMetrics: config.enableMetrics !== false,
            enablePerformanceMonitoring: config.enablePerformanceMonitoring !== false,
            ...config
        };

        // Components
        this.analyticsMonitor = null;
        this.threadManager = null;
        this.threadListView = null;
        this.voiceOrchestrator = null;

        // UI Elements
        this.container = null;
        this.voiceButton = null;
        this.statusIndicator = null;
        this.transcriptDisplay = null;
        this.metricsPanel = null;

        // State
        this.isVoiceActive = false;

        console.log('[VoiceUIIntegration] Initialized');
    }

    /**
     * Initialize all components
     */
    async initialize() {
        console.log('[VoiceUIIntegration] Starting initialization...');

        // Get container
        this.container = document.querySelector(this.config.containerSelector);
        if (!this.container) {
            throw new Error(`Container not found: ${this.config.containerSelector}`);
        }

        // Create analytics monitor (if enabled)
        if (this.config.enablePerformanceMonitoring) {
            this.analyticsMonitor = new AnalyticsMonitor();
            this.analyticsMonitor.performanceMode.voiceUX.enabled = true;
            console.log('[VoiceUIIntegration] Performance monitoring enabled');
        }

        // Create thread manager
        this.threadManager = new ThreadStateManager();

        // Create thread list view
        const listContainer = this._createListContainer();
        this.threadListView = new ThreadListView(listContainer, this.threadManager, {
            onCreate: (topic) => this._handleCreateThread(topic),
            onActivate: (thread) => this._handleActivateThread(thread),
            onArchive: (thread) => this._handleArchiveThread(thread)
        });

        // Create voice orchestrator
        this.voiceOrchestrator = new VoiceOrchestrator(this.analyticsMonitor);

        // Set voice orchestrator callbacks
        this.voiceOrchestrator.onResponse((response) => this._handleVoiceResponse(response));
        this.voiceOrchestrator.onThreadChange((event) => this._handleThreadChange(event));
        this.voiceOrchestrator.onError((error) => this._handleVoiceError(error));

        // Create UI controls
        if (this.config.enableVoiceButton) {
            this._createVoiceControls();
        }

        if (this.config.enableMetrics) {
            this._createMetricsPanel();
        }

        console.log('[VoiceUIIntegration] Initialization complete');
    }

    /**
     * Create list container
     */
    _createListContainer() {
        const listContainer = document.createElement('div');
        listContainer.id = 'thread-list-container';
        this.container.appendChild(listContainer);
        return listContainer;
    }

    /**
     * Create voice controls (button, status, transcript)
     */
    _createVoiceControls() {
        const controlsContainer = document.createElement('div');
        controlsContainer.className = 'voice-controls';

        // Voice button
        this.voiceButton = document.createElement('button');
        this.voiceButton.className = 'voice-controls__button voice-controls__button--inactive';
        this.voiceButton.innerHTML = '🎤 Start Voice';
        this.voiceButton.onclick = () => this._toggleVoice();

        // Status indicator
        this.statusIndicator = document.createElement('div');
        this.statusIndicator.className = 'voice-controls__status';
        this.statusIndicator.textContent = 'Voice: Inactive';

        // Transcript display
        this.transcriptDisplay = document.createElement('div');
        this.transcriptDisplay.className = 'voice-controls__transcript';
        this.transcriptDisplay.textContent = 'Transcript will appear here...';

        controlsContainer.appendChild(this.voiceButton);
        controlsContainer.appendChild(this.statusIndicator);
        controlsContainer.appendChild(this.transcriptDisplay);

        this.container.insertBefore(controlsContainer, this.container.firstChild);
    }

    /**
     * Create metrics panel
     */
    _createMetricsPanel() {
        this.metricsPanel = document.createElement('div');
        this.metricsPanel.className = 'voice-metrics-panel';
        this.metricsPanel.innerHTML = '<h3>Voice Metrics</h3><div class="voice-metrics-panel__content"></div>';

        this.container.appendChild(this.metricsPanel);

        // Update metrics every 2 seconds
        setInterval(() => this._updateMetrics(), 2000);
    }

    /**
     * Toggle voice on/off
     */
    _toggleVoice() {
        if (this.isVoiceActive) {
            this._stopVoice();
        } else {
            this._startVoice();
        }
    }

    /**
     * Start voice interaction
     */
    _startVoice() {
        // Check performance constraints
        if (this.analyticsMonitor) {
            const shouldDisable = this.analyticsMonitor.shouldDisableVoice();
            if (shouldDisable.shouldDisable) {
                const reasons = {
                    'low_battery': 'Battery too low (<30%). Please charge device.',
                    'high_memory': 'Memory usage too high (>85%). Please close apps.',
                    'very_slow_network': 'Network too slow. Voice may not work properly.'
                };

                alert(reasons[shouldDisable.reason] || 'Voice cannot start due to performance constraints.');
                return;
            }
        }

        // Start voice orchestrator
        const started = this.voiceOrchestrator.start();

        if (started) {
            this.isVoiceActive = true;
            this.voiceButton.className = 'voice-controls__button voice-controls__button--active';
            this.voiceButton.innerHTML = '🎤 Stop Voice';
            this.statusIndicator.textContent = 'Voice: Listening...';
            this.statusIndicator.className = 'voice-controls__status voice-controls__status--active';

            // Start voice session in analytics
            if (this.analyticsMonitor) {
                this.analyticsMonitor.startVoiceSession();
            }

            console.log('[VoiceUIIntegration] Voice started');
        }
    }

    /**
     * Stop voice interaction
     */
    _stopVoice() {
        this.voiceOrchestrator.stop();

        this.isVoiceActive = false;
        this.voiceButton.className = 'voice-controls__button voice-controls__button--inactive';
        this.voiceButton.innerHTML = '🎤 Start Voice';
        this.statusIndicator.textContent = 'Voice: Inactive';
        this.statusIndicator.className = 'voice-controls__status';

        // Stop voice session in analytics
        if (this.analyticsMonitor) {
            this.analyticsMonitor.stopVoiceSession();
        }

        console.log('[VoiceUIIntegration] Voice stopped');
    }

    /**
     * Handle voice response
     */
    _handleVoiceResponse(response) {
        console.log('[VoiceUIIntegration] Voice response:', response);

        // Update transcript display
        if (this.transcriptDisplay) {
            if (response.type === 'interim_transcript') {
                this.transcriptDisplay.textContent = `🎤 ${response.text}`;
                this.transcriptDisplay.className = 'voice-controls__transcript voice-controls__transcript--interim';
            } else {
                this.transcriptDisplay.textContent = `💬 ${response.text}`;
                this.transcriptDisplay.className = 'voice-controls__transcript voice-controls__transcript--final';

                // Clear after 5 seconds
                setTimeout(() => {
                    if (this.transcriptDisplay.textContent === `💬 ${response.text}`) {
                        this.transcriptDisplay.textContent = 'Listening...';
                        this.transcriptDisplay.className = 'voice-controls__transcript';
                    }
                }, 5000);
            }
        }

        // Handle visual feedback
        if (response.visual_feedback) {
            this._handleVisualFeedback(response.visual_feedback);
        }
    }

    /**
     * Handle visual feedback
     */
    _handleVisualFeedback(feedback) {
        if (feedback.type === 'thread_created') {
            // Thread will be added via onThreadChange callback
            console.log('[VoiceUIIntegration] Thread created:', feedback.threadName);
        }

        if (feedback.type === 'thread_activated') {
            // Thread will be updated via onThreadChange callback
            console.log('[VoiceUIIntegration] Thread activated:', feedback.threadName);
        }

        if (feedback.type === 'thread_closed') {
            console.log('[VoiceUIIntegration] Thread closed:', feedback.threadId);
        }

        if (feedback.type === 'disambiguation') {
            // Show disambiguation UI (future enhancement)
            console.log('[VoiceUIIntegration] Multiple threads found:', feedback.options);
        }
    }

    /**
     * Handle thread change event from voice
     */
    _handleThreadChange(event) {
        console.log('[VoiceUIIntegration] Thread change:', event);

        if (event.type === 'thread_created') {
            this.threadListView.addThread(event.thread);

            // Track in analytics
            if (this.analyticsMonitor) {
                this.analyticsMonitor.trackThreadCreated();
            }
        }

        if (event.type === 'thread_activated') {
            this.threadListView.updateThread(event.thread);

            // Update all other threads to background
            const allThreads = Array.from(this.threadManager.threads.values());
            for (const thread of allThreads) {
                if (thread.id !== event.thread.id) {
                    this.threadListView.updateThread(thread);
                }
            }

            // Track in analytics
            if (this.analyticsMonitor) {
                this.analyticsMonitor.trackThreadSwitch();
            }
        }

        if (event.type === 'thread_closed') {
            // Update thread to archived state
            const thread = this.threadManager.threads.get(event.threadId);
            if (thread) {
                this.threadListView.updateThread(thread);
            }
        }
    }

    /**
     * Handle voice error
     */
    _handleVoiceError(error) {
        console.error('[VoiceUIIntegration] Voice error:', error);

        // Show error in transcript
        if (this.transcriptDisplay) {
            this.transcriptDisplay.textContent = `❌ Error: ${error.message}`;
            this.transcriptDisplay.className = 'voice-controls__transcript voice-controls__transcript--error';

            setTimeout(() => {
                this.transcriptDisplay.textContent = 'Listening...';
                this.transcriptDisplay.className = 'voice-controls__transcript';
            }, 5000);
        }

        // Track in analytics
        if (this.analyticsMonitor) {
            this.analyticsMonitor.trackVoiceError(error.type);
        }
    }

    /**
     * Handle create thread (from UI button)
     */
    _handleCreateThread(topic) {
        const thread = this.threadManager.createThread(topic);
        this.threadManager.activateThread(thread.id);
        this.threadListView.renderAll();

        console.log('[VoiceUIIntegration] Thread created via UI:', topic);
    }

    /**
     * Handle activate thread (from UI click)
     */
    _handleActivateThread(thread) {
        this.threadManager.activateThread(thread.id);
        this.threadListView.renderAll();

        console.log('[VoiceUIIntegration] Thread activated via UI:', thread.name);
    }

    /**
     * Handle archive thread (from UI button)
     */
    _handleArchiveThread(thread) {
        this.threadManager.closeThread(thread.id);
        this.threadListView.renderAll();

        console.log('[VoiceUIIntegration] Thread archived via UI:', thread.name);
    }

    /**
     * Update metrics panel
     */
    _updateMetrics() {
        if (!this.metricsPanel) return;

        const content = this.metricsPanel.querySelector('.voice-metrics-panel__content');
        if (!content) return;

        // Get voice metrics
        const voiceMetrics = this.voiceOrchestrator.getMetrics();
        const threadMetrics = this.threadListView.getMetrics();

        let html = `
            <div class="metric">
                <span class="metric__label">Voice Active:</span>
                <span class="metric__value">${voiceMetrics.isActive ? 'Yes' : 'No'}</span>
            </div>
            <div class="metric">
                <span class="metric__label">Mode:</span>
                <span class="metric__value">${voiceMetrics.currentMode}</span>
            </div>
            <div class="metric">
                <span class="metric__label">Commands Processed:</span>
                <span class="metric__value">${voiceMetrics.voiceInput.totalCommands}</span>
            </div>
            <div class="metric">
                <span class="metric__label">Success Rate:</span>
                <span class="metric__value">${voiceMetrics.voiceInput.successRate.toFixed(0)}%</span>
            </div>
            <div class="metric">
                <span class="metric__label">Avg Latency:</span>
                <span class="metric__value">${voiceMetrics.voiceInput.lastLatencyMs}ms</span>
            </div>
            <div class="metric">
                <span class="metric__label">Total Threads:</span>
                <span class="metric__value">${threadMetrics.totalThreads}</span>
            </div>
            <div class="metric">
                <span class="metric__label">Active Threads:</span>
                <span class="metric__value">${threadMetrics.activeThreads}</span>
            </div>
        `;

        // Add performance metrics if available
        if (this.analyticsMonitor) {
            const perf = this.analyticsMonitor.performanceMode;
            html += `
                <div class="metric">
                    <span class="metric__label">Battery:</span>
                    <span class="metric__value">${perf.currentBatteryLevel}%</span>
                </div>
                <div class="metric">
                    <span class="metric__label">Network:</span>
                    <span class="metric__value">${perf.networkMonitoring.effectiveType}</span>
                </div>
                <div class="metric">
                    <span class="metric__label">Memory:</span>
                    <span class="metric__value">${perf.memoryMonitoring.currentUsagePercent.toFixed(0)}%</span>
                </div>
            `;
        }

        content.innerHTML = html;
    }

    /**
     * Get current state
     */
    getState() {
        return {
            isVoiceActive: this.isVoiceActive,
            voiceMetrics: this.voiceOrchestrator.getMetrics(),
            threadMetrics: this.threadListView.getMetrics()
        };
    }

    /**
     * Cleanup
     */
    destroy() {
        if (this.isVoiceActive) {
            this._stopVoice();
        }

        console.log('[VoiceUIIntegration] Destroyed');
    }
}


// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = VoiceUIIntegration;
}
