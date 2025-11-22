/**
 * Voice Orchestrator
 * Milestone 1: Complete voice UX integration
 * Date: November 2025
 *
 * Integrates:
 * - VoiceInputPipeline (STT + NLU)
 * - ThreadStateManager (thread lifecycle)
 * - ConversationalEngine (natural responses)
 * - Phase 3.11 Performance Monitoring
 */

class VoiceOrchestrator {
    constructor(analyticsMonitor = null) {
        // Components
        this.voiceInput = new VoiceInputPipeline({
            continuous: true,
            interimResults: true,
            targetLatencyMs: 750
        });

        this.threadManager = new ThreadStateManager();

        // Performance monitoring integration (Phase 3.11)
        this.analyticsMonitor = analyticsMonitor;

        // Voice mode
        this.currentMode = 'conversational'; // conversational, command, streaming, disabled

        // Text-to-Speech
        this.ttsEnabled = true;
        this.ttsVoice = null;

        // State
        this.isActive = false;
        this.waitingForConfirmation = false;
        this.pendingAction = null;

        // Callbacks
        this.onResponseCallback = null;
        this.onThreadChangeCallback = null;
        this.onErrorCallback = null;

        // Metrics
        this.sessionMetrics = {
            startTime: null,
            commandsProcessed: 0,
            threadsCreated: 0,
            threadSwitches: 0,
            errors: 0
        };

        // Initialize
        this._initializeTTS();
        this._setupEventHandlers();

        console.log('[VoiceOrchestrator] Initialized');
    }

    /**
     * Initialize Text-to-Speech
     */
    _initializeTTS() {
        if (!('speechSynthesis' in window)) {
            console.warn('[VoiceOrchestrator] Text-to-Speech not supported');
            this.ttsEnabled = false;
            return;
        }

        // Get available voices
        const loadVoices = () => {
            const voices = window.speechSynthesis.getVoices();
            // Prefer natural-sounding voices
            this.ttsVoice = voices.find(v => v.lang === 'en-US' && v.name.includes('Natural')) ||
                            voices.find(v => v.lang === 'en-US') ||
                            voices[0];

            console.log('[VoiceOrchestrator] TTS voice selected:', this.ttsVoice?.name || 'default');
        };

        loadVoices();
        if (speechSynthesis.onvoiceschanged !== undefined) {
            speechSynthesis.onvoiceschanged = loadVoices;
        }
    }

    /**
     * Setup event handlers for voice input
     */
    _setupEventHandlers() {
        // Transcript callback (interim + final)
        this.voiceInput.onTranscript((transcript) => {
            console.log(`[VoiceOrchestrator] Transcript: "${transcript.transcript}" (${transcript.isFinal ? 'final' : 'interim'})`);

            // Show interim results in UI (if callback set)
            if (this.onResponseCallback && !transcript.isFinal) {
                this.onResponseCallback({
                    type: 'interim_transcript',
                    text: transcript.transcript,
                    shouldSpeak: false
                });
            }
        });

        // Intent callback (final only)
        this.voiceInput.onIntent((intent) => {
            this._handleIntent(intent);
        });

        // Error callback
        this.voiceInput.onError((error) => {
            this._handleError(error);
        });
    }

    /**
     * Start voice interaction
     */
    start() {
        if (this.isActive) {
            console.warn('[VoiceOrchestrator] Already active');
            return false;
        }

        // Check Phase 3.11 conditions before starting
        if (this.analyticsMonitor) {
            const shouldDisable = this._checkPerformanceConstraints();
            if (shouldDisable) {
                console.warn('[VoiceOrchestrator] Voice disabled due to performance constraints');
                return false;
            }
        }

        // Start voice input
        const started = this.voiceInput.start();
        if (started) {
            this.isActive = true;
            this.sessionMetrics.startTime = new Date();
            this.speak("I'm listening. How can I help?");
            console.log('[VoiceOrchestrator] Voice interaction started');
        }

        return started;
    }

    /**
     * Stop voice interaction
     */
    stop() {
        if (!this.isActive) {
            return false;
        }

        const stopped = this.voiceInput.stop();
        if (stopped) {
            this.isActive = false;
            this.speak("Voice interaction stopped.");
            console.log('[VoiceOrchestrator] Voice interaction stopped');
        }

        return stopped;
    }

    /**
     * Check Phase 3.11 performance constraints
     */
    _checkPerformanceConstraints() {
        if (!this.analyticsMonitor) {
            return false; // No constraints if no monitor
        }

        const perf = this.analyticsMonitor.performanceMode;

        // Check battery level
        if (perf.currentBatteryLevel < 30 && !perf.isCharging) {
            console.warn('[VoiceOrchestrator] Low battery - voice disabled');
            return true; // Disable
        }

        // Check network quality (for future cloud STT)
        const network = perf.networkMonitoring;
        if (network.effectiveType === 'slow-2g' || network.effectiveType === '2g') {
            console.warn('[VoiceOrchestrator] Slow network - voice may degrade');
            // Don't disable, but warn
        }

        // Check memory usage
        const memory = perf.memoryMonitoring;
        if (memory.currentUsagePercent > 85) {
            console.warn('[VoiceOrchestrator] High memory usage - voice disabled');
            return true; // Disable
        }

        return false; // OK to proceed
    }

    /**
     * Handle classified intent
     */
    async _handleIntent(intent) {
        const startTime = Date.now();

        console.log(`[VoiceOrchestrator] Handling intent: ${intent.type}`);

        try {
            let response = null;

            // Route to appropriate handler
            switch (intent.type) {
                case 'create_thread':
                    response = await this._handleCreateThread(intent);
                    break;

                case 'switch_thread':
                    response = await this._handleSwitchThread(intent);
                    break;

                case 'close_thread':
                    response = await this._handleCloseThread(intent);
                    break;

                case 'summarize_threads':
                    response = await this._handleSummarizeThreads(intent);
                    break;

                case 'pause':
                    response = await this._handlePause(intent);
                    break;

                case 'help':
                    response = await this._handleHelp(intent);
                    break;

                case 'question':
                case 'statement':
                case 'acknowledgment':
                    response = await this._handleConversational(intent);
                    break;

                default:
                    response = {
                        text: "I didn't quite understand that. Could you rephrase?",
                        shouldSpeak: true
                    };
            }

            // Calculate total latency
            const latencyMs = Date.now() - startTime;
            response.latencyMs = latencyMs;

            // Track metrics
            this.sessionMetrics.commandsProcessed++;
            if (this.analyticsMonitor) {
                this._trackVoiceMetrics(intent, response, latencyMs);
            }

            // Send response
            this._sendResponse(response);

        } catch (error) {
            console.error('[VoiceOrchestrator] Intent handling error:', error);
            this.sessionMetrics.errors++;
            this._handleError({
                type: 'execution_error',
                message: error.message,
                timestamp: new Date()
            });
        }
    }

    /**
     * Handle: Create new thread
     */
    async _handleCreateThread(intent) {
        const topic = intent.entities.topic || 'New Thread';

        // Create thread
        const thread = this.threadManager.createThread(topic);

        // Activate it
        this.threadManager.activateThread(thread.id);

        // Track
        this.sessionMetrics.threadsCreated++;

        // Notify callback
        if (this.onThreadChangeCallback) {
            this.onThreadChangeCallback({
                type: 'thread_created',
                thread: thread
            });
        }

        return {
            text: `Opening '${topic}.' What do you want to focus on?`,
            shouldSpeak: true,
            visual_feedback: {
                type: 'thread_created',
                threadId: thread.id,
                threadName: thread.name
            }
        };
    }

    /**
     * Handle: Switch to existing thread
     */
    async _handleSwitchThread(intent) {
        const threadName = intent.entities.threadName;

        // Find matching threads
        const matches = this.threadManager.findThreadsByName(threadName);

        if (matches.length === 0) {
            return {
                text: `I don't see any threads with that name. Would you like to create a new one?`,
                shouldSpeak: true
            };
        }

        if (matches.length === 1) {
            // Single match - activate it
            const thread = matches[0].thread;
            this.threadManager.activateThread(thread.id);
            this.sessionMetrics.threadSwitches++;

            // Notify callback
            if (this.onThreadChangeCallback) {
                this.onThreadChangeCallback({
                    type: 'thread_activated',
                    thread: thread
                });
            }

            return {
                text: `Returning to '${thread.name}.' Picking up where we left off...`,
                shouldSpeak: true,
                visual_feedback: {
                    type: 'thread_activated',
                    threadId: thread.id,
                    threadName: thread.name
                }
            };
        }

        // Multiple matches - ask for clarification
        const threadList = matches.slice(0, 3).map(m => m.thread.name).join(', ');
        return {
            text: `I found ${matches.length} threads matching '${threadName}': ${threadList}. Which one?`,
            shouldSpeak: true,
            visual_feedback: {
                type: 'disambiguation',
                options: matches.map(m => ({
                    threadId: m.thread.id,
                    threadName: m.thread.name,
                    score: m.score
                }))
            }
        };
    }

    /**
     * Handle: Close thread
     */
    async _handleCloseThread(intent) {
        const threadName = intent.entities.threadName;
        const currentThread = this.threadManager.getActiveThread();

        // If no thread name specified, close current thread
        let threadToClose = null;
        if (!threadName && currentThread) {
            threadToClose = currentThread;
        } else if (threadName) {
            const matches = this.threadManager.findThreadsByName(threadName);
            if (matches.length > 0) {
                threadToClose = matches[0].thread;
            }
        }

        if (!threadToClose) {
            return {
                text: "Which thread would you like to close?",
                shouldSpeak: true
            };
        }

        // Confirmation required (high-impact action)
        if (!this.waitingForConfirmation) {
            this.waitingForConfirmation = true;
            this.pendingAction = {
                type: 'close_thread',
                threadId: threadToClose.id,
                threadName: threadToClose.name
            };

            return {
                text: `Are you sure you want to close '${threadToClose.name}'? Say 'yes' to confirm or 'no' to cancel.`,
                shouldSpeak: true
            };
        }

        // Confirmation received - execute
        this.threadManager.closeThread(threadToClose.id);
        this.waitingForConfirmation = false;
        this.pendingAction = null;

        return {
            text: `Thread '${threadToClose.name}' closed.`,
            shouldSpeak: true,
            visual_feedback: {
                type: 'thread_closed',
                threadId: threadToClose.id
            }
        };
    }

    /**
     * Handle: Summarize all threads
     */
    async _handleSummarizeThreads(intent) {
        const summaries = this.threadManager.summarizeAllThreads();

        if (summaries.length === 0) {
            return {
                text: "You don't have any active threads right now.",
                shouldSpeak: true
            };
        }

        // Build response
        let text = `${summaries.length} thread${summaries.length > 1 ? 's' : ''} currently active. `;

        for (let i = 0; i < Math.min(summaries.length, 3); i++) {
            const s = summaries[i];
            text += `${i + 1}. ${s.name}: ${s.summary}. `;
        }

        if (summaries.length > 3) {
            text += `And ${summaries.length - 3} more.`;
        }

        return {
            text: text,
            shouldSpeak: true,
            visual_feedback: {
                type: 'thread_summaries',
                summaries: summaries
            }
        };
    }

    /**
     * Handle: Pause (stop all tasks)
     */
    async _handlePause(intent) {
        this.stop();

        return {
            text: "All tasks stopped. I'm listening.",
            shouldSpeak: true
        };
    }

    /**
     * Handle: Help
     */
    async _handleHelp(intent) {
        const helpText = "I can help you manage conversation threads. " +
                        "You can create a new thread, switch between threads, " +
                        "summarize active threads, or close a thread. " +
                        "Just speak naturally and I'll understand.";

        return {
            text: helpText,
            shouldSpeak: true
        };
    }

    /**
     * Handle: Conversational (question/statement/acknowledgment)
     */
    async _handleConversational(intent) {
        const activeThread = this.threadManager.getActiveThread();

        if (!activeThread) {
            // No active thread - create one
            const thread = this.threadManager.createThread('General');
            this.threadManager.activateThread(thread.id);
        }

        // Add message to thread
        const thread = this.threadManager.getActiveThread();
        this.threadManager.addMessage(thread.id, 'user', intent.rawTranscript);

        // Generate response (Milestone 1: Simple echo/acknowledgment)
        // Milestone 2+ can integrate with LLM for better responses
        let responseText = "";

        if (intent.type === 'question') {
            responseText = "That's an interesting question. Let me think about that...";
        } else if (intent.type === 'acknowledgment') {
            responseText = "Got it!";
        } else {
            responseText = "I'm listening. Tell me more.";
        }

        // Add response to thread
        this.threadManager.addMessage(thread.id, 'assistant', responseText);

        return {
            text: responseText,
            shouldSpeak: true,
            visual_feedback: {
                type: 'conversational_response',
                threadId: thread.id
            }
        };
    }

    /**
     * Send response to user
     */
    _sendResponse(response) {
        // Speak if enabled
        if (response.shouldSpeak && this.ttsEnabled) {
            this.speak(response.text);
        }

        // Send to callback (for UI update)
        if (this.onResponseCallback) {
            this.onResponseCallback(response);
        }
    }

    /**
     * Speak text using TTS
     */
    speak(text) {
        if (!this.ttsEnabled || !('speechSynthesis' in window)) {
            return;
        }

        // Cancel any ongoing speech
        window.speechSynthesis.cancel();

        // Create utterance
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.voice = this.ttsVoice;
        utterance.rate = 1.0;
        utterance.pitch = 1.0;
        utterance.volume = 0.8;

        // Speak
        window.speechSynthesis.speak(utterance);
    }

    /**
     * Handle error
     */
    _handleError(error) {
        console.error('[VoiceOrchestrator] Error:', error);

        const errorMessage = error.message || 'An error occurred';

        // Send error response
        const response = {
            text: errorMessage,
            shouldSpeak: true,
            isError: true
        };

        this._sendResponse(response);

        // Notify callback
        if (this.onErrorCallback) {
            this.onErrorCallback(error);
        }
    }

    /**
     * Track voice metrics (Phase 3.11 integration)
     */
    _trackVoiceMetrics(intent, response, latencyMs) {
        if (!this.analyticsMonitor) {
            return;
        }

        // Get performance state
        const perf = this.analyticsMonitor.performanceMode;

        // Create voice metrics entry
        const metrics = {
            timestamp: new Date(),
            intentType: intent.type,
            intentConfidence: intent.confidence,
            latencyMs: latencyMs,
            success: !response.isError,
            batteryLevel: perf.currentBatteryLevel,
            networkType: perf.networkMonitoring.effectiveType,
            memoryUsagePercent: perf.memoryMonitoring.currentUsagePercent
        };

        // Log metrics
        console.log('[VoiceOrchestrator] Metrics:', metrics);

        // TODO: Store in analytics monitor's voice history
        // Will be implemented in next step
    }

    /**
     * Set callbacks
     */
    onResponse(callback) {
        this.onResponseCallback = callback;
    }

    onThreadChange(callback) {
        this.onThreadChangeCallback = callback;
    }

    onError(callback) {
        this.onErrorCallback = callback;
    }

    /**
     * Get session metrics
     */
    getMetrics() {
        return {
            ...this.sessionMetrics,
            voiceInput: this.voiceInput.getMetrics(),
            threadManager: this.threadManager.getMetrics(),
            isActive: this.isActive,
            currentMode: this.currentMode
        };
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = VoiceOrchestrator;
}
