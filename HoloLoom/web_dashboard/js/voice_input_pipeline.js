/**
 * Voice Input Pipeline
 * Milestone 1: Web Speech API Integration (6-week MVP)
 * Date: November 2025
 *
 * Handles:
 * - Speech-to-Text (Web Speech API)
 * - Natural Language Understanding (pattern matching)
 * - Intent classification
 * - Phase 3.11 performance monitoring integration
 */

class VoiceInputPipeline {
    constructor(config = {}) {
        // Configuration
        this.config = {
            language: config.language || 'en-US',
            continuous: config.continuous !== undefined ? config.continuous : true,
            interimResults: config.interimResults !== undefined ? config.interimResults : true,
            maxAlternatives: config.maxAlternatives || 1,
            targetLatencyMs: config.targetLatencyMs || 750,
            intentConfidenceThreshold: config.intentConfidenceThreshold || 0.7,
            ...config
        };

        // Web Speech API
        this.recognition = null;
        this.isListening = false;

        // Callbacks
        this.onTranscriptCallback = null;
        this.onIntentCallback = null;
        this.onErrorCallback = null;

        // Metrics
        this.metrics = {
            totalCommands: 0,
            successfulCommands: 0,
            averageLatencyMs: 0,
            lastLatencyMs: 0,
            sttErrors: 0
        };

        // Performance monitoring (Phase 3.11 integration)
        this.performanceMonitor = null;

        // Initialize
        this._initializeWebSpeech();
    }

    /**
     * Initialize Web Speech API
     */
    _initializeWebSpeech() {
        if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
            console.error('[VoiceInputPipeline] Web Speech API not supported');
            return;
        }

        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        this.recognition = new SpeechRecognition();

        // Configure recognition
        this.recognition.continuous = this.config.continuous;
        this.recognition.interimResults = this.config.interimResults;
        this.recognition.lang = this.config.language;
        this.recognition.maxAlternatives = this.config.maxAlternatives;

        // Event handlers
        this.recognition.onstart = () => this._handleStart();
        this.recognition.onresult = (event) => this._handleResult(event);
        this.recognition.onerror = (event) => this._handleError(event);
        this.recognition.onend = () => this._handleEnd();

        console.log('[VoiceInputPipeline] Web Speech API initialized');
    }

    /**
     * Set performance monitor (Phase 3.11 integration)
     */
    setPerformanceMonitor(monitor) {
        this.performanceMonitor = monitor;
    }

    /**
     * Start listening for voice input
     */
    start() {
        if (!this.recognition) {
            console.error('[VoiceInputPipeline] Speech recognition not initialized');
            return false;
        }

        if (this.isListening) {
            console.warn('[VoiceInputPipeline] Already listening');
            return false;
        }

        try {
            this.recognition.start();
            return true;
        } catch (error) {
            console.error('[VoiceInputPipeline] Failed to start:', error);
            return false;
        }
    }

    /**
     * Stop listening
     */
    stop() {
        if (!this.recognition || !this.isListening) {
            return false;
        }

        try {
            this.recognition.stop();
            return true;
        } catch (error) {
            console.error('[VoiceInputPipeline] Failed to stop:', error);
            return false;
        }
    }

    /**
     * Handle recognition start
     */
    _handleStart() {
        this.isListening = true;
        console.log('[VoiceInputPipeline] Listening started');
    }

    /**
     * Handle recognition result
     */
    _handleResult(event) {
        const startTime = Date.now();

        // Extract transcript
        const results = event.results;
        const lastResult = results[results.length - 1];
        const transcript = lastResult[0].transcript;
        const confidence = lastResult[0].confidence;
        const isFinal = lastResult.isFinal;

        // Create voice input
        const voiceInput = {
            transcript: transcript,
            confidence: confidence,
            language: this.config.language,
            timestamp: new Date(),
            isFinal: isFinal,
            engine: 'web_speech_api'
        };

        // Call transcript callback
        if (this.onTranscriptCallback) {
            this.onTranscriptCallback(voiceInput);
        }

        // Process final results only
        if (isFinal) {
            // Classify intent
            const intent = this._classifyIntent(transcript);

            // Calculate latency
            const latencyMs = Date.now() - startTime;
            this.metrics.lastLatencyMs = latencyMs;
            this.metrics.totalCommands++;

            // Create intent with metadata
            const intentWithMetrics = {
                ...intent,
                metrics: {
                    sttLatencyMs: latencyMs,
                    sttConfidence: confidence,
                    totalLatencyMs: latencyMs // Will be updated after execution
                }
            };

            // Call intent callback
            if (this.onIntentCallback) {
                this.onIntentCallback(intentWithMetrics);
            }

            // Track successful command
            if (intent.confidence >= this.config.intentConfidenceThreshold) {
                this.metrics.successfulCommands++;
            }

            console.log(`[VoiceInputPipeline] Intent: ${intent.type} (${(intent.confidence * 100).toFixed(0)}% confident)`);
        }
    }

    /**
     * Classify intent from transcript using pattern matching
     * Milestone 1: Basic patterns for thread management
     */
    _classifyIntent(transcript) {
        const text = transcript.toLowerCase().trim();

        // Intent patterns with confidence scoring
        const patterns = [
            // Thread creation
            {
                regex: /^(?:start|create|open|new) (?:a )?(?:new )?thread(?: (?:for|about))? (.+)$/i,
                type: 'create_thread',
                confidence: 0.95,
                extract: (match) => ({ topic: match[1] })
            },

            // Thread switching
            {
                regex: /^(?:go back to|switch to|return to|open)(?: the)? (.+) thread$/i,
                type: 'switch_thread',
                confidence: 0.90,
                extract: (match) => ({ threadName: match[1] })
            },
            {
                regex: /^(?:switch|change|go) to (.+)$/i,
                type: 'switch_thread',
                confidence: 0.80,
                extract: (match) => ({ threadName: match[1] })
            },

            // Thread closing
            {
                regex: /^(?:close|end|finish)(?: the)? (.+) thread$/i,
                type: 'close_thread',
                confidence: 0.90,
                extract: (match) => ({ threadName: match[1] })
            },

            // Thread summary
            {
                regex: /^(?:summarize|summary of|what are)(?: the)? (?:active )?threads?$/i,
                type: 'summarize_threads',
                confidence: 0.95,
                extract: () => ({})
            },

            // System control
            {
                regex: /^(?:loom,? )?(?:pause|stop|halt)$/i,
                type: 'pause',
                confidence: 0.98,
                extract: () => ({})
            },
            {
                regex: /^(?:loom,? )?(?:help|what can you do)$/i,
                type: 'help',
                confidence: 0.95,
                extract: () => ({})
            },

            // Conversational
            {
                regex: /^(?:thanks?|thank you|got it|okay|ok|yes|no)$/i,
                type: 'acknowledgment',
                confidence: 0.90,
                extract: () => ({})
            },
            {
                regex: /\?$/,
                type: 'question',
                confidence: 0.70,
                extract: () => ({ question: text })
            }
        ];

        // Try to match patterns
        for (const pattern of patterns) {
            const match = text.match(pattern.regex);
            if (match) {
                return {
                    type: pattern.type,
                    confidence: pattern.confidence,
                    entities: pattern.extract(match),
                    rawTranscript: transcript,
                    requiresConfirmation: pattern.type === 'close_thread',
                    canExecuteImmediately: pattern.type !== 'close_thread',
                    timestamp: new Date()
                };
            }
        }

        // Default to statement if no pattern matches
        return {
            type: 'statement',
            confidence: 0.60,
            entities: { statement: transcript },
            rawTranscript: transcript,
            requiresConfirmation: false,
            canExecuteImmediately: true,
            timestamp: new Date()
        };
    }

    /**
     * Handle recognition error
     */
    _handleError(event) {
        this.metrics.sttErrors++;

        const errorMessage = this._getErrorMessage(event.error);
        console.error('[VoiceInputPipeline] Recognition error:', errorMessage);

        if (this.onErrorCallback) {
            this.onErrorCallback({
                type: event.error,
                message: errorMessage,
                timestamp: new Date()
            });
        }
    }

    /**
     * Handle recognition end
     */
    _handleEnd() {
        this.isListening = false;
        console.log('[VoiceInputPipeline] Listening stopped');

        // Auto-restart if continuous mode enabled
        if (this.config.continuous && this.recognition) {
            setTimeout(() => {
                if (!this.isListening) {
                    this.start();
                }
            }, 100);
        }
    }

    /**
     * Get user-friendly error message
     */
    _getErrorMessage(error) {
        const errors = {
            'no-speech': 'No speech detected. Please try again.',
            'audio-capture': 'Microphone not available. Please check permissions.',
            'not-allowed': 'Microphone access denied. Please enable in browser settings.',
            'network': 'Network error. Please check connection.',
            'aborted': 'Recognition aborted.',
            'service-not-allowed': 'Speech recognition service not available.'
        };

        return errors[error] || `Unknown error: ${error}`;
    }

    /**
     * Set callback for transcript events
     */
    onTranscript(callback) {
        this.onTranscriptCallback = callback;
    }

    /**
     * Set callback for intent events
     */
    onIntent(callback) {
        this.onIntentCallback = callback;
    }

    /**
     * Set callback for error events
     */
    onError(callback) {
        this.onErrorCallback = callback;
    }

    /**
     * Get metrics
     */
    getMetrics() {
        return {
            ...this.metrics,
            successRate: this.metrics.totalCommands > 0
                ? (this.metrics.successfulCommands / this.metrics.totalCommands) * 100
                : 0,
            averageLatencyMs: this.metrics.lastLatencyMs, // Simplified for now
            isListening: this.isListening
        };
    }

    /**
     * Reset metrics
     */
    resetMetrics() {
        this.metrics = {
            totalCommands: 0,
            successfulCommands: 0,
            averageLatencyMs: 0,
            lastLatencyMs: 0,
            sttErrors: 0
        };
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = VoiceInputPipeline;
}
