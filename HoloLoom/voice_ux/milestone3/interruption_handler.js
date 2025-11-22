/**
 * Interruption Handler - Barge-in Support
 * Milestone 3: User can interrupt ongoing responses
 * Date: November 2025
 *
 * Features:
 * - Real-time barge-in detection (user starts speaking)
 * - Graceful interruption (stop response, handle new input)
 * - Interruption state management
 * - Recovery and resumption support
 * - Metrics tracking (interruption rate, patterns)
 */

/**
 * Interruption States
 */
const InterruptionState = {
    IDLE: 'idle',                    // Not speaking, not listening for interruptions
    SPEAKING: 'speaking',            // System is speaking, monitoring for interruptions
    INTERRUPTED: 'interrupted',      // User has interrupted, processing new input
    RECOVERING: 'recovering'         // Resuming after interruption
};

/**
 * Interruption Handler
 * Manages barge-in detection and graceful interruption handling
 */
class InterruptionHandler {
    constructor(config = {}) {
        this.config = {
            enabled: config.enabled !== false,

            // Barge-in detection
            interruptionThresholdMs: config.interruptionThresholdMs || 300, // Min duration to trigger interrupt
            confidenceThreshold: config.confidenceThreshold || 0.5,         // Min confidence for interrupt
            volumeThreshold: config.volumeThreshold || 0.15,                // Min volume level (0.0-1.0)

            // Recovery
            allowResumption: config.allowResumption !== false,              // Can resume interrupted response
            resumptionTimeoutMs: config.resumptionTimeoutMs || 5000,        // Timeout before discarding interrupted response

            // Behavior
            gracefulStopMs: config.gracefulStopMs || 200,                   // Time to gracefully stop response
            clearQueueOnInterrupt: config.clearQueueOnInterrupt !== false,  // Clear pending actions

            ...config
        };

        this.state = InterruptionState.IDLE;
        this.currentResponse = null;
        this.interruptedResponse = null;
        this.interruptionStartTime = null;
        this.resumptionTimer = null;

        // Audio monitoring
        this.audioContext = null;
        this.analyser = null;
        this.volumeDataArray = null;

        // Callbacks
        this.onInterruptCallback = null;
        this.onResumeCallback = null;
        this.onDiscardCallback = null;

        // Metrics
        this.metrics = {
            totalInterruptions: 0,
            totalResumptions: 0,
            totalDiscards: 0,
            avgInterruptionDuration: 0,
            interruptionPatterns: {
                early: 0,      // Interrupted in first 25% of response
                mid: 0,        // Interrupted in middle 50%
                late: 0        // Interrupted in last 25%
            }
        };

        console.log('[InterruptionHandler] Initialized', this.config);
    }

    /**
     * Initialize audio monitoring for barge-in detection
     */
    async initialize(audioStream) {
        if (!this.config.enabled) {
            console.log('[InterruptionHandler] Disabled, skipping initialization');
            return false;
        }

        try {
            // Create audio context
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();

            // Create analyser for volume monitoring
            this.analyser = this.audioContext.createAnalyser();
            this.analyser.fftSize = 256;
            this.analyser.smoothingTimeConstant = 0.3;

            const bufferLength = this.analyser.frequencyBinCount;
            this.volumeDataArray = new Uint8Array(bufferLength);

            // Connect audio stream
            const source = this.audioContext.createMediaStreamSource(audioStream);
            source.connect(this.analyser);

            console.log('[InterruptionHandler] Audio monitoring initialized');
            return true;

        } catch (error) {
            console.error('[InterruptionHandler] Initialization failed:', error);
            return false;
        }
    }

    /**
     * Start monitoring for interruptions
     * @param {Object} response - Current response being delivered
     */
    startMonitoring(response) {
        if (!this.config.enabled || this.state === InterruptionState.SPEAKING) {
            return;
        }

        this.state = InterruptionState.SPEAKING;
        this.currentResponse = {
            ...response,
            startTime: Date.now(),
            estimatedDuration: this._estimateResponseDuration(response),
            interrupted: false
        };

        // Clear any existing resumption timer
        if (this.resumptionTimer) {
            clearTimeout(this.resumptionTimer);
            this.resumptionTimer = null;
        }

        // Start barge-in detection loop
        this._startBargeInDetection();

        console.log('[InterruptionHandler] Monitoring started for response:', response.text?.substring(0, 50));
    }

    /**
     * Stop monitoring for interruptions
     */
    stopMonitoring() {
        if (this.state !== InterruptionState.SPEAKING) {
            return;
        }

        this.state = InterruptionState.IDLE;
        this.currentResponse = null;

        console.log('[InterruptionHandler] Monitoring stopped');
    }

    /**
     * Handle user interruption (barge-in)
     * @param {Object} userInput - User's interrupting input
     */
    async handleInterruption(userInput) {
        if (this.state !== InterruptionState.SPEAKING) {
            console.warn('[InterruptionHandler] Not in SPEAKING state, ignoring interruption');
            return null;
        }

        const interruptionTime = Date.now();
        const responseDuration = interruptionTime - this.currentResponse.startTime;

        // Update state
        this.state = InterruptionState.INTERRUPTED;
        this.interruptionStartTime = interruptionTime;

        // Save interrupted response
        this.interruptedResponse = {
            ...this.currentResponse,
            interruptedAt: interruptionTime,
            progress: responseDuration / this.currentResponse.estimatedDuration
        };

        // Update metrics
        this.metrics.totalInterruptions++;
        this._updateInterruptionPattern(this.interruptedResponse.progress);
        this.metrics.avgInterruptionDuration =
            (this.metrics.avgInterruptionDuration * (this.metrics.totalInterruptions - 1) + responseDuration) /
            this.metrics.totalInterruptions;

        // Gracefully stop current response
        await this._gracefulStop();

        // Emit interruption event
        if (this.onInterruptCallback) {
            this.onInterruptCallback({
                type: 'interrupted',
                interruptedResponse: this.interruptedResponse,
                userInput: userInput,
                timestamp: new Date()
            });
        }

        // Start resumption timer if enabled
        if (this.config.allowResumption) {
            this._startResumptionTimer();
        } else {
            this._discardInterruptedResponse();
        }

        console.log('[InterruptionHandler] Interrupted after', responseDuration, 'ms');

        return {
            interrupted: true,
            savedResponse: this.config.allowResumption ? this.interruptedResponse : null,
            userInput: userInput
        };
    }

    /**
     * Resume interrupted response
     */
    async resumeInterruptedResponse() {
        if (!this.interruptedResponse) {
            console.warn('[InterruptionHandler] No interrupted response to resume');
            return null;
        }

        this.state = InterruptionState.RECOVERING;

        const resumedResponse = {
            ...this.interruptedResponse,
            resumed: true,
            resumeTime: Date.now()
        };

        // Update metrics
        this.metrics.totalResumptions++;

        // Emit resume event
        if (this.onResumeCallback) {
            this.onResumeCallback({
                type: 'resumed',
                response: resumedResponse,
                timestamp: new Date()
            });
        }

        // Clear interrupted response
        this.interruptedResponse = null;
        this.interruptionStartTime = null;

        // Clear resumption timer
        if (this.resumptionTimer) {
            clearTimeout(this.resumptionTimer);
            this.resumptionTimer = null;
        }

        console.log('[InterruptionHandler] Resumed interrupted response');

        return resumedResponse;
    }

    /**
     * Discard interrupted response
     */
    _discardInterruptedResponse() {
        if (!this.interruptedResponse) {
            return;
        }

        // Update metrics
        this.metrics.totalDiscards++;

        // Emit discard event
        if (this.onDiscardCallback) {
            this.onDiscardCallback({
                type: 'discarded',
                response: this.interruptedResponse,
                timestamp: new Date()
            });
        }

        console.log('[InterruptionHandler] Discarded interrupted response');

        this.interruptedResponse = null;
        this.interruptionStartTime = null;
        this.state = InterruptionState.IDLE;
    }

    /**
     * Start barge-in detection loop
     */
    _startBargeInDetection() {
        if (!this.analyser || this.state !== InterruptionState.SPEAKING) {
            return;
        }

        const detect = () => {
            if (this.state !== InterruptionState.SPEAKING) {
                return; // Stop detection loop
            }

            // Get current volume level
            const volume = this._getCurrentVolume();

            // Check if user is speaking (volume above threshold)
            if (volume > this.config.volumeThreshold) {
                // User might be interrupting
                console.log('[InterruptionHandler] Possible barge-in detected, volume:', volume.toFixed(2));

                // Note: Actual interruption handling triggered by transcript confidence
                // This is just pre-detection via volume monitoring
            }

            // Continue detection loop (every 100ms)
            setTimeout(detect, 100);
        };

        detect();
    }

    /**
     * Get current volume level from audio analyser
     */
    _getCurrentVolume() {
        if (!this.analyser || !this.volumeDataArray) {
            return 0.0;
        }

        this.analyser.getByteFrequencyData(this.volumeDataArray);

        // Calculate average volume (0-255 range)
        let sum = 0;
        for (let i = 0; i < this.volumeDataArray.length; i++) {
            sum += this.volumeDataArray[i];
        }
        const average = sum / this.volumeDataArray.length;

        // Normalize to 0.0-1.0
        return average / 255.0;
    }

    /**
     * Gracefully stop current response
     */
    async _gracefulStop() {
        // Allow graceful stopping period
        await new Promise(resolve => setTimeout(resolve, this.config.gracefulStopMs));

        // Stop TTS if active
        if (window.speechSynthesis && window.speechSynthesis.speaking) {
            window.speechSynthesis.cancel();
        }

        // Clear action queue if configured
        if (this.config.clearQueueOnInterrupt) {
            // Signal to clear any pending actions
            console.log('[InterruptionHandler] Clearing action queue');
        }

        console.log('[InterruptionHandler] Graceful stop complete');
    }

    /**
     * Start resumption timer
     */
    _startResumptionTimer() {
        if (this.resumptionTimer) {
            clearTimeout(this.resumptionTimer);
        }

        this.resumptionTimer = setTimeout(() => {
            console.log('[InterruptionHandler] Resumption timeout, discarding response');
            this._discardInterruptedResponse();
        }, this.config.resumptionTimeoutMs);
    }

    /**
     * Estimate response duration based on text length
     */
    _estimateResponseDuration(response) {
        if (!response.text) {
            return 1000; // Default 1 second
        }

        // Rough estimate: ~150 words per minute = 2.5 words per second
        const wordCount = response.text.split(/\s+/).length;
        const estimatedSeconds = wordCount / 2.5;

        return estimatedSeconds * 1000; // Convert to milliseconds
    }

    /**
     * Update interruption pattern metrics
     */
    _updateInterruptionPattern(progress) {
        if (progress < 0.25) {
            this.metrics.interruptionPatterns.early++;
        } else if (progress > 0.75) {
            this.metrics.interruptionPatterns.late++;
        } else {
            this.metrics.interruptionPatterns.mid++;
        }
    }

    /**
     * Get interruption metrics
     */
    getMetrics() {
        return {
            ...this.metrics,
            currentState: this.state,
            hasInterruptedResponse: this.interruptedResponse !== null,
            interruptionRate: this.metrics.totalInterruptions > 0
                ? this.metrics.totalDiscards / this.metrics.totalInterruptions
                : 0.0,
            resumptionRate: this.metrics.totalInterruptions > 0
                ? this.metrics.totalResumptions / this.metrics.totalInterruptions
                : 0.0
        };
    }

    /**
     * Check if currently monitoring for interruptions
     */
    isMonitoring() {
        return this.state === InterruptionState.SPEAKING;
    }

    /**
     * Check if currently interrupted
     */
    isInterrupted() {
        return this.state === InterruptionState.INTERRUPTED;
    }

    /**
     * Cleanup
     */
    async destroy() {
        // Stop monitoring
        this.stopMonitoring();

        // Clear timers
        if (this.resumptionTimer) {
            clearTimeout(this.resumptionTimer);
            this.resumptionTimer = null;
        }

        // Close audio context
        if (this.audioContext) {
            await this.audioContext.close();
            this.audioContext = null;
        }

        console.log('[InterruptionHandler] Destroyed');
    }

    // Public API for callbacks

    onInterrupt(callback) {
        this.onInterruptCallback = callback;
    }

    onResume(callback) {
        this.onResumeCallback = callback;
    }

    onDiscard(callback) {
        this.onDiscardCallback = callback;
    }
}


/**
 * Integration with StreamingEngine
 * Demonstrates how to use InterruptionHandler with continuous streaming
 */
class StreamingWithInterruption {
    constructor(streamingEngine, interruptionHandler) {
        this.streamingEngine = streamingEngine;
        this.interruptionHandler = interruptionHandler;

        this.isResponding = false;
        this.currentResponseId = null;

        // Wire up event handlers
        this._setupEventHandlers();
    }

    _setupEventHandlers() {
        // Listen for streaming transcripts
        this.streamingEngine.onSegment((event) => {
            if (event.type === 'final' && this.isResponding) {
                // User is speaking while system is responding
                // Check if this is an interruption
                this._checkForInterruption(event.segment);
            }
        });

        // Listen for interruption events
        this.interruptionHandler.onInterrupt((event) => {
            console.log('[StreamingWithInterruption] Interrupted:', event.userInput);
            this.isResponding = false;

            // Handle new user input
            this._processNewInput(event.userInput);
        });

        this.interruptionHandler.onResume((event) => {
            console.log('[StreamingWithInterruption] Resuming response:', event.response.text?.substring(0, 50));

            // Resume delivering interrupted response
            this._deliverResponse(event.response);
        });
    }

    /**
     * Start responding to user query
     */
    async startResponse(response) {
        this.isResponding = true;
        this.currentResponseId = `resp_${Date.now()}`;

        // Start interruption monitoring
        this.interruptionHandler.startMonitoring(response);

        // Deliver response (TTS, UI, etc.)
        await this._deliverResponse(response);

        // Stop monitoring when response complete
        this.interruptionHandler.stopMonitoring();
        this.isResponding = false;
    }

    /**
     * Check if streaming transcript is an interruption
     */
    _checkForInterruption(segment) {
        // Check confidence and duration
        if (segment.confidence >= this.interruptionHandler.config.confidenceThreshold &&
            segment.duration >= this.interruptionHandler.config.interruptionThresholdMs) {

            // This is a valid interruption
            this.interruptionHandler.handleInterruption({
                text: segment.transcript,
                confidence: segment.confidence,
                segmentId: segment.id
            });
        }
    }

    /**
     * Deliver response (stub - implement with TTS/UI)
     */
    async _deliverResponse(response) {
        console.log('[StreamingWithInterruption] Delivering:', response.text);

        // Simulate response delivery
        // In real implementation: TTS, text-to-speech, UI updates, etc.
        return new Promise(resolve => {
            setTimeout(resolve, response.estimatedDuration || 1000);
        });
    }

    /**
     * Process new user input after interruption
     */
    async _processNewInput(userInput) {
        console.log('[StreamingWithInterruption] Processing new input:', userInput.text);

        // In real implementation: route to orchestrator, get new response, etc.
    }
}


// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        InterruptionHandler,
        InterruptionState,
        StreamingWithInterruption
    };
}
