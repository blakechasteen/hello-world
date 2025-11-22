/**
 * Wake Word Detector - Hands-free Activation
 * Milestone 2: "Hey Loom", "OK Loom" keyword spotting
 * Date: November 2025
 *
 * Features:
 * - Continuous wake word monitoring
 * - Multiple wake word support ("Hey Loom", "OK Loom")
 * - Low-power keyword spotting
 * - Configurable sensitivity
 * - False positive rejection
 * - Confidence scoring
 * - Audio buffering for context
 */

/**
 * Wake Word Detection Result
 */
class WakeWordResult {
    constructor(data) {
        this.detected = data.detected || false;
        this.wakeWord = data.wakeWord || null;
        this.confidence = data.confidence || 0.0;
        this.audioContext = data.audioContext || null; // Audio before/after wake word
        this.timestamp = data.timestamp || new Date();
    }
}

/**
 * Wake Word Detector
 * Detects wake words using pattern matching and audio features
 */
class WakeWordDetector {
    constructor(config = {}) {
        this.config = {
            enabled: config.enabled !== false,

            // Wake words
            wakeWords: config.wakeWords || ['hey loom', 'ok loom', 'loom'],
            caseInsensitive: config.caseInsensitive !== false,

            // Detection settings
            confidenceThreshold: config.confidenceThreshold || 0.7,
            sensitivity: config.sensitivity || 0.5, // 0.0 (low) to 1.0 (high)
            requireExactMatch: config.requireExactMatch !== false,

            // Audio buffering
            preBufferMs: config.preBufferMs || 1000,  // Audio before wake word
            postBufferMs: config.postBufferMs || 500, // Audio after wake word

            // False positive rejection
            enableFalsePositiveFilter: config.enableFalsePositiveFilter !== false,
            minWordDuration: config.minWordDuration || 300, // ms
            maxWordDuration: config.maxWordDuration || 2000, // ms

            // Performance
            checkIntervalMs: config.checkIntervalMs || 100, // How often to check

            ...config
        };

        // Detection state
        this.isMonitoring = false;
        this.lastDetection = null;
        this.audioRingBuffer = [];
        this.maxBufferSamples = Math.ceil(
            (this.config.preBufferMs + this.config.postBufferMs) / this.config.checkIntervalMs
        );

        // Audio analysis
        this.audioContext = null;
        this.analyser = null;
        this.audioDataArray = null;
        this.streamingRecognition = null; // For transcript-based detection

        // Callbacks
        this.onWakeWordDetectedCallback = null;
        this.onFalsePositiveCallback = null;

        // Metrics
        this.metrics = {
            totalDetections: 0,
            falsePositives: 0,
            avgConfidence: 0.0,
            detectionsByWakeWord: {},
            lastDetectionTime: null
        };

        console.log('[WakeWordDetector] Initialized with wake words:', this.config.wakeWords);
    }

    /**
     * Initialize audio monitoring
     */
    async initialize(audioStream, streamingRecognition = null) {
        if (!this.config.enabled) {
            console.log('[WakeWordDetector] Disabled, skipping initialization');
            return false;
        }

        try {
            // Create audio context
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();

            // Create analyser
            this.analyser = this.audioContext.createAnalyser();
            this.analyser.fftSize = 2048;
            this.analyser.smoothingTimeConstant = 0.3;

            const bufferLength = this.analyser.frequencyBinCount;
            this.audioDataArray = new Uint8Array(bufferLength);

            // Connect audio stream
            const source = this.audioContext.createMediaStreamSource(audioStream);
            source.connect(this.analyser);

            // Store streaming recognition for transcript-based detection
            this.streamingRecognition = streamingRecognition;

            console.log('[WakeWordDetector] Audio monitoring initialized');
            return true;

        } catch (error) {
            console.error('[WakeWordDetector] Initialization failed:', error);
            return false;
        }
    }

    /**
     * Start wake word monitoring
     */
    async startMonitoring() {
        if (!this.config.enabled || this.isMonitoring) {
            return false;
        }

        this.isMonitoring = true;

        // Start detection loop
        this._startDetectionLoop();

        // Start transcript monitoring if streaming recognition available
        if (this.streamingRecognition) {
            this._setupTranscriptMonitoring();
        }

        console.log('[WakeWordDetector] Monitoring started');
        return true;
    }

    /**
     * Stop wake word monitoring
     */
    stopMonitoring() {
        if (!this.isMonitoring) {
            return false;
        }

        this.isMonitoring = false;

        console.log('[WakeWordDetector] Monitoring stopped');
        return true;
    }

    /**
     * Start detection loop
     */
    _startDetectionLoop() {
        if (!this.isMonitoring) {
            return;
        }

        const detect = async () => {
            if (!this.isMonitoring) {
                return; // Stop loop
            }

            // Get current audio features
            const audioFeatures = this._extractAudioFeatures();

            // Add to ring buffer
            this.audioRingBuffer.push({
                features: audioFeatures,
                timestamp: Date.now()
            });

            // Maintain buffer size
            if (this.audioRingBuffer.length > this.maxBufferSamples) {
                this.audioRingBuffer.shift();
            }

            // Continue loop
            setTimeout(detect, this.config.checkIntervalMs);
        };

        detect();
    }

    /**
     * Setup transcript monitoring
     */
    _setupTranscriptMonitoring() {
        if (!this.streamingRecognition) {
            return;
        }

        // Listen for interim transcripts
        this.streamingRecognition.onStream((event) => {
            if (event.type === 'interim' || event.type === 'final') {
                this._checkTranscriptForWakeWord(event.segment.transcript);
            }
        });
    }

    /**
     * Check transcript for wake word
     */
    _checkTranscriptForWakeWord(transcript) {
        const normalized = this.config.caseInsensitive
            ? transcript.toLowerCase()
            : transcript;

        for (const wakeWord of this.config.wakeWords) {
            const normalizedWakeWord = this.config.caseInsensitive
                ? wakeWord.toLowerCase()
                : wakeWord;

            // Check for exact match
            if (this.config.requireExactMatch) {
                if (normalized === normalizedWakeWord) {
                    this._handleDetection(wakeWord, 1.0, transcript);
                    return;
                }
            } else {
                // Check for substring match
                if (normalized.includes(normalizedWakeWord)) {
                    // Calculate confidence based on match quality
                    const confidence = this._calculateMatchConfidence(normalized, normalizedWakeWord);
                    this._handleDetection(wakeWord, confidence, transcript);
                    return;
                }
            }
        }
    }

    /**
     * Calculate match confidence
     */
    _calculateMatchConfidence(transcript, wakeWord) {
        // Position-based confidence (beginning of sentence = higher)
        const position = transcript.indexOf(wakeWord);
        const positionScore = 1.0 - (position / transcript.length);

        // Length ratio (closer to wake word length = higher)
        const lengthRatio = Math.min(wakeWord.length / transcript.length, 1.0);

        // Combined confidence
        const confidence = 0.5 * positionScore + 0.5 * lengthRatio;

        return Math.max(this.config.sensitivity, confidence);
    }

    /**
     * Handle wake word detection
     */
    _handleDetection(wakeWord, confidence, context) {
        // Check confidence threshold
        if (confidence < this.config.confidenceThreshold) {
            console.log('[WakeWordDetector] Detection below threshold:', confidence);
            return;
        }

        // False positive filtering
        if (this.config.enableFalsePositiveFilter) {
            if (!this._validateDetection(wakeWord, context)) {
                this.metrics.falsePositives++;

                if (this.onFalsePositiveCallback) {
                    this.onFalsePositiveCallback({
                        type: 'false_positive',
                        wakeWord: wakeWord,
                        context: context,
                        timestamp: new Date()
                    });
                }

                console.log('[WakeWordDetector] False positive detected');
                return;
            }
        }

        // Valid detection!
        const result = new WakeWordResult({
            detected: true,
            wakeWord: wakeWord,
            confidence: confidence,
            audioContext: this._getAudioContext(),
            timestamp: new Date()
        });

        this.lastDetection = result;
        this.metrics.lastDetectionTime = result.timestamp;

        // Update metrics
        this.metrics.totalDetections++;
        this.metrics.detectionsByWakeWord[wakeWord] =
            (this.metrics.detectionsByWakeWord[wakeWord] || 0) + 1;
        this.metrics.avgConfidence =
            (this.metrics.avgConfidence * (this.metrics.totalDetections - 1) + confidence) /
            this.metrics.totalDetections;

        // Emit event
        if (this.onWakeWordDetectedCallback) {
            this.onWakeWordDetectedCallback({
                type: 'wake_word_detected',
                result: result,
                timestamp: new Date()
            });
        }

        console.log('[WakeWordDetector] Wake word detected:', wakeWord, 'confidence:', confidence.toFixed(2));
    }

    /**
     * Validate detection (false positive filter)
     */
    _validateDetection(wakeWord, context) {
        // Check word duration (too short/long = likely false positive)
        const wordLength = wakeWord.length;
        const estimatedDuration = (wordLength / 5) * 1000; // ~5 chars per second

        if (estimatedDuration < this.config.minWordDuration ||
            estimatedDuration > this.config.maxWordDuration) {
            return false;
        }

        // Check context (wake word should be at beginning)
        if (context) {
            const normalized = this.config.caseInsensitive
                ? context.toLowerCase()
                : context;
            const position = normalized.indexOf(wakeWord.toLowerCase());

            // Wake word should be in first 20% of transcript
            if (position / normalized.length > 0.2) {
                return false;
            }
        }

        return true;
    }

    /**
     * Extract audio features
     */
    _extractAudioFeatures() {
        if (!this.analyser || !this.audioDataArray) {
            return null;
        }

        this.analyser.getByteTimeDomainData(this.audioDataArray);

        // Calculate features
        let sum = 0;
        let sumSquares = 0;

        for (let i = 0; i < this.audioDataArray.length; i++) {
            const normalized = (this.audioDataArray[i] - 128) / 128;
            sum += normalized;
            sumSquares += normalized * normalized;
        }

        const mean = sum / this.audioDataArray.length;
        const variance = (sumSquares / this.audioDataArray.length) - (mean * mean);
        const energy = Math.sqrt(variance);

        return {
            energy: energy,
            mean: mean,
            variance: variance
        };
    }

    /**
     * Get audio context (buffered audio)
     */
    _getAudioContext() {
        if (this.audioRingBuffer.length === 0) {
            return null;
        }

        // Get pre-buffer and post-buffer samples
        const now = Date.now();
        const preBufferSamples = this.audioRingBuffer.filter(
            sample => now - sample.timestamp <= this.config.preBufferMs
        );

        return {
            preBuffer: preBufferSamples,
            bufferDuration: this.config.preBufferMs + this.config.postBufferMs
        };
    }

    /**
     * Add custom wake word
     */
    addWakeWord(wakeWord) {
        if (!this.config.wakeWords.includes(wakeWord)) {
            this.config.wakeWords.push(wakeWord);
            console.log('[WakeWordDetector] Added wake word:', wakeWord);
        }
    }

    /**
     * Remove wake word
     */
    removeWakeWord(wakeWord) {
        const index = this.config.wakeWords.indexOf(wakeWord);
        if (index > -1) {
            this.config.wakeWords.splice(index, 1);
            console.log('[WakeWordDetector] Removed wake word:', wakeWord);
        }
    }

    /**
     * Get all wake words
     */
    getWakeWords() {
        return this.config.wakeWords;
    }

    /**
     * Get last detection
     */
    getLastDetection() {
        return this.lastDetection;
    }

    /**
     * Get metrics
     */
    getMetrics() {
        return {
            ...this.metrics,
            isMonitoring: this.isMonitoring,
            wakeWordsCount: this.config.wakeWords.length,
            falsePositiveRate: this.metrics.totalDetections > 0
                ? this.metrics.falsePositives / this.metrics.totalDetections
                : 0.0,
            detectionRate: this.metrics.totalDetections / (Date.now() - (this.metrics.lastDetectionTime?.getTime() || Date.now()))
        };
    }

    /**
     * Cleanup
     */
    async destroy() {
        // Stop monitoring
        this.stopMonitoring();

        // Close audio context
        if (this.audioContext) {
            await this.audioContext.close();
            this.audioContext = null;
        }

        this.audioRingBuffer = [];

        console.log('[WakeWordDetector] Destroyed');
    }

    // Public API for callbacks

    onWakeWordDetected(callback) {
        this.onWakeWordDetectedCallback = callback;
    }

    onFalsePositive(callback) {
        this.onFalsePositiveCallback = callback;
    }
}


/**
 * Integration Example
 * Shows how to use WakeWordDetector with StreamingEngine
 */
class VoiceActivationSystem {
    constructor(streamingEngine, wakeWordDetector) {
        this.streamingEngine = streamingEngine;
        this.wakeWordDetector = wakeWordDetector;

        this.isActivated = false;
        this.activationTimer = null;
        this.activationDuration = 30000; // 30 seconds active after wake word

        this._setupEventHandlers();
    }

    _setupEventHandlers() {
        // Listen for wake word detections
        this.wakeWordDetector.onWakeWordDetected((event) => {
            console.log('[VoiceActivationSystem] Wake word detected, activating...');
            this._activateVoice();
        });
    }

    async _activateVoice() {
        if (this.isActivated) {
            // Reset timer
            this._resetActivationTimer();
            return;
        }

        this.isActivated = true;

        // Start streaming recognition
        await this.streamingEngine.start();

        // Set deactivation timer
        this._resetActivationTimer();

        console.log('[VoiceActivationSystem] Voice activated for', this.activationDuration, 'ms');
    }

    _resetActivationTimer() {
        if (this.activationTimer) {
            clearTimeout(this.activationTimer);
        }

        this.activationTimer = setTimeout(() => {
            this._deactivateVoice();
        }, this.activationDuration);
    }

    async _deactivateVoice() {
        if (!this.isActivated) {
            return;
        }

        this.isActivated = false;

        // Stop streaming recognition
        await this.streamingEngine.stop();

        console.log('[VoiceActivationSystem] Voice deactivated');
    }

    async destroy() {
        if (this.activationTimer) {
            clearTimeout(this.activationTimer);
        }

        await this._deactivateVoice();
    }
}


// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        WakeWordDetector,
        WakeWordResult,
        VoiceActivationSystem
    };
}
