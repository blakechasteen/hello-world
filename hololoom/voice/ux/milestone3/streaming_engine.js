/**
 * Streaming Recognition Engine
 * Milestone 3: Continuous cognition with real-time streaming
 * Date: November 2025
 *
 * Features:
 * - Continuous background recognition (always listening)
 * - Real-time transcript streaming (no wait for "final")
 * - Voice Activity Detection (VAD) for silence detection
 * - Automatic phrase segmentation
 * - Buffer management for long conversations
 * - Low-latency mode (<100ms for interim results)
 */

class StreamingEngine extends VoiceEngineProtocol {
    constructor(config = {}) {
        super();

        this.config = {
            continuous: true, // Always on
            maxBufferSeconds: config.maxBufferSeconds || 30, // Max buffer before flush
            silenceThresholdMs: config.silenceThresholdMs || 1500, // Silence detection
            interimUpdateIntervalMs: config.interimUpdateIntervalMs || 100, // Interim update frequency
            vad: config.vad !== false, // Voice Activity Detection
            autoSegment: config.autoSegment !== false, // Automatic phrase segmentation
            language: config.language || 'en-US',
            ...config
        };

        this.baseEngine = null; // Underlying engine (Web Speech or Whisper)
        this.isStreaming = false;

        this.streamBuffer = [];
        this.currentSegment = null;
        this.lastVoiceActivityTime = null;
        this.silenceTimer = null;

        this.onStreamCallback = null;
        this.onSegmentCallback = null;
        this.onErrorCallback = null;

        this.metrics = {
            totalSegments: 0,
            totalWords: 0,
            avgSegmentDuration: 0,
            avgInterimLatency: 0,
            silenceDetections: 0,
            bufferFlushes: 0
        };
    }

    static isSupported() {
        // Streaming requires Web Speech API or compatible streaming STT
        return WebSpeechEngine.isSupported();
    }

    async initialize() {
        // Create base engine (use Web Speech for streaming)
        this.baseEngine = new WebSpeechEngine({
            continuous: true,
            interimResults: true,
            language: this.config.language
        });

        await this.baseEngine.initialize();

        // Set up base engine callbacks
        this.baseEngine.onTranscript((result) => {
            this._handleBaseTranscript(result);
        });

        this.baseEngine.onError((error) => {
            if (this.onErrorCallback) {
                this.onErrorCallback(error);
            }
        });

        console.log('[StreamingEngine] Initialized');
        return true;
    }

    async start(config = {}) {
        if (this.isStreaming) {
            console.warn('[StreamingEngine] Already streaming');
            return false;
        }

        // Update language if provided
        if (config.language) {
            this.config.language = config.language;
        }

        // Start base engine
        const started = await this.baseEngine.start({ language: this.config.language });

        if (started) {
            this.isStreaming = true;
            this.currentSegment = this._createSegment();
            this.lastVoiceActivityTime = Date.now();

            // Start silence detection timer
            if (this.config.vad) {
                this._startSilenceDetection();
            }

            console.log('[StreamingEngine] Streaming started');
        }

        return started;
    }

    async stop() {
        if (!this.isStreaming) {
            return false;
        }

        // Stop base engine
        const stopped = await this.baseEngine.stop();

        if (stopped) {
            this.isStreaming = false;

            // Flush current segment
            if (this.currentSegment && this.currentSegment.transcript) {
                this._finalizeSegment(this.currentSegment);
            }

            // Clear timers
            if (this.silenceTimer) {
                clearTimeout(this.silenceTimer);
                this.silenceTimer = null;
            }

            console.log('[StreamingEngine] Streaming stopped');
        }

        return stopped;
    }

    async processAudio(audioChunk) {
        // For custom audio processing (if base engine supports it)
        if (this.baseEngine.processAudio) {
            return await this.baseEngine.processAudio(audioChunk);
        }

        throw new Error('Base engine does not support audio processing');
    }

    /**
     * Handle transcript from base engine
     */
    _handleBaseTranscript(result) {
        const now = Date.now();

        // Update voice activity time
        this.lastVoiceActivityTime = now;

        // Reset silence timer
        if (this.config.vad) {
            this._resetSilenceDetection();
        }

        if (result.isFinal) {
            // Final result - finalize current segment
            this._finalizeSegment(this.currentSegment, result.transcript, result.confidence);

            // Create new segment
            this.currentSegment = this._createSegment();

        } else {
            // Interim result - update current segment
            this.currentSegment.transcript = result.transcript;
            this.currentSegment.confidence = result.confidence;
            this.currentSegment.wordCount = result.transcript.split(/\s+/).length;
            this.currentSegment.lastUpdateTime = now;

            // Emit streaming update
            if (this.onStreamCallback) {
                this.onStreamCallback({
                    type: 'interim',
                    segment: this.currentSegment,
                    timestamp: new Date()
                });
            }

            // Update interim latency metric
            const latency = now - this.currentSegment.startTime;
            this.metrics.avgInterimLatency =
                (this.metrics.avgInterimLatency * this.metrics.totalSegments + latency) /
                (this.metrics.totalSegments + 1);
        }

        // Check buffer size
        this._checkBufferSize();
    }

    /**
     * Create a new segment
     */
    _createSegment() {
        return {
            id: `seg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
            transcript: '',
            confidence: 0.0,
            wordCount: 0,
            startTime: Date.now(),
            endTime: null,
            duration: 0,
            isFinal: false
        };
    }

    /**
     * Finalize a segment
     */
    _finalizeSegment(segment, finalTranscript = null, finalConfidence = null) {
        if (!segment) return;

        const now = Date.now();

        segment.transcript = finalTranscript || segment.transcript;
        segment.confidence = finalConfidence || segment.confidence;
        segment.endTime = now;
        segment.duration = now - segment.startTime;
        segment.isFinal = true;

        // Add to buffer
        this.streamBuffer.push(segment);

        // Update metrics
        this.metrics.totalSegments++;
        this.metrics.totalWords += segment.wordCount;
        this.metrics.avgSegmentDuration =
            (this.metrics.avgSegmentDuration * (this.metrics.totalSegments - 1) + segment.duration) /
            this.metrics.totalSegments;

        // Emit segment callback
        if (this.onSegmentCallback) {
            this.onSegmentCallback({
                type: 'final',
                segment: segment,
                timestamp: new Date()
            });
        }

        console.log(`[StreamingEngine] Segment finalized: "${segment.transcript}"`);
    }

    /**
     * Start silence detection
     */
    _startSilenceDetection() {
        this._resetSilenceDetection();
    }

    /**
     * Reset silence detection timer
     */
    _resetSilenceDetection() {
        if (this.silenceTimer) {
            clearTimeout(this.silenceTimer);
        }

        this.silenceTimer = setTimeout(() => {
            this._handleSilenceDetected();
        }, this.config.silenceThresholdMs);
    }

    /**
     * Handle silence detected
     */
    _handleSilenceDetected() {
        console.log('[StreamingEngine] Silence detected');

        this.metrics.silenceDetections++;

        // Finalize current segment if it has content
        if (this.currentSegment && this.currentSegment.transcript) {
            this._finalizeSegment(this.currentSegment);
            this.currentSegment = this._createSegment();
        }

        // Emit silence event
        if (this.onStreamCallback) {
            this.onStreamCallback({
                type: 'silence',
                timestamp: new Date()
            });
        }
    }

    /**
     * Check buffer size and flush if needed
     */
    _checkBufferSize() {
        const totalDuration = this.streamBuffer.reduce((sum, seg) => sum + seg.duration, 0);
        const totalSeconds = totalDuration / 1000;

        if (totalSeconds >= this.config.maxBufferSeconds) {
            this._flushBuffer();
        }
    }

    /**
     * Flush buffer
     */
    _flushBuffer() {
        console.log(`[StreamingEngine] Flushing buffer (${this.streamBuffer.length} segments)`);

        this.metrics.bufferFlushes++;

        // Emit flush event with buffer contents
        if (this.onStreamCallback) {
            this.onStreamCallback({
                type: 'flush',
                segments: [...this.streamBuffer],
                timestamp: new Date()
            });
        }

        // Clear buffer
        this.streamBuffer = [];
    }

    /**
     * Get full transcript from buffer
     */
    getFullTranscript() {
        return this.streamBuffer
            .map(seg => seg.transcript)
            .join(' ')
            .trim();
    }

    /**
     * Get recent segments
     */
    getRecentSegments(count = 5) {
        return this.streamBuffer.slice(-count);
    }

    /**
     * Clear buffer
     */
    clearBuffer() {
        this.streamBuffer = [];
        console.log('[StreamingEngine] Buffer cleared');
    }

    getSupportedLanguages() {
        return this.baseEngine.getSupportedLanguages();
    }

    getCapabilities() {
        return new EngineCapabilities({
            streaming: true,
            interimResults: true,
            multiLanguage: true,
            punctuation: false,
            profanityFilter: true,
            voiceActivityDetection: this.config.vad,
            speakerDiarization: false,
            emotionDetection: false,
            wakeWordDetection: false,
            multimodal: false
        });
    }

    getMetrics() {
        return {
            ...this.metrics,
            bufferSize: this.streamBuffer.length,
            currentSegmentDuration: this.currentSegment
                ? Date.now() - this.currentSegment.startTime
                : 0
        };
    }

    async destroy() {
        if (this.isStreaming) {
            await this.stop();
        }

        if (this.baseEngine) {
            await this.baseEngine.destroy();
        }

        if (this.silenceTimer) {
            clearTimeout(this.silenceTimer);
        }

        this.streamBuffer = [];
        this.currentSegment = null;
    }

    // Public API for callbacks

    onStream(callback) {
        this.onStreamCallback = callback;
    }

    onSegment(callback) {
        this.onSegmentCallback = callback;
    }

    onError(callback) {
        this.onErrorCallback = callback;
    }
}


/**
 * Streaming Visualization Helper
 * Real-time visualization of streaming transcripts
 */
class StreamingVisualizer {
    constructor(container) {
        this.container = container;
        this.segments = [];
        this.currentSegmentElement = null;
    }

    /**
     * Update with interim transcript
     */
    updateInterim(segment) {
        if (!this.currentSegmentElement) {
            this.currentSegmentElement = document.createElement('div');
            this.currentSegmentElement.className = 'streaming-segment streaming-segment--interim';
            this.container.appendChild(this.currentSegmentElement);
        }

        this.currentSegmentElement.textContent = segment.transcript;
        this.currentSegmentElement.style.opacity = segment.confidence;
    }

    /**
     * Finalize segment
     */
    finalizeSegment(segment) {
        if (this.currentSegmentElement) {
            this.currentSegmentElement.className = 'streaming-segment streaming-segment--final';
            this.currentSegmentElement.textContent = segment.transcript;
            this.currentSegmentElement.style.opacity = 1.0;
        }

        this.segments.push(segment);
        this.currentSegmentElement = null;

        // Auto-scroll to bottom
        this.container.scrollTop = this.container.scrollHeight;
    }

    /**
     * Show silence indicator
     */
    showSilence() {
        const silenceElement = document.createElement('div');
        silenceElement.className = 'streaming-silence';
        silenceElement.textContent = '---';
        this.container.appendChild(silenceElement);
    }

    /**
     * Clear all segments
     */
    clear() {
        this.container.innerHTML = '';
        this.segments = [];
        this.currentSegmentElement = null;
    }
}


// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        StreamingEngine,
        StreamingVisualizer
    };
}
