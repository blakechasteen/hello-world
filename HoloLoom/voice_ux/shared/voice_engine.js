/**
 * Voice Engine - Shared Abstraction Layer
 * Milestones 2-4: Unified voice recognition interface
 * Date: November 2025
 *
 * Provides unified interface for:
 * - Web Speech API (Milestone 1)
 * - Whisper API (Milestone 2)
 * - Google Cloud Speech-to-Text (Milestone 2)
 * - Streaming recognition (Milestone 3)
 * - Multimodal fusion (Milestone 4)
 */

/**
 * Voice Engine Protocol
 * All voice recognition backends must implement this interface
 */
class VoiceEngineProtocol {
    constructor() {
        if (this.constructor === VoiceEngineProtocol) {
            throw new Error('VoiceEngineProtocol is abstract - cannot instantiate');
        }
    }

    /**
     * Initialize the engine
     * @returns {Promise<boolean>} Success status
     */
    async initialize() {
        throw new Error('initialize() must be implemented');
    }

    /**
     * Start voice recognition
     * @param {Object} config - Recognition configuration
     * @returns {Promise<boolean>} Success status
     */
    async start(config = {}) {
        throw new Error('start() must be implemented');
    }

    /**
     * Stop voice recognition
     * @returns {Promise<boolean>} Success status
     */
    async stop() {
        throw new Error('stop() must be implemented');
    }

    /**
     * Process audio chunk (for streaming)
     * @param {ArrayBuffer} audioChunk - Audio data
     * @returns {Promise<TranscriptResult>} Transcript result
     */
    async processAudio(audioChunk) {
        throw new Error('processAudio() must be implemented');
    }

    /**
     * Get supported languages
     * @returns {Array<string>} Language codes (e.g., ['en-US', 'es-ES'])
     */
    getSupportedLanguages() {
        throw new Error('getSupportedLanguages() must be implemented');
    }

    /**
     * Get engine capabilities
     * @returns {EngineCapabilities} Capability flags
     */
    getCapabilities() {
        throw new Error('getCapabilities() must be implemented');
    }

    /**
     * Get metrics
     * @returns {EngineMetrics} Performance metrics
     */
    getMetrics() {
        throw new Error('getMetrics() must be implemented');
    }

    /**
     * Cleanup resources
     * @returns {Promise<void>}
     */
    async destroy() {
        throw new Error('destroy() must be implemented');
    }
}


/**
 * Voice Engine Factory
 * Creates appropriate voice engine based on requirements
 */
class VoiceEngineFactory {
    static create(engineType, config = {}) {
        switch (engineType) {
            case 'web_speech':
                return new WebSpeechEngine(config);

            case 'whisper':
                return new WhisperEngine(config);

            case 'google_cloud':
                return new GoogleCloudEngine(config);

            case 'streaming':
                return new StreamingEngine(config);

            case 'multimodal':
                return new MultimodalEngine(config);

            case 'auto':
                return VoiceEngineFactory._selectBestEngine(config);

            default:
                throw new Error(`Unknown engine type: ${engineType}`);
        }
    }

    /**
     * Auto-select best available engine
     */
    static _selectBestEngine(config) {
        // Priority order:
        // 1. Whisper API (best accuracy, multi-language)
        // 2. Google Cloud (good accuracy, enterprise)
        // 3. Web Speech API (local, fast, free)

        if (config.apiKey && config.preferCloud) {
            // Try Whisper first if API key available
            if (WhisperEngine.isSupported()) {
                return new WhisperEngine(config);
            }

            // Fall back to Google Cloud
            if (GoogleCloudEngine.isSupported()) {
                return new GoogleCloudEngine(config);
            }
        }

        // Default to Web Speech API (always available in modern browsers)
        if (WebSpeechEngine.isSupported()) {
            return new WebSpeechEngine(config);
        }

        throw new Error('No voice engine available');
    }

    /**
     * Get all available engines
     */
    static getAvailableEngines() {
        const engines = [];

        if (WebSpeechEngine.isSupported()) {
            engines.push({
                type: 'web_speech',
                name: 'Web Speech API',
                local: true,
                free: true,
                languages: 50,
                streaming: true
            });
        }

        if (WhisperEngine.isSupported()) {
            engines.push({
                type: 'whisper',
                name: 'OpenAI Whisper',
                local: false,
                free: false,
                languages: 99,
                streaming: false
            });
        }

        if (GoogleCloudEngine.isSupported()) {
            engines.push({
                type: 'google_cloud',
                name: 'Google Cloud Speech-to-Text',
                local: false,
                free: false,
                languages: 125,
                streaming: true
            });
        }

        return engines;
    }
}


/**
 * Web Speech Engine (Milestone 1 - already implemented)
 * Wraps browser Web Speech API
 */
class WebSpeechEngine extends VoiceEngineProtocol {
    constructor(config = {}) {
        super();
        this.config = {
            language: config.language || 'en-US',
            continuous: config.continuous !== false,
            interimResults: config.interimResults !== false,
            maxAlternatives: config.maxAlternatives || 1,
            ...config
        };

        this.recognition = null;
        this.isListening = false;
        this.onTranscriptCallback = null;
        this.onErrorCallback = null;

        this.metrics = {
            totalTranscripts: 0,
            avgLatencyMs: 0,
            errors: 0
        };
    }

    static isSupported() {
        return 'webkitSpeechRecognition' in window || 'SpeechRecognition' in window;
    }

    async initialize() {
        if (!WebSpeechEngine.isSupported()) {
            throw new Error('Web Speech API not supported');
        }

        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        this.recognition = new SpeechRecognition();

        this.recognition.continuous = this.config.continuous;
        this.recognition.interimResults = this.config.interimResults;
        this.recognition.lang = this.config.language;
        this.recognition.maxAlternatives = this.config.maxAlternatives;

        this.recognition.onresult = (event) => this._handleResult(event);
        this.recognition.onerror = (event) => this._handleError(event);
        this.recognition.onend = () => this._handleEnd();

        return true;
    }

    async start(config = {}) {
        if (config.language) {
            this.recognition.lang = config.language;
        }

        try {
            this.recognition.start();
            this.isListening = true;
            return true;
        } catch (error) {
            console.error('[WebSpeechEngine] Start failed:', error);
            return false;
        }
    }

    async stop() {
        if (!this.isListening) return false;

        try {
            this.recognition.stop();
            this.isListening = false;
            return true;
        } catch (error) {
            console.error('[WebSpeechEngine] Stop failed:', error);
            return false;
        }
    }

    async processAudio(audioChunk) {
        // Web Speech API doesn't support manual audio feeding
        throw new Error('processAudio() not supported by Web Speech API');
    }

    getSupportedLanguages() {
        // Web Speech API supports ~50 languages (varies by browser)
        return [
            'en-US', 'en-GB', 'es-ES', 'es-MX', 'fr-FR', 'de-DE',
            'it-IT', 'ja-JP', 'ko-KR', 'pt-BR', 'ru-RU', 'zh-CN'
        ];
    }

    getCapabilities() {
        return {
            streaming: true,
            interimResults: true,
            multiLanguage: true,
            punctuation: false,
            profanityFilter: true,
            voiceActivityDetection: true,
            speakerDiarization: false,
            emotionDetection: false
        };
    }

    getMetrics() {
        return this.metrics;
    }

    async destroy() {
        if (this.isListening) {
            await this.stop();
        }
        this.recognition = null;
    }

    // Event handlers

    _handleResult(event) {
        const startTime = Date.now();

        const results = event.results;
        const lastResult = results[results.length - 1];
        const transcript = lastResult[0].transcript;
        const confidence = lastResult[0].confidence || 1.0;
        const isFinal = lastResult.isFinal;

        if (this.onTranscriptCallback) {
            this.onTranscriptCallback({
                transcript: transcript,
                confidence: confidence,
                isFinal: isFinal,
                language: this.config.language,
                engine: 'web_speech',
                latencyMs: Date.now() - startTime,
                timestamp: new Date()
            });
        }

        if (isFinal) {
            this.metrics.totalTranscripts++;
        }
    }

    _handleError(event) {
        this.metrics.errors++;

        if (this.onErrorCallback) {
            this.onErrorCallback({
                type: event.error,
                message: this._getErrorMessage(event.error),
                timestamp: new Date()
            });
        }
    }

    _handleEnd() {
        this.isListening = false;

        // Auto-restart if continuous mode
        if (this.config.continuous && this.recognition) {
            setTimeout(() => {
                if (!this.isListening) {
                    this.start();
                }
            }, 100);
        }
    }

    _getErrorMessage(error) {
        const errors = {
            'no-speech': 'No speech detected',
            'audio-capture': 'Microphone not available',
            'not-allowed': 'Microphone access denied',
            'network': 'Network error',
            'aborted': 'Recognition aborted',
            'service-not-allowed': 'Speech recognition service not available'
        };
        return errors[error] || `Unknown error: ${error}`;
    }

    // Public API for callbacks

    onTranscript(callback) {
        this.onTranscriptCallback = callback;
    }

    onError(callback) {
        this.onErrorCallback = callback;
    }
}


/**
 * Transcript Result (common format across all engines)
 */
class TranscriptResult {
    constructor(data) {
        this.transcript = data.transcript || '';
        this.confidence = data.confidence || 0.0;
        this.isFinal = data.isFinal !== false;
        this.language = data.language || 'en-US';
        this.engine = data.engine || 'unknown';
        this.latencyMs = data.latencyMs || 0;
        this.timestamp = data.timestamp || new Date();
        this.alternatives = data.alternatives || [];
        this.words = data.words || []; // Word-level timestamps
        this.speakerId = data.speakerId || null; // For speaker diarization
        this.emotion = data.emotion || null; // For emotion detection
    }
}


/**
 * Engine Capabilities
 */
class EngineCapabilities {
    constructor(data = {}) {
        this.streaming = data.streaming || false;
        this.interimResults = data.interimResults || false;
        this.multiLanguage = data.multiLanguage || false;
        this.punctuation = data.punctuation || false;
        this.profanityFilter = data.profanityFilter || false;
        this.voiceActivityDetection = data.voiceActivityDetection || false;
        this.speakerDiarization = data.speakerDiarization || false;
        this.emotionDetection = data.emotionDetection || false;
        this.wakeWordDetection = data.wakeWordDetection || false;
        this.multimodal = data.multimodal || false;
    }
}


/**
 * Engine Metrics
 */
class EngineMetrics {
    constructor() {
        this.totalTranscripts = 0;
        this.totalWords = 0;
        this.avgLatencyMs = 0;
        this.avgConfidence = 0.0;
        this.errors = 0;
        this.engineSwitches = 0;
    }
}


// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        VoiceEngineProtocol,
        VoiceEngineFactory,
        WebSpeechEngine,
        TranscriptResult,
        EngineCapabilities,
        EngineMetrics
    };
}
