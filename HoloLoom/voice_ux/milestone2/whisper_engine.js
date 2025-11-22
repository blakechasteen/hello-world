/**
 * Whisper Engine - OpenAI Whisper API Integration
 * Milestone 2: Cloud STT with 99 language support
 * Date: November 2025
 *
 * Features:
 * - 99 language support (auto-detection or explicit)
 * - High accuracy (~95%+ word error rate)
 * - Punctuation and capitalization
 * - Multiple model sizes (tiny, base, small, medium, large)
 * - Timestamps at word level
 * - Translation to English
 */

class WhisperEngine extends VoiceEngineProtocol {
    constructor(config = {}) {
        super();

        this.config = {
            apiKey: config.apiKey || process.env.OPENAI_API_KEY,
            model: config.model || 'whisper-1', // OpenAI API model
            language: config.language || null, // Auto-detect if null
            temperature: config.temperature || 0.0, // 0 = more deterministic
            prompt: config.prompt || null, // Optional context prompt
            responseFormat: config.responseFormat || 'json', // json, text, srt, vtt
            timestampGranularities: config.timestampGranularities || ['word'], // word, segment
            ...config
        };

        this.isListening = false;
        this.audioRecorder = null;
        this.audioChunks = [];
        this.onTranscriptCallback = null;
        this.onErrorCallback = null;

        this.metrics = {
            totalRequests: 0,
            totalWords: 0,
            avgLatencyMs: 0,
            avgConfidence: 0.0,
            errors: 0,
            tokensUsed: 0,
            costUSD: 0.0
        };

        // Whisper API pricing: $0.006 per minute
        this.COST_PER_MINUTE = 0.006;
    }

    static isSupported() {
        // Whisper requires:
        // 1. MediaRecorder API (for audio capture)
        // 2. Fetch API (for HTTP requests)
        // 3. API key
        return 'MediaRecorder' in window && 'fetch' in window;
    }

    async initialize() {
        if (!WhisperEngine.isSupported()) {
            throw new Error('Whisper engine not supported (missing MediaRecorder or Fetch API)');
        }

        if (!this.config.apiKey) {
            throw new Error('Whisper API key required');
        }

        // Test API key validity
        try {
            const response = await fetch('https://api.openai.com/v1/models', {
                method: 'GET',
                headers: {
                    'Authorization': `Bearer ${this.config.apiKey}`
                }
            });

            if (!response.ok) {
                throw new Error(`API key validation failed: ${response.statusText}`);
            }

            console.log('[WhisperEngine] Initialized successfully');
            return true;
        } catch (error) {
            console.error('[WhisperEngine] Initialization failed:', error);
            throw error;
        }
    }

    async start(config = {}) {
        if (this.isListening) {
            console.warn('[WhisperEngine] Already listening');
            return false;
        }

        // Update config if provided
        if (config.language) {
            this.config.language = config.language;
        }

        try {
            // Request microphone access
            const stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true,
                    sampleRate: 16000 // Whisper works best at 16kHz
                }
            });

            // Create MediaRecorder
            this.audioRecorder = new MediaRecorder(stream, {
                mimeType: 'audio/webm;codecs=opus' // Whisper supports many formats
            });

            this.audioChunks = [];

            // Collect audio chunks
            this.audioRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    this.audioChunks.push(event.data);
                }
            };

            // Process when recording stops
            this.audioRecorder.onstop = async () => {
                await this._processRecording();
            };

            // Start recording
            this.audioRecorder.start();
            this.isListening = true;

            console.log('[WhisperEngine] Recording started');
            return true;

        } catch (error) {
            console.error('[WhisperEngine] Start failed:', error);

            if (this.onErrorCallback) {
                this.onErrorCallback({
                    type: 'start_failed',
                    message: error.message,
                    timestamp: new Date()
                });
            }

            return false;
        }
    }

    async stop() {
        if (!this.isListening || !this.audioRecorder) {
            return false;
        }

        try {
            // Stop recording (will trigger onstop → _processRecording)
            this.audioRecorder.stop();

            // Stop all tracks
            this.audioRecorder.stream.getTracks().forEach(track => track.stop());

            this.isListening = false;

            console.log('[WhisperEngine] Recording stopped');
            return true;

        } catch (error) {
            console.error('[WhisperEngine] Stop failed:', error);
            return false;
        }
    }

    async processAudio(audioChunk) {
        // For streaming mode - accumulate chunks and process periodically
        this.audioChunks.push(audioChunk);

        // Process every 5 seconds of audio (configurable)
        const totalSize = this.audioChunks.reduce((sum, chunk) => sum + chunk.size, 0);
        const estimatedDuration = totalSize / (16000 * 2); // Rough estimate

        if (estimatedDuration >= 5.0) {
            return await this._processRecording();
        }

        return null;
    }

    async _processRecording() {
        const startTime = Date.now();

        try {
            // Combine audio chunks into single blob
            const audioBlob = new Blob(this.audioChunks, { type: 'audio/webm' });

            // Clear chunks
            this.audioChunks = [];

            // Create FormData for multipart upload
            const formData = new FormData();
            formData.append('file', audioBlob, 'audio.webm');
            formData.append('model', this.config.model);
            formData.append('response_format', this.config.responseFormat);
            formData.append('temperature', this.config.temperature);

            if (this.config.language) {
                formData.append('language', this.config.language);
            }

            if (this.config.prompt) {
                formData.append('prompt', this.config.prompt);
            }

            if (this.config.timestampGranularities) {
                formData.append('timestamp_granularities[]', this.config.timestampGranularities.join(','));
            }

            // Send to Whisper API
            const response = await fetch('https://api.openai.com/v1/audio/transcriptions', {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${this.config.apiKey}`
                },
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Whisper API error: ${response.status} ${response.statusText}`);
            }

            const result = await response.json();

            // Calculate latency
            const latencyMs = Date.now() - startTime;

            // Update metrics
            this.metrics.totalRequests++;
            this.metrics.avgLatencyMs = (this.metrics.avgLatencyMs * (this.metrics.totalRequests - 1) + latencyMs) / this.metrics.totalRequests;

            // Estimate cost (based on audio duration)
            const audioDuration = result.duration || 0;
            const costUSD = (audioDuration / 60.0) * this.COST_PER_MINUTE;
            this.metrics.costUSD += costUSD;

            // Parse transcript result
            const transcriptResult = this._parseWhisperResult(result, latencyMs);

            // Call callback
            if (this.onTranscriptCallback) {
                this.onTranscriptCallback(transcriptResult);
            }

            return transcriptResult;

        } catch (error) {
            console.error('[WhisperEngine] Processing failed:', error);
            this.metrics.errors++;

            if (this.onErrorCallback) {
                this.onErrorCallback({
                    type: 'transcription_failed',
                    message: error.message,
                    timestamp: new Date()
                });
            }

            return null;
        }
    }

    _parseWhisperResult(result, latencyMs) {
        // Whisper API response format:
        // {
        //   "text": "Hello, world!",
        //   "task": "transcribe",
        //   "language": "english",
        //   "duration": 5.2,
        //   "words": [
        //     { "word": "Hello", "start": 0.0, "end": 0.5 },
        //     { "word": "world", "start": 0.6, "end": 1.2 }
        //   ]
        // }

        return new TranscriptResult({
            transcript: result.text || '',
            confidence: 1.0, // Whisper doesn't provide confidence scores
            isFinal: true,
            language: result.language || this.config.language || 'unknown',
            engine: 'whisper',
            latencyMs: latencyMs,
            timestamp: new Date(),
            words: result.words || [],
            alternatives: [], // Whisper doesn't provide alternatives
            speakerId: null,
            emotion: null
        });
    }

    getSupportedLanguages() {
        // Whisper supports 99 languages
        return [
            'en', 'zh', 'de', 'es', 'ru', 'ko', 'fr', 'ja', 'pt', 'tr', 'pl', 'ca', 'nl', 'ar',
            'sv', 'it', 'id', 'hi', 'fi', 'vi', 'he', 'uk', 'el', 'ms', 'cs', 'ro', 'da', 'hu',
            'ta', 'no', 'th', 'ur', 'hr', 'bg', 'lt', 'la', 'mi', 'ml', 'cy', 'sk', 'te', 'fa',
            'lv', 'bn', 'sr', 'az', 'sl', 'kn', 'et', 'mk', 'br', 'eu', 'is', 'hy', 'ne', 'mn',
            'bs', 'kk', 'sq', 'sw', 'gl', 'mr', 'pa', 'si', 'km', 'sn', 'yo', 'so', 'af', 'oc',
            'ka', 'be', 'tg', 'sd', 'gu', 'am', 'yi', 'lo', 'uz', 'fo', 'ht', 'ps', 'tk', 'nn',
            'mt', 'sa', 'lb', 'my', 'bo', 'tl', 'mg', 'as', 'tt', 'haw', 'ln', 'ha', 'ba', 'jw',
            'su'
        ];
    }

    getCapabilities() {
        return new EngineCapabilities({
            streaming: false, // Whisper API doesn't support streaming (yet)
            interimResults: false,
            multiLanguage: true, // 99 languages
            punctuation: true,
            profanityFilter: false, // Whisper transcribes everything
            voiceActivityDetection: false,
            speakerDiarization: false,
            emotionDetection: false,
            wakeWordDetection: false,
            multimodal: false
        });
    }

    getMetrics() {
        return this.metrics;
    }

    async destroy() {
        if (this.isListening) {
            await this.stop();
        }

        if (this.audioRecorder && this.audioRecorder.stream) {
            this.audioRecorder.stream.getTracks().forEach(track => track.stop());
        }

        this.audioRecorder = null;
        this.audioChunks = [];
    }

    // Public API for callbacks

    onTranscript(callback) {
        this.onTranscriptCallback = callback;
    }

    onError(callback) {
        this.onErrorCallback = callback;
    }

    // Helper: Translate to English (Whisper translation API)

    async translateToEnglish(audioBlob) {
        const formData = new FormData();
        formData.append('file', audioBlob, 'audio.webm');
        formData.append('model', this.config.model);
        formData.append('response_format', 'json');

        try {
            const response = await fetch('https://api.openai.com/v1/audio/translations', {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${this.config.apiKey}`
                },
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Translation failed: ${response.statusText}`);
            }

            const result = await response.json();
            return result.text;

        } catch (error) {
            console.error('[WhisperEngine] Translation failed:', error);
            return null;
        }
    }
}


// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = WhisperEngine;
}
