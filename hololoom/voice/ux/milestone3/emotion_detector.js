/**
 * Emotion Detection System
 * Milestone 3: Sentiment analysis and emotional context awareness
 * Date: November 2025
 *
 * Features:
 * - Real-time emotion detection from voice
 * - Sentiment analysis (positive, negative, neutral)
 * - Arousal and valence scoring
 * - Emotion history tracking
 * - Prosodic feature analysis (pitch, energy, speech rate)
 * - Integration with dialogue context
 */

/**
 * Emotion Categories (Plutchik's wheel simplified)
 */
const EmotionCategory = {
    // Primary emotions
    JOY: 'joy',
    SADNESS: 'sadness',
    ANGER: 'anger',
    FEAR: 'fear',
    SURPRISE: 'surprise',
    DISGUST: 'disgust',
    TRUST: 'trust',
    ANTICIPATION: 'anticipation',

    // Neutral
    NEUTRAL: 'neutral'
};

/**
 * Sentiment Polarity
 */
const SentimentPolarity = {
    POSITIVE: 'positive',
    NEGATIVE: 'negative',
    NEUTRAL: 'neutral'
};

/**
 * Emotion Detection Result
 */
class EmotionResult {
    constructor(data) {
        this.emotion = data.emotion || EmotionCategory.NEUTRAL;
        this.confidence = data.confidence || 0.0;
        this.sentiment = data.sentiment || SentimentPolarity.NEUTRAL;
        this.sentimentScore = data.sentimentScore || 0.0; // -1.0 to +1.0
        this.arousal = data.arousal || 0.0; // 0.0 (calm) to 1.0 (excited)
        this.valence = data.valence || 0.0; // -1.0 (negative) to +1.0 (positive)
        this.prosody = data.prosody || null; // Prosodic features
        this.timestamp = data.timestamp || new Date();
    }
}

/**
 * Prosodic Features
 */
class ProsodicFeatures {
    constructor(data) {
        this.pitch = data.pitch || 0.0; // Average pitch (Hz)
        this.pitchVariation = data.pitchVariation || 0.0; // Pitch variance
        this.energy = data.energy || 0.0; // Volume/intensity
        this.speechRate = data.speechRate || 0.0; // Words per minute
        this.pauseDuration = data.pauseDuration || 0.0; // Average pause length (ms)
    }
}

/**
 * Emotion Detector
 * Detects emotions from voice input using prosody and text analysis
 */
class EmotionDetector {
    constructor(config = {}) {
        this.config = {
            enabled: config.enabled !== false,

            // Analysis modes
            useProsody: config.useProsody !== false,        // Prosodic feature analysis
            useTextSentiment: config.useTextSentiment !== false, // Text-based sentiment
            useCombined: config.useCombined !== false,      // Combine prosody + text

            // Thresholds
            confidenceThreshold: config.confidenceThreshold || 0.6,
            arousalThreshold: config.arousalThreshold || 0.5,

            // Feature extraction
            pitchAnalysisWindow: config.pitchAnalysisWindow || 100, // ms
            energyAnalysisWindow: config.energyAnalysisWindow || 50, // ms

            ...config
        };

        // Audio analysis
        this.audioContext = null;
        this.analyser = null;
        this.frequencyDataArray = null;
        this.timeDataArray = null;

        // Emotion history
        this.emotionHistory = [];
        this.maxHistorySize = 50;

        // Sentiment lexicon (simple keyword-based)
        this._initializeSentimentLexicon();

        // Callbacks
        this.onEmotionDetectedCallback = null;
        this.onSentimentChangeCallback = null;

        // Metrics
        this.metrics = {
            totalDetections: 0,
            emotionCounts: Object.fromEntries(
                Object.values(EmotionCategory).map(e => [e, 0])
            ),
            avgConfidence: 0.0,
            avgArousal: 0.0,
            avgValence: 0.0
        };

        console.log('[EmotionDetector] Initialized', this.config);
    }

    /**
     * Initialize sentiment lexicon
     */
    _initializeSentimentLexicon() {
        this.sentimentLexicon = {
            // Positive words
            positive: [
                'happy', 'joy', 'love', 'wonderful', 'excellent', 'great', 'amazing',
                'fantastic', 'good', 'nice', 'thanks', 'thank', 'appreciate', 'perfect',
                'awesome', 'brilliant', 'excited', 'delighted', 'pleased'
            ],

            // Negative words
            negative: [
                'sad', 'angry', 'hate', 'terrible', 'awful', 'bad', 'horrible',
                'disappointing', 'frustrated', 'annoyed', 'upset', 'worried', 'concerned',
                'wrong', 'problem', 'issue', 'error', 'fail', 'broken'
            ],

            // Emotion indicators
            emotions: {
                joy: ['happy', 'joy', 'delighted', 'excited', 'pleased'],
                sadness: ['sad', 'unhappy', 'depressed', 'down', 'disappointed'],
                anger: ['angry', 'mad', 'furious', 'annoyed', 'frustrated'],
                fear: ['afraid', 'scared', 'worried', 'anxious', 'nervous'],
                surprise: ['surprised', 'shocked', 'amazed', 'astonished'],
                disgust: ['disgusted', 'revolted', 'awful', 'terrible'],
                trust: ['trust', 'reliable', 'confident', 'secure'],
                anticipation: ['excited', 'eager', 'looking forward', 'expecting']
            }
        };
    }

    /**
     * Initialize audio analysis
     */
    async initialize(audioStream) {
        if (!this.config.enabled || !this.config.useProsody) {
            console.log('[EmotionDetector] Prosody analysis disabled');
            return false;
        }

        try {
            // Create audio context
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();

            // Create analyser for prosody extraction
            this.analyser = this.audioContext.createAnalyser();
            this.analyser.fftSize = 2048;
            this.analyser.smoothingTimeConstant = 0.8;

            const bufferLength = this.analyser.frequencyBinCount;
            this.frequencyDataArray = new Uint8Array(bufferLength);
            this.timeDataArray = new Uint8Array(bufferLength);

            // Connect audio stream
            const source = this.audioContext.createMediaStreamSource(audioStream);
            source.connect(this.analyser);

            console.log('[EmotionDetector] Audio analysis initialized');
            return true;

        } catch (error) {
            console.error('[EmotionDetector] Initialization failed:', error);
            return false;
        }
    }

    /**
     * Detect emotion from transcript and audio
     */
    detectEmotion(transcript, audioFeatures = null) {
        if (!this.config.enabled) {
            return new EmotionResult({ emotion: EmotionCategory.NEUTRAL });
        }

        // Extract text-based sentiment
        const textSentiment = this.config.useTextSentiment
            ? this._analyzeTextSentiment(transcript)
            : null;

        // Extract prosodic features
        const prosody = this.config.useProsody && audioFeatures
            ? this._extractProsodicFeatures(audioFeatures)
            : this._extractProsodicFeatures(); // Extract from analyser

        // Combine analyses
        const emotionResult = this.config.useCombined
            ? this._combinedEmotionDetection(textSentiment, prosody)
            : this._textOnlyEmotionDetection(textSentiment);

        // Add prosody to result
        emotionResult.prosody = prosody;

        // Update history
        this.emotionHistory.push(emotionResult);
        if (this.emotionHistory.length > this.maxHistorySize) {
            this.emotionHistory.shift();
        }

        // Update metrics
        this._updateMetrics(emotionResult);

        // Emit event if confidence high enough
        if (emotionResult.confidence >= this.config.confidenceThreshold) {
            if (this.onEmotionDetectedCallback) {
                this.onEmotionDetectedCallback({
                    type: 'emotion_detected',
                    emotion: emotionResult,
                    transcript: transcript,
                    timestamp: new Date()
                });
            }
        }

        console.log('[EmotionDetector] Detected:', emotionResult.emotion, 'confidence:', emotionResult.confidence.toFixed(2));

        return emotionResult;
    }

    /**
     * Analyze text sentiment
     */
    _analyzeTextSentiment(text) {
        const lowerText = text.toLowerCase();
        let positiveScore = 0;
        let negativeScore = 0;

        // Count positive words
        for (const word of this.sentimentLexicon.positive) {
            if (lowerText.includes(word)) {
                positiveScore++;
            }
        }

        // Count negative words
        for (const word of this.sentimentLexicon.negative) {
            if (lowerText.includes(word)) {
                negativeScore++;
            }
        }

        // Calculate sentiment score (-1.0 to +1.0)
        const total = positiveScore + negativeScore;
        const sentimentScore = total > 0
            ? (positiveScore - negativeScore) / total
            : 0.0;

        // Determine polarity
        let sentiment = SentimentPolarity.NEUTRAL;
        if (sentimentScore > 0.2) sentiment = SentimentPolarity.POSITIVE;
        if (sentimentScore < -0.2) sentiment = SentimentPolarity.NEGATIVE;

        // Detect specific emotions from keywords
        let detectedEmotion = EmotionCategory.NEUTRAL;
        let maxMatches = 0;

        for (const [emotion, keywords] of Object.entries(this.sentimentLexicon.emotions)) {
            let matches = 0;
            for (const keyword of keywords) {
                if (lowerText.includes(keyword)) {
                    matches++;
                }
            }

            if (matches > maxMatches) {
                maxMatches = matches;
                detectedEmotion = emotion;
            }
        }

        return {
            sentiment: sentiment,
            sentimentScore: sentimentScore,
            emotion: detectedEmotion,
            confidence: total > 0 ? Math.min(total / 3.0, 1.0) : 0.3 // Rough confidence
        };
    }

    /**
     * Extract prosodic features from audio
     */
    _extractProsodicFeatures(providedFeatures = null) {
        if (providedFeatures) {
            return new ProsodicFeatures(providedFeatures);
        }

        if (!this.analyser) {
            return new ProsodicFeatures({});
        }

        // Get frequency and time domain data
        this.analyser.getByteFrequencyData(this.frequencyDataArray);
        this.analyser.getByteTimeDomainData(this.timeDataArray);

        // Calculate pitch (dominant frequency)
        const pitch = this._calculateDominantFrequency(this.frequencyDataArray);

        // Calculate energy (average amplitude)
        let sum = 0;
        for (let i = 0; i < this.timeDataArray.length; i++) {
            const normalized = (this.timeDataArray[i] - 128) / 128;
            sum += normalized * normalized;
        }
        const energy = Math.sqrt(sum / this.timeDataArray.length);

        // Estimate pitch variation (simple std dev)
        const pitchVariation = this._calculatePitchVariation();

        return new ProsodicFeatures({
            pitch: pitch,
            pitchVariation: pitchVariation,
            energy: energy,
            speechRate: 0.0, // Would need transcript timing
            pauseDuration: 0.0
        });
    }

    /**
     * Calculate dominant frequency (pitch)
     */
    _calculateDominantFrequency(frequencyData) {
        let maxValue = 0;
        let maxIndex = 0;

        for (let i = 0; i < frequencyData.length; i++) {
            if (frequencyData[i] > maxValue) {
                maxValue = frequencyData[i];
                maxIndex = i;
            }
        }

        // Convert bin index to frequency
        const nyquist = this.audioContext.sampleRate / 2;
        const frequency = (maxIndex * nyquist) / frequencyData.length;

        return frequency;
    }

    /**
     * Calculate pitch variation (simplified)
     */
    _calculatePitchVariation() {
        // Placeholder - would need historical pitch values
        return 0.1;
    }

    /**
     * Combined emotion detection (text + prosody)
     */
    _combinedEmotionDetection(textSentiment, prosody) {
        if (!textSentiment) {
            return new EmotionResult({ emotion: EmotionCategory.NEUTRAL });
        }

        // Base emotion from text
        let emotion = textSentiment.emotion;
        let confidence = textSentiment.confidence;

        // Adjust based on prosody
        if (prosody) {
            // High energy + positive text = joy
            if (prosody.energy > 0.6 && textSentiment.sentimentScore > 0.2) {
                emotion = EmotionCategory.JOY;
                confidence = Math.min(confidence + 0.2, 1.0);
            }

            // High energy + negative text = anger
            if (prosody.energy > 0.6 && textSentiment.sentimentScore < -0.2) {
                emotion = EmotionCategory.ANGER;
                confidence = Math.min(confidence + 0.2, 1.0);
            }

            // Low energy + negative text = sadness
            if (prosody.energy < 0.3 && textSentiment.sentimentScore < -0.2) {
                emotion = EmotionCategory.SADNESS;
                confidence = Math.min(confidence + 0.1, 1.0);
            }

            // High pitch variation = surprise/excitement
            if (prosody.pitchVariation > 0.5) {
                if (textSentiment.sentimentScore > 0) {
                    emotion = EmotionCategory.SURPRISE;
                }
            }
        }

        // Calculate arousal and valence
        const arousal = prosody ? prosody.energy : 0.0;
        const valence = textSentiment.sentimentScore;

        return new EmotionResult({
            emotion: emotion,
            confidence: confidence,
            sentiment: textSentiment.sentiment,
            sentimentScore: textSentiment.sentimentScore,
            arousal: arousal,
            valence: valence,
            prosody: prosody
        });
    }

    /**
     * Text-only emotion detection
     */
    _textOnlyEmotionDetection(textSentiment) {
        if (!textSentiment) {
            return new EmotionResult({ emotion: EmotionCategory.NEUTRAL });
        }

        return new EmotionResult({
            emotion: textSentiment.emotion,
            confidence: textSentiment.confidence,
            sentiment: textSentiment.sentiment,
            sentimentScore: textSentiment.sentimentScore,
            arousal: Math.abs(textSentiment.sentimentScore), // Rough arousal
            valence: textSentiment.sentimentScore
        });
    }

    /**
     * Update metrics
     */
    _updateMetrics(emotionResult) {
        this.metrics.totalDetections++;
        this.metrics.emotionCounts[emotionResult.emotion]++;

        this.metrics.avgConfidence =
            (this.metrics.avgConfidence * (this.metrics.totalDetections - 1) + emotionResult.confidence) /
            this.metrics.totalDetections;

        this.metrics.avgArousal =
            (this.metrics.avgArousal * (this.metrics.totalDetections - 1) + emotionResult.arousal) /
            this.metrics.totalDetections;

        this.metrics.avgValence =
            (this.metrics.avgValence * (this.metrics.totalDetections - 1) + emotionResult.valence) /
            this.metrics.totalDetections;
    }

    /**
     * Get recent emotion trend
     */
    getEmotionTrend(windowSize = 5) {
        const recent = this.emotionHistory.slice(-windowSize);

        if (recent.length === 0) {
            return null;
        }

        // Calculate average valence and arousal
        const avgValence = recent.reduce((sum, e) => sum + e.valence, 0) / recent.length;
        const avgArousal = recent.reduce((sum, e) => sum + e.arousal, 0) / recent.length;

        // Determine trend
        let trend = 'stable';
        if (recent.length >= 3) {
            const first = recent[0].valence;
            const last = recent[recent.length - 1].valence;

            if (last - first > 0.3) trend = 'improving';
            if (last - first < -0.3) trend = 'declining';
        }

        return {
            avgValence: avgValence,
            avgArousal: avgArousal,
            trend: trend,
            windowSize: recent.length
        };
    }

    /**
     * Get dominant emotion over recent history
     */
    getDominantEmotion(windowSize = 10) {
        const recent = this.emotionHistory.slice(-windowSize);

        if (recent.length === 0) {
            return EmotionCategory.NEUTRAL;
        }

        // Count emotions
        const counts = {};
        for (const result of recent) {
            counts[result.emotion] = (counts[result.emotion] || 0) + 1;
        }

        // Find most common
        let maxCount = 0;
        let dominant = EmotionCategory.NEUTRAL;

        for (const [emotion, count] of Object.entries(counts)) {
            if (count > maxCount) {
                maxCount = count;
                dominant = emotion;
            }
        }

        return dominant;
    }

    /**
     * Get metrics
     */
    getMetrics() {
        return {
            ...this.metrics,
            historySize: this.emotionHistory.length,
            dominantEmotion: this.getDominantEmotion(),
            recentTrend: this.getEmotionTrend()
        };
    }

    /**
     * Cleanup
     */
    async destroy() {
        if (this.audioContext) {
            await this.audioContext.close();
            this.audioContext = null;
        }

        this.emotionHistory = [];

        console.log('[EmotionDetector] Destroyed');
    }

    // Public API for callbacks

    onEmotionDetected(callback) {
        this.onEmotionDetectedCallback = callback;
    }

    onSentimentChange(callback) {
        this.onSentimentChangeCallback = callback;
    }
}


// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        EmotionDetector,
        EmotionResult,
        ProsodicFeatures,
        EmotionCategory,
        SentimentPolarity
    };
}
