/**
 * Vocal Prosody Analyzer - Advanced Voice Nuance Detection
 * Milestone 3 Enhancement: Detailed vocal expression analysis
 * Date: November 2025
 *
 * Features:
 * - Advanced prosodic feature extraction (pitch, intensity, duration, rhythm)
 * - Speaking style classification (conversational, formal, excited, etc.)
 * - Voice quality analysis (breathiness, creaky voice, falsetto)
 * - Micro-prosodic events (voice breaks, hesitations, emphasis)
 * - Paralinguistic cues (laughter, sighs, crying)
 * - Expressive nuance detection (sarcasm, irony, enthusiasm)
 * - Real-time prosody tracking
 * - Integration with Web Audio API for detailed analysis
 */

/**
 * Speaking Styles
 */
const SpeakingStyle = {
    CONVERSATIONAL: 'conversational',      // Casual, relaxed
    FORMAL: 'formal',                       // Professional, measured
    EXCITED: 'excited',                     // High energy, fast
    CALM: 'calm',                           // Slow, soothing
    EMPHATIC: 'emphatic',                   // Strong stress patterns
    MONOTONE: 'monotone',                   // Flat, little variation
    QUESTIONING: 'questioning',             // Rising intonation
    DECLARATIVE: 'declarative'              // Falling intonation
};

/**
 * Voice Quality Types
 */
const VoiceQuality = {
    MODAL: 'modal',                         // Normal voice
    BREATHY: 'breathy',                     // Soft, airy
    CREAKY: 'creaky',                       // Vocal fry, low
    FALSETTO: 'falsetto',                   // High, strained
    TENSE: 'tense',                         // Tight, strained
    RELAXED: 'relaxed'                      // Loose, easy
};

/**
 * Paralinguistic Events
 */
const ParalinguisticEvent = {
    LAUGHTER: 'laughter',
    SIGH: 'sigh',
    CRYING: 'crying',
    COUGH: 'cough',
    THROAT_CLEAR: 'throat_clear',
    HESITATION: 'hesitation',               // "uh", "um"
    VOICE_BREAK: 'voice_break',             // Crack in voice
    EMPHASIS: 'emphasis'                    // Strong stress
};

/**
 * Prosodic Features
 */
class ProsodicFeatures {
    constructor(data) {
        // Fundamental frequency (pitch)
        this.f0Mean = data.f0Mean || 0;           // Hz
        this.f0Std = data.f0Std || 0;             // Standard deviation
        this.f0Min = data.f0Min || 0;
        this.f0Max = data.f0Max || 0;
        this.f0Range = data.f0Range || 0;

        // Pitch dynamics
        this.pitchContour = data.pitchContour || []; // Array of pitch values over time
        this.pitchSlope = data.pitchSlope || 0;      // Overall rising/falling trend

        // Intensity (loudness)
        this.intensityMean = data.intensityMean || 0;
        this.intensityStd = data.intensityStd || 0;
        this.intensityRange = data.intensityRange || 0;

        // Timing features
        this.speechRate = data.speechRate || 0;      // syllables/second
        this.articulationRate = data.articulationRate || 0; // syllables/second (excluding pauses)
        this.pauseDuration = data.pauseDuration || 0; // Average pause length (ms)
        this.pauseFrequency = data.pauseFrequency || 0; // Pauses per second

        // Rhythm
        this.jitter = data.jitter || 0;              // Pitch period variability
        this.shimmer = data.shimmer || 0;            // Amplitude variability

        // Voice quality indicators
        this.harmonicToNoiseRatio = data.harmonicToNoiseRatio || 0; // HNR
        this.spectralTilt = data.spectralTilt || 0;  // High vs low frequency balance

        // Formants (resonance frequencies)
        this.formants = data.formants || [];          // [F1, F2, F3, F4]

        // Timestamp
        this.timestamp = data.timestamp || new Date();
    }

    /**
     * Calculate pitch variability (normalized)
     */
    getPitchVariability() {
        if (this.f0Mean === 0) return 0;
        return this.f0Std / this.f0Mean;
    }

    /**
     * Calculate intensity variability (normalized)
     */
    getIntensityVariability() {
        if (this.intensityMean === 0) return 0;
        return this.intensityStd / this.intensityMean;
    }

    /**
     * Detect rising intonation (question-like)
     */
    hasRisingIntonation() {
        return this.pitchSlope > 0.1; // Positive slope = rising
    }

    /**
     * Detect falling intonation (statement-like)
     */
    hasFallingIntonation() {
        return this.pitchSlope < -0.1; // Negative slope = falling
    }
}

/**
 * Vocal Prosody Result
 */
class VocalProsodyResult {
    constructor(data) {
        // Prosodic features
        this.features = data.features || new ProsodicFeatures({});

        // Speaking style
        this.speakingStyle = data.speakingStyle || SpeakingStyle.CONVERSATIONAL;
        this.styleConfidence = data.styleConfidence || 0.0;

        // Voice quality
        this.voiceQuality = data.voiceQuality || VoiceQuality.MODAL;
        this.qualityConfidence = data.qualityConfidence || 0.0;

        // Paralinguistic events detected
        this.paralinguisticEvents = data.paralinguisticEvents || [];

        // Expressive nuances
        this.expressiveness = data.expressiveness || 0.5; // 0.0 (flat) to 1.0 (very expressive)
        this.enthusiasm = data.enthusiasm || 0.5;
        this.confidence = data.confidence || 0.5;
        this.stress = data.stress || 0.5; // Emotional stress level

        // Detected patterns
        this.sarcasm = data.sarcasm || 0.0;       // 0.0-1.0
        this.irony = data.irony || 0.0;
        this.hesitation = data.hesitation || 0.0;

        // Overall analysis confidence
        this.analysisConfidence = data.analysisConfidence || 0.0;

        // Metadata
        this.timestamp = data.timestamp || new Date();
        this.duration = data.duration || 0; // Analysis window duration (ms)
    }

    /**
     * Get expressive signature (for speaker identification)
     */
    getExpressiveSignature() {
        return {
            pitchVariability: this.features.getPitchVariability(),
            intensityVariability: this.features.getIntensityVariability(),
            speechRate: this.features.speechRate,
            expressiveness: this.expressiveness,
            voiceQuality: this.voiceQuality
        };
    }

    /**
     * Detect if speaker is stressed/anxious
     */
    isStressed() {
        return this.stress > 0.7 ||
               this.features.speechRate > 5.0 || // Fast speech
               this.features.jitter > 0.02 ||     // High pitch variability
               this.hesitation > 0.5;              // Frequent hesitations
    }

    /**
     * Detect if speaker is confident
     */
    isConfident() {
        return this.confidence > 0.7 &&
               this.features.speechRate > 3.0 &&
               this.features.intensityMean > 60 &&
               this.hesitation < 0.3;
    }
}

/**
 * Vocal Prosody Analyzer
 * Extracts detailed prosodic features from voice audio
 */
class VocalProsodyAnalyzer {
    constructor(config = {}) {
        this.config = {
            enabled: config.enabled !== false,

            // Analysis settings
            analysisWindowMs: config.analysisWindowMs || 1000, // 1 second windows
            hopSizeMs: config.hopSizeMs || 500,                // 50% overlap

            // Audio analysis
            fftSize: config.fftSize || 2048,
            smoothingTimeConstant: config.smoothingTimeConstant || 0.3,

            // Pitch detection
            minPitch: config.minPitch || 75,    // Hz (typical male lower bound)
            maxPitch: config.maxPitch || 500,   // Hz (typical female upper bound)

            // Feature extraction
            enableFormants: config.enableFormants !== false,
            enableMicroProsody: config.enableMicroProsody !== false,
            enableParalinguistics: config.enableParalinguistics !== false,

            ...config
        };

        // Audio analysis components
        this.audioContext = null;
        this.analyser = null;
        this.audioDataArray = null;

        // Feature buffers
        this.pitchHistory = [];
        this.intensityHistory = [];
        this.maxHistorySize = 50;

        // Paralinguistic detection
        this.laughterDetector = null;
        this.hesitationDetector = null;

        // Analysis state
        this.isAnalyzing = false;

        // Callbacks
        this.onProsodyDetectedCallback = null;
        this.onParalinguisticEventCallback = null;

        // Metrics
        this.metrics = {
            totalAnalyses: 0,
            avgProcessingTimeMs: 0,
            styleDistribution: {},
            lastAnalysisTime: null
        };

        console.log('[VocalProsodyAnalyzer] Initialized');
    }

    /**
     * Initialize with audio stream
     */
    async initialize(audioStream) {
        if (!this.config.enabled) {
            console.log('[VocalProsodyAnalyzer] Disabled, skipping initialization');
            return false;
        }

        try {
            // Create audio context
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();

            // Create analyser
            this.analyser = this.audioContext.createAnalyser();
            this.analyser.fftSize = this.config.fftSize;
            this.analyser.smoothingTimeConstant = this.config.smoothingTimeConstant;

            const bufferLength = this.analyser.frequencyBinCount;
            this.audioDataArray = new Uint8Array(bufferLength);
            this.frequencyDataArray = new Uint8Array(bufferLength);

            // Connect audio stream
            const source = this.audioContext.createMediaStreamSource(audioStream);
            source.connect(this.analyser);

            console.log('[VocalProsodyAnalyzer] Initialized successfully');
            return true;

        } catch (error) {
            console.error('[VocalProsodyAnalyzer] Initialization failed:', error);
            return false;
        }
    }

    /**
     * Start prosody analysis
     */
    startAnalysis() {
        if (!this.audioContext || this.isAnalyzing) {
            return false;
        }

        this.isAnalyzing = true;
        console.log('[VocalProsodyAnalyzer] Analysis started');
        return true;
    }

    /**
     * Stop prosody analysis
     */
    stopAnalysis() {
        if (!this.isAnalyzing) {
            return false;
        }

        this.isAnalyzing = false;
        console.log('[VocalProsodyAnalyzer] Analysis stopped');
        return true;
    }

    /**
     * Analyze current audio window
     */
    analyzeProsody() {
        if (!this.analyser || !this.isAnalyzing) {
            return null;
        }

        const startTime = Date.now();

        // Extract raw audio features
        const features = this._extractProsodicFeatures();

        // Classify speaking style
        const speakingStyle = this._classifySpeakingStyle(features);

        // Classify voice quality
        const voiceQuality = this._classifyVoiceQuality(features);

        // Detect paralinguistic events
        const paralinguisticEvents = this.config.enableParalinguistics
            ? this._detectParalinguisticEvents(features)
            : [];

        // Calculate expressive metrics
        const expressiveness = this._calculateExpressiveness(features);
        const enthusiasm = this._calculateEnthusiasm(features);
        const confidence = this._calculateConfidence(features);
        const stress = this._calculateStress(features);

        // Detect nuances
        const sarcasm = this._detectSarcasm(features);
        const irony = this._detectIrony(features);
        const hesitation = this._detectHesitation(features);

        const result = new VocalProsodyResult({
            features: features,
            speakingStyle: speakingStyle.style,
            styleConfidence: speakingStyle.confidence,
            voiceQuality: voiceQuality.quality,
            qualityConfidence: voiceQuality.confidence,
            paralinguisticEvents: paralinguisticEvents,
            expressiveness: expressiveness,
            enthusiasm: enthusiasm,
            confidence: confidence,
            stress: stress,
            sarcasm: sarcasm,
            irony: irony,
            hesitation: hesitation,
            analysisConfidence: 0.8, // Overall confidence
            duration: this.config.analysisWindowMs
        });

        // Update metrics
        this._updateMetrics(result, Date.now() - startTime);

        // Emit events
        if (this.onProsodyDetectedCallback) {
            this.onProsodyDetectedCallback({
                type: 'prosody_detected',
                result: result,
                timestamp: new Date()
            });
        }

        return result;
    }

    /**
     * Extract prosodic features from audio
     */
    _extractProsodicFeatures() {
        // Get time domain data (waveform)
        this.analyser.getByteTimeDomainData(this.audioDataArray);

        // Get frequency domain data (spectrum)
        this.analyser.getByteFrequencyData(this.frequencyDataArray);

        // Calculate fundamental frequency (pitch)
        const f0 = this._estimatePitch(this.audioDataArray);

        // Track pitch history
        this.pitchHistory.push(f0);
        if (this.pitchHistory.length > this.maxHistorySize) {
            this.pitchHistory.shift();
        }

        // Calculate pitch statistics
        const f0Mean = this._mean(this.pitchHistory);
        const f0Std = this._std(this.pitchHistory, f0Mean);
        const f0Min = Math.min(...this.pitchHistory);
        const f0Max = Math.max(...this.pitchHistory);

        // Calculate intensity (energy)
        const intensity = this._calculateIntensity(this.audioDataArray);

        // Track intensity history
        this.intensityHistory.push(intensity);
        if (this.intensityHistory.length > this.maxHistorySize) {
            this.intensityHistory.shift();
        }

        const intensityMean = this._mean(this.intensityHistory);
        const intensityStd = this._std(this.intensityHistory, intensityMean);
        const intensityMin = Math.min(...this.intensityHistory);
        const intensityMax = Math.max(...this.intensityHistory);

        // Calculate pitch slope (rising/falling intonation)
        const pitchSlope = this._calculateSlope(this.pitchHistory);

        // Estimate speech rate (very approximate without transcript)
        const speechRate = this._estimateSpeechRate(this.audioDataArray);

        // Calculate jitter and shimmer
        const jitter = this._calculateJitter(this.pitchHistory);
        const shimmer = this._calculateShimmer(this.intensityHistory);

        // Calculate harmonic-to-noise ratio
        const hnr = this._calculateHNR(this.frequencyDataArray);

        // Calculate spectral tilt
        const spectralTilt = this._calculateSpectralTilt(this.frequencyDataArray);

        // Extract formants (if enabled)
        const formants = this.config.enableFormants
            ? this._extractFormants(this.frequencyDataArray)
            : [];

        return new ProsodicFeatures({
            f0Mean: f0Mean,
            f0Std: f0Std,
            f0Min: f0Min,
            f0Max: f0Max,
            f0Range: f0Max - f0Min,
            pitchContour: [...this.pitchHistory],
            pitchSlope: pitchSlope,
            intensityMean: intensityMean,
            intensityStd: intensityStd,
            intensityRange: intensityMax - intensityMin,
            speechRate: speechRate,
            articulationRate: speechRate * 1.2, // Approximation
            pauseDuration: 0, // Would need VAD for accurate detection
            pauseFrequency: 0,
            jitter: jitter,
            shimmer: shimmer,
            harmonicToNoiseRatio: hnr,
            spectralTilt: spectralTilt,
            formants: formants
        });
    }

    /**
     * Estimate fundamental frequency (pitch) using autocorrelation
     */
    _estimatePitch(audioData) {
        // Normalize audio data
        const normalized = new Float32Array(audioData.length);
        for (let i = 0; i < audioData.length; i++) {
            normalized[i] = (audioData[i] - 128) / 128;
        }

        // Autocorrelation
        const sampleRate = this.audioContext.sampleRate;
        const minLag = Math.floor(sampleRate / this.config.maxPitch);
        const maxLag = Math.floor(sampleRate / this.config.minPitch);

        let bestLag = minLag;
        let bestCorrelation = -1;

        for (let lag = minLag; lag < Math.min(maxLag, normalized.length / 2); lag++) {
            let sum = 0;
            for (let i = 0; i < normalized.length - lag; i++) {
                sum += normalized[i] * normalized[i + lag];
            }
            const correlation = sum / (normalized.length - lag);

            if (correlation > bestCorrelation) {
                bestCorrelation = correlation;
                bestLag = lag;
            }
        }

        return sampleRate / bestLag; // Convert lag to frequency (Hz)
    }

    /**
     * Calculate intensity (RMS energy)
     */
    _calculateIntensity(audioData) {
        let sum = 0;
        for (let i = 0; i < audioData.length; i++) {
            const normalized = (audioData[i] - 128) / 128;
            sum += normalized * normalized;
        }
        const rms = Math.sqrt(sum / audioData.length);

        // Convert to approximate dB
        return 20 * Math.log10(rms + 1e-10);
    }

    /**
     * Estimate speech rate (syllables per second)
     */
    _estimateSpeechRate(audioData) {
        // Count intensity peaks as proxy for syllables
        let peaks = 0;
        let threshold = 150; // Intensity threshold

        for (let i = 1; i < audioData.length - 1; i++) {
            if (audioData[i] > threshold &&
                audioData[i] > audioData[i - 1] &&
                audioData[i] > audioData[i + 1]) {
                peaks++;
            }
        }

        const duration = audioData.length / this.audioContext.sampleRate;
        return peaks / duration; // Peaks per second (approximate syllables/s)
    }

    /**
     * Calculate jitter (pitch period variability)
     */
    _calculateJitter(pitchHistory) {
        if (pitchHistory.length < 2) return 0;

        let sum = 0;
        for (let i = 1; i < pitchHistory.length; i++) {
            sum += Math.abs(pitchHistory[i] - pitchHistory[i - 1]);
        }

        const avgDifference = sum / (pitchHistory.length - 1);
        const avgPitch = this._mean(pitchHistory);

        return avgPitch > 0 ? avgDifference / avgPitch : 0;
    }

    /**
     * Calculate shimmer (amplitude variability)
     */
    _calculateShimmer(intensityHistory) {
        if (intensityHistory.length < 2) return 0;

        let sum = 0;
        for (let i = 1; i < intensityHistory.length; i++) {
            sum += Math.abs(intensityHistory[i] - intensityHistory[i - 1]);
        }

        const avgDifference = sum / (intensityHistory.length - 1);
        const avgIntensity = this._mean(intensityHistory);

        return avgIntensity > 0 ? avgDifference / avgIntensity : 0;
    }

    /**
     * Calculate harmonic-to-noise ratio
     */
    _calculateHNR(frequencyData) {
        // Simplified HNR: ratio of low frequency to high frequency energy
        const lowFreqSum = frequencyData.slice(0, frequencyData.length / 4)
            .reduce((a, b) => a + b, 0);
        const highFreqSum = frequencyData.slice(3 * frequencyData.length / 4)
            .reduce((a, b) => a + b, 0);

        return highFreqSum > 0 ? lowFreqSum / highFreqSum : 0;
    }

    /**
     * Calculate spectral tilt
     */
    _calculateSpectralTilt(frequencyData) {
        // Measure slope of frequency spectrum
        const mid = Math.floor(frequencyData.length / 2);
        const lowAvg = this._mean(frequencyData.slice(0, mid));
        const highAvg = this._mean(frequencyData.slice(mid));

        return lowAvg - highAvg; // Positive = more low frequency energy
    }

    /**
     * Extract formant frequencies
     */
    _extractFormants(frequencyData) {
        // Simplified formant extraction: find peaks in spectrum
        const peaks = [];
        for (let i = 2; i < frequencyData.length - 2; i++) {
            if (frequencyData[i] > frequencyData[i - 1] &&
                frequencyData[i] > frequencyData[i + 1] &&
                frequencyData[i] > 100) { // Threshold
                const freq = i * this.audioContext.sampleRate / this.config.fftSize;
                peaks.push(freq);
            }
        }

        // Return first 4 formants (F1-F4)
        return peaks.slice(0, 4);
    }

    /**
     * Classify speaking style from features
     */
    _classifySpeakingStyle(features) {
        // Decision tree based on prosodic features
        if (features.speechRate > 5.0 && features.f0Range > 100) {
            return { style: SpeakingStyle.EXCITED, confidence: 0.8 };
        } else if (features.speechRate < 2.0 && features.f0Range < 50) {
            return { style: SpeakingStyle.CALM, confidence: 0.75 };
        } else if (features.f0Range < 30 && features.getPitchVariability() < 0.1) {
            return { style: SpeakingStyle.MONOTONE, confidence: 0.85 };
        } else if (features.hasRisingIntonation()) {
            return { style: SpeakingStyle.QUESTIONING, confidence: 0.7 };
        } else if (features.hasFallingIntonation()) {
            return { style: SpeakingStyle.DECLARATIVE, confidence: 0.7 };
        } else if (features.intensityRange > 20) {
            return { style: SpeakingStyle.EMPHATIC, confidence: 0.75 };
        } else if (features.speechRate < 3.5 && features.f0Range < 80) {
            return { style: SpeakingStyle.FORMAL, confidence: 0.7 };
        }

        return { style: SpeakingStyle.CONVERSATIONAL, confidence: 0.6 };
    }

    /**
     * Classify voice quality from features
     */
    _classifyVoiceQuality(features) {
        if (features.harmonicToNoiseRatio < 5) {
            return { quality: VoiceQuality.BREATHY, confidence: 0.7 };
        } else if (features.f0Mean < 100 && features.jitter > 0.03) {
            return { quality: VoiceQuality.CREAKY, confidence: 0.75 };
        } else if (features.f0Mean > 300) {
            return { quality: VoiceQuality.FALSETTO, confidence: 0.8 };
        } else if (features.shimmer > 0.05) {
            return { quality: VoiceQuality.TENSE, confidence: 0.7 };
        } else if (features.jitter < 0.01 && features.shimmer < 0.02) {
            return { quality: VoiceQuality.RELAXED, confidence: 0.75 };
        }

        return { quality: VoiceQuality.MODAL, confidence: 0.8 };
    }

    /**
     * Detect paralinguistic events
     */
    _detectParalinguisticEvents(features) {
        const events = [];

        // Laughter: high pitch variation + high intensity
        if (features.f0Range > 150 && features.intensityRange > 30) {
            events.push({ type: ParalinguisticEvent.LAUGHTER, confidence: 0.7 });
        }

        // Sigh: falling pitch + decreasing intensity
        if (features.pitchSlope < -0.2 && features.intensityMean < 50) {
            events.push({ type: ParalinguisticEvent.SIGH, confidence: 0.6 });
        }

        // Voice break: sudden pitch jump
        if (features.jitter > 0.05) {
            events.push({ type: ParalinguisticEvent.VOICE_BREAK, confidence: 0.65 });
        }

        // Emphasis: high intensity + pitch peak
        if (features.intensityMean > 70 && features.f0Max > features.f0Mean * 1.5) {
            events.push({ type: ParalinguisticEvent.EMPHASIS, confidence: 0.75 });
        }

        return events;
    }

    /**
     * Calculate expressiveness (0.0-1.0)
     */
    _calculateExpressiveness(features) {
        const pitchVar = features.getPitchVariability();
        const intensityVar = features.getIntensityVariability();

        return Math.min(1.0, (pitchVar + intensityVar) / 2);
    }

    /**
     * Calculate enthusiasm (0.0-1.0)
     */
    _calculateEnthusiasm(features) {
        const factors = [
            features.speechRate / 6.0,              // Fast speech
            features.intensityMean / 80.0,          // Loud
            features.f0Range / 150.0,               // Wide pitch range
            features.getPitchVariability() * 2      // Variable pitch
        ];

        return Math.min(1.0, this._mean(factors));
    }

    /**
     * Calculate confidence (0.0-1.0)
     */
    _calculateConfidence(features) {
        const factors = [
            Math.min(1.0, features.speechRate / 4.0),  // Steady pace
            Math.min(1.0, features.intensityMean / 70.0), // Adequate volume
            1.0 - features.jitter * 20,                // Low jitter
            features.hasRisingIntonation() ? 0.3 : 0.8 // Declarative > questioning
        ];

        return Math.max(0.0, this._mean(factors));
    }

    /**
     * Calculate stress level (0.0-1.0)
     */
    _calculateStress(features) {
        const factors = [
            Math.min(1.0, features.speechRate / 6.0),  // Fast speech
            features.jitter * 30,                       // High jitter
            features.shimmer * 15,                      // High shimmer
            features.getPitchVariability() * 1.5        // Erratic pitch
        ];

        return Math.min(1.0, this._mean(factors));
    }

    /**
     * Detect sarcasm (0.0-1.0)
     */
    _detectSarcasm(features) {
        // Sarcasm often has exaggerated prosody + slow speech
        if (features.speechRate < 2.5 &&
            features.f0Range > 100 &&
            features.intensityRange > 25) {
            return 0.6;
        }
        return 0.0;
    }

    /**
     * Detect irony (0.0-1.0)
     */
    _detectIrony(features) {
        // Irony: flat affect with occasional emphasis
        if (features.getPitchVariability() < 0.15 &&
            features.intensityRange > 20) {
            return 0.5;
        }
        return 0.0;
    }

    /**
     * Detect hesitation (0.0-1.0)
     */
    _detectHesitation(features) {
        // Hesitation: slow speech + pauses + pitch drops
        if (features.speechRate < 2.0 &&
            features.pitchSlope < -0.1 &&
            features.intensityMean < 55) {
            return 0.7;
        }
        return 0.0;
    }

    /**
     * Update metrics
     */
    _updateMetrics(result, processingTime) {
        this.metrics.totalAnalyses++;

        this.metrics.avgProcessingTimeMs =
            (this.metrics.avgProcessingTimeMs * (this.metrics.totalAnalyses - 1) + processingTime) /
            this.metrics.totalAnalyses;

        this.metrics.styleDistribution[result.speakingStyle] =
            (this.metrics.styleDistribution[result.speakingStyle] || 0) + 1;

        this.metrics.lastAnalysisTime = result.timestamp;
    }

    // Utility functions

    _mean(arr) {
        return arr.length > 0 ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;
    }

    _std(arr, mean = null) {
        if (arr.length === 0) return 0;
        const m = mean !== null ? mean : this._mean(arr);
        const variance = arr.reduce((sum, val) => sum + Math.pow(val - m, 2), 0) / arr.length;
        return Math.sqrt(variance);
    }

    _calculateSlope(arr) {
        if (arr.length < 2) return 0;

        const n = arr.length;
        const xMean = (n - 1) / 2;
        const yMean = this._mean(arr);

        let numerator = 0;
        let denominator = 0;

        for (let i = 0; i < n; i++) {
            numerator += (i - xMean) * (arr[i] - yMean);
            denominator += Math.pow(i - xMean, 2);
        }

        return denominator > 0 ? numerator / denominator : 0;
    }

    /**
     * Get metrics
     */
    getMetrics() {
        return {
            ...this.metrics,
            isAnalyzing: this.isAnalyzing
        };
    }

    /**
     * Cleanup
     */
    async destroy() {
        this.stopAnalysis();

        if (this.audioContext) {
            await this.audioContext.close();
            this.audioContext = null;
        }

        this.pitchHistory = [];
        this.intensityHistory = [];

        console.log('[VocalProsodyAnalyzer] Destroyed');
    }

    // Public API for callbacks

    onProsodyDetected(callback) {
        this.onProsodyDetectedCallback = callback;
    }

    onParalinguisticEvent(callback) {
        this.onParalinguisticEventCallback = callback;
    }
}


// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        VocalProsodyAnalyzer,
        VocalProsodyResult,
        ProsodicFeatures,
        SpeakingStyle,
        VoiceQuality,
        ParalinguisticEvent
    };
}
