/**
 * Emotion Fusion System - Multimodal Emotion Integration
 * Milestone 3 Enhancement: Combined facial + vocal + text emotion analysis
 * Date: November 2025
 *
 * Features:
 * - Multi-channel emotion fusion (face + voice + text)
 * - Conflict detection and resolution
 * - Weighted fusion strategies
 * - Temporal emotion tracking (mood over time)
 * - Emotion state machine (transitions, persistence)
 * - Confidence-weighted aggregation
 * - Modality reliability scoring
 * - Complete emotional profile generation
 */

/**
 * Emotion Modalities
 */
const EmotionModality = {
    FACIAL: 'facial',
    VOCAL: 'vocal',
    TEXT: 'text'
};

/**
 * Fusion Strategies
 */
const FusionStrategy = {
    AVERAGE: 'average',                     // Simple average
    WEIGHTED: 'weighted',                   // Confidence-weighted
    MAX_CONFIDENCE: 'max_confidence',       // Choose highest confidence
    MAJORITY_VOTE: 'majority_vote',         // Most common emotion
    PRIORITY: 'priority',                   // Modality priority (face > voice > text)
    BAYESIAN: 'bayesian'                    // Bayesian combination
};

/**
 * Unified Emotion Categories
 */
const UnifiedEmotion = {
    // Ekman's 6 basic emotions
    HAPPY: 'happy',
    SAD: 'sad',
    ANGRY: 'angry',
    FEARFUL: 'fearful',
    DISGUSTED: 'disgusted',
    SURPRISED: 'surprised',

    // Additional nuanced emotions
    NEUTRAL: 'neutral',
    EXCITED: 'excited',
    CALM: 'calm',
    ANXIOUS: 'anxious',
    CONFUSED: 'confused',
    CONTEMPTUOUS: 'contemptuous'
};

/**
 * Emotional Valence (positive/negative dimension)
 */
const EmotionalValence = {
    VERY_NEGATIVE: -2,
    NEGATIVE: -1,
    NEUTRAL: 0,
    POSITIVE: 1,
    VERY_POSITIVE: 2
};

/**
 * Fused Emotion Result
 */
class FusedEmotionResult {
    constructor(data) {
        // Primary emotion
        this.emotion = data.emotion || UnifiedEmotion.NEUTRAL;
        this.confidence = data.confidence || 0.0;

        // Emotion probabilities across all categories
        this.emotionProbabilities = data.emotionProbabilities || {};

        // Dimensional metrics
        this.valence = data.valence || 0.0;         // -1.0 (negative) to +1.0 (positive)
        this.arousal = data.arousal || 0.0;         // 0.0 (calm) to 1.0 (excited)
        this.dominance = data.dominance || 0.5;     // 0.0 (submissive) to 1.0 (dominant)

        // Source contributions
        this.facialContribution = data.facialContribution || null;
        this.vocalContribution = data.vocalContribution || null;
        this.textContribution = data.textContribution || null;

        // Fusion metadata
        this.fusionStrategy = data.fusionStrategy || FusionStrategy.WEIGHTED;
        this.modalitiesUsed = data.modalitiesUsed || [];
        this.conflictDetected = data.conflictDetected || false;
        this.conflictResolution = data.conflictResolution || null;

        // Temporal context
        this.previousEmotion = data.previousEmotion || null;
        this.emotionDuration = data.emotionDuration || 0; // ms
        this.emotionStability = data.emotionStability || 0.0; // 0.0-1.0

        // Nuanced analysis
        this.sarcasm = data.sarcasm || 0.0;
        this.genuineness = data.genuineness || 1.0;  // How genuine the emotion appears
        this.intensity = data.intensity || 0.5;      // Emotion intensity

        // Metadata
        this.timestamp = data.timestamp || new Date();
        this.fusionConfidence = data.fusionConfidence || 0.0;
    }

    /**
     * Get emotional state as VAD (Valence-Arousal-Dominance) vector
     */
    getVADVector() {
        return {
            valence: this.valence,
            arousal: this.arousal,
            dominance: this.dominance
        };
    }

    /**
     * Check if emotion is stable (not rapidly changing)
     */
    isStable() {
        return this.emotionStability > 0.7 && this.emotionDuration > 3000;
    }

    /**
     * Get emotional intensity category
     */
    getIntensityCategory() {
        if (this.intensity < 0.3) return 'low';
        if (this.intensity < 0.7) return 'moderate';
        return 'high';
    }

    /**
     * Check for mixed emotions (conflict between modalities)
     */
    hasMixedEmotions() {
        return this.conflictDetected && this.confidence < 0.7;
    }
}

/**
 * Emotion Fusion System
 * Combines emotion signals from multiple modalities
 */
class EmotionFusionSystem {
    constructor(config = {}) {
        this.config = {
            enabled: config.enabled !== false,

            // Fusion settings
            fusionStrategy: config.fusionStrategy || FusionStrategy.WEIGHTED,

            // Modality weights (when using weighted fusion)
            modalityWeights: config.modalityWeights || {
                facial: 0.5,    // Face is most reliable
                vocal: 0.3,     // Voice is good indicator
                text: 0.2       // Text can be misleading (sarcasm)
            },

            // Conflict detection
            conflictThreshold: config.conflictThreshold || 0.4, // Max emotion difference
            enableConflictResolution: config.enableConflictResolution !== false,

            // Temporal tracking
            emotionHistorySize: config.emotionHistorySize || 10,
            stabilityThreshold: config.stabilityThreshold || 0.7,

            // Minimum confidence thresholds
            minFacialConfidence: config.minFacialConfidence || 0.5,
            minVocalConfidence: config.minVocalConfidence || 0.4,
            minTextConfidence: config.minTextConfidence || 0.3,

            ...config
        };

        // Emotion history (for temporal analysis)
        this.emotionHistory = [];

        // Current emotional state
        this.currentEmotion = null;
        this.emotionStartTime = null;

        // Callbacks
        this.onEmotionChangedCallback = null;
        this.onConflictDetectedCallback = null;

        // Metrics
        this.metrics = {
            totalFusions: 0,
            fusionsByStrategy: {},
            conflictsDetected: 0,
            avgFusionConfidence: 0.0,
            modalityUsage: { facial: 0, vocal: 0, text: 0 }
        };

        console.log('[EmotionFusionSystem] Initialized with strategy:', this.config.fusionStrategy);
    }

    /**
     * Fuse emotions from multiple modalities
     */
    fuseEmotions(inputs) {
        const { facial, vocal, text } = inputs;

        // Filter inputs by confidence threshold
        const validInputs = this._filterValidInputs({ facial, vocal, text });

        if (validInputs.length === 0) {
            console.log('[EmotionFusionSystem] No valid emotion inputs');
            return this._createNeutralResult();
        }

        // Apply fusion strategy
        let fusedResult;

        switch (this.config.fusionStrategy) {
            case FusionStrategy.AVERAGE:
                fusedResult = this._averageFusion(validInputs);
                break;
            case FusionStrategy.WEIGHTED:
                fusedResult = this._weightedFusion(validInputs);
                break;
            case FusionStrategy.MAX_CONFIDENCE:
                fusedResult = this._maxConfidenceFusion(validInputs);
                break;
            case FusionStrategy.MAJORITY_VOTE:
                fusedResult = this._majorityVoteFusion(validInputs);
                break;
            case FusionStrategy.PRIORITY:
                fusedResult = this._priorityFusion(validInputs);
                break;
            case FusionStrategy.BAYESIAN:
                fusedResult = this._bayesianFusion(validInputs);
                break;
            default:
                fusedResult = this._weightedFusion(validInputs);
        }

        // Detect conflicts
        const conflict = this._detectConflict(validInputs);
        fusedResult.conflictDetected = conflict.detected;
        fusedResult.conflictResolution = conflict.resolution;

        // Add temporal context
        fusedResult = this._addTemporalContext(fusedResult);

        // Detect nuances
        fusedResult.sarcasm = this._detectSarcasm(validInputs);
        fusedResult.genuineness = this._assessGenuineness(validInputs);

        // Store in history
        this._addToHistory(fusedResult);

        // Update current emotion
        this._updateCurrentEmotion(fusedResult);

        // Update metrics
        this._updateMetrics(fusedResult, validInputs);

        // Emit events
        this._emitEvents(fusedResult);

        console.log('[EmotionFusionSystem] Fused emotion:', fusedResult.emotion,
            'confidence:', fusedResult.confidence.toFixed(2));

        return fusedResult;
    }

    /**
     * Filter valid inputs by confidence thresholds
     */
    _filterValidInputs(inputs) {
        const validInputs = [];

        if (inputs.facial && inputs.facial.confidence >= this.config.minFacialConfidence) {
            validInputs.push({ modality: EmotionModality.FACIAL, data: inputs.facial });
            this.metrics.modalityUsage.facial++;
        }

        if (inputs.vocal && inputs.vocal.confidence >= this.config.minVocalConfidence) {
            validInputs.push({ modality: EmotionModality.VOCAL, data: inputs.vocal });
            this.metrics.modalityUsage.vocal++;
        }

        if (inputs.text && inputs.text.confidence >= this.config.minTextConfidence) {
            validInputs.push({ modality: EmotionModality.TEXT, data: inputs.text });
            this.metrics.modalityUsage.text++;
        }

        return validInputs;
    }

    /**
     * Simple average fusion
     */
    _averageFusion(inputs) {
        const emotions = {};

        for (const input of inputs) {
            const emotion = this._normalizeEmotion(input.data.emotion);
            emotions[emotion] = (emotions[emotion] || 0) + 1;
        }

        // Find most common emotion
        const primaryEmotion = Object.entries(emotions)
            .sort((a, b) => b[1] - a[1])[0][0];

        const confidence = emotions[primaryEmotion] / inputs.length;

        return this._createFusedResult(primaryEmotion, confidence, inputs);
    }

    /**
     * Confidence-weighted fusion
     */
    _weightedFusion(inputs) {
        const emotionScores = {};
        let totalWeight = 0;

        for (const input of inputs) {
            const emotion = this._normalizeEmotion(input.data.emotion);
            const weight = input.data.confidence * this.config.modalityWeights[input.modality];

            emotionScores[emotion] = (emotionScores[emotion] || 0) + weight;
            totalWeight += weight;
        }

        // Normalize scores
        const primaryEmotion = Object.entries(emotionScores)
            .sort((a, b) => b[1] - a[1])[0][0];

        const confidence = totalWeight > 0 ? emotionScores[primaryEmotion] / totalWeight : 0;

        return this._createFusedResult(primaryEmotion, confidence, inputs);
    }

    /**
     * Maximum confidence fusion (choose most confident)
     */
    _maxConfidenceFusion(inputs) {
        const best = inputs.reduce((max, input) =>
            input.data.confidence > max.data.confidence ? input : max
        );

        const emotion = this._normalizeEmotion(best.data.emotion);

        return this._createFusedResult(emotion, best.data.confidence, inputs);
    }

    /**
     * Majority vote fusion
     */
    _majorityVoteFusion(inputs) {
        const votes = {};

        for (const input of inputs) {
            const emotion = this._normalizeEmotion(input.data.emotion);
            votes[emotion] = (votes[emotion] || 0) + 1;
        }

        const primaryEmotion = Object.entries(votes)
            .sort((a, b) => b[1] - a[1])[0][0];

        const confidence = votes[primaryEmotion] / inputs.length;

        return this._createFusedResult(primaryEmotion, confidence, inputs);
    }

    /**
     * Priority-based fusion (face > voice > text)
     */
    _priorityFusion(inputs) {
        const priorities = [
            EmotionModality.FACIAL,
            EmotionModality.VOCAL,
            EmotionModality.TEXT
        ];

        for (const priority of priorities) {
            const input = inputs.find(i => i.modality === priority);
            if (input) {
                const emotion = this._normalizeEmotion(input.data.emotion);
                return this._createFusedResult(emotion, input.data.confidence, inputs);
            }
        }

        return this._createNeutralResult();
    }

    /**
     * Bayesian fusion (probabilistic combination)
     */
    _bayesianFusion(inputs) {
        // Simplified Bayesian: multiply probabilities and normalize
        const emotionProbs = {};

        // Initialize with uniform prior
        for (const emotion of Object.values(UnifiedEmotion)) {
            emotionProbs[emotion] = 1.0 / Object.keys(UnifiedEmotion).length;
        }

        // Update with each modality
        for (const input of inputs) {
            const emotion = this._normalizeEmotion(input.data.emotion);
            const confidence = input.data.confidence;
            const weight = this.config.modalityWeights[input.modality];

            // Bayes update: P(emotion|data) ∝ P(data|emotion) * P(emotion)
            emotionProbs[emotion] *= confidence * weight;
        }

        // Normalize
        const total = Object.values(emotionProbs).reduce((a, b) => a + b, 0);
        for (const emotion in emotionProbs) {
            emotionProbs[emotion] /= total;
        }

        // Find primary emotion
        const primaryEmotion = Object.entries(emotionProbs)
            .sort((a, b) => b[1] - a[1])[0][0];

        return this._createFusedResult(primaryEmotion, emotionProbs[primaryEmotion], inputs);
    }

    /**
     * Create fused result
     */
    _createFusedResult(emotion, confidence, inputs) {
        // Calculate VAD (Valence-Arousal-Dominance)
        const vad = this._calculateVAD(emotion, inputs);

        // Extract contributions from each modality
        const contributions = {
            facial: inputs.find(i => i.modality === EmotionModality.FACIAL)?.data || null,
            vocal: inputs.find(i => i.modality === EmotionModality.VOCAL)?.data || null,
            text: inputs.find(i => i.modality === EmotionModality.TEXT)?.data || null
        };

        return new FusedEmotionResult({
            emotion: emotion,
            confidence: confidence,
            valence: vad.valence,
            arousal: vad.arousal,
            dominance: vad.dominance,
            facialContribution: contributions.facial,
            vocalContribution: contributions.vocal,
            textContribution: contributions.text,
            fusionStrategy: this.config.fusionStrategy,
            modalitiesUsed: inputs.map(i => i.modality),
            fusionConfidence: this._calculateFusionConfidence(inputs)
        });
    }

    /**
     * Create neutral result (fallback)
     */
    _createNeutralResult() {
        return new FusedEmotionResult({
            emotion: UnifiedEmotion.NEUTRAL,
            confidence: 1.0,
            fusionStrategy: this.config.fusionStrategy
        });
    }

    /**
     * Normalize emotion labels to unified categories
     */
    _normalizeEmotion(emotion) {
        const mapping = {
            'joy': UnifiedEmotion.HAPPY,
            'happiness': UnifiedEmotion.HAPPY,
            'sadness': UnifiedEmotion.SAD,
            'anger': UnifiedEmotion.ANGRY,
            'fear': UnifiedEmotion.FEARFUL,
            'disgust': UnifiedEmotion.DISGUSTED,
            'surprise': UnifiedEmotion.SURPRISED,
            'excited': UnifiedEmotion.EXCITED,
            'anxiety': UnifiedEmotion.ANXIOUS
        };

        return mapping[emotion.toLowerCase()] || emotion;
    }

    /**
     * Calculate VAD (Valence-Arousal-Dominance) from emotion
     */
    _calculateVAD(emotion, inputs) {
        // Emotion → VAD mapping
        const vadMapping = {
            [UnifiedEmotion.HAPPY]: { valence: 0.8, arousal: 0.6, dominance: 0.7 },
            [UnifiedEmotion.SAD]: { valence: -0.7, arousal: 0.3, dominance: 0.3 },
            [UnifiedEmotion.ANGRY]: { valence: -0.6, arousal: 0.8, dominance: 0.8 },
            [UnifiedEmotion.FEARFUL]: { valence: -0.8, arousal: 0.9, dominance: 0.2 },
            [UnifiedEmotion.DISGUSTED]: { valence: -0.7, arousal: 0.5, dominance: 0.5 },
            [UnifiedEmotion.SURPRISED]: { valence: 0.2, arousal: 0.9, dominance: 0.5 },
            [UnifiedEmotion.NEUTRAL]: { valence: 0.0, arousal: 0.3, dominance: 0.5 },
            [UnifiedEmotion.EXCITED]: { valence: 0.9, arousal: 0.9, dominance: 0.7 },
            [UnifiedEmotion.CALM]: { valence: 0.5, arousal: 0.1, dominance: 0.6 },
            [UnifiedEmotion.ANXIOUS]: { valence: -0.5, arousal: 0.8, dominance: 0.3 }
        };

        return vadMapping[emotion] || { valence: 0, arousal: 0.5, dominance: 0.5 };
    }

    /**
     * Detect conflict between modalities
     */
    _detectConflict(inputs) {
        if (inputs.length < 2) {
            return { detected: false, resolution: null };
        }

        const emotions = inputs.map(i => this._normalizeEmotion(i.data.emotion));
        const uniqueEmotions = [...new Set(emotions)];

        // Conflict if emotions disagree significantly
        if (uniqueEmotions.length > 1) {
            this.metrics.conflictsDetected++;

            if (this.onConflictDetectedCallback) {
                this.onConflictDetectedCallback({
                    type: 'emotion_conflict',
                    emotions: emotions,
                    timestamp: new Date()
                });
            }

            return {
                detected: true,
                resolution: `Resolved via ${this.config.fusionStrategy} fusion`
            };
        }

        return { detected: false, resolution: null };
    }

    /**
     * Add temporal context to result
     */
    _addTemporalContext(result) {
        if (this.currentEmotion === result.emotion) {
            // Same emotion - calculate duration
            result.emotionDuration = Date.now() - this.emotionStartTime;
            result.previousEmotion = this.currentEmotion;

            // Calculate stability
            result.emotionStability = this._calculateStability();
        } else {
            // Emotion changed
            result.emotionDuration = 0;
            result.previousEmotion = this.currentEmotion;
            result.emotionStability = 0.0;
        }

        return result;
    }

    /**
     * Calculate emotion stability (consistency over time)
     */
    _calculateStability() {
        if (this.emotionHistory.length < 3) {
            return 0.0;
        }

        const recent = this.emotionHistory.slice(-5);
        const sameEmotion = recent.filter(e => e.emotion === this.currentEmotion).length;

        return sameEmotion / recent.length;
    }

    /**
     * Detect sarcasm (mismatch between text and tone)
     */
    _detectSarcasm(inputs) {
        const textInput = inputs.find(i => i.modality === EmotionModality.TEXT);
        const vocalInput = inputs.find(i => i.modality === EmotionModality.VOCAL);

        if (!textInput || !vocalInput) {
            return 0.0;
        }

        const textEmotion = this._normalizeEmotion(textInput.data.emotion);
        const vocalEmotion = this._normalizeEmotion(vocalInput.data.emotion);

        // Sarcasm: positive text + negative voice (or vice versa)
        const textValence = this._getEmotionValence(textEmotion);
        const vocalValence = this._getEmotionValence(vocalEmotion);

        if ((textValence > 0 && vocalValence < 0) ||
            (textValence < 0 && vocalValence > 0)) {
            return 0.7; // High sarcasm probability
        }

        return 0.0;
    }

    /**
     * Assess genuineness (agreement between modalities)
     */
    _assessGenuineness(inputs) {
        if (inputs.length < 2) {
            return 1.0; // Single modality = assume genuine
        }

        const emotions = inputs.map(i => this._normalizeEmotion(i.data.emotion));
        const uniqueEmotions = [...new Set(emotions)];

        // High agreement = high genuineness
        return 1.0 - (uniqueEmotions.length - 1) * 0.3;
    }

    /**
     * Get emotion valence (positive/negative)
     */
    _getEmotionValence(emotion) {
        const positiveEmotions = [UnifiedEmotion.HAPPY, UnifiedEmotion.EXCITED, UnifiedEmotion.CALM];
        const negativeEmotions = [UnifiedEmotion.SAD, UnifiedEmotion.ANGRY, UnifiedEmotion.FEARFUL, UnifiedEmotion.DISGUSTED];

        if (positiveEmotions.includes(emotion)) return 1;
        if (negativeEmotions.includes(emotion)) return -1;
        return 0;
    }

    /**
     * Calculate fusion confidence
     */
    _calculateFusionConfidence(inputs) {
        const avgConfidence = inputs.reduce((sum, i) => sum + i.data.confidence, 0) / inputs.length;
        const modalityBonus = inputs.length * 0.1; // More modalities = higher confidence

        return Math.min(1.0, avgConfidence + modalityBonus);
    }

    /**
     * Add result to history
     */
    _addToHistory(result) {
        this.emotionHistory.push(result);

        if (this.emotionHistory.length > this.config.emotionHistorySize) {
            this.emotionHistory.shift();
        }
    }

    /**
     * Update current emotion state
     */
    _updateCurrentEmotion(result) {
        if (this.currentEmotion !== result.emotion) {
            this.currentEmotion = result.emotion;
            this.emotionStartTime = Date.now();

            // Emit emotion changed event
            if (this.onEmotionChangedCallback) {
                this.onEmotionChangedCallback({
                    type: 'emotion_changed',
                    newEmotion: result.emotion,
                    previousEmotion: result.previousEmotion,
                    timestamp: new Date()
                });
            }
        }
    }

    /**
     * Update metrics
     */
    _updateMetrics(result, inputs) {
        this.metrics.totalFusions++;

        this.metrics.fusionsByStrategy[this.config.fusionStrategy] =
            (this.metrics.fusionsByStrategy[this.config.fusionStrategy] || 0) + 1;

        this.metrics.avgFusionConfidence =
            (this.metrics.avgFusionConfidence * (this.metrics.totalFusions - 1) + result.fusionConfidence) /
            this.metrics.totalFusions;
    }

    /**
     * Emit events
     */
    _emitEvents(result) {
        // Events emitted via callbacks
    }

    /**
     * Get recent emotional trend
     */
    getEmotionalTrend(windowSize = 5) {
        if (this.emotionHistory.length < windowSize) {
            return null;
        }

        const recent = this.emotionHistory.slice(-windowSize);
        const avgValence = recent.reduce((sum, e) => sum + e.valence, 0) / windowSize;
        const avgArousal = recent.reduce((sum, e) => sum + e.arousal, 0) / windowSize;

        return {
            valence: avgValence,
            arousal: avgArousal,
            trend: avgValence > 0 ? 'positive' : avgValence < 0 ? 'negative' : 'neutral'
        };
    }

    /**
     * Get dominant mood (most common emotion over time)
     */
    getDominantMood() {
        if (this.emotionHistory.length === 0) {
            return null;
        }

        const emotionCounts = {};
        for (const result of this.emotionHistory) {
            emotionCounts[result.emotion] = (emotionCounts[result.emotion] || 0) + 1;
        }

        return Object.entries(emotionCounts)
            .sort((a, b) => b[1] - a[1])[0][0];
    }

    /**
     * Get metrics
     */
    getMetrics() {
        return {
            ...this.metrics,
            currentEmotion: this.currentEmotion,
            historySize: this.emotionHistory.length
        };
    }

    /**
     * Clear emotion history
     */
    clearHistory() {
        this.emotionHistory = [];
        this.currentEmotion = null;
        this.emotionStartTime = null;
    }

    // Public API for callbacks

    onEmotionChanged(callback) {
        this.onEmotionChangedCallback = callback;
    }

    onConflictDetected(callback) {
        this.onConflictDetectedCallback = callback;
    }
}


// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        EmotionFusionSystem,
        FusedEmotionResult,
        EmotionModality,
        FusionStrategy,
        UnifiedEmotion,
        EmotionalValence
    };
}
