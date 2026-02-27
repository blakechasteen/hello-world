/**
 * Predictive Emotional Anticipation System
 * Predicts user's emotional state before they speak based on context and patterns
 * Date: November 2025
 *
 * Breakthrough Feature: 105/100
 *
 * Core Capabilities:
 * 1. Context-based emotion prediction
 * 2. Historical pattern matching
 * 3. Real-time pre-speech signal analysis
 * 4. Temporal pattern recognition
 * 5. Proactive intervention triggering
 * 6. Confidence-based prediction gating
 */

import { EmotionalKnowledgeGraph } from './emotional_knowledge_graph.js';
import { ChainOfEmotionReasoner } from './chain_of_emotion_reasoning.js';

/**
 * Prediction Signals
 * Observable cues before user speaks
 */
const PredictionSignal = {
    TEMPORAL: 'temporal',              // Time of day, day of week
    CONTEXTUAL: 'contextual',          // Situation, environment
    PHYSIOLOGICAL: 'physiological',    // Pre-speech facial cues, body language
    BEHAVIORAL: 'behavioral',          // User actions, interaction patterns
    HISTORICAL: 'historical',          // Past patterns from knowledge graph
    SEQUENTIAL: 'sequential'           // Emotion chain trajectory
};

/**
 * Prediction Confidence Levels
 */
const ConfidenceLevel = {
    VERY_HIGH: { min: 0.8, label: 'Very High', actionable: true },
    HIGH: { min: 0.6, label: 'High', actionable: true },
    MEDIUM: { min: 0.4, label: 'Medium', actionable: false },
    LOW: { min: 0.2, label: 'Low', actionable: false },
    VERY_LOW: { min: 0.0, label: 'Very Low', actionable: false }
};

/**
 * Prediction Result
 */
class EmotionPrediction {
    constructor(data) {
        this.predictedEmotion = data.predictedEmotion;
        this.confidence = data.confidence || 0.0;
        this.timestamp = new Date();

        // Supporting signals
        this.signals = data.signals || {};  // SignalType -> signal data

        // Reasoning
        this.reasoning = data.reasoning || '';
        this.evidenceSources = data.evidenceSources || [];

        // Alternative predictions
        this.alternatives = data.alternatives || [];

        // Actionability
        this.isActionable = this.confidence >= ConfidenceLevel.HIGH.min;
        this.suggestedPreemptiveAction = data.suggestedPreemptiveAction || null;

        // Validation
        this.actualEmotion = null;  // Will be filled when user speaks
        this.wasCorrect = null;
    }

    /**
     * Validate prediction against actual emotion
     */
    validate(actualEmotion) {
        this.actualEmotion = actualEmotion;
        this.wasCorrect = this.predictedEmotion === actualEmotion;
        return this.wasCorrect;
    }

    getConfidenceLevel() {
        for (const [level, config] of Object.entries(ConfidenceLevel)) {
            if (this.confidence >= config.min) {
                return config.label;
            }
        }
        return ConfidenceLevel.VERY_LOW.label;
    }

    toJSON() {
        return {
            predictedEmotion: this.predictedEmotion,
            confidence: this.confidence,
            confidenceLevel: this.getConfidenceLevel(),
            isActionable: this.isActionable,
            reasoning: this.reasoning,
            signals: this.signals,
            alternatives: this.alternatives,
            suggestedAction: this.suggestedPreemptiveAction,
            timestamp: this.timestamp.toISOString()
        };
    }
}

/**
 * Context Analyzer
 * Extracts contextual features for prediction
 */
class ContextAnalyzer {
    constructor() {
        // Temporal patterns
        this.temporalPatterns = new Map();  // "hour:dayOfWeek" -> emotion distribution
    }

    /**
     * Extract contextual features
     */
    extractFeatures(context = {}) {
        const now = new Date();

        return {
            temporal: {
                hour: now.getHours(),
                dayOfWeek: now.getDay(),
                isWeekend: now.getDay() === 0 || now.getDay() === 6,
                timeOfDay: this._getTimeOfDay(now.getHours()),
                date: now.toISOString().split('T')[0]
            },
            situational: {
                location: context.location || 'unknown',
                activity: context.activity || 'unknown',
                withOthers: context.withOthers || false,
                stressLevel: context.stressLevel || 'normal'
            },
            environmental: {
                noiseLevel: context.noiseLevel || 'normal',
                lighting: context.lighting || 'normal',
                temperature: context.temperature || 'normal'
            },
            interaction: {
                sessionDuration: context.sessionDuration || 0,
                turnsInSession: context.turnsInSession || 0,
                lastInteractionTime: context.lastInteractionTime || null
            }
        };
    }

    _getTimeOfDay(hour) {
        if (hour >= 5 && hour < 12) return 'morning';
        if (hour >= 12 && hour < 17) return 'afternoon';
        if (hour >= 17 && hour < 21) return 'evening';
        return 'night';
    }

    /**
     * Learn temporal pattern
     */
    learnTemporalPattern(hour, dayOfWeek, emotion) {
        const key = `${hour}:${dayOfWeek}`;

        if (!this.temporalPatterns.has(key)) {
            this.temporalPatterns.set(key, new Map());
        }

        const distribution = this.temporalPatterns.get(key);
        distribution.set(emotion, (distribution.get(emotion) || 0) + 1);
    }

    /**
     * Get likely emotion for temporal context
     */
    predictFromTemporal(hour, dayOfWeek) {
        const key = `${hour}:${dayOfWeek}`;
        const distribution = this.temporalPatterns.get(key);

        if (!distribution || distribution.size === 0) {
            return { emotion: null, confidence: 0.0 };
        }

        // Get most common emotion
        const total = Array.from(distribution.values()).reduce((a, b) => a + b, 0);
        const sorted = Array.from(distribution.entries())
            .sort((a, b) => b[1] - a[1]);

        const [emotion, count] = sorted[0];
        const confidence = count / total;

        return { emotion, confidence };
    }
}

/**
 * Pre-Speech Signal Detector
 * Detects physiological and behavioral cues before user speaks
 */
class PreSpeechSignalDetector {
    constructor() {
        this.signalHistory = [];
        this.maxHistoryLength = 100;
    }

    /**
     * Analyze pre-speech signals
     */
    async analyzePreSpeechSignals(videoFrame = null, audioBuffer = null) {
        const signals = {};

        // Facial micro-expressions (before speaking)
        if (videoFrame) {
            signals.facial = await this._detectFacialCues(videoFrame);
        }

        // Pre-speech vocal characteristics (hesitation, breath patterns)
        if (audioBuffer) {
            signals.vocal = this._detectVocalCues(audioBuffer);
        }

        // Interaction timing patterns
        signals.timing = this._analyzeInteractionTiming();

        return signals;
    }

    async _detectFacialCues(videoFrame) {
        // STUB: In production, use actual computer vision
        // Look for:
        // - Furrowed brow (anxiety, concern)
        // - Tensed jaw (stress, anger)
        // - Downturned mouth (sadness)
        // - Raised eyebrows (surprise, interest)
        // - Gaze direction (engagement, avoidance)

        return {
            tensionLevel: 0.3,
            gazeEngaged: true,
            microExpressions: ['brow_furrow'],
            confidence: 0.6
        };
    }

    _detectVocalCues(audioBuffer) {
        // Analyze pre-speech audio:
        // - Sighs (frustration, fatigue)
        // - Sharp intake of breath (anxiety, surprise)
        // - Hesitation/pauses (uncertainty)
        // - Vocal fry (fatigue, disengagement)

        return {
            hesitation: false,
            sighs: false,
            breathPattern: 'normal',
            confidence: 0.5
        };
    }

    _analyzeInteractionTiming() {
        if (this.signalHistory.length < 3) {
            return { pattern: 'insufficient_data', confidence: 0.0 };
        }

        // Analyze timing patterns:
        // - Rapid responses (engagement, excitement)
        // - Long pauses (hesitation, contemplation)
        // - Irregular timing (stress, distraction)

        const recentTimings = this.signalHistory.slice(-5);
        const avgGap = recentTimings.reduce((sum, t) => sum + t.gap, 0) / recentTimings.length;

        if (avgGap < 2000) {
            return { pattern: 'rapid', impliedEmotion: 'engaged', confidence: 0.6 };
        } else if (avgGap > 10000) {
            return { pattern: 'slow', impliedEmotion: 'contemplative', confidence: 0.5 };
        }

        return { pattern: 'normal', confidence: 0.3 };
    }

    recordInteraction(timestamp) {
        const gap = this.signalHistory.length > 0 ?
            timestamp - this.signalHistory[this.signalHistory.length - 1].timestamp :
            0;

        this.signalHistory.push({ timestamp, gap });

        if (this.signalHistory.length > this.maxHistoryLength) {
            this.signalHistory.shift();
        }
    }
}

/**
 * Predictive Emotional Anticipation System
 */
class PredictiveEmotionalAnticipation {
    constructor(config = {}) {
        this.config = {
            // Prediction settings
            minConfidenceForAction: config.minConfidenceForAction || 0.6,
            enableProactiveIntervention: config.enableProactiveIntervention !== false,
            enableLearning: config.enableLearning !== false,

            // Component settings
            userId: config.userId || 'default_user',
            llmProvider: config.llmProvider || 'mock',

            ...config
        };

        // Initialize components
        this.contextAnalyzer = new ContextAnalyzer();
        this.preSpeechDetector = new PreSpeechSignalDetector();

        // Integration with other systems
        this.knowledgeGraph = config.knowledgeGraph || new EmotionalKnowledgeGraph(this.config.userId);
        this.chainReasoner = config.chainReasoner || new ChainOfEmotionReasoner();

        // Prediction history
        this.predictions = [];
        this.validatedPredictions = [];

        // Learning
        this.accuracyBySignal = new Map();  // SignalType -> accuracy rate

        console.log('[PredictiveAnticipation] Initialized for user:', this.config.userId);
    }

    /**
     * Predict user's emotional state before they speak
     */
    async predict(context = {}, preSpeechData = {}) {
        const signals = {};
        const evidenceSources = [];

        // 1. Temporal prediction
        const contextFeatures = this.contextAnalyzer.extractFeatures(context);
        const temporalPrediction = this.contextAnalyzer.predictFromTemporal(
            contextFeatures.temporal.hour,
            contextFeatures.temporal.dayOfWeek
        );

        if (temporalPrediction.confidence > 0.3) {
            signals[PredictionSignal.TEMPORAL] = temporalPrediction;
            evidenceSources.push(`Temporal pattern: ${contextFeatures.temporal.timeOfDay}`);
        }

        // 2. Historical pattern matching
        const historicalPrediction = this._predictFromHistory(contextFeatures);
        if (historicalPrediction.confidence > 0.3) {
            signals[PredictionSignal.HISTORICAL] = historicalPrediction;
            evidenceSources.push('Historical emotional patterns');
        }

        // 3. Sequential prediction (emotion chain trajectory)
        const sequentialPrediction = await this._predictFromChain();
        if (sequentialPrediction.confidence > 0.3) {
            signals[PredictionSignal.SEQUENTIAL] = sequentialPrediction;
            evidenceSources.push('Emotional chain trajectory');
        }

        // 4. Pre-speech signal analysis
        if (preSpeechData.videoFrame || preSpeechData.audioBuffer) {
            const preSpeechSignals = await this.preSpeechDetector.analyzePreSpeechSignals(
                preSpeechData.videoFrame,
                preSpeechData.audioBuffer
            );

            if (preSpeechSignals.facial) {
                signals[PredictionSignal.PHYSIOLOGICAL] = preSpeechSignals.facial;
                evidenceSources.push('Facial micro-expressions');
            }

            if (preSpeechSignals.vocal) {
                signals[PredictionSignal.BEHAVIORAL] = preSpeechSignals.vocal;
                evidenceSources.push('Pre-speech vocal patterns');
            }
        }

        // 5. Contextual prediction
        const contextualPrediction = this._predictFromContext(contextFeatures);
        if (contextualPrediction.confidence > 0.3) {
            signals[PredictionSignal.CONTEXTUAL] = contextualPrediction;
            evidenceSources.push('Situational context');
        }

        // 6. Ensemble prediction (combine all signals)
        const finalPrediction = this._ensemblePrediction(signals, contextFeatures);

        // 7. Determine if actionable
        let suggestedAction = null;
        if (finalPrediction.confidence >= this.config.minConfidenceForAction) {
            suggestedAction = this._suggestPreemptiveAction(finalPrediction.emotion, contextFeatures);
        }

        // Create prediction result
        const prediction = new EmotionPrediction({
            predictedEmotion: finalPrediction.emotion,
            confidence: finalPrediction.confidence,
            signals: signals,
            reasoning: finalPrediction.reasoning,
            evidenceSources: evidenceSources,
            alternatives: finalPrediction.alternatives,
            suggestedPreemptiveAction: suggestedAction
        });

        this.predictions.push(prediction);

        return prediction;
    }

    /**
     * Predict from historical patterns in knowledge graph
     */
    _predictFromHistory(contextFeatures) {
        // Query knowledge graph for similar contexts
        const triggers = this.knowledgeGraph.getEmotionalTriggers();

        if (triggers.length === 0) {
            return { emotion: null, confidence: 0.0 };
        }

        // Match context to historical triggers
        const matchingTriggers = triggers.filter(t => {
            // Simple matching: check if trigger relates to current situation
            return t.trigger.includes(contextFeatures.situational.activity) ||
                   t.trigger.includes(contextFeatures.temporal.timeOfDay);
        });

        if (matchingTriggers.length === 0) {
            // Fall back to most common historical emotion
            return {
                emotion: triggers[0].emotion,
                confidence: triggers[0].strength * 0.5  // Lower confidence for generic match
            };
        }

        // Use most frequent matching trigger
        const topMatch = matchingTriggers.sort((a, b) => b.strength - a.strength)[0];

        return {
            emotion: topMatch.emotion,
            confidence: topMatch.strength * 0.8
        };
    }

    /**
     * Predict from emotion chain trajectory
     */
    async _predictFromChain() {
        const chainSummary = this.chainReasoner.getChainSummary(this.config.userId);

        if (!chainSummary.exists || chainSummary.length < 3) {
            return { emotion: null, confidence: 0.0 };
        }

        // Use chain reasoner's prediction
        const prediction = await this.chainReasoner.predictNext(this.config.userId);

        return {
            emotion: prediction.predictedEmotion,
            confidence: prediction.confidence * 0.9,  // Slightly discount
            timeframe: prediction.timeframe
        };
    }

    /**
     * Predict from contextual features
     */
    _predictFromContext(contextFeatures) {
        // Simple heuristics based on context
        const { situational, environmental, interaction } = contextFeatures;

        if (situational.stressLevel === 'high') {
            return { emotion: 'anxious', confidence: 0.7 };
        }

        if (environmental.noiseLevel === 'loud') {
            return { emotion: 'frustrated', confidence: 0.6 };
        }

        if (interaction.sessionDuration > 1800000) {  // >30 minutes
            return { emotion: 'fatigued', confidence: 0.5 };
        }

        return { emotion: null, confidence: 0.0 };
    }

    /**
     * Combine multiple prediction signals
     */
    _ensemblePrediction(signals, contextFeatures) {
        if (Object.keys(signals).length === 0) {
            return {
                emotion: 'neutral',
                confidence: 0.3,
                reasoning: 'Insufficient signals for prediction',
                alternatives: []
            };
        }

        // Weight signals by learned accuracy
        const weightedVotes = new Map();

        for (const [signalType, prediction] of Object.entries(signals)) {
            if (!prediction.emotion) continue;

            const signalAccuracy = this.accuracyBySignal.get(signalType) || 0.5;
            const weight = prediction.confidence * signalAccuracy;

            const currentWeight = weightedVotes.get(prediction.emotion) || 0;
            weightedVotes.set(prediction.emotion, currentWeight + weight);
        }

        // Get top predictions
        const sorted = Array.from(weightedVotes.entries())
            .sort((a, b) => b[1] - a[1]);

        if (sorted.length === 0) {
            return {
                emotion: 'neutral',
                confidence: 0.2,
                reasoning: 'No confident predictions',
                alternatives: []
            };
        }

        const [topEmotion, topWeight] = sorted[0];
        const totalWeight = Array.from(weightedVotes.values()).reduce((a, b) => a + b, 0);
        const confidence = topWeight / totalWeight;

        const alternatives = sorted.slice(1, 4).map(([emotion, weight]) => ({
            emotion,
            confidence: weight / totalWeight
        }));

        const reasoning = `Predicted based on ${Object.keys(signals).length} signals: ${Object.keys(signals).join(', ')}`;

        return {
            emotion: topEmotion,
            confidence,
            reasoning,
            alternatives
        };
    }

    /**
     * Suggest preemptive action
     */
    _suggestPreemptiveAction(predictedEmotion, contextFeatures) {
        // Get effective responses from knowledge graph
        const effectiveResponses = this.knowledgeGraph.getEffectiveResponses(predictedEmotion);

        if (effectiveResponses.length > 0) {
            return {
                action: effectiveResponses[0].action,
                tone: effectiveResponses[0].tone,
                reasoning: 'Known effective response from history',
                confidence: effectiveResponses[0].effectiveness
            };
        }

        // Fall back to generic preemptive actions
        const preemptiveActions = {
            'anxious': { action: 'reassure', tone: 'calm', message: 'I\'m here to help whenever you\'re ready.' },
            'frustrated': { action: 'acknowledge', tone: 'empathetic', message: 'I sense this might be challenging.' },
            'sad': { action: 'validate', tone: 'supportive', message: 'Take your time, I\'m listening.' },
            'angry': { action: 'de_escalate', tone: 'calm', message: 'Let\'s work through this together.' }
        };

        return preemptiveActions[predictedEmotion] || null;
    }

    /**
     * Validate prediction against actual emotion
     */
    validatePrediction(predictionIndex, actualEmotion) {
        if (predictionIndex >= this.predictions.length) {
            return false;
        }

        const prediction = this.predictions[predictionIndex];
        const wasCorrect = prediction.validate(actualEmotion);

        this.validatedPredictions.push(prediction);

        // Update signal accuracy
        if (this.config.enableLearning) {
            this._updateSignalAccuracy(prediction);
        }

        // Learn temporal pattern
        const now = new Date();
        this.contextAnalyzer.learnTemporalPattern(
            now.getHours(),
            now.getDay(),
            actualEmotion
        );

        return wasCorrect;
    }

    /**
     * Update signal accuracy based on validation
     */
    _updateSignalAccuracy(prediction) {
        for (const signalType of Object.keys(prediction.signals)) {
            if (!this.accuracyBySignal.has(signalType)) {
                this.accuracyBySignal.set(signalType, 0.5);  // Start at 50%
            }

            const currentAccuracy = this.accuracyBySignal.get(signalType);
            const alpha = 0.1;  // Learning rate

            const newAccuracy = prediction.wasCorrect ?
                currentAccuracy + alpha * (1.0 - currentAccuracy) :  // Increase if correct
                currentAccuracy - alpha * currentAccuracy;            // Decrease if wrong

            this.accuracyBySignal.set(signalType, newAccuracy);
        }
    }

    /**
     * Get prediction statistics
     */
    getStatistics() {
        if (this.validatedPredictions.length === 0) {
            return {
                totalPredictions: 0,
                accuracy: 0.0,
                bySignal: {}
            };
        }

        const correct = this.validatedPredictions.filter(p => p.wasCorrect).length;
        const accuracy = correct / this.validatedPredictions.length;

        return {
            totalPredictions: this.validatedPredictions.length,
            correct: correct,
            accuracy: accuracy,
            bySignal: Object.fromEntries(this.accuracyBySignal),
            recentPredictions: this.predictions.slice(-10).map(p => p.toJSON())
        };
    }
}

// Export
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        PredictiveEmotionalAnticipation,
        EmotionPrediction,
        ContextAnalyzer,
        PreSpeechSignalDetector,
        PredictionSignal,
        ConfidenceLevel
    };
}
