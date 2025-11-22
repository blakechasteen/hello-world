/**
 * Meta-Emotion Interpreter - LLM-Powered Contextual Emotion Understanding
 * Uses metaprompting to enhance emotion analysis with reasoning and context
 * Date: November 2025
 *
 * Features:
 * - Dynamic fusion strategy selection via LLM
 * - Contextual emotion interpretation beyond categorical labels
 * - Complex conflict resolution using reasoning
 * - Conversation-aware emotion understanding
 * - Adaptive calibration from feedback
 * - Nuanced emotional state descriptions
 */

/**
 * Meta-Interpretation Modes
 */
const MetaMode = {
    STRATEGY_SELECTION: 'strategy_selection',  // LLM chooses best fusion strategy
    NUANCE_INTERPRETATION: 'nuance_interpretation',  // LLM describes emotional nuance
    CONFLICT_RESOLUTION: 'conflict_resolution',  // LLM resolves contradictory signals
    CONTEXT_ADAPTATION: 'context_adaptation',  // LLM adapts to conversation context
    FULL_META: 'full_meta'  // All meta-capabilities enabled
};

/**
 * LLM Provider Types
 */
const LLMProvider = {
    OPENAI: 'openai',
    ANTHROPIC: 'anthropic',
    OLLAMA: 'ollama',  // Local models
    MOCK: 'mock'  // For testing without API
};

/**
 * Prompt Template Library
 */
class PromptTemplates {
    static STRATEGY_SELECTION = `You are an emotion analysis expert. Given the following emotion signals from different modalities, select the best fusion strategy.

**Emotion Signals:**
{{signals}}

**Available Fusion Strategies:**
1. **average**: Simple mean of all emotions (good for consistent signals)
2. **weighted**: Weight by confidence × modality weight (good for varying confidence)
3. **max_confidence**: Trust most confident signal (good when one modality is clearly reliable)
4. **majority_vote**: Democratic consensus (good for discrete choices)
5. **priority**: Hierarchical trust (facial > vocal > text) (good for known modality reliability)
6. **bayesian**: Probabilistic integration (good for uncertainty quantification)

**Context:**
{{context}}

**Conflict Level:** {{conflict_level}}

Respond with JSON only:
{
  "strategy": "strategy_name",
  "reasoning": "brief explanation",
  "confidence": 0.0-1.0
}`;

    static NUANCE_INTERPRETATION = `You are an emotion analysis expert. Given raw emotion signals, provide nuanced interpretation.

**Facial Emotion:** {{facial_emotion}} (confidence: {{facial_confidence}})
{{facial_details}}

**Vocal Prosody:** {{vocal_emotion}} (confidence: {{vocal_confidence}})
{{vocal_details}}

**Text Sentiment:** {{text_emotion}} (confidence: {{text_confidence}})
{{text_details}}

**VAD Coordinates:**
- Valence: {{valence}} (negative ↔ positive)
- Arousal: {{arousal}} (calm ↔ excited)
- Dominance: {{dominance}} (submissive ↔ dominant)

**Context:** {{context}}

Provide nuanced emotional interpretation as JSON:
{
  "primary_emotion": "emotion name",
  "secondary_emotion": "emotion name or null",
  "intensity": 0.0-1.0,
  "genuineness": 0.0-1.0,
  "nuance_description": "2-3 sentence description of emotional state",
  "social_implications": "how this emotion affects interaction",
  "confidence": 0.0-1.0
}`;

    static CONFLICT_RESOLUTION = `You are an emotion analysis expert. The following modalities show conflicting emotion signals:

**Conflicts Detected:**
{{conflicts}}

**Facial:** {{facial_emotion}} ({{facial_confidence}})
**Vocal:** {{vocal_emotion}} ({{vocal_confidence}})
**Text:** {{text_emotion}} ({{text_confidence}})

**Possible Explanations:**
1. Sarcasm (positive text, negative voice)
2. Politeness masking (positive text, negative face)
3. Suppressed emotion (neutral face, emotional voice)
4. Cultural display rules (emotion felt ≠ emotion shown)
5. Technical error (one modality unreliable)

**Context:** {{context}}

Resolve conflict as JSON:
{
  "likely_true_emotion": "emotion name",
  "explanation": "why this is likely correct",
  "alternative_interpretations": ["other possibilities"],
  "recommended_action": "how system should respond",
  "confidence": 0.0-1.0
}`;

    static CONTEXT_ADAPTATION = `You are an emotion analysis expert. Adapt emotion interpretation to conversation context.

**Current Emotion:** {{current_emotion}} ({{current_confidence}})
**Current VAD:** valence={{valence}}, arousal={{arousal}}, dominance={{dominance}}

**Conversation History:**
{{conversation_history}}

**User Profile:**
{{user_profile}}

**Temporal Pattern:**
{{temporal_pattern}}

Consider:
1. Is this emotion consistent with conversation flow?
2. Does user typically express emotions this way?
3. Is emotional trajectory natural or anomalous?
4. What triggered this emotional state?
5. How should system adapt interaction?

Respond as JSON:
{
  "adjusted_emotion": "emotion name (may differ from raw signal)",
  "adjustment_reasoning": "why adjustment was made",
  "conversation_coherence": 0.0-1.0,
  "interaction_recommendations": ["specific suggestions"],
  "confidence": 0.0-1.0
}`;

    static FULL_META_ANALYSIS = `You are an emotion analysis expert. Perform comprehensive meta-analysis of emotional state.

**Raw Signals:**
- Facial: {{facial_emotion}} ({{facial_confidence}})
- Vocal: {{vocal_emotion}} ({{vocal_confidence}})
- Text: {{text_emotion}} ({{text_confidence}})

**Fusion Result:** {{fused_emotion}} ({{fused_confidence}})
**Fusion Strategy Used:** {{fusion_strategy}}

**VAD:** v={{valence}}, a={{arousal}}, d={{dominance}}

**Conflicts:** {{conflicts}}
**Sarcasm Score:** {{sarcasm}}
**Genuineness:** {{genuineness}}

**Context:** {{context}}
**History:** {{history}}

Provide comprehensive analysis as JSON:
{
  "interpreted_emotion": "final emotion after meta-analysis",
  "confidence": 0.0-1.0,
  "nuance_description": "detailed emotional state",
  "key_insights": ["important observations"],
  "uncertainty_factors": ["sources of uncertainty"],
  "interaction_strategy": "recommended system response approach",
  "calibration_suggestions": ["how to improve future analysis"]
}`;
}

/**
 * LLM Client Abstraction
 */
class LLMClient {
    constructor(config = {}) {
        this.config = {
            provider: config.provider || LLMProvider.MOCK,
            apiKey: config.apiKey || null,
            model: config.model || 'gpt-4',
            temperature: config.temperature || 0.7,
            maxTokens: config.maxTokens || 500,
            ...config
        };
    }

    async complete(prompt, options = {}) {
        switch (this.config.provider) {
            case LLMProvider.OPENAI:
                return await this._openaiComplete(prompt, options);
            case LLMProvider.ANTHROPIC:
                return await this._anthropicComplete(prompt, options);
            case LLMProvider.OLLAMA:
                return await this._ollamaComplete(prompt, options);
            case LLMProvider.MOCK:
                return await this._mockComplete(prompt, options);
            default:
                throw new Error(`Unknown LLM provider: ${this.config.provider}`);
        }
    }

    async _openaiComplete(prompt, options) {
        // STUB: In production, call actual OpenAI API
        /*
        const response = await fetch('https://api.openai.com/v1/chat/completions', {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${this.config.apiKey}`,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                model: this.config.model,
                messages: [{ role: 'user', content: prompt }],
                temperature: options.temperature || this.config.temperature,
                max_tokens: options.maxTokens || this.config.maxTokens
            })
        });
        const data = await response.json();
        return data.choices[0].message.content;
        */

        console.log('[LLMClient] OpenAI stub - replace with actual API call');
        return this._mockComplete(prompt, options);
    }

    async _anthropicComplete(prompt, options) {
        // STUB: In production, call actual Anthropic API
        console.log('[LLMClient] Anthropic stub - replace with actual API call');
        return this._mockComplete(prompt, options);
    }

    async _ollamaComplete(prompt, options) {
        // STUB: In production, call local Ollama instance
        /*
        const response = await fetch('http://localhost:11434/api/generate', {
            method: 'POST',
            body: JSON.stringify({
                model: this.config.model,
                prompt: prompt,
                stream: false
            })
        });
        const data = await response.json();
        return data.response;
        */

        console.log('[LLMClient] Ollama stub - replace with actual API call');
        return this._mockComplete(prompt, options);
    }

    async _mockComplete(prompt, options) {
        // Mock responses for testing
        await new Promise(resolve => setTimeout(resolve, 100)); // Simulate latency

        if (prompt.includes('strategy_selection')) {
            return JSON.stringify({
                strategy: 'weighted',
                reasoning: 'Varying confidence levels across modalities suggest weighted fusion',
                confidence: 0.85
            });
        } else if (prompt.includes('nuance_interpretation')) {
            return JSON.stringify({
                primary_emotion: 'anxious_hopeful',
                secondary_emotion: 'determined',
                intensity: 0.7,
                genuineness: 0.8,
                nuance_description: 'User shows genuine anxiety about current situation but maintains hopeful outlook and determination to succeed.',
                social_implications: 'User may need reassurance and concrete next steps to reduce anxiety while maintaining motivation.',
                confidence: 0.82
            });
        } else if (prompt.includes('conflict_resolution')) {
            return JSON.stringify({
                likely_true_emotion: 'frustrated',
                explanation: 'Polite text masking frustrated vocal tone suggests professional context where emotions must be controlled',
                alternative_interpretations: ['genuinely neutral', 'sarcastic politeness'],
                recommended_action: 'Acknowledge underlying frustration tactfully while maintaining professional tone',
                confidence: 0.78
            });
        } else if (prompt.includes('context_adaptation')) {
            return JSON.stringify({
                adjusted_emotion: 'relieved_cautious',
                adjustment_reasoning: 'Previous conversation showed concern; current positive emotion likely relief but tempered by lingering caution',
                conversation_coherence: 0.88,
                interaction_recommendations: [
                    'Acknowledge relief without dismissing concerns',
                    'Offer continued support',
                    'Suggest concrete steps forward'
                ],
                confidence: 0.84
            });
        } else {
            return JSON.stringify({
                interpreted_emotion: 'complex_mixed',
                confidence: 0.75,
                nuance_description: 'Multi-faceted emotional state',
                key_insights: ['Multiple emotions present', 'Context-dependent interpretation'],
                uncertainty_factors: ['Limited context', 'Conflicting signals'],
                interaction_strategy: 'Empathetic, adaptive response',
                calibration_suggestions: ['Gather more context', 'Track temporal patterns']
            });
        }
    }
}

/**
 * Meta-Emotion Interpretation Result
 */
class MetaEmotionResult {
    constructor(data) {
        this.interpretedEmotion = data.interpretedEmotion;
        this.confidence = data.confidence || 0.0;
        this.nuanceDescription = data.nuanceDescription || '';
        this.secondaryEmotion = data.secondaryEmotion || null;
        this.intensity = data.intensity || 0.0;
        this.genuineness = data.genuineness || 0.0;

        // Meta-insights
        this.keyInsights = data.keyInsights || [];
        this.uncertaintyFactors = data.uncertaintyFactors || [];
        this.interactionStrategy = data.interactionStrategy || '';
        this.interactionRecommendations = data.interactionRecommendations || [];

        // Strategy selection
        this.selectedStrategy = data.selectedStrategy || null;
        this.strategyReasoning = data.strategyReasoning || '';

        // Conflict resolution
        this.conflictResolution = data.conflictResolution || null;

        // Context adaptation
        this.conversationCoherence = data.conversationCoherence || 0.0;
        this.adjustmentReasoning = data.adjustmentReasoning || '';

        // Calibration
        this.calibrationSuggestions = data.calibrationSuggestions || [];

        // Raw signals (preserved)
        this.rawFusion = data.rawFusion || null;

        this.timestamp = new Date();
    }
}

/**
 * Meta-Emotion Interpreter
 * Uses LLM prompting for contextual emotion understanding
 */
class MetaEmotionInterpreter {
    constructor(config = {}) {
        this.config = {
            enabled: config.enabled !== false,
            mode: config.mode || MetaMode.FULL_META,

            // LLM configuration
            llmProvider: config.llmProvider || LLMProvider.MOCK,
            llmApiKey: config.llmApiKey || null,
            llmModel: config.llmModel || 'gpt-4',
            llmTemperature: config.llmTemperature || 0.7,

            // Context window
            conversationHistorySize: config.conversationHistorySize || 10,

            // Thresholds
            conflictThreshold: config.conflictThreshold || 0.3,  // Emotion difference threshold
            lowConfidenceThreshold: config.lowConfidenceThreshold || 0.6,

            // Fallback behavior
            fallbackToRaw: config.fallbackToRaw !== false,

            ...config
        };

        // Initialize LLM client
        this.llm = new LLMClient({
            provider: this.config.llmProvider,
            apiKey: this.config.llmApiKey,
            model: this.config.llmModel,
            temperature: this.config.llmTemperature
        });

        // Context tracking
        this.conversationHistory = [];
        this.userProfile = {
            emotionalBaseline: null,
            expressionStyle: null,
            culturalContext: null
        };

        // Metrics
        this.metrics = {
            totalInterpretations: 0,
            strategySelections: 0,
            conflictResolutions: 0,
            contextAdaptations: 0,
            llmCallLatencyMs: [],
            fallbackCount: 0
        };

        console.log('[MetaEmotionInterpreter] Initialized with mode:', this.config.mode);
    }

    /**
     * Main interpretation method
     */
    async interpret(fusionResult, signals, context = {}) {
        if (!this.config.enabled) {
            return this._createPassthroughResult(fusionResult);
        }

        const startTime = Date.now();

        try {
            let result;

            switch (this.config.mode) {
                case MetaMode.STRATEGY_SELECTION:
                    result = await this._selectStrategy(signals, context);
                    break;
                case MetaMode.NUANCE_INTERPRETATION:
                    result = await this._interpretNuance(fusionResult, signals, context);
                    break;
                case MetaMode.CONFLICT_RESOLUTION:
                    result = await this._resolveConflicts(fusionResult, signals, context);
                    break;
                case MetaMode.CONTEXT_ADAPTATION:
                    result = await this._adaptToContext(fusionResult, signals, context);
                    break;
                case MetaMode.FULL_META:
                    result = await this._fullMetaAnalysis(fusionResult, signals, context);
                    break;
                default:
                    result = this._createPassthroughResult(fusionResult);
            }

            // Track metrics
            const latency = Date.now() - startTime;
            this.metrics.totalInterpretations++;
            this.metrics.llmCallLatencyMs.push(latency);

            // Update conversation history
            this._updateConversationHistory(result, context);

            return result;

        } catch (error) {
            console.error('[MetaEmotionInterpreter] Error during interpretation:', error);

            if (this.config.fallbackToRaw) {
                this.metrics.fallbackCount++;
                return this._createPassthroughResult(fusionResult);
            } else {
                throw error;
            }
        }
    }

    /**
     * Strategy Selection Mode
     */
    async _selectStrategy(signals, context) {
        const prompt = this._fillTemplate(PromptTemplates.STRATEGY_SELECTION, {
            signals: this._formatSignals(signals),
            context: this._formatContext(context),
            conflict_level: this._calculateConflictLevel(signals)
        });

        const response = await this.llm.complete(prompt);
        const parsed = JSON.parse(response);

        this.metrics.strategySelections++;

        return new MetaEmotionResult({
            interpretedEmotion: null,  // Strategy selection doesn't interpret
            confidence: parsed.confidence,
            selectedStrategy: parsed.strategy,
            strategyReasoning: parsed.reasoning,
            rawFusion: null
        });
    }

    /**
     * Nuance Interpretation Mode
     */
    async _interpretNuance(fusionResult, signals, context) {
        const facialSignal = signals.find(s => s.modality === 'facial');
        const vocalSignal = signals.find(s => s.modality === 'vocal');
        const textSignal = signals.find(s => s.modality === 'text');

        const prompt = this._fillTemplate(PromptTemplates.NUANCE_INTERPRETATION, {
            facial_emotion: facialSignal?.data.emotion || 'none',
            facial_confidence: facialSignal?.data.confidence || 0,
            facial_details: this._formatFacialDetails(facialSignal),
            vocal_emotion: vocalSignal?.data.emotion || 'none',
            vocal_confidence: vocalSignal?.data.confidence || 0,
            vocal_details: this._formatVocalDetails(vocalSignal),
            text_emotion: textSignal?.data.emotion || 'none',
            text_confidence: textSignal?.data.confidence || 0,
            text_details: this._formatTextDetails(textSignal),
            valence: fusionResult.valence,
            arousal: fusionResult.arousal,
            dominance: fusionResult.dominance,
            context: this._formatContext(context)
        });

        const response = await this.llm.complete(prompt);
        const parsed = JSON.parse(response);

        return new MetaEmotionResult({
            interpretedEmotion: parsed.primary_emotion,
            secondaryEmotion: parsed.secondary_emotion,
            confidence: parsed.confidence,
            intensity: parsed.intensity,
            genuineness: parsed.genuineness,
            nuanceDescription: parsed.nuance_description,
            interactionStrategy: parsed.social_implications,
            rawFusion: fusionResult
        });
    }

    /**
     * Conflict Resolution Mode
     */
    async _resolveConflicts(fusionResult, signals, context) {
        const conflicts = this._detectConflicts(signals);

        if (conflicts.length === 0) {
            // No conflicts - passthrough
            return this._createPassthroughResult(fusionResult);
        }

        const facialSignal = signals.find(s => s.modality === 'facial');
        const vocalSignal = signals.find(s => s.modality === 'vocal');
        const textSignal = signals.find(s => s.modality === 'text');

        const prompt = this._fillTemplate(PromptTemplates.CONFLICT_RESOLUTION, {
            conflicts: this._formatConflicts(conflicts),
            facial_emotion: facialSignal?.data.emotion || 'none',
            facial_confidence: facialSignal?.data.confidence || 0,
            vocal_emotion: vocalSignal?.data.emotion || 'none',
            vocal_confidence: vocalSignal?.data.confidence || 0,
            text_emotion: textSignal?.data.emotion || 'none',
            text_confidence: textSignal?.data.confidence || 0,
            context: this._formatContext(context)
        });

        const response = await this.llm.complete(prompt);
        const parsed = JSON.parse(response);

        this.metrics.conflictResolutions++;

        return new MetaEmotionResult({
            interpretedEmotion: parsed.likely_true_emotion,
            confidence: parsed.confidence,
            nuanceDescription: parsed.explanation,
            conflictResolution: {
                conflicts: conflicts,
                resolution: parsed.likely_true_emotion,
                alternativeInterpretations: parsed.alternative_interpretations
            },
            interactionStrategy: parsed.recommended_action,
            rawFusion: fusionResult
        });
    }

    /**
     * Context Adaptation Mode
     */
    async _adaptToContext(fusionResult, signals, context) {
        const prompt = this._fillTemplate(PromptTemplates.CONTEXT_ADAPTATION, {
            current_emotion: fusionResult.emotion,
            current_confidence: fusionResult.confidence,
            valence: fusionResult.valence,
            arousal: fusionResult.arousal,
            dominance: fusionResult.dominance,
            conversation_history: this._formatConversationHistory(),
            user_profile: this._formatUserProfile(),
            temporal_pattern: this._formatTemporalPattern()
        });

        const response = await this.llm.complete(prompt);
        const parsed = JSON.parse(response);

        this.metrics.contextAdaptations++;

        return new MetaEmotionResult({
            interpretedEmotion: parsed.adjusted_emotion,
            confidence: parsed.confidence,
            adjustmentReasoning: parsed.adjustment_reasoning,
            conversationCoherence: parsed.conversation_coherence,
            interactionRecommendations: parsed.interaction_recommendations,
            rawFusion: fusionResult
        });
    }

    /**
     * Full Meta-Analysis Mode
     */
    async _fullMetaAnalysis(fusionResult, signals, context) {
        const facialSignal = signals.find(s => s.modality === 'facial');
        const vocalSignal = signals.find(s => s.modality === 'vocal');
        const textSignal = signals.find(s => s.modality === 'text');

        const prompt = this._fillTemplate(PromptTemplates.FULL_META_ANALYSIS, {
            facial_emotion: facialSignal?.data.emotion || 'none',
            facial_confidence: facialSignal?.data.confidence || 0,
            vocal_emotion: vocalSignal?.data.emotion || 'none',
            vocal_confidence: vocalSignal?.data.confidence || 0,
            text_emotion: textSignal?.data.emotion || 'none',
            text_confidence: textSignal?.data.confidence || 0,
            fused_emotion: fusionResult.emotion,
            fused_confidence: fusionResult.confidence,
            fusion_strategy: fusionResult.fusionStrategy || 'unknown',
            valence: fusionResult.valence,
            arousal: fusionResult.arousal,
            dominance: fusionResult.dominance,
            conflicts: this._formatConflicts(this._detectConflicts(signals)),
            sarcasm: fusionResult.sarcasm || 0,
            genuineness: fusionResult.genuineness || 0,
            context: this._formatContext(context),
            history: this._formatConversationHistory()
        });

        const response = await this.llm.complete(prompt);
        const parsed = JSON.parse(response);

        return new MetaEmotionResult({
            interpretedEmotion: parsed.interpreted_emotion,
            confidence: parsed.confidence,
            nuanceDescription: parsed.nuance_description,
            keyInsights: parsed.key_insights,
            uncertaintyFactors: parsed.uncertainty_factors,
            interactionStrategy: parsed.interaction_strategy,
            calibrationSuggestions: parsed.calibration_suggestions,
            rawFusion: fusionResult
        });
    }

    // Helper methods

    _fillTemplate(template, vars) {
        let filled = template;
        for (const [key, value] of Object.entries(vars)) {
            filled = filled.replace(new RegExp(`{{${key}}}`, 'g'), value);
        }
        return filled;
    }

    _formatSignals(signals) {
        return signals.map(s =>
            `- ${s.modality}: ${s.data.emotion} (confidence: ${s.data.confidence.toFixed(2)})`
        ).join('\n');
    }

    _formatContext(context) {
        return JSON.stringify(context, null, 2);
    }

    _formatFacialDetails(signal) {
        if (!signal) return 'none';
        const data = signal.data;
        return `Head pose: yaw=${data.headPose?.yaw || 0}, Eye contact: ${data.eyeContact}, Attention: ${data.attention}`;
    }

    _formatVocalDetails(signal) {
        if (!signal) return 'none';
        const data = signal.data;
        return `Speaking style: ${data.speakingStyle?.style || 'unknown'}, Voice quality: ${data.voiceQuality?.type || 'unknown'}`;
    }

    _formatTextDetails(signal) {
        if (!signal) return 'none';
        return `Sentiment: ${signal.data.sentiment || 'unknown'}, Intensity: ${signal.data.intensity || 0}`;
    }

    _formatConflicts(conflicts) {
        if (conflicts.length === 0) return 'none';
        return conflicts.map(c => `${c.modality1} (${c.emotion1}) vs ${c.modality2} (${c.emotion2})`).join(', ');
    }

    _formatConversationHistory() {
        return this.conversationHistory.slice(-5).map((h, i) =>
            `${i + 1}. ${h.emotion} (${h.confidence.toFixed(2)}) - "${h.context?.text || ''}"`
        ).join('\n');
    }

    _formatUserProfile() {
        return JSON.stringify(this.userProfile, null, 2);
    }

    _formatTemporalPattern() {
        if (this.conversationHistory.length < 2) return 'insufficient data';

        const recent = this.conversationHistory.slice(-5);
        const emotions = recent.map(h => h.emotion);
        const avgConfidence = recent.reduce((sum, h) => sum + h.confidence, 0) / recent.length;

        return `Recent emotions: ${emotions.join(' → ')}, Avg confidence: ${avgConfidence.toFixed(2)}`;
    }

    _calculateConflictLevel(signals) {
        const conflicts = this._detectConflicts(signals);
        return (conflicts.length / Math.max(1, signals.length)).toFixed(2);
    }

    _detectConflicts(signals) {
        const conflicts = [];

        for (let i = 0; i < signals.length; i++) {
            for (let j = i + 1; j < signals.length; j++) {
                const emotion1 = signals[i].data.emotion;
                const emotion2 = signals[j].data.emotion;

                if (emotion1 !== emotion2) {
                    // Check if emotions are actually conflicting (not just different)
                    const valence1 = this._getEmotionValence(emotion1);
                    const valence2 = this._getEmotionValence(emotion2);

                    if (Math.abs(valence1 - valence2) > this.config.conflictThreshold) {
                        conflicts.push({
                            modality1: signals[i].modality,
                            emotion1: emotion1,
                            modality2: signals[j].modality,
                            emotion2: emotion2,
                            valenceDifference: Math.abs(valence1 - valence2)
                        });
                    }
                }
            }
        }

        return conflicts;
    }

    _getEmotionValence(emotion) {
        const valenceMap = {
            'happy': 0.8, 'joyful': 0.9, 'excited': 0.7,
            'sad': -0.7, 'depressed': -0.9, 'disappointed': -0.5,
            'angry': -0.6, 'frustrated': -0.5, 'annoyed': -0.4,
            'fearful': -0.6, 'anxious': -0.5, 'worried': -0.4,
            'disgusted': -0.7, 'contempt': -0.6,
            'surprised': 0.3, 'curious': 0.4,
            'neutral': 0.0, 'calm': 0.2
        };
        return valenceMap[emotion.toLowerCase()] || 0.0;
    }

    _createPassthroughResult(fusionResult) {
        return new MetaEmotionResult({
            interpretedEmotion: fusionResult.emotion,
            confidence: fusionResult.confidence,
            nuanceDescription: 'Passthrough (meta-interpretation disabled or fallback)',
            rawFusion: fusionResult
        });
    }

    _updateConversationHistory(result, context) {
        this.conversationHistory.push({
            emotion: result.interpretedEmotion,
            confidence: result.confidence,
            context: context,
            timestamp: new Date()
        });

        // Maintain window size
        if (this.conversationHistory.length > this.config.conversationHistorySize) {
            this.conversationHistory.shift();
        }
    }

    /**
     * Get metrics
     */
    getMetrics() {
        const avgLatency = this.metrics.llmCallLatencyMs.length > 0
            ? this.metrics.llmCallLatencyMs.reduce((a, b) => a + b, 0) / this.metrics.llmCallLatencyMs.length
            : 0;

        return {
            ...this.metrics,
            avgLLMLatencyMs: avgLatency,
            fallbackRate: this.metrics.totalInterpretations > 0
                ? this.metrics.fallbackCount / this.metrics.totalInterpretations
                : 0
        };
    }
}


// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        MetaEmotionInterpreter,
        MetaEmotionResult,
        MetaMode,
        LLMProvider,
        LLMClient,
        PromptTemplates
    };
}
