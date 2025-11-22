/**
 * Complete Emotional Intelligence Pipeline
 * End-to-end integration: Signal Detection → Fusion → Meta-Interpretation → Action Planning
 * Date: November 2025
 *
 * Architecture:
 * 1. Signal Detection (facial + vocal + text)
 * 2. Emotion Fusion (6 strategies)
 * 3. Meta-Interpretation (5 LLM modes)
 * 4. Next Steps Planning (action selection)
 * 5. Response Generation (templated outputs)
 */

import { FacialEmotionAnalyzer } from './facial_emotion_analyzer.js';
import { VocalProsodyAnalyzer } from './vocal_prosody_analyzer.js';
import { EmotionDetector } from './emotion_detector.js';
import { EmotionFusionSystem, FusionStrategy } from './emotion_fusion.js';
import { MetaEmotionInterpreter, MetaMode, LLMProvider } from './meta_emotion_interpreter.js';
import { NextStepsPipeline, PlanningStrategy, ConversationGoal } from './next_steps_pipeline.js';
import { ActionTemplates } from './action_templates.js';

/**
 * Pipeline Configuration
 */
class PipelineConfig {
    constructor(config = {}) {
        // Signal detection
        this.facialBackend = config.facialBackend || 'face-api';
        this.enableFacial = config.enableFacial !== false;
        this.enableVocal = config.enableVocal !== false;
        this.enableText = config.enableText !== false;

        // Fusion
        this.fusionStrategy = config.fusionStrategy || FusionStrategy.BAYESIAN;
        this.modalityWeights = config.modalityWeights || {
            facial: 0.4,
            vocal: 0.35,
            text: 0.25
        };

        // Meta-interpretation
        this.enableMetaInterpretation = config.enableMetaInterpretation !== false;
        this.metaMode = config.metaMode || MetaMode.FULL_META;

        // LLM
        this.llmProvider = config.llmProvider || LLMProvider.ANTHROPIC;
        this.llmApiKey = config.llmApiKey || null;
        this.llmModel = config.llmModel || 'claude-3-5-sonnet-20241022';

        // Planning
        this.enableActionPlanning = config.enableActionPlanning !== false;
        this.planningStrategy = config.planningStrategy || PlanningStrategy.ADAPTIVE;
        this.conversationGoal = config.conversationGoal || ConversationGoal.SUPPORT;

        // Response generation
        this.useTemplates = config.useTemplates !== false;
    }

    static minimal() {
        return new PipelineConfig({
            enableMetaInterpretation: false,
            enableActionPlanning: false,
            fusionStrategy: FusionStrategy.AVERAGE
        });
    }

    static standard() {
        return new PipelineConfig({
            metaMode: MetaMode.NUANCE_INTERPRETATION,
            planningStrategy: PlanningStrategy.IMMEDIATE
        });
    }

    static advanced() {
        return new PipelineConfig({
            metaMode: MetaMode.FULL_META,
            planningStrategy: PlanningStrategy.ADAPTIVE
        });
    }
}

/**
 * Pipeline Result
 */
class PipelineResult {
    constructor(data) {
        // Signal-level
        this.facialResult = data.facialResult || null;
        this.vocalResult = data.vocalResult || null;
        this.textResult = data.textResult || null;

        // Fusion
        this.fusedEmotion = data.fusedEmotion || null;

        // Meta-interpretation
        this.metaInterpretation = data.metaInterpretation || null;

        // Action plan
        this.actionPlan = data.actionPlan || null;

        // Generated response
        this.response = data.response || '';
        this.tone = data.tone || 'empathetic';

        // Metadata
        this.confidence = data.confidence || 0.0;
        this.processingTimeMs = data.processingTimeMs || 0;
        this.timestamp = new Date();
    }

    /**
     * Get summary for logging/debugging
     */
    getSummary() {
        return {
            emotion: this.metaInterpretation?.interpretedEmotion || this.fusedEmotion?.emotion,
            confidence: this.confidence,
            action: this.actionPlan?.immediateAction?.action,
            response: this.response,
            processingTime: `${this.processingTimeMs}ms`
        };
    }
}

/**
 * Complete Emotional Intelligence Pipeline
 */
class EmotionalIntelligencePipeline {
    constructor(config = {}) {
        this.config = config instanceof PipelineConfig ? config : new PipelineConfig(config);

        // Initialize components
        this._initializeComponents();

        // Conversation state
        this.conversationTurn = 0;
        this.conversationHistory = [];

        // Metrics
        this.metrics = {
            totalProcessed: 0,
            avgProcessingTimeMs: 0,
            avgConfidence: 0,
            modalityUsage: {
                facial: 0,
                vocal: 0,
                text: 0
            }
        };

        console.log('[EmotionalIntelligencePipeline] Initialized with config:', this.config);
    }

    _initializeComponents() {
        // Signal detectors
        if (this.config.enableFacial) {
            this.facialAnalyzer = new FacialEmotionAnalyzer({
                backend: this.config.facialBackend
            });
        }

        if (this.config.enableVocal) {
            this.vocalAnalyzer = new VocalProsodyAnalyzer();
        }

        if (this.config.enableText) {
            this.textAnalyzer = new EmotionDetector();
        }

        // Fusion system
        this.fusionSystem = new EmotionFusionSystem({
            fusionStrategy: this.config.fusionStrategy,
            modalityWeights: this.config.modalityWeights
        });

        // Meta-interpreter
        if (this.config.enableMetaInterpretation) {
            this.metaInterpreter = new MetaEmotionInterpreter({
                mode: this.config.metaMode,
                llmProvider: this.config.llmProvider,
                llmApiKey: this.config.llmApiKey,
                llmModel: this.config.llmModel
            });
        }

        // Action planner
        if (this.config.enableActionPlanning) {
            this.actionPlanner = new NextStepsPipeline({
                planningStrategy: this.config.planningStrategy,
                defaultGoal: this.config.conversationGoal,
                llmProvider: this.config.llmProvider,
                llmApiKey: this.config.llmApiKey,
                llmModel: this.config.llmModel
            });
        }

        // Response templates
        if (this.config.useTemplates) {
            this.templates = new ActionTemplates();
        }
    }

    /**
     * Process input through complete pipeline
     */
    async process(input) {
        const startTime = Date.now();
        this.conversationTurn++;
        this.metrics.totalProcessed++;

        try {
            // 1. Signal Detection
            const signals = await this._detectSignals(input);

            // 2. Emotion Fusion
            const fusedEmotion = this.fusionSystem.fuse(signals);

            // 3. Meta-Interpretation (optional)
            let metaInterpretation = null;
            if (this.config.enableMetaInterpretation && this.metaInterpreter) {
                metaInterpretation = await this.metaInterpreter.interpret(
                    fusedEmotion,
                    signals,
                    {
                        text: input.text,
                        conversationTurn: this.conversationTurn,
                        ...input.context
                    }
                );
            }

            // 4. Action Planning (optional)
            let actionPlan = null;
            if (this.config.enableActionPlanning && this.actionPlanner) {
                const emotionResult = metaInterpretation || fusedEmotion;
                actionPlan = await this.actionPlanner.plan(
                    emotionResult,
                    input.text,
                    {
                        conversationTurn: this.conversationTurn,
                        ...input.context
                    }
                );
            }

            // 5. Response Generation
            const response = this._generateResponse(
                fusedEmotion,
                metaInterpretation,
                actionPlan,
                input
            );

            // Calculate metrics
            const processingTime = Date.now() - startTime;
            const confidence = metaInterpretation?.confidence ||
                             fusedEmotion?.confidence ||
                             0.0;

            // Update metrics
            this._updateMetrics(processingTime, confidence, signals);

            // Create result
            const result = new PipelineResult({
                facialResult: signals.find(s => s.modality === 'facial')?.data || null,
                vocalResult: signals.find(s => s.modality === 'vocal')?.data || null,
                textResult: signals.find(s => s.modality === 'text')?.data || null,
                fusedEmotion: fusedEmotion,
                metaInterpretation: metaInterpretation,
                actionPlan: actionPlan,
                response: response.text,
                tone: response.tone,
                confidence: confidence,
                processingTimeMs: processingTime
            });

            // Update conversation history
            this._updateConversationHistory(result, input);

            return result;

        } catch (error) {
            console.error('[EmotionalIntelligencePipeline] Error during processing:', error);
            throw error;
        }
    }

    /**
     * Detect signals from all modalities
     */
    async _detectSignals(input) {
        const signals = [];

        // Facial analysis
        if (this.config.enableFacial && input.videoFrame && this.facialAnalyzer) {
            try {
                const facialResult = await this.facialAnalyzer.analyze(input.videoFrame);
                signals.push({
                    modality: 'facial',
                    data: facialResult
                });
                this.metrics.modalityUsage.facial++;
            } catch (error) {
                console.warn('[Pipeline] Facial analysis failed:', error.message);
            }
        }

        // Vocal analysis
        if (this.config.enableVocal && input.audioBuffer && this.vocalAnalyzer) {
            try {
                const vocalResult = await this.vocalAnalyzer.analyze(input.audioBuffer);
                signals.push({
                    modality: 'vocal',
                    data: vocalResult
                });
                this.metrics.modalityUsage.vocal++;
            } catch (error) {
                console.warn('[Pipeline] Vocal analysis failed:', error.message);
            }
        }

        // Text analysis
        if (this.config.enableText && input.text && this.textAnalyzer) {
            try {
                const textResult = await this.textAnalyzer.detectEmotion(input.text);
                signals.push({
                    modality: 'text',
                    data: textResult
                });
                this.metrics.modalityUsage.text++;
            } catch (error) {
                console.warn('[Pipeline] Text analysis failed:', error.message);
            }
        }

        if (signals.length === 0) {
            throw new Error('No signals detected - at least one modality must be available');
        }

        return signals;
    }

    /**
     * Generate response
     */
    _generateResponse(fusedEmotion, metaInterpretation, actionPlan, input) {
        let text = '';
        let tone = 'empathetic';

        if (actionPlan && actionPlan.immediateAction) {
            // Use action plan response
            text = actionPlan.immediateAction.response;
            tone = actionPlan.immediateAction.tone;

            // Enhance with template if available
            if (this.config.useTemplates && this.templates) {
                const emotion = metaInterpretation?.interpretedEmotion || fusedEmotion.emotion;
                const action = actionPlan.immediateAction.action;

                const template = this.templates.get(action, emotion, tone);
                if (template) {
                    // Extract context from input
                    const context = {
                        topic: input.context?.topic || 'this situation',
                        situation: input.context?.situation || 'what you\'re experiencing',
                        ...input.context
                    };

                    text = template.render(context);
                }
            }
        } else if (metaInterpretation) {
            // Use meta-interpretation strategy
            text = metaInterpretation.interactionStrategy ||
                   `I understand you're feeling ${metaInterpretation.interpretedEmotion}.`;
        } else {
            // Fallback to fusion result
            text = `I sense you're feeling ${fusedEmotion.emotion}.`;
        }

        return { text, tone };
    }

    /**
     * Update metrics
     */
    _updateMetrics(processingTime, confidence, signals) {
        // Processing time
        this.metrics.avgProcessingTimeMs =
            (this.metrics.avgProcessingTimeMs * (this.metrics.totalProcessed - 1) + processingTime) /
            this.metrics.totalProcessed;

        // Confidence
        this.metrics.avgConfidence =
            (this.metrics.avgConfidence * (this.metrics.totalProcessed - 1) + confidence) /
            this.metrics.totalProcessed;
    }

    /**
     * Update conversation history
     */
    _updateConversationHistory(result, input) {
        this.conversationHistory.push({
            turn: this.conversationTurn,
            userInput: input.text,
            emotion: result.metaInterpretation?.interpretedEmotion || result.fusedEmotion?.emotion,
            response: result.response,
            confidence: result.confidence,
            timestamp: new Date()
        });

        // Keep last 10 turns
        if (this.conversationHistory.length > 10) {
            this.conversationHistory.shift();
        }
    }

    /**
     * Set conversation goal
     */
    setGoal(goal) {
        this.config.conversationGoal = goal;
        if (this.actionPlanner) {
            this.actionPlanner.setGoal(goal);
        }
    }

    /**
     * Get metrics
     */
    getMetrics() {
        return {
            pipeline: this.metrics,
            fusion: this.fusionSystem?.getMetrics() || {},
            metaInterpreter: this.metaInterpreter?.getMetrics() || {},
            actionPlanner: this.actionPlanner?.getMetrics() || {}
        };
    }

    /**
     * Get conversation history
     */
    getConversationHistory() {
        return this.conversationHistory;
    }
}

// Export
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        EmotionalIntelligencePipeline,
        PipelineConfig,
        PipelineResult
    };
}
