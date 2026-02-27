/**
 * Next Steps Pipeline - Metaprompting-Based Action Planning
 * Determines optimal system responses based on emotional analysis
 * Date: November 2025
 *
 * Features:
 * - LLM-powered action planning from emotional state
 * - Multi-step conversation strategies
 * - Context-aware response generation
 * - Adaptive goal-driven planning
 * - Emotional trajectory prediction
 * - Intervention detection (escalation/de-escalation)
 */

import { LLMClient, LLMProvider } from './meta_emotion_interpreter.js';

/**
 * Action Types
 */
const ActionType = {
    // Immediate responses
    ACKNOWLEDGE: 'acknowledge',              // Acknowledge emotion
    VALIDATE: 'validate',                    // Validate feelings
    REASSURE: 'reassure',                    // Provide reassurance
    EMPATHIZE: 'empathize',                  // Show empathy
    CLARIFY: 'clarify',                      // Request clarification

    // Information gathering
    ASK_OPEN: 'ask_open',                    // Open-ended question
    ASK_CLOSED: 'ask_closed',                // Yes/no question
    PROBE: 'probe',                          // Probe deeper

    // Problem solving
    SUGGEST: 'suggest',                      // Suggest action
    BRAINSTORM: 'brainstorm',                // Generate options
    PLAN: 'plan',                            // Create plan
    EXECUTE: 'execute',                      // Execute action

    // Emotional regulation
    DE_ESCALATE: 'de_escalate',              // Calm down
    REFRAME: 'reframe',                      // Reframe perspective
    ENCOURAGE: 'encourage',                  // Boost confidence

    // Escalation
    ESCALATE_HUMAN: 'escalate_human',        // Hand off to human
    ESCALATE_SPECIALIST: 'escalate_specialist', // Specialist needed
    EMERGENCY: 'emergency',                  // Emergency intervention

    // Continuation
    WAIT: 'wait',                            // Wait for user
    SUMMARIZE: 'summarize',                  // Summarize progress
    CLOSE: 'close'                           // End conversation
};

/**
 * Planning Strategy
 */
const PlanningStrategy = {
    IMMEDIATE: 'immediate',          // Single immediate action
    SHORT_TERM: 'short_term',        // 2-3 step plan
    LONG_TERM: 'long_term',          // Multi-step conversation arc
    ADAPTIVE: 'adaptive'             // Dynamic replanning
};

/**
 * Conversation Goal
 */
const ConversationGoal = {
    SUPPORT: 'support',              // Emotional support
    PROBLEM_SOLVE: 'problem_solve',  // Solve specific problem
    INFORMATION: 'information',      // Gather/provide information
    EDUCATION: 'education',          // Teach/explain
    ENTERTAINMENT: 'entertainment',  // Engage/entertain
    TRANSACTION: 'transaction'       // Complete task/transaction
};

/**
 * Prompt Templates for Next Steps Planning
 */
class NextStepsPrompts {
    static IMMEDIATE_ACTION = `You are an empathetic AI assistant. Based on the user's emotional state, determine the best immediate response.

**User Emotional State:**
- Emotion: {{emotion}}
- Confidence: {{confidence}}
- Intensity: {{intensity}}
- Genuineness: {{genuineness}}
- Nuance: {{nuance}}

**Interaction Strategy:** {{interaction_strategy}}

**Conversation Context:**
- Goal: {{goal}}
- Turn: {{turn}}
- Recent History: {{history}}

**What the user just said:** "{{user_text}}"

**Available Actions:**
- acknowledge: Acknowledge their emotion
- validate: Validate their feelings
- reassure: Provide reassurance
- empathize: Show empathy
- clarify: Ask for clarification
- ask_open: Open-ended question
- suggest: Suggest next step
- de_escalate: Help calm down
- escalate_human: Hand off to human

Select the best immediate action and generate a response. Respond as JSON:
{
  "action": "action_type",
  "reasoning": "why this action",
  "response": "what to say to user",
  "tone": "empathetic|professional|casual|supportive",
  "confidence": 0.0-1.0
}`;

    static MULTI_STEP_PLAN = `You are an empathetic AI assistant planning a conversation strategy.

**User Emotional State:**
- Emotion: {{emotion}}
- Nuance: {{nuance}}
- Trajectory: {{trajectory}}

**Interaction Strategy:** {{interaction_strategy}}

**Conversation Goal:** {{goal}}

**Current Situation:** {{situation}}

**Constraints:**
- Time available: {{time_constraint}}
- User patience level: {{patience}}
- Complexity tolerance: {{complexity}}

Plan a 3-5 step conversation strategy to achieve the goal while being emotionally sensitive.

Respond as JSON:
{
  "steps": [
    {
      "step": 1,
      "action": "action_type",
      "purpose": "what this accomplishes",
      "response": "what to say",
      "expected_emotion": "how user might feel after",
      "success_criteria": "how to know it worked"
    }
  ],
  "overall_strategy": "high-level approach",
  "contingencies": ["if user becomes more upset", "if user disengages"],
  "confidence": 0.0-1.0
}`;

    static INTERVENTION_DETECTION = `You are an AI safety monitor. Analyze whether intervention is needed.

**User Emotional State:**
- Emotion: {{emotion}}
- Intensity: {{intensity}}
- Trajectory: {{trajectory}}
- Red Flags: {{red_flags}}

**What user said:** "{{user_text}}"

**Conversation Context:** {{context}}

Determine if intervention is needed (escalate to human, emergency services, etc.).

Respond as JSON:
{
  "intervention_needed": true/false,
  "intervention_type": "none|human_escalation|specialist|emergency",
  "urgency": "low|medium|high|critical",
  "reasoning": "why intervention is/isn't needed",
  "suggested_action": "what to do right now",
  "confidence": 0.0-1.0
}`;

    static EMOTIONAL_TRAJECTORY = `You are an emotion prediction expert. Predict how user emotion will evolve.

**Current Emotional State:**
- Emotion: {{emotion}}
- Intensity: {{intensity}}
- Nuance: {{nuance}}

**Recent Emotional History:**
{{history}}

**Planned Next Action:** {{next_action}}

**Context:** {{context}}

Predict emotional trajectory and suggest adjustments if needed.

Respond as JSON:
{
  "predicted_emotion": "likely emotion after action",
  "predicted_intensity": 0.0-1.0,
  "trajectory": "improving|stable|declining",
  "risks": ["potential negative outcomes"],
  "adjustments": ["suggested modifications to plan"],
  "confidence": 0.0-1.0
}`;

    static GOAL_ALIGNMENT = `You are a conversation strategist. Ensure actions align with conversation goal.

**Current Goal:** {{goal}}

**User Emotional State:** {{emotion}} ({{nuance}})

**Proposed Action:** {{proposed_action}}

**Context:** {{context}}

Evaluate if proposed action supports the goal given user's emotional state.

Respond as JSON:
{
  "aligned": true/false,
  "alignment_score": 0.0-1.0,
  "reasoning": "why action does/doesn't support goal",
  "suggested_alternative": "better action if not aligned",
  "potential_conflicts": ["ways action might work against goal"],
  "confidence": 0.0-1.0
}`;
}

/**
 * Action Plan Step
 */
class ActionStep {
    constructor(data) {
        this.step = data.step || 1;
        this.action = data.action;
        this.purpose = data.purpose || '';
        this.response = data.response || '';
        this.tone = data.tone || 'empathetic';
        this.expectedEmotion = data.expectedEmotion || null;
        this.successCriteria = data.successCriteria || null;
        this.confidence = data.confidence || 0.0;
    }
}

/**
 * Action Plan Result
 */
class ActionPlanResult {
    constructor(data) {
        this.immediateAction = data.immediateAction || null;
        this.multiStepPlan = data.multiStepPlan || [];
        this.overallStrategy = data.overallStrategy || '';
        this.contingencies = data.contingencies || [];

        // Intervention
        this.interventionNeeded = data.interventionNeeded || false;
        this.interventionType = data.interventionType || 'none';
        this.urgency = data.urgency || 'low';

        // Trajectory
        this.predictedEmotion = data.predictedEmotion || null;
        this.predictedIntensity = data.predictedIntensity || 0.0;
        this.trajectory = data.trajectory || 'stable';

        // Alignment
        this.goalAligned = data.goalAligned !== false;
        this.alignmentScore = data.alignmentScore || 1.0;

        // Metadata
        this.confidence = data.confidence || 0.0;
        this.reasoning = data.reasoning || '';
        this.timestamp = new Date();
    }
}

/**
 * Next Steps Pipeline
 * Uses metaprompting to plan optimal system responses based on emotional analysis
 */
class NextStepsPipeline {
    constructor(config = {}) {
        this.config = {
            enabled: config.enabled !== false,

            // LLM configuration
            llmProvider: config.llmProvider || LLMProvider.MOCK,
            llmApiKey: config.llmApiKey || null,
            llmModel: config.llmModel || 'gpt-4',
            llmTemperature: config.llmTemperature || 0.7,

            // Planning strategy
            planningStrategy: config.planningStrategy || PlanningStrategy.ADAPTIVE,

            // Default goal
            defaultGoal: config.defaultGoal || ConversationGoal.SUPPORT,

            // Intervention thresholds
            interventionIntensityThreshold: config.interventionIntensityThreshold || 0.8,
            interventionRedFlags: config.interventionRedFlags || [
                'self-harm', 'suicide', 'violence', 'crisis', 'emergency'
            ],

            // Trajectory tracking
            emotionHistorySize: config.emotionHistorySize || 5,

            ...config
        };

        // Initialize LLM client
        this.llm = new LLMClient({
            provider: this.config.llmProvider,
            apiKey: this.config.llmApiKey,
            model: this.config.llmModel,
            temperature: this.config.llmTemperature
        });

        // State tracking
        this.emotionHistory = [];
        this.actionHistory = [];
        this.currentGoal = this.config.defaultGoal;
        this.currentPlan = null;

        // Metrics
        this.metrics = {
            totalPlans: 0,
            interventionsDetected: 0,
            goalsAchieved: 0,
            planAdjustments: 0,
            avgConfidence: 0.0
        };

        console.log('[NextStepsPipeline] Initialized with strategy:', this.config.planningStrategy);
    }

    /**
     * Main planning method
     */
    async plan(metaEmotionResult, userText, context = {}) {
        if (!this.config.enabled) {
            return this._createPassthroughPlan(metaEmotionResult);
        }

        this.metrics.totalPlans++;

        // Update emotion history
        this._updateEmotionHistory(metaEmotionResult);

        // Extract goal from context or use default
        const goal = context.goal || this.currentGoal;

        // 1. Check for intervention needs (safety first)
        const intervention = await this._checkIntervention(
            metaEmotionResult,
            userText,
            context
        );

        if (intervention.interventionNeeded) {
            this.metrics.interventionsDetected++;
            return new ActionPlanResult({
                immediateAction: this._createInterventionAction(intervention),
                interventionNeeded: true,
                interventionType: intervention.interventionType,
                urgency: intervention.urgency,
                reasoning: intervention.reasoning,
                confidence: intervention.confidence
            });
        }

        // 2. Plan actions based on strategy
        let actionPlan;

        switch (this.config.planningStrategy) {
            case PlanningStrategy.IMMEDIATE:
                actionPlan = await this._planImmediate(metaEmotionResult, userText, goal, context);
                break;
            case PlanningStrategy.SHORT_TERM:
                actionPlan = await this._planShortTerm(metaEmotionResult, userText, goal, context);
                break;
            case PlanningStrategy.LONG_TERM:
                actionPlan = await this._planLongTerm(metaEmotionResult, userText, goal, context);
                break;
            case PlanningStrategy.ADAPTIVE:
                actionPlan = await this._planAdaptive(metaEmotionResult, userText, goal, context);
                break;
            default:
                actionPlan = await this._planImmediate(metaEmotionResult, userText, goal, context);
        }

        // 3. Predict trajectory
        const trajectory = await this._predictTrajectory(
            metaEmotionResult,
            actionPlan.immediateAction,
            context
        );

        actionPlan.predictedEmotion = trajectory.predicted_emotion;
        actionPlan.predictedIntensity = trajectory.predicted_intensity;
        actionPlan.trajectory = trajectory.trajectory;

        // 4. Verify goal alignment
        const alignment = await this._checkGoalAlignment(
            goal,
            metaEmotionResult,
            actionPlan.immediateAction,
            context
        );

        actionPlan.goalAligned = alignment.aligned;
        actionPlan.alignmentScore = alignment.alignment_score;

        // If not aligned, adjust
        if (!alignment.aligned && alignment.suggested_alternative) {
            actionPlan.immediateAction.action = alignment.suggested_alternative;
            actionPlan.reasoning += ` (Adjusted for goal alignment: ${alignment.reasoning})`;
            this.metrics.planAdjustments++;
        }

        // Track action
        this._updateActionHistory(actionPlan);

        // Update metrics
        this.metrics.avgConfidence =
            (this.metrics.avgConfidence * (this.metrics.totalPlans - 1) + actionPlan.confidence) /
            this.metrics.totalPlans;

        return actionPlan;
    }

    /**
     * Plan immediate action
     */
    async _planImmediate(metaEmotionResult, userText, goal, context) {
        const prompt = this._fillTemplate(NextStepsPrompts.IMMEDIATE_ACTION, {
            emotion: metaEmotionResult.interpretedEmotion,
            confidence: metaEmotionResult.confidence,
            intensity: metaEmotionResult.intensity || 0.5,
            genuineness: metaEmotionResult.genuineness || 0.5,
            nuance: metaEmotionResult.nuanceDescription || '',
            interaction_strategy: metaEmotionResult.interactionStrategy || '',
            goal: goal,
            turn: context.conversationTurn || 1,
            history: this._formatEmotionHistory(),
            user_text: userText
        });

        const response = await this.llm.complete(prompt);
        const parsed = JSON.parse(response);

        return new ActionPlanResult({
            immediateAction: new ActionStep({
                action: parsed.action,
                response: parsed.response,
                tone: parsed.tone,
                confidence: parsed.confidence
            }),
            reasoning: parsed.reasoning,
            confidence: parsed.confidence
        });
    }

    /**
     * Plan short-term (2-3 steps)
     */
    async _planShortTerm(metaEmotionResult, userText, goal, context) {
        const prompt = this._fillTemplate(NextStepsPrompts.MULTI_STEP_PLAN, {
            emotion: metaEmotionResult.interpretedEmotion,
            nuance: metaEmotionResult.nuanceDescription || '',
            trajectory: this._calculateTrajectory(),
            interaction_strategy: metaEmotionResult.interactionStrategy || '',
            goal: goal,
            situation: context.situation || 'General conversation',
            time_constraint: context.timeConstraint || 'moderate',
            patience: context.userPatience || 'moderate',
            complexity: context.complexityTolerance || 'moderate'
        });

        const response = await this.llm.complete(prompt);
        const parsed = JSON.parse(response);

        const steps = parsed.steps.map(s => new ActionStep(s));

        return new ActionPlanResult({
            immediateAction: steps[0],
            multiStepPlan: steps,
            overallStrategy: parsed.overall_strategy,
            contingencies: parsed.contingencies,
            confidence: parsed.confidence
        });
    }

    /**
     * Plan long-term (full conversation arc)
     */
    async _planLongTerm(metaEmotionResult, userText, goal, context) {
        // Similar to short-term but with more steps
        return await this._planShortTerm(metaEmotionResult, userText, goal, {
            ...context,
            timeConstraint: 'extended'
        });
    }

    /**
     * Adaptive planning (adjust based on state)
     */
    async _planAdaptive(metaEmotionResult, userText, goal, context) {
        // Choose strategy based on emotional state and context
        const intensity = metaEmotionResult.intensity || 0.5;
        const conversationTurn = context.conversationTurn || 1;

        // High intensity or early conversation → immediate
        if (intensity > 0.7 || conversationTurn < 3) {
            return await this._planImmediate(metaEmotionResult, userText, goal, context);
        }

        // Complex emotional state → short-term
        if (metaEmotionResult.interpretedEmotion.includes('_')) {
            return await this._planShortTerm(metaEmotionResult, userText, goal, context);
        }

        // Default → immediate
        return await this._planImmediate(metaEmotionResult, userText, goal, context);
    }

    /**
     * Check if intervention needed
     */
    async _checkIntervention(metaEmotionResult, userText, context) {
        // Quick check for red flags
        const hasRedFlag = this.config.interventionRedFlags.some(
            flag => userText.toLowerCase().includes(flag)
        );

        const highIntensity = (metaEmotionResult.intensity || 0) >
            this.config.interventionIntensityThreshold;

        if (!hasRedFlag && !highIntensity) {
            return { interventionNeeded: false };
        }

        // LLM analysis
        const prompt = this._fillTemplate(NextStepsPrompts.INTERVENTION_DETECTION, {
            emotion: metaEmotionResult.interpretedEmotion,
            intensity: metaEmotionResult.intensity || 0,
            trajectory: this._calculateTrajectory(),
            red_flags: hasRedFlag ? 'Detected' : 'None',
            user_text: userText,
            context: JSON.stringify(context)
        });

        const response = await this.llm.complete(prompt);
        const parsed = JSON.parse(response);

        return parsed;
    }

    /**
     * Predict emotional trajectory
     */
    async _predictTrajectory(metaEmotionResult, nextAction, context) {
        const prompt = this._fillTemplate(NextStepsPrompts.EMOTIONAL_TRAJECTORY, {
            emotion: metaEmotionResult.interpretedEmotion,
            intensity: metaEmotionResult.intensity || 0,
            nuance: metaEmotionResult.nuanceDescription || '',
            history: this._formatEmotionHistory(),
            next_action: nextAction.action,
            context: JSON.stringify(context)
        });

        const response = await this.llm.complete(prompt);
        const parsed = JSON.parse(response);

        return parsed;
    }

    /**
     * Check goal alignment
     */
    async _checkGoalAlignment(goal, metaEmotionResult, action, context) {
        const prompt = this._fillTemplate(NextStepsPrompts.GOAL_ALIGNMENT, {
            goal: goal,
            emotion: metaEmotionResult.interpretedEmotion,
            nuance: metaEmotionResult.nuanceDescription || '',
            proposed_action: action.action,
            context: JSON.stringify(context)
        });

        const response = await this.llm.complete(prompt);
        const parsed = JSON.parse(response);

        return parsed;
    }

    // Helper methods

    _fillTemplate(template, vars) {
        let filled = template;
        for (const [key, value] of Object.entries(vars)) {
            filled = filled.replace(new RegExp(`{{${key}}}`, 'g'), value);
        }
        return filled;
    }

    _updateEmotionHistory(metaResult) {
        this.emotionHistory.push({
            emotion: metaResult.interpretedEmotion,
            intensity: metaResult.intensity || 0,
            confidence: metaResult.confidence,
            timestamp: new Date()
        });

        if (this.emotionHistory.length > this.config.emotionHistorySize) {
            this.emotionHistory.shift();
        }
    }

    _updateActionHistory(actionPlan) {
        this.actionHistory.push({
            action: actionPlan.immediateAction.action,
            response: actionPlan.immediateAction.response,
            timestamp: new Date()
        });

        if (this.actionHistory.length > this.config.emotionHistorySize) {
            this.actionHistory.shift();
        }
    }

    _formatEmotionHistory() {
        return this.emotionHistory.map((h, i) =>
            `${i + 1}. ${h.emotion} (intensity: ${h.intensity.toFixed(2)})`
        ).join(', ');
    }

    _calculateTrajectory() {
        if (this.emotionHistory.length < 2) return 'insufficient data';

        const recent = this.emotionHistory.slice(-3);
        const intensities = recent.map(h => h.intensity);

        const trend = intensities[intensities.length - 1] - intensities[0];

        if (trend > 0.2) return 'escalating';
        if (trend < -0.2) return 'de-escalating';
        return 'stable';
    }

    _createInterventionAction(intervention) {
        return new ActionStep({
            action: intervention.intervention_type === 'emergency'
                ? ActionType.EMERGENCY
                : ActionType.ESCALATE_HUMAN,
            response: intervention.suggested_action,
            tone: 'professional',
            confidence: intervention.confidence
        });
    }

    _createPassthroughPlan(metaEmotionResult) {
        return new ActionPlanResult({
            immediateAction: new ActionStep({
                action: ActionType.ACKNOWLEDGE,
                response: `I understand you're feeling ${metaEmotionResult.interpretedEmotion}.`,
                tone: 'empathetic',
                confidence: 0.5
            }),
            reasoning: 'Passthrough (pipeline disabled)',
            confidence: 0.5
        });
    }

    /**
     * Set conversation goal
     */
    setGoal(goal) {
        this.currentGoal = goal;
        console.log('[NextStepsPipeline] Goal set to:', goal);
    }

    /**
     * Get metrics
     */
    getMetrics() {
        return {
            ...this.metrics,
            emotionHistorySize: this.emotionHistory.length,
            actionHistorySize: this.actionHistory.length,
            currentGoal: this.currentGoal,
            interventionRate: this.metrics.totalPlans > 0
                ? this.metrics.interventionsDetected / this.metrics.totalPlans
                : 0
        };
    }
}


// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        NextStepsPipeline,
        ActionPlanResult,
        ActionStep,
        ActionType,
        PlanningStrategy,
        ConversationGoal,
        NextStepsPrompts
    };
}
