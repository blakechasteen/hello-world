/**
 * Action Templates - Pre-built Response Templates for Emotional States
 * Provides structured, emotionally-appropriate responses
 * Date: November 2025
 *
 * Features:
 * - Emotion-specific response templates
 * - Tone variations (empathetic, professional, casual)
 * - Multi-step conversation flows
 * - Escalation/de-escalation strategies
 * - Customizable template variables
 */

import { ActionType } from './next_steps_pipeline.js';

/**
 * Response Template
 */
class ResponseTemplate {
    constructor(data) {
        this.action = data.action;
        this.emotion = data.emotion;
        this.tone = data.tone || 'empathetic';
        this.template = data.template;
        this.variables = data.variables || [];
        this.followUp = data.followUp || null;
        this.successCriteria = data.successCriteria || null;
    }

    render(context = {}) {
        let response = this.template;

        // Replace variables
        for (const variable of this.variables) {
            const value = context[variable] || `{${variable}}`;
            response = response.replace(new RegExp(`{${variable}}`, 'g'), value);
        }

        return response;
    }
}

/**
 * Action Templates Library
 */
class ActionTemplates {
    constructor() {
        this.templates = new Map();
        this._initializeTemplates();
    }

    _initializeTemplates() {
        // ===== ACKNOWLEDGMENT TEMPLATES =====

        this.add(new ResponseTemplate({
            action: ActionType.ACKNOWLEDGE,
            emotion: 'anxious',
            tone: 'empathetic',
            template: "I can see you're feeling anxious about {topic}. That's completely understandable.",
            variables: ['topic'],
            followUp: ActionType.VALIDATE
        }));

        this.add(new ResponseTemplate({
            action: ActionType.ACKNOWLEDGE,
            emotion: 'frustrated',
            tone: 'empathetic',
            template: "I hear your frustration with {situation}. It sounds like this has been challenging.",
            variables: ['situation'],
            followUp: ActionType.EMPATHIZE
        }));

        this.add(new ResponseTemplate({
            action: ActionType.ACKNOWLEDGE,
            emotion: 'happy',
            tone: 'warm',
            template: "I'm glad to hear you're feeling positive about {topic}!",
            variables: ['topic'],
            followUp: ActionType.ENCOURAGE
        }));

        this.add(new ResponseTemplate({
            action: ActionType.ACKNOWLEDGE,
            emotion: 'confused',
            tone: 'supportive',
            template: "I understand this feels confusing. Let me help clarify {topic} for you.",
            variables: ['topic'],
            followUp: ActionType.CLARIFY
        }));

        // ===== VALIDATION TEMPLATES =====

        this.add(new ResponseTemplate({
            action: ActionType.VALIDATE,
            emotion: 'anxious',
            tone: 'empathetic',
            template: "Your concerns about {topic} are valid. Many people feel this way when {situation}.",
            variables: ['topic', 'situation'],
            followUp: ActionType.REASSURE
        }));

        this.add(new ResponseTemplate({
            action: ActionType.VALIDATE,
            emotion: 'frustrated',
            tone: 'empathetic',
            template: "Your frustration is completely justified. {situation} would be frustrating for anyone.",
            variables: ['situation'],
            followUp: ActionType.SUGGEST
        }));

        this.add(new ResponseTemplate({
            action: ActionType.VALIDATE,
            emotion: 'sad',
            tone: 'gentle',
            template: "It's okay to feel sad about {topic}. Your feelings matter.",
            variables: ['topic'],
            followUp: ActionType.EMPATHIZE
        }));

        // ===== REASSURANCE TEMPLATES =====

        this.add(new ResponseTemplate({
            action: ActionType.REASSURE,
            emotion: 'anxious',
            tone: 'calm',
            template: "I want you to know that {reassurance}. We can work through this together.",
            variables: ['reassurance'],
            followUp: ActionType.SUGGEST
        }));

        this.add(new ResponseTemplate({
            action: ActionType.REASSURE,
            emotion: 'worried',
            tone: 'supportive',
            template: "While I understand your concern about {topic}, {positive_fact}. You're not alone in this.",
            variables: ['topic', 'positive_fact'],
            followUp: ActionType.PLAN
        }));

        // ===== EMPATHY TEMPLATES =====

        this.add(new ResponseTemplate({
            action: ActionType.EMPATHIZE,
            emotion: 'frustrated',
            tone: 'understanding',
            template: "I can imagine how frustrating {situation} must be, especially when {complication}.",
            variables: ['situation', 'complication'],
            followUp: ActionType.SUGGEST
        }));

        this.add(new ResponseTemplate({
            action: ActionType.EMPATHIZE,
            emotion: 'sad',
            tone: 'gentle',
            template: "I'm sorry you're going through this. Feeling sad about {topic} is a natural response.",
            variables: ['topic'],
            followUp: ActionType.ASK_OPEN
        }));

        this.add(new ResponseTemplate({
            action: ActionType.EMPATHIZE,
            emotion: 'overwhelmed',
            tone: 'calm',
            template: "It sounds like you have a lot on your plate with {situation}. That would be overwhelming for anyone.",
            variables: ['situation'],
            followUp: ActionType.BRAINSTORM
        }));

        // ===== CLARIFICATION TEMPLATES =====

        this.add(new ResponseTemplate({
            action: ActionType.CLARIFY,
            emotion: 'confused',
            tone: 'patient',
            template: "I want to make sure I understand. When you say {user_statement}, do you mean {interpretation}?",
            variables: ['user_statement', 'interpretation'],
            followUp: ActionType.ASK_CLOSED
        }));

        this.add(new ResponseTemplate({
            action: ActionType.CLARIFY,
            emotion: 'uncertain',
            tone: 'supportive',
            template: "Could you help me understand more about {topic}? What specifically concerns you?",
            variables: ['topic'],
            followUp: ActionType.ASK_OPEN
        }));

        // ===== OPEN QUESTION TEMPLATES =====

        this.add(new ResponseTemplate({
            action: ActionType.ASK_OPEN,
            emotion: 'anxious',
            tone: 'gentle',
            template: "What would help you feel more comfortable about {topic}?",
            variables: ['topic'],
            successCriteria: 'User shares specific concern or need'
        }));

        this.add(new ResponseTemplate({
            action: ActionType.ASK_OPEN,
            emotion: 'frustrated',
            tone: 'calm',
            template: "What do you think would help resolve {situation}?",
            variables: ['situation'],
            successCriteria: 'User proposes solution or direction'
        }));

        this.add(new ResponseTemplate({
            action: ActionType.ASK_OPEN,
            emotion: 'overwhelmed',
            tone: 'supportive',
            template: "If you could change one thing about {situation} right now, what would it be?",
            variables: ['situation'],
            successCriteria: 'User identifies priority'
        }));

        // ===== SUGGESTION TEMPLATES =====

        this.add(new ResponseTemplate({
            action: ActionType.SUGGEST,
            emotion: 'anxious',
            tone: 'calm',
            template: "One approach that might help is {suggestion}. Would you like to explore that?",
            variables: ['suggestion'],
            followUp: ActionType.ASK_CLOSED
        }));

        this.add(new ResponseTemplate({
            action: ActionType.SUGGEST,
            emotion: 'frustrated',
            tone: 'practical',
            template: "Here's what we could try: {suggestion}. Does that sound workable?",
            variables: ['suggestion'],
            followUp: ActionType.PLAN
        }));

        this.add(new ResponseTemplate({
            action: ActionType.SUGGEST,
            emotion: 'overwhelmed',
            tone: 'structured',
            template: "Let's break this down. First, we could {first_step}. How does that sound?",
            variables: ['first_step'],
            followUp: ActionType.PLAN
        }));

        // ===== DE-ESCALATION TEMPLATES =====

        this.add(new ResponseTemplate({
            action: ActionType.DE_ESCALATE,
            emotion: 'angry',
            tone: 'calm',
            template: "I understand you're upset about {situation}. Let's take a moment and work through this together.",
            variables: ['situation'],
            followUp: ActionType.VALIDATE
        }));

        this.add(new ResponseTemplate({
            action: ActionType.DE_ESCALATE,
            emotion: 'anxious_intense',
            tone: 'grounding',
            template: "I hear you, and I'm here to help. Let's focus on one thing at a time. First, {grounding_question}?",
            variables: ['grounding_question'],
            followUp: ActionType.CLARIFY
        }));

        // ===== REFRAME TEMPLATES =====

        this.add(new ResponseTemplate({
            action: ActionType.REFRAME,
            emotion: 'negative',
            tone: 'constructive',
            template: "I hear that {situation} feels {negative_emotion}. Another way to look at it might be {reframe}.",
            variables: ['situation', 'negative_emotion', 'reframe'],
            followUp: ActionType.ASK_OPEN
        }));

        this.add(new ResponseTemplate({
            action: ActionType.REFRAME,
            emotion: 'discouraged',
            tone: 'encouraging',
            template: "While {obstacle} is challenging, you've already {achievement}. That shows {strength}.",
            variables: ['obstacle', 'achievement', 'strength'],
            followUp: ActionType.ENCOURAGE
        }));

        // ===== ENCOURAGEMENT TEMPLATES =====

        this.add(new ResponseTemplate({
            action: ActionType.ENCOURAGE,
            emotion: 'uncertain',
            tone: 'supportive',
            template: "You've handled {past_success} before, and you can do this too. I believe in your ability to {goal}.",
            variables: ['past_success', 'goal'],
            followUp: ActionType.SUGGEST
        }));

        this.add(new ResponseTemplate({
            action: ActionType.ENCOURAGE,
            emotion: 'discouraged',
            tone: 'motivating',
            template: "Progress isn't always linear. The fact that you're {current_effort} shows real determination.",
            variables: ['current_effort'],
            followUp: ActionType.PLAN
        }));

        // ===== BRAINSTORM TEMPLATES =====

        this.add(new ResponseTemplate({
            action: ActionType.BRAINSTORM,
            emotion: 'stuck',
            tone: 'creative',
            template: "Let's think outside the box. What if we tried {option1}? Or {option2}? What sounds most feasible?",
            variables: ['option1', 'option2'],
            followUp: ActionType.ASK_CLOSED
        }));

        // ===== PLAN TEMPLATES =====

        this.add(new ResponseTemplate({
            action: ActionType.PLAN,
            emotion: 'anxious',
            tone: 'structured',
            template: "Let's create a clear plan. Step 1: {step1}. Step 2: {step2}. How does that feel?",
            variables: ['step1', 'step2'],
            followUp: ActionType.ASK_CLOSED
        }));

        this.add(new ResponseTemplate({
            action: ActionType.PLAN,
            emotion: 'overwhelmed',
            tone: 'organized',
            template: "Here's how we can tackle this systematically: First, {priority1}. Then, {priority2}. Sound good?",
            variables: ['priority1', 'priority2'],
            followUp: ActionType.EXECUTE
        }));

        // ===== ESCALATION TEMPLATES =====

        this.add(new ResponseTemplate({
            action: ActionType.ESCALATE_HUMAN,
            emotion: 'crisis',
            tone: 'professional',
            template: "I want to make sure you get the best support. Let me connect you with a human specialist who can help with {issue}.",
            variables: ['issue'],
            successCriteria: 'User accepts human handoff'
        }));

        this.add(new ResponseTemplate({
            action: ActionType.EMERGENCY,
            emotion: 'emergency',
            tone: 'urgent_calm',
            template: "I'm concerned about your safety regarding {concern}. Please reach out to {emergency_resource} immediately at {contact}.",
            variables: ['concern', 'emergency_resource', 'contact'],
            successCriteria: 'User acknowledges emergency resources'
        }));

        // ===== MULTI-EMOTION TEMPLATES =====

        this.add(new ResponseTemplate({
            action: ActionType.ACKNOWLEDGE,
            emotion: 'anxious_hopeful',
            tone: 'balanced',
            template: "I can see you're both concerned and hopeful about {topic}. Those feelings can coexist, and that's okay.",
            variables: ['topic'],
            followUp: ActionType.VALIDATE
        }));

        this.add(new ResponseTemplate({
            action: ActionType.ACKNOWLEDGE,
            emotion: 'frustrated_determined',
            tone: 'motivating',
            template: "Your frustration with {obstacle} is understandable, but I also hear your determination to {goal}. Let's channel that energy.",
            variables: ['obstacle', 'goal'],
            followUp: ActionType.PLAN
        }));
    }

    /**
     * Add template to library
     */
    add(template) {
        const key = `${template.action}_${template.emotion}_${template.tone}`;
        this.templates.set(key, template);
    }

    /**
     * Get template by action, emotion, and tone
     */
    get(action, emotion, tone = 'empathetic') {
        const key = `${action}_${emotion}_${tone}`;
        let template = this.templates.get(key);

        if (!template) {
            // Try without tone
            for (const [k, t] of this.templates.entries()) {
                if (k.startsWith(`${action}_${emotion}_`)) {
                    template = t;
                    break;
                }
            }
        }

        if (!template) {
            // Try just action
            for (const [k, t] of this.templates.entries()) {
                if (k.startsWith(`${action}_`)) {
                    template = t;
                    break;
                }
            }
        }

        return template;
    }

    /**
     * Get all templates for an action
     */
    getByAction(action) {
        const templates = [];
        for (const [key, template] of this.templates.entries()) {
            if (template.action === action) {
                templates.push(template);
            }
        }
        return templates;
    }

    /**
     * Get all templates for an emotion
     */
    getByEmotion(emotion) {
        const templates = [];
        for (const [key, template] of this.templates.entries()) {
            if (template.emotion === emotion || template.emotion.includes(emotion)) {
                templates.push(template);
            }
        }
        return templates;
    }

    /**
     * Render template with context
     */
    render(action, emotion, context, tone = 'empathetic') {
        const template = this.get(action, emotion, tone);

        if (!template) {
            // Fallback
            return `I understand you're feeling ${emotion}. How can I help?`;
        }

        return template.render(context);
    }
}

/**
 * Conversation Flow Templates
 * Multi-step templates for common conversation patterns
 */
class ConversationFlowTemplates {
    static SUPPORT_ANXIOUS_USER = [
        {
            step: 1,
            action: ActionType.ACKNOWLEDGE,
            template: "I can see you're feeling anxious about {topic}.",
            variables: ['topic']
        },
        {
            step: 2,
            action: ActionType.VALIDATE,
            template: "That's a completely normal reaction when {situation}.",
            variables: ['situation']
        },
        {
            step: 3,
            action: ActionType.ASK_OPEN,
            template: "What would help you feel more comfortable right now?",
            variables: []
        },
        {
            step: 4,
            action: ActionType.SUGGEST,
            template: "Based on what you've shared, {suggestion} might help. Shall we try that?",
            variables: ['suggestion']
        }
    ];

    static RESOLVE_FRUSTRATION = [
        {
            step: 1,
            action: ActionType.ACKNOWLEDGE,
            template: "I hear your frustration with {situation}.",
            variables: ['situation']
        },
        {
            step: 2,
            action: ActionType.EMPATHIZE,
            template: "That would be frustrating for anyone, especially when {complication}.",
            variables: ['complication']
        },
        {
            step: 3,
            action: ActionType.CLARIFY,
            template: "Help me understand what outcome you're hoping for.",
            variables: []
        },
        {
            step: 4,
            action: ActionType.BRAINSTORM,
            template: "Let's explore options: {option1} or {option2}. What sounds better?",
            variables: ['option1', 'option2']
        },
        {
            step: 5,
            action: ActionType.PLAN,
            template: "Great. Here's how we'll proceed: {plan}.",
            variables: ['plan']
        }
    ];

    static DE_ESCALATE_ANGER = [
        {
            step: 1,
            action: ActionType.DE_ESCALATE,
            template: "I understand you're upset. Let's take a moment.",
            variables: []
        },
        {
            step: 2,
            action: ActionType.VALIDATE,
            template: "Your feelings about {situation} are valid.",
            variables: ['situation']
        },
        {
            step: 3,
            action: ActionType.ASK_OPEN,
            template: "What's the most important thing we need to address?",
            variables: []
        },
        {
            step: 4,
            action: ActionType.SUGGEST,
            template: "Let's focus on {priority} first. Does that work for you?",
            variables: ['priority']
        }
    ];

    static BUILD_CONFIDENCE = [
        {
            step: 1,
            action: ActionType.ACKNOWLEDGE,
            template: "I can see you're uncertain about {topic}.",
            variables: ['topic']
        },
        {
            step: 2,
            action: ActionType.REFRAME,
            template: "Remember, you successfully handled {past_success}. This is similar.",
            variables: ['past_success']
        },
        {
            step: 3,
            action: ActionType.ENCOURAGE,
            template: "You have the skills needed: {skill1} and {skill2}.",
            variables: ['skill1', 'skill2']
        },
        {
            step: 4,
            action: ActionType.PLAN,
            template: "Let's break it into manageable steps: {step1}, then {step2}.",
            variables: ['step1', 'step2']
        }
    ];
}

// Export
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        ActionTemplates,
        ResponseTemplate,
        ConversationFlowTemplates
    };
}
