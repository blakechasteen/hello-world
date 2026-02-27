# Meta-Emotion Interpreter Integration Guide

**Date**: November 2025
**Purpose**: LLM-powered contextual emotion understanding using metaprompting

---

## Overview

The **MetaEmotionInterpreter** adds a powerful reasoning layer on top of signal-level emotion detection, using LLM prompting to:

1. **Dynamically select fusion strategies** based on context
2. **Interpret emotional nuance** beyond categorical labels
3. **Resolve complex conflicts** between modalities using reasoning
4. **Adapt to conversation context** and user patterns
5. **Provide actionable interaction strategies**

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    User Input                            │
│              (voice + video + text)                      │
└────────────────┬────────────────────────────────────────┘
                 │
    ┌────────────┼────────────┐
    │            │            │
┌───▼────┐  ┌───▼────┐  ┌───▼────┐
│Facial  │  │Vocal   │  │Text    │  Signal-Level
│Analyzer│  │Prosody │  │Emotion │  Detection
└───┬────┘  └───┬────┘  └───┬────┘
    │            │            │
    └────────────┼────────────┘
                 │
         ┌───────▼────────┐
         │ EmotionFusion  │        Fusion Layer
         │    System      │        (6 strategies)
         └───────┬────────┘
                 │
         ┌───────▼────────┐
         │MetaEmotion     │        Meta Layer
         │ Interpreter    │        (LLM reasoning)
         │  (5 modes)     │
         └───────┬────────┘
                 │
         ┌───────▼────────┐
         │  Final Result  │        Enhanced
         │ + Interaction  │        Understanding
         │   Strategy     │
         └────────────────┘
```

---

## Quick Start

### 1. Basic Integration

```javascript
import { MetaEmotionInterpreter, MetaMode, LLMProvider } from './meta_emotion_interpreter.js';
import { EmotionFusionSystem } from './emotion_fusion.js';
import { FacialEmotionAnalyzer } from './facial_emotion_analyzer.js';
import { VocalProsodyAnalyzer } from './vocal_prosody_analyzer.js';
import { EmotionDetector } from './emotion_detector.js';

// Initialize analyzers
const facialAnalyzer = new FacialEmotionAnalyzer({ backend: 'face-api' });
const vocalAnalyzer = new VocalProsodyAnalyzer();
const textAnalyzer = new EmotionDetector();

// Initialize fusion system
const fusionSystem = new EmotionFusionSystem({
    fusionStrategy: 'weighted',
    modalityWeights: { facial: 0.4, vocal: 0.35, text: 0.25 }
});

// Initialize meta-interpreter
const metaInterpreter = new MetaEmotionInterpreter({
    enabled: true,
    mode: MetaMode.FULL_META,
    llmProvider: LLMProvider.OPENAI,
    llmApiKey: 'your-api-key',
    llmModel: 'gpt-4'
});

// Analyze emotions with meta-interpretation
async function analyzeEmotionWithMeta(videoFrame, audioBuffer, text) {
    // 1. Signal-level detection
    const facialResult = await facialAnalyzer.analyze(videoFrame);
    const vocalResult = await vocalAnalyzer.analyze(audioBuffer);
    const textResult = await textAnalyzer.detectEmotion(text);

    // 2. Fusion
    const signals = [
        { modality: 'facial', data: facialResult },
        { modality: 'vocal', data: vocalResult },
        { modality: 'text', data: textResult }
    ];
    const fusedResult = fusionSystem.fuse(signals);

    // 3. Meta-interpretation (LLM reasoning)
    const metaResult = await metaInterpreter.interpret(
        fusedResult,
        signals,
        { text: text, conversationTurn: 5 }
    );

    return metaResult;
}

// Usage
const result = await analyzeEmotionWithMeta(frame, audio, "I'm fine, really.");

console.log('Interpreted Emotion:', result.interpretedEmotion);
console.log('Nuance:', result.nuanceDescription);
console.log('Interaction Strategy:', result.interactionStrategy);
```

---

## 5 Meta-Interpretation Modes

### Mode 1: Strategy Selection

**Use Case**: Let LLM choose the best fusion strategy based on context

```javascript
const metaInterpreter = new MetaEmotionInterpreter({
    mode: MetaMode.STRATEGY_SELECTION,
    llmProvider: LLMProvider.ANTHROPIC,
    llmModel: 'claude-3-5-sonnet-20241022'
});

const strategyResult = await metaInterpreter.interpret(null, signals, context);

console.log('Selected Strategy:', strategyResult.selectedStrategy);
// => "weighted"

console.log('Reasoning:', strategyResult.strategyReasoning);
// => "Varying confidence levels across modalities suggest weighted fusion"

// Use selected strategy
const fusionSystem = new EmotionFusionSystem({
    fusionStrategy: strategyResult.selectedStrategy
});
```

**Prompt Template**:
```
Given emotion signals from facial (happy, 0.8), vocal (neutral, 0.6), text (positive, 0.9)
with conflict level 0.33, select best fusion strategy from:
- average, weighted, max_confidence, majority_vote, priority, bayesian

Context: Professional customer service call, user appears satisfied but voice is flat
```

**LLM Response**:
```json
{
  "strategy": "weighted",
  "reasoning": "High facial and text confidence (0.8, 0.9) but lower vocal confidence (0.6) suggests weighted fusion to trust high-confidence signals more",
  "confidence": 0.87
}
```

---

### Mode 2: Nuance Interpretation

**Use Case**: Understand emotional complexity beyond categorical labels

```javascript
const metaInterpreter = new MetaEmotionInterpreter({
    mode: MetaMode.NUANCE_INTERPRETATION,
    llmProvider: LLMProvider.OPENAI,
    llmModel: 'gpt-4'
});

const nuanceResult = await metaInterpreter.interpret(fusedResult, signals, context);

console.log('Primary:', nuanceResult.interpretedEmotion);
// => "anxious_hopeful"

console.log('Secondary:', nuanceResult.secondaryEmotion);
// => "determined"

console.log('Nuance:', nuanceResult.nuanceDescription);
// => "User shows genuine anxiety about current situation but maintains
//     hopeful outlook and determination to succeed. Seeking reassurance
//     while actively problem-solving."

console.log('Intensity:', nuanceResult.intensity);
// => 0.7

console.log('Genuineness:', nuanceResult.genuineness);
// => 0.8
```

**Prompt Template**:
```
Facial: worried (0.7), furrowed brow, gaze downward
Vocal: hesitant speaking style, rising pitch (questioning), pauses
Text: "I think... maybe we can still fix this?"

VAD: valence=-0.2, arousal=0.6, dominance=0.4

Provide nuanced interpretation considering all signals.
```

**LLM Response**:
```json
{
  "primary_emotion": "anxious_hopeful",
  "secondary_emotion": "determined",
  "intensity": 0.7,
  "genuineness": 0.8,
  "nuance_description": "User experiences anxiety about situation (worried face, hesitant voice) but maintains hope and determination (rising questioning tone, 'maybe we can fix'). Seeking validation while actively trying to problem-solve.",
  "social_implications": "User needs reassurance and concrete next steps to reduce anxiety while maintaining motivation. Acknowledge concern, validate effort, provide clear action plan.",
  "confidence": 0.82
}
```

---

### Mode 3: Conflict Resolution

**Use Case**: Resolve contradictory emotion signals using reasoning

```javascript
const metaInterpreter = new MetaEmotionInterpreter({
    mode: MetaMode.CONFLICT_RESOLUTION,
    conflictThreshold: 0.3
});

const conflictResult = await metaInterpreter.interpret(fusedResult, signals, context);

console.log('True Emotion:', conflictResult.interpretedEmotion);
// => "frustrated"

console.log('Resolution:', conflictResult.conflictResolution);
// => {
//      conflicts: [{ modality1: 'text', emotion1: 'positive',
//                     modality2: 'vocal', emotion2: 'negative' }],
//      resolution: 'frustrated',
//      alternativeInterpretations: ['genuinely neutral', 'sarcastic politeness']
//    }

console.log('Explanation:', conflictResult.nuanceDescription);
// => "Polite text masking frustrated vocal tone suggests professional
//     context where emotions must be controlled"

console.log('Recommended Action:', conflictResult.interactionStrategy);
// => "Acknowledge underlying frustration tactfully while maintaining
//     professional tone"
```

**Scenario**: User says "Everything is fine." (text: positive) but voice is tight, clipped (vocal: frustrated)

**Prompt Template**:
```
Conflicts Detected:
- text (positive, 0.8) vs vocal (frustrated, 0.75)

Facial: neutral (0.6), slight jaw tension
Vocal: clipped speech, low pitch, fast rate
Text: "Everything is fine."

Possible Explanations:
1. Sarcasm
2. Politeness masking
3. Suppressed emotion
4. Cultural display rules
5. Technical error

Context: Professional workplace chat
```

**LLM Response**:
```json
{
  "likely_true_emotion": "frustrated",
  "explanation": "Polite text masking frustrated vocal tone (clipped, fast, low pitch) and facial tension suggests professional context where direct expression of frustration is inappropriate. User is controlling display but vocal leakage reveals true state.",
  "alternative_interpretations": [
    "genuinely neutral but rushed/stressed for unrelated reasons",
    "sarcastic politeness (less likely given context)"
  ],
  "recommended_action": "Acknowledge underlying frustration tactfully ('I sense some concern about this...') while maintaining professional tone. Offer constructive problem-solving.",
  "confidence": 0.78
}
```

---

### Mode 4: Context Adaptation

**Use Case**: Adapt emotion interpretation to conversation flow and user patterns

```javascript
const metaInterpreter = new MetaEmotionInterpreter({
    mode: MetaMode.CONTEXT_ADAPTATION,
    conversationHistorySize: 10
});

// After 5 turns of conversation
const adaptedResult = await metaInterpreter.interpret(fusedResult, signals, {
    text: "That might work!",
    conversationTurn: 5
});

console.log('Adjusted Emotion:', adaptedResult.interpretedEmotion);
// => "relieved_cautious"

console.log('Adjustment Reasoning:', adaptedResult.adjustmentReasoning);
// => "Previous conversation showed concern; current positive emotion
//     likely relief but tempered by lingering caution from earlier
//     uncertainty"

console.log('Conversation Coherence:', adaptedResult.conversationCoherence);
// => 0.88

console.log('Recommendations:', adaptedResult.interactionRecommendations);
// => [
//      "Acknowledge relief without dismissing concerns",
//      "Offer continued support",
//      "Suggest concrete steps forward"
//    ]
```

**Conversation History**:
```
Turn 1: worried (0.75) - "I'm not sure this will work"
Turn 2: anxious (0.80) - "What if it fails?"
Turn 3: uncertain (0.65) - "Maybe we should try something else?"
Turn 4: tentatively_hopeful (0.60) - "Well, I suppose we could..."
Turn 5: positive (0.70) - "That might work!"
```

**Prompt Template**:
```
Current Emotion: positive (0.70)
Current VAD: valence=0.5, arousal=0.6, dominance=0.5

Conversation History:
1. worried (0.75) - "I'm not sure this will work"
2. anxious (0.80) - "What if it fails?"
3. uncertain (0.65) - "Maybe we should try something else?"
4. tentatively_hopeful (0.60) - "Well, I suppose we could..."
5. positive (0.70) - "That might work!"

User Profile: Tends toward analytical, cautious decision-making

Temporal Pattern: worried → anxious → uncertain → hopeful → positive

Consider: Is this emotion consistent with conversation flow?
         What triggered this emotional state?
         How should system adapt interaction?
```

**LLM Response**:
```json
{
  "adjusted_emotion": "relieved_cautious",
  "adjustment_reasoning": "Emotional trajectory shows progression from high anxiety (turn 2) through uncertainty toward cautious optimism. Current 'positive' (0.70) is likely relief that a solution path emerged, but confidence remains modest (not euphoric) due to lingering caution. User's analytical style suggests they won't fully commit until validated.",
  "conversation_coherence": 0.88,
  "interaction_recommendations": [
    "Acknowledge relief: 'I'm glad this feels more promising'",
    "Don't dismiss earlier concerns: 'I know you had reservations...'",
    "Offer validation: 'Let's walk through how this addresses your concerns'",
    "Suggest concrete next step to build confidence"
  ],
  "confidence": 0.84
}
```

---

### Mode 5: Full Meta-Analysis

**Use Case**: Comprehensive LLM reasoning combining all capabilities

```javascript
const metaInterpreter = new MetaEmotionInterpreter({
    mode: MetaMode.FULL_META,
    llmProvider: LLMProvider.ANTHROPIC,
    llmModel: 'claude-3-5-sonnet-20241022',
    conversationHistorySize: 10
});

const fullResult = await metaInterpreter.interpret(fusedResult, signals, context);

console.log('Interpreted Emotion:', fullResult.interpretedEmotion);
// => "complex_mixed_anxious_determined"

console.log('Nuance:', fullResult.nuanceDescription);
// => "Multi-layered emotional state combining anxiety about outcome
//     with strong determination to succeed. User is managing fear
//     through active problem-solving."

console.log('Key Insights:', fullResult.keyInsights);
// => [
//      "Anxiety is motivating rather than paralyzing",
//      "User seeks control through planning",
//      "Determination masks underlying fear of failure"
//    ]

console.log('Uncertainty Factors:', fullResult.uncertaintyFactors);
// => [
//      "Limited facial data (low lighting)",
//      "Vocal prosody suggests stress but unclear if task-related",
//      "Text positive but could be performative"
//    ]

console.log('Interaction Strategy:', fullResult.interactionStrategy);
// => "Empathetic, adaptive response that acknowledges both anxiety
//     and determination. Provide structured support to channel
//     determination productively while addressing anxiety."

console.log('Calibration Suggestions:', fullResult.calibrationSuggestions);
// => [
//      "Request better lighting for facial analysis",
//      "Track task-specific stress vs general anxiety baseline",
//      "Validate genuine emotion vs performative positivity over time"
//    ]
```

**Full Prompt** (combines all signals, context, history):
```
Raw Signals:
- Facial: worried (0.65), furrowed brow, tense jaw
- Vocal: determined speaking style (0.72), high pitch variation
- Text: positive (0.78), "I'm going to make this work!"

Fusion Result: anxious_determined (0.71)
Fusion Strategy Used: bayesian

VAD: v=0.2, a=0.7, d=0.6

Conflicts: facial (worried) vs text (positive)
Sarcasm Score: 0.15
Genuineness: 0.78

Context: User working on critical project with tight deadline
History: Last 3 turns showed increasing anxiety then sudden determination

Perform comprehensive meta-analysis.
```

**LLM Response**:
```json
{
  "interpreted_emotion": "anxious_determination_with_controlled_fear",
  "confidence": 0.75,
  "nuance_description": "User experiences genuine anxiety about project outcome (facial tension, worried expression) but has shifted into determined problem-solving mode (vocal determination, positive text). The anxiety is being actively managed through action rather than avoidance. Facial-text conflict suggests user is 'putting on a brave face' while acknowledging internal worry.",
  "key_insights": [
    "Anxiety is adaptive/motivating rather than paralyzing - driving action",
    "User copes with fear through control-seeking (determination, planning)",
    "Genuineness score (0.78) suggests authentic determination, not just performative",
    "VAD shows high arousal (0.7) + moderate dominance (0.6) = engaged agency",
    "Tight deadline context explains intensity of both anxiety and determination"
  ],
  "uncertainty_factors": [
    "Facial data has lower confidence (0.65) - lighting, angle, or genuine ambiguity?",
    "Vocal 'determination' could also indicate stress-induced intensity",
    "Text positivity ('make this work') might be self-encouragement vs social display",
    "Limited conversation history (3 turns) - unclear if this pattern is typical for user"
  ],
  "interaction_strategy": "Empathetic, structured support strategy: (1) Validate both emotions: 'I can see you're really committed to this, and I also sense some understandable concern about the timeline.' (2) Channel determination productively: 'Let's break this down into clear next steps.' (3) Address anxiety without amplifying: 'What would help you feel more confident about the outcome?' (4) Provide agency: Offer concrete tools/resources rather than just reassurance.",
  "calibration_suggestions": [
    "Request better lighting/angle for improved facial confidence in future",
    "Establish user's anxiety baseline (trait) vs current state (state) over multiple sessions",
    "Track whether 'positive self-talk' text (like 'I'm going to...') predicts success or masks overwhelm",
    "Monitor if determination sustains or gives way to burnout as deadline approaches",
    "Calibrate sarcasm threshold - 0.15 is low, but watch for passive-aggressive patterns"
  ]
}
```

---

## Production Integration

### With Real LLM Providers

**OpenAI Integration**:
```javascript
const metaInterpreter = new MetaEmotionInterpreter({
    llmProvider: LLMProvider.OPENAI,
    llmApiKey: process.env.OPENAI_API_KEY,
    llmModel: 'gpt-4-turbo-preview',
    llmTemperature: 0.7
});
```

**Anthropic Integration**:
```javascript
const metaInterpreter = new MetaEmotionInterpreter({
    llmProvider: LLMProvider.ANTHROPIC,
    llmApiKey: process.env.ANTHROPIC_API_KEY,
    llmModel: 'claude-3-5-sonnet-20241022',
    llmTemperature: 0.7
});
```

**Local Ollama Integration**:
```javascript
const metaInterpreter = new MetaEmotionInterpreter({
    llmProvider: LLMProvider.OLLAMA,
    llmModel: 'llama3.2:3b',
    llmTemperature: 0.8
});
```

---

## Performance Characteristics

| Mode | Avg Latency | Use Case |
|------|-------------|----------|
| **Strategy Selection** | ~800ms | Pre-fusion strategy optimization |
| **Nuance Interpretation** | ~1200ms | Rich emotional understanding |
| **Conflict Resolution** | ~1000ms | Contradictory signal handling |
| **Context Adaptation** | ~1400ms | Conversation-aware interpretation |
| **Full Meta-Analysis** | ~2000ms | Comprehensive reasoning |

**Optimization Tips**:
1. Use **Strategy Selection** mode for pre-fusion optimization (fastest)
2. Use **Nuance Interpretation** for real-time interaction (balanced)
3. Use **Full Meta** for asynchronous analysis (offline processing)
4. Enable caching for repeated context patterns
5. Use local Ollama for privacy-sensitive applications (no cloud calls)

---

## Fallback Behavior

The meta-interpreter gracefully degrades when LLM is unavailable:

```javascript
const metaInterpreter = new MetaEmotionInterpreter({
    enabled: true,
    fallbackToRaw: true  // Return fusion result if LLM fails
});

try {
    const result = await metaInterpreter.interpret(fusedResult, signals, context);
    // LLM succeeded
} catch (error) {
    // Falls back to fusedResult automatically
    console.log('Meta-interpretation failed, using raw fusion');
}
```

**Fallback Metrics**:
```javascript
const metrics = metaInterpreter.getMetrics();
console.log('Fallback Rate:', metrics.fallbackRate);
// => 0.03 (3% of interpretations fell back)
```

---

## Metrics and Monitoring

```javascript
const metrics = metaInterpreter.getMetrics();

console.log('Total Interpretations:', metrics.totalInterpretations);
// => 127

console.log('Strategy Selections:', metrics.strategySelections);
// => 15

console.log('Conflict Resolutions:', metrics.conflictResolutions);
// => 8

console.log('Context Adaptations:', metrics.contextAdaptations);
// => 12

console.log('Avg LLM Latency:', metrics.avgLLMLatencyMs);
// => 1243ms

console.log('Fallback Rate:', metrics.fallbackRate);
// => 0.02 (2%)
```

---

## Best Practices

### 1. **Mode Selection Strategy**

```javascript
function selectMetaMode(fusedResult, signals, context) {
    // High conflict → Conflict Resolution
    if (hasHighConflict(signals)) {
        return MetaMode.CONFLICT_RESOLUTION;
    }

    // Long conversation → Context Adaptation
    if (context.conversationTurn > 5) {
        return MetaMode.CONTEXT_ADAPTATION;
    }

    // Low fusion confidence → Strategy Selection
    if (fusedResult.confidence < 0.6) {
        return MetaMode.STRATEGY_SELECTION;
    }

    // Complex emotional state → Nuance Interpretation
    if (fusedResult.emotion.includes('_')) {
        return MetaMode.NUANCE_INTERPRETATION;
    }

    // Default → Full Meta (when latency allows)
    return MetaMode.FULL_META;
}
```

### 2. **Context Enrichment**

Provide rich context for better LLM reasoning:

```javascript
const context = {
    text: transcript,
    conversationTurn: 7,
    userProfile: {
        name: 'Alice',
        emotionalBaseline: 'calm_analytical',
        culturalContext: 'Western professional',
        communicationStyle: 'direct'
    },
    taskContext: {
        type: 'problem_solving',
        urgency: 'high',
        complexity: 'moderate'
    },
    environmentContext: {
        setting: 'remote_video_call',
        timeOfDay: 'afternoon',
        interruptions: 2
    }
};

const result = await metaInterpreter.interpret(fusedResult, signals, context);
```

### 3. **Prompt Customization**

Customize prompts for domain-specific needs:

```javascript
import { PromptTemplates } from './meta_emotion_interpreter.js';

// Add custom prompt
PromptTemplates.CUSTOMER_SERVICE = `
You are analyzing customer emotion in service context.

Customer Emotion: {{current_emotion}}
Service Stage: {{service_stage}}
Issue Severity: {{issue_severity}}
Wait Time: {{wait_time}}

Interpret emotion considering:
1. Frustration may be issue-related, not agent-related
2. Politeness may mask dissatisfaction
3. Relief indicates problem resolution progress

Respond with service-specific recommendations...
`;

// Use in meta-interpreter
// (requires extending MetaEmotionInterpreter for custom modes)
```

### 4. **User Profile Learning**

Track user patterns over time:

```javascript
class UserEmotionProfile {
    constructor(userId) {
        this.userId = userId;
        this.emotionalBaseline = null;
        this.expressionStyle = null;
        this.calibrationData = [];
    }

    update(metaResult) {
        this.calibrationData.push({
            emotion: metaResult.interpretedEmotion,
            confidence: metaResult.confidence,
            timestamp: new Date()
        });

        // Learn baseline over 10+ interactions
        if (this.calibrationData.length > 10) {
            this.emotionalBaseline = this._calculateBaseline();
            this.expressionStyle = this._detectStyle();
        }
    }

    _calculateBaseline() {
        const avgValence = this.calibrationData
            .reduce((sum, d) => sum + d.valence, 0) / this.calibrationData.length;

        if (avgValence > 0.3) return 'positive_baseline';
        if (avgValence < -0.3) return 'negative_baseline';
        return 'neutral_baseline';
    }

    _detectStyle() {
        const genuinenessAvg = this.calibrationData
            .reduce((sum, d) => sum + (d.genuineness || 0), 0) / this.calibrationData.length;

        if (genuinenessAvg > 0.8) return 'expressive';
        if (genuinenessAvg < 0.5) return 'controlled';
        return 'moderate';
    }
}

// Use profile in meta-interpretation
const userProfile = getUserProfile(userId);
metaInterpreter.userProfile = {
    emotionalBaseline: userProfile.emotionalBaseline,
    expressionStyle: userProfile.expressionStyle
};
```

---

## Complete Example: Voice Assistant with Meta-Emotion

```javascript
import { MetaEmotionInterpreter, MetaMode, LLMProvider } from './meta_emotion_interpreter.js';
import { EmotionFusionSystem } from './emotion_fusion.js';
import { FacialEmotionAnalyzer } from './facial_emotion_analyzer.js';
import { VocalProsodyAnalyzer } from './vocal_prosody_analyzer.js';
import { EmotionDetector } from './emotion_detector.js';

class EmotionalVoiceAssistant {
    constructor(config = {}) {
        // Signal analyzers
        this.facialAnalyzer = new FacialEmotionAnalyzer({ backend: 'face-api' });
        this.vocalAnalyzer = new VocalProsodyAnalyzer();
        this.textAnalyzer = new EmotionDetector();

        // Fusion system
        this.fusionSystem = new EmotionFusionSystem({
            fusionStrategy: 'bayesian',
            modalityWeights: { facial: 0.4, vocal: 0.35, text: 0.25 }
        });

        // Meta-interpreter
        this.metaInterpreter = new MetaEmotionInterpreter({
            mode: MetaMode.FULL_META,
            llmProvider: LLMProvider.ANTHROPIC,
            llmApiKey: config.anthropicApiKey,
            llmModel: 'claude-3-5-sonnet-20241022',
            conversationHistorySize: 10
        });

        this.conversationTurn = 0;
    }

    async processInput(videoFrame, audioBuffer, transcript) {
        this.conversationTurn++;

        // 1. Signal-level detection
        const [facialResult, vocalResult, textResult] = await Promise.all([
            this.facialAnalyzer.analyze(videoFrame),
            this.vocalAnalyzer.analyze(audioBuffer),
            this.textAnalyzer.detectEmotion(transcript)
        ]);

        // 2. Fusion
        const signals = [
            { modality: 'facial', data: facialResult },
            { modality: 'vocal', data: vocalResult },
            { modality: 'text', data: textResult }
        ];
        const fusedResult = this.fusionSystem.fuse(signals);

        // 3. Meta-interpretation
        const metaResult = await this.metaInterpreter.interpret(
            fusedResult,
            signals,
            {
                text: transcript,
                conversationTurn: this.conversationTurn,
                userProfile: this.getUserProfile(),
                taskContext: this.getTaskContext()
            }
        );

        // 4. Generate response based on meta-interpretation
        const response = this.generateResponse(metaResult);

        return {
            emotion: metaResult.interpretedEmotion,
            confidence: metaResult.confidence,
            nuance: metaResult.nuanceDescription,
            interactionStrategy: metaResult.interactionStrategy,
            response: response
        };
    }

    generateResponse(metaResult) {
        // Use LLM's interaction strategy to guide response
        const strategy = metaResult.interactionStrategy;

        if (strategy.includes('reassurance')) {
            return this.generateReassuranceResponse(metaResult);
        } else if (strategy.includes('empathetic')) {
            return this.generateEmpatheticResponse(metaResult);
        } else if (strategy.includes('structured')) {
            return this.generateStructuredResponse(metaResult);
        } else {
            return this.generateDefaultResponse(metaResult);
        }
    }

    generateEmpatheticResponse(metaResult) {
        // Example: Acknowledge emotion + offer support
        const emotion = metaResult.interpretedEmotion;
        const recommendations = metaResult.interactionRecommendations || [];

        return {
            acknowledgment: `I understand you're feeling ${emotion}.`,
            support: recommendations[0] || 'How can I help?',
            action: recommendations[1] || 'Let me know what you need.'
        };
    }

    getUserProfile() {
        // Stub - in production, load from database
        return {
            emotionalBaseline: 'calm_analytical',
            communicationStyle: 'direct'
        };
    }

    getTaskContext() {
        // Stub - in production, extract from current task
        return {
            type: 'problem_solving',
            urgency: 'moderate'
        };
    }
}

// Usage
const assistant = new EmotionalVoiceAssistant({
    anthropicApiKey: 'your-api-key'
});

const result = await assistant.processInput(videoFrame, audioBuffer, "I'm not sure what to do.");

console.log('Emotion:', result.emotion);
// => "uncertain_anxious"

console.log('Response:', result.response);
// => {
//      acknowledgment: "I understand you're feeling uncertain and anxious.",
//      support: "Let's break this down into clear steps.",
//      action: "What's the main thing you're uncertain about?"
//    }
```

---

## Summary

The **MetaEmotionInterpreter** adds powerful LLM-based reasoning to emotion analysis:

✅ **5 specialized modes** for different use cases
✅ **Context-aware interpretation** beyond categorical labels
✅ **Conflict resolution** using reasoning
✅ **Conversation adaptation** tracking emotional trajectories
✅ **Actionable interaction strategies** for system responses
✅ **Graceful fallback** when LLM unavailable
✅ **Production-ready** with OpenAI, Anthropic, Ollama support

**Total Enhancement**: Signal detection (facial/vocal/text) → Fusion (6 strategies) → **Meta-reasoning (5 modes)** = **Comprehensive emotional intelligence**
