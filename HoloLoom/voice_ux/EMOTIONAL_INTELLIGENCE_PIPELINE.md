# Complete Emotional Intelligence Pipeline

**Date**: November 2025
**Purpose**: End-to-end emotional understanding with metaprompting-based action planning

---

## Overview

The **Emotional Intelligence Pipeline** is a complete system for understanding and responding to human emotions through multimodal analysis and LLM-powered reasoning.

```
Input (voice + video + text)
         ↓
┌────────────────────────────────────────────────┐
│ 1. SIGNAL DETECTION                            │
│    • Facial Emotion (Ekman + FACS)            │
│    • Vocal Prosody (F0, intensity, style)     │
│    • Text Sentiment (emotion + intensity)      │
└────────────────────────────────────────────────┘
         ↓
┌────────────────────────────────────────────────┐
│ 2. EMOTION FUSION                              │
│    • 6 strategies (average, weighted, etc.)    │
│    • Conflict detection                        │
│    • VAD space projection                      │
└────────────────────────────────────────────────┘
         ↓
┌────────────────────────────────────────────────┐
│ 3. META-INTERPRETATION (LLM)                   │
│    • Contextual nuance understanding           │
│    • Conflict resolution via reasoning         │
│    • Conversation-aware adaptation             │
└────────────────────────────────────────────────┘
         ↓
┌────────────────────────────────────────────────┐
│ 4. NEXT STEPS PLANNING (LLM)                   │
│    • Action selection based on emotion         │
│    • Multi-step conversation strategies        │
│    • Safety intervention detection             │
└────────────────────────────────────────────────┘
         ↓
┌────────────────────────────────────────────────┐
│ 5. RESPONSE GENERATION                         │
│    • Template-based responses                  │
│    • Tone-appropriate messaging                │
│    • Emotionally-aligned interaction           │
└────────────────────────────────────────────────┘
         ↓
Output (response + action plan)
```

---

## Quick Start

### Basic Usage

```javascript
import { EmotionalIntelligencePipeline, PipelineConfig } from './complete_emotional_pipeline.js';

// Initialize with standard configuration
const pipeline = new EmotionalIntelligencePipeline(PipelineConfig.standard());

// Process multimodal input
const result = await pipeline.process({
    videoFrame: cameraFrame,    // Video frame (optional)
    audioBuffer: micBuffer,     // Audio buffer (optional)
    text: "I'm not sure what to do...",  // Transcribed text
    context: {
        topic: 'project deadline',
        situation: 'feeling overwhelmed'
    }
});

// Get response
console.log('Response:', result.response);
// => "I can see you're feeling uncertain about project deadline.
//     That's completely understandable. What would help you feel
//     more comfortable right now?"

console.log('Emotion:', result.metaInterpretation.interpretedEmotion);
// => "uncertain_overwhelmed"

console.log('Action:', result.actionPlan.immediateAction.action);
// => "ask_open"
```

---

## Configuration Modes

### 1. Minimal Mode

**Use Case**: Fast processing, no LLM calls

```javascript
const pipeline = new EmotionalIntelligencePipeline(PipelineConfig.minimal());

// Only signal detection + fusion (no meta-interpretation or planning)
// Latency: ~100ms
```

**Components Active**:
- ✅ Signal Detection
- ✅ Emotion Fusion (simple averaging)
- ❌ Meta-Interpretation
- ❌ Action Planning

### 2. Standard Mode

**Use Case**: Balanced performance + intelligence

```javascript
const pipeline = new EmotionalIntelligencePipeline(PipelineConfig.standard());

// Signal + fusion + nuance interpretation + immediate action
// Latency: ~1.4s (LLM calls)
```

**Components Active**:
- ✅ Signal Detection
- ✅ Emotion Fusion (Bayesian)
- ✅ Meta-Interpretation (nuance only)
- ✅ Action Planning (immediate actions)

### 3. Advanced Mode (Full Intelligence)

**Use Case**: Maximum understanding, complex conversations

```javascript
const pipeline = new EmotionalIntelligencePipeline(PipelineConfig.advanced());

// All components active with full reasoning
// Latency: ~2.5s (multiple LLM calls)
```

**Components Active**:
- ✅ Signal Detection
- ✅ Emotion Fusion (Bayesian)
- ✅ Meta-Interpretation (full meta-analysis)
- ✅ Action Planning (adaptive multi-step)

### 4. Custom Configuration

```javascript
const pipeline = new EmotionalIntelligencePipeline({
    // Signal detection
    enableFacial: true,
    enableVocal: true,
    enableText: true,
    facialBackend: 'face-api',

    // Fusion
    fusionStrategy: FusionStrategy.WEIGHTED,
    modalityWeights: {
        facial: 0.5,   // Trust facial more
        vocal: 0.3,
        text: 0.2
    },

    // Meta-interpretation
    enableMetaInterpretation: true,
    metaMode: MetaMode.CONFLICT_RESOLUTION,

    // LLM
    llmProvider: LLMProvider.OPENAI,
    llmApiKey: 'your-api-key',
    llmModel: 'gpt-4-turbo-preview',

    // Planning
    enableActionPlanning: true,
    planningStrategy: PlanningStrategy.SHORT_TERM,
    conversationGoal: ConversationGoal.PROBLEM_SOLVE,

    // Response generation
    useTemplates: true
});
```

---

## Complete Example: Emotional Voice Assistant

```javascript
import { EmotionalIntelligencePipeline, PipelineConfig } from './complete_emotional_pipeline.js';
import { ConversationGoal } from './next_steps_pipeline.js';
import { LLMProvider } from './meta_emotion_interpreter.js';

class EmotionalVoiceAssistant {
    constructor(config = {}) {
        this.pipeline = new EmotionalIntelligencePipeline({
            llmProvider: LLMProvider.ANTHROPIC,
            llmApiKey: config.anthropicApiKey,
            llmModel: 'claude-3-5-sonnet-20241022',
            conversationGoal: ConversationGoal.SUPPORT
        });

        this.conversationActive = false;
    }

    async startConversation(goal = ConversationGoal.SUPPORT) {
        this.pipeline.setGoal(goal);
        this.conversationActive = true;
        console.log('[Assistant] Conversation started with goal:', goal);
    }

    async processUserInput(videoFrame, audioBuffer, transcript) {
        if (!this.conversationActive) {
            throw new Error('No active conversation - call startConversation() first');
        }

        // Extract context from transcript (could be more sophisticated)
        const context = this._extractContext(transcript);

        // Process through pipeline
        const result = await this.pipeline.process({
            videoFrame: videoFrame,
            audioBuffer: audioBuffer,
            text: transcript,
            context: context
        });

        // Log for debugging
        console.log('[Assistant] Processed:', result.getSummary());

        // Check for intervention needs
        if (result.actionPlan?.interventionNeeded) {
            await this._handleIntervention(result.actionPlan);
        }

        return {
            response: result.response,
            tone: result.tone,
            emotion: result.metaInterpretation?.interpretedEmotion,
            confidence: result.confidence,
            shouldContinue: !result.actionPlan?.immediateAction?.action.includes('close')
        };
    }

    _extractContext(transcript) {
        // Simple keyword-based context extraction
        // In production, use more sophisticated NLP

        const context = {
            topic: 'general',
            situation: 'conversation'
        };

        // Detect topics
        if (transcript.toLowerCase().includes('work') ||
            transcript.toLowerCase().includes('job')) {
            context.topic = 'work';
        } else if (transcript.toLowerCase().includes('relationship') ||
                   transcript.toLowerCase().includes('family')) {
            context.topic = 'relationships';
        } else if (transcript.toLowerCase().includes('health') ||
                   transcript.toLowerCase().includes('feeling')) {
            context.topic = 'health';
        }

        // Detect situation
        if (transcript.toLowerCase().includes('deadline') ||
            transcript.toLowerCase().includes('urgent')) {
            context.situation = 'time pressure';
        } else if (transcript.toLowerCase().includes('overwhelm') ||
                   transcript.toLowerCase().includes('too much')) {
            context.situation = 'feeling overwhelmed';
        } else if (transcript.toLowerCase().includes('unsure') ||
                   transcript.toLowerCase().includes('don\'t know')) {
            context.situation = 'uncertainty';
        }

        return context;
    }

    async _handleIntervention(actionPlan) {
        console.warn('[Assistant] INTERVENTION NEEDED:', actionPlan.interventionType);
        console.warn('[Assistant] Urgency:', actionPlan.urgency);
        console.warn('[Assistant] Reasoning:', actionPlan.reasoning);

        // In production:
        // - Escalate to human operator
        // - Contact emergency services if needed
        // - Log incident for review
    }

    async endConversation() {
        const metrics = this.pipeline.getMetrics();
        const history = this.pipeline.getConversationHistory();

        this.conversationActive = false;

        return {
            metrics: metrics,
            history: history,
            summary: this._generateConversationSummary(history)
        };
    }

    _generateConversationSummary(history) {
        const turns = history.length;
        const emotions = history.map(h => h.emotion).filter(e => e);
        const avgConfidence = history.reduce((sum, h) => sum + h.confidence, 0) / turns;

        return {
            totalTurns: turns,
            emotionsEncountered: [...new Set(emotions)],
            averageConfidence: avgConfidence.toFixed(2),
            duration: history.length > 0
                ? (history[history.length - 1].timestamp - history[0].timestamp) / 1000
                : 0
        };
    }

    getMetrics() {
        return this.pipeline.getMetrics();
    }
}

// Usage
const assistant = new EmotionalVoiceAssistant({
    anthropicApiKey: 'your-api-key'
});

await assistant.startConversation(ConversationGoal.SUPPORT);

// Process turn 1
const response1 = await assistant.processUserInput(
    videoFrame1,
    audioBuffer1,
    "I'm really stressed about this project deadline."
);

console.log('Assistant:', response1.response);
// => "I can see you're feeling stressed about this project deadline.
//     That would be stressful for anyone, especially when time is tight.
//     What would help you feel more comfortable right now?"

// Process turn 2
const response2 = await assistant.processUserInput(
    videoFrame2,
    audioBuffer2,
    "I'm not sure where to start. There's so much to do."
);

console.log('Assistant:', response2.response);
// => "I hear that feeling of being overwhelmed. Let's break this down
//     into manageable steps. First, what's the most important thing
//     we need to address?"

// End conversation
const summary = await assistant.endConversation();
console.log('Summary:', summary);
// => {
//      totalTurns: 2,
//      emotionsEncountered: ['stressed', 'overwhelmed_uncertain'],
//      averageConfidence: '0.83',
//      duration: 45.2
//    }
```

---

## Pipeline Components

### 1. Signal Detection Layer

**Files**:
- `facial_emotion_analyzer.js` (~650 lines)
- `vocal_prosody_analyzer.js` (~750 lines)
- `emotion_detector.js` (~517 lines)

**Capabilities**:
- **Facial**: 7 Ekman emotions, 15 FACS Action Units, head pose, gaze tracking
- **Vocal**: F0 extraction, prosody features, speaking style, voice quality, paralinguistics
- **Text**: Sentiment analysis, intensity, keyword detection

### 2. Emotion Fusion Layer

**File**: `emotion_fusion.js` (~700 lines)

**Strategies**:
1. **Average**: Simple mean across modalities
2. **Weighted**: Confidence × modality weight
3. **Max Confidence**: Trust most confident signal
4. **Majority Vote**: Democratic consensus
5. **Priority**: Hierarchical (facial > vocal > text)
6. **Bayesian**: Probabilistic integration

**Features**:
- VAD space projection (Valence-Arousal-Dominance)
- Conflict detection
- Sarcasm detection (text/vocal mismatch)
- Temporal emotion tracking

### 3. Meta-Interpretation Layer

**File**: `meta_emotion_interpreter.js` (~890 lines)

**Modes**:
1. **Strategy Selection**: LLM chooses fusion strategy (~800ms)
2. **Nuance Interpretation**: Rich emotional understanding (~1200ms)
3. **Conflict Resolution**: Resolve contradictory signals (~1000ms)
4. **Context Adaptation**: Conversation-aware adjustment (~1400ms)
5. **Full Meta-Analysis**: Comprehensive reasoning (~2000ms)

**Features**:
- LLM-powered reasoning (OpenAI, Anthropic, Ollama)
- Conversation history tracking
- User profile adaptation
- Calibration suggestions

### 4. Action Planning Layer

**File**: `next_steps_pipeline.js` (~650 lines)

**Planning Strategies**:
1. **Immediate**: Single action (~800ms)
2. **Short-Term**: 2-3 step plan (~1.2s)
3. **Long-Term**: Full conversation arc (~1.5s)
4. **Adaptive**: Dynamic strategy selection (~varies)

**Action Types** (15 total):
- **Immediate**: acknowledge, validate, reassure, empathize, clarify
- **Information**: ask_open, ask_closed, probe
- **Problem Solving**: suggest, brainstorm, plan, execute
- **Emotional**: de_escalate, reframe, encourage
- **Escalation**: escalate_human, emergency

**Safety Features**:
- Intervention detection (self-harm, crisis keywords)
- Emotional trajectory prediction
- Goal alignment verification

### 5. Response Generation Layer

**File**: `action_templates.js** (~580 lines)

**Template Categories**:
- Acknowledgment (4 emotions)
- Validation (3 emotions)
- Reassurance (2 emotions)
- Empathy (3 emotions)
- Clarification (2 emotions)
- Open questions (3 emotions)
- Suggestions (3 emotions)
- De-escalation (2 emotions)
- Reframing (2 emotions)
- Encouragement (2 emotions)
- Escalation (2 types)

**Conversation Flows** (pre-built):
- Support anxious user (4 steps)
- Resolve frustration (5 steps)
- De-escalate anger (4 steps)
- Build confidence (4 steps)

---

## Performance Characteristics

| Configuration | Latency | Use Case |
|---------------|---------|----------|
| **Minimal** | ~100ms | Real-time, no LLM |
| **Standard** | ~1.4s | Balanced |
| **Advanced** | ~2.5s | Maximum intelligence |
| **Custom (immediate)** | ~1.0s | Fast planning |
| **Custom (short-term)** | ~1.5s | Multi-step planning |

**Breakdown (Standard Mode)**:
- Signal Detection: ~50ms (parallel)
- Emotion Fusion: ~5ms
- Meta-Interpretation (nuance): ~1200ms
- Action Planning (immediate): ~800ms
- Response Generation: ~2ms
- **Total**: ~1405ms

---

## Metrics & Monitoring

```javascript
const metrics = pipeline.getMetrics();

console.log('Pipeline Metrics:', metrics.pipeline);
// => {
//      totalProcessed: 42,
//      avgProcessingTimeMs: 1387,
//      avgConfidence: 0.81,
//      modalityUsage: { facial: 42, vocal: 42, text: 42 }
//    }

console.log('Fusion Metrics:', metrics.fusion);
// => {
//      totalFusions: 42,
//      fusionsByStrategy: { bayesian: 42 },
//      conflictsDetected: 3,
//      avgFusionConfidence: 0.78
//    }

console.log('Meta-Interpreter Metrics:', metrics.metaInterpreter);
// => {
//      totalInterpretations: 42,
//      avgLLMLatencyMs: 1243,
//      fallbackRate: 0.02
//    }

console.log('Action Planner Metrics:', metrics.actionPlanner);
// => {
//      totalPlans: 42,
//      interventionsDetected: 1,
//      planAdjustments: 5,
//      avgConfidence: 0.83
//    }
```

---

## Best Practices

### 1. Start Minimal, Scale Up

```javascript
// Development: minimal mode
const devPipeline = new EmotionalIntelligencePipeline(PipelineConfig.minimal());

// Staging: standard mode
const stagingPipeline = new EmotionalIntelligencePipeline(PipelineConfig.standard());

// Production: advanced mode
const prodPipeline = new EmotionalIntelligencePipeline(PipelineConfig.advanced());
```

### 2. Handle Missing Modalities Gracefully

```javascript
// Process with any combination of modalities
const result = await pipeline.process({
    videoFrame: null,           // No camera
    audioBuffer: micBuffer,     // Vocal only
    text: transcript,
    context: {}
});
// Pipeline automatically adapts
```

### 3. Set Appropriate Goals

```javascript
// Customer support
pipeline.setGoal(ConversationGoal.SUPPORT);

// Technical troubleshooting
pipeline.setGoal(ConversationGoal.PROBLEM_SOLVE);

// Educational context
pipeline.setGoal(ConversationGoal.EDUCATION);
```

### 4. Monitor Intervention Needs

```javascript
const result = await pipeline.process(input);

if (result.actionPlan?.interventionNeeded) {
    const urgency = result.actionPlan.urgency;

    if (urgency === 'critical') {
        // Immediate escalation
        await escalateToEmergency(result);
    } else if (urgency === 'high') {
        // Human operator
        await escalateToHuman(result);
    }
}
```

### 5. Provide Rich Context

```javascript
await pipeline.process({
    videoFrame: frame,
    audioBuffer: audio,
    text: transcript,
    context: {
        // Domain context
        topic: 'work stress',
        situation: 'approaching deadline',

        // User profile
        userPatience: 'low',
        complexityTolerance: 'moderate',

        // Task context
        timeConstraint: 'urgent',
        previousAttempts: 2,

        // Environmental
        setting: 'remote_call',
        timeOfDay: 'late_night'
    }
});
```

---

## Production Deployment

### Environment Variables

```bash
# LLM Configuration
export LLM_PROVIDER=anthropic
export ANTHROPIC_API_KEY=your-key
export LLM_MODEL=claude-3-5-sonnet-20241022

# Pipeline Mode
export PIPELINE_MODE=standard  # or minimal, advanced

# Conversation Goal
export DEFAULT_GOAL=support

# Logging
export LOG_LEVEL=info
```

### Error Handling

```javascript
try {
    const result = await pipeline.process(input);
    // Success
} catch (error) {
    if (error.message.includes('No signals detected')) {
        // Handle missing modalities
        console.error('At least one modality required');
    } else if (error.message.includes('LLM')) {
        // LLM failure - fall back to fusion only
        const fusionOnly = await pipeline.process({
            ...input,
            enableMetaInterpretation: false,
            enableActionPlanning: false
        });
    } else {
        // Unknown error
        console.error('Pipeline error:', error);
    }
}
```

### Caching Strategies

```javascript
// Cache LLM responses for common patterns
const responseCache = new Map();

async function processWithCache(input) {
    const cacheKey = JSON.stringify({
        emotion: input.emotion,
        goal: input.goal,
        context: input.context
    });

    if (responseCache.has(cacheKey)) {
        return responseCache.get(cacheKey);
    }

    const result = await pipeline.process(input);
    responseCache.set(cacheKey, result);

    return result;
}
```

---

## Architecture Summary

**Total System**:
- **Lines of Code**: ~5,300
- **Components**: 8 major systems
- **LLM Integration**: 3 providers (OpenAI, Anthropic, Ollama)
- **Processing Layers**: 5 (signal → fusion → meta → planning → response)
- **Emotion Capabilities**: 7 base + unlimited nuanced combinations
- **Action Types**: 15
- **Templates**: 40+ pre-built responses

**Key Innovation**: Combining signal-level emotion detection with LLM-powered reasoning to create truly contextual emotional intelligence—understanding not just *what* someone feels, but *why* they feel it and *how* to respond appropriately.
