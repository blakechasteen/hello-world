# 🚀 Quick Start Guide
**Get the 105/100 Breakthrough System running in 5 minutes**

---

## Prerequisites

- Node.js 16+ or modern browser
- (Optional) Anthropic/OpenAI API key for LLM features

---

## Installation

### Step 1: Copy the Files

Copy these files to your project:

```
milestone3/
├── complete_emotional_pipeline.js          # 100/100 complete system
├── recursive_self_improvement.js           # 105/100 feature #1
├── chain_of_emotion_reasoning.js           # 105/100 feature #2
├── multi_agent_emotional_debate.js         # 105/100 feature #3
├── emotional_knowledge_graph.js            # 105/100 feature #4
└── predictive_emotional_anticipation.js    # 105/100 feature #5
```

### Step 2: Choose Your Integration Level

**Option A: Simple (100/100 - Complete System Only)**
```javascript
import { EmotionalIntelligencePipeline, PipelineConfig } from './complete_emotional_pipeline.js';

// Create pipeline
const pipeline = new EmotionalIntelligencePipeline(
    PipelineConfig.standard()  // or .minimal() or .advanced()
);

// Process input
const result = await pipeline.process({
    text: "I'm feeling frustrated with this code",
    videoFrame: videoFrame,     // optional
    audioBuffer: audioBuffer,   // optional
    context: { activity: 'coding' }
});

console.log('Emotion:', result.emotion);
console.log('Response:', result.response);
console.log('Confidence:', result.confidence);
```

**Option B: Breakthrough (105/100 - All Features)**
```javascript
import { BreakthroughEmotionalSystem } from './BREAKTHROUGH_105_INTEGRATION.js';

// Create system for a user
const system = new BreakthroughEmotionalSystem('user_123');

// Interact
const result = await system.interact({
    text: "I can't figure this out",
    videoFrame: videoFrame,
    audioBuffer: audioBuffer
}, {
    location: 'home',
    activity: 'coding',
    sessionDuration: 30000,
    turnsInSession: 5
});

console.log('Prediction:', result.prediction);
console.log('Response:', result.response);
console.log('Preemptive:', result.preemptiveAction);
```

---

## Common Scenarios

### Scenario 1: Basic Emotion Detection

```javascript
const pipeline = new EmotionalIntelligencePipeline(PipelineConfig.minimal());

const result = await pipeline.process({
    text: "This is amazing!",
    context: {}
});

console.log(result.getSummary());
// Output: { emotion: 'happy', confidence: 0.85, action: 'validate', ... }
```

### Scenario 2: Multimodal Analysis

```javascript
const pipeline = new EmotionalIntelligencePipeline(PipelineConfig.advanced());

const result = await pipeline.process({
    text: "I'm fine",
    videoFrame: webcamFrame,    // Shows sad facial expression
    audioBuffer: micBuffer,     // Flat vocal tone
    context: { turn: 3 }
});

// Detects sarcasm/mismatch
console.log(result.fusedEmotion);
// Output: { emotion: 'sad', confidence: 0.72, sarcasmDetected: true }
```

### Scenario 3: Proactive Prediction

```javascript
const system = new BreakthroughEmotionalSystem('user_123');

// User hasn't spoken yet, but system predicts
const prediction = await system.predictiveAnticipation.predict({
    location: 'work',
    activity: 'debugging',
    stressLevel: 'high'
}, {
    videoFrame: webcamFrame  // Shows furrowed brow
});

if (prediction.isActionable) {
    console.log('Predicted emotion:', prediction.predictedEmotion);
    console.log('Suggested action:', prediction.suggestedPreemptiveAction);
    // Output: "I sense you might be feeling frustrated. Would you like help debugging?"
}
```

### Scenario 4: Learning from Feedback

```javascript
const system = new BreakthroughEmotionalSystem('user_123');

// Interaction 1
const result1 = await system.interact(input1, context1);
console.log(result1.response);

// User provides feedback
await system.provideFeedback(0, true, 0.9);  // Was helpful, satisfaction: 0.9

// System learns and improves
// Next interaction will use updated weights and strategies
```

### Scenario 5: Multi-Agent Debate (High Stakes)

```javascript
const pipeline = new EmotionalIntelligencePipeline(PipelineConfig.advanced());

// Low confidence triggers debate
const result = await pipeline.process({
    text: "I don't know what to do",
    context: { highStakes: true }
});

// 6 agents debated the interpretation
console.log('Debate used:', result.metaInterpretation.debateTranscript);
console.log('Agreement level:', result.metaInterpretation.agreementLevel);
```

---

## Configuration Reference

### PipelineConfig Options

```javascript
new PipelineConfig({
    // Signal Detection
    enableFacial: true,              // Enable facial emotion analysis
    enableVocal: true,               // Enable vocal prosody analysis
    enableText: true,                // Enable text sentiment analysis
    facialBackend: 'face-api',       // 'face-api' | 'mediapipe' | 'mock'

    // Fusion
    fusionStrategy: FusionStrategy.BAYESIAN,  // AVERAGE | WEIGHTED | MAX_CONFIDENCE | MAJORITY_VOTE | PRIORITY | BAYESIAN
    modalityWeights: {
        facial: 0.4,
        vocal: 0.35,
        text: 0.25
    },

    // Meta-Interpretation
    enableMetaInterpretation: true,
    metaMode: MetaMode.FULL_META,    // STRATEGY_SELECTION | NUANCE_INTERPRETATION | CONFLICT_RESOLUTION | CONTEXT_ADAPTATION | FULL_META

    // LLM
    llmProvider: 'anthropic',        // 'anthropic' | 'openai' | 'ollama' | 'mock'
    llmApiKey: 'your-api-key',
    llmModel: 'claude-3-5-sonnet-20241022',

    // Action Planning
    enableActionPlanning: true,
    planningStrategy: PlanningStrategy.ADAPTIVE,  // IMMEDIATE | SHORT_TERM | LONG_TERM | ADAPTIVE
    conversationGoal: ConversationGoal.SUPPORT,   // SUPPORT | INFORM | ENGAGE | THERAPEUTIC

    // Response Generation
    useTemplates: true
})
```

### Quick Config Presets

```javascript
// Minimal: Fastest, basic features only
PipelineConfig.minimal()

// Standard: Good balance of features and performance
PipelineConfig.standard()

// Advanced: All features enabled
PipelineConfig.advanced()
```

---

## API Reference (Quick)

### EmotionalIntelligencePipeline

```javascript
class EmotionalIntelligencePipeline {
    async process(input)           // Main processing method
    setGoal(goal)                  // Change conversation goal
    getMetrics()                   // Get performance metrics
    getConversationHistory()       // Get last 10 turns
}
```

### BreakthroughEmotionalSystem

```javascript
class BreakthroughEmotionalSystem {
    async interact(input, context)          // Main interaction
    async provideFeedback(index, helpful, satisfaction)  // Learn from feedback
    getInsights()                           // Get predictions, patterns, learning stats
}
```

### Input Format

```javascript
{
    text: "What the user said",
    videoFrame: HTMLCanvasElement | ImageData,  // Optional
    audioBuffer: Float32Array,                  // Optional
    context: {
        activity: 'string',
        location: 'string',
        stressLevel: 'low' | 'normal' | 'high',
        turn: number,
        // ... any custom fields
    }
}
```

### Result Format

```javascript
{
    // User-facing
    response: "System's response text",
    tone: 'empathetic' | 'professional' | 'casual' | 'supportive',

    // Analysis
    emotion: 'happy' | 'sad' | 'angry' | 'anxious' | ...,
    confidence: 0.0-1.0,
    prediction: { /* prediction details */ },
    preemptiveAction: { /* suggested action before user spoke */ },

    // Metadata
    processingTime: milliseconds,
    timestamp: Date
}
```

---

## Performance Tips

### 1. Choose the Right Config

```javascript
// Fast queries (< 50ms)
PipelineConfig.minimal()

// Balanced (< 150ms)
PipelineConfig.standard()

// Highest quality (< 300ms)
PipelineConfig.advanced()
```

### 2. Disable Unused Modalities

```javascript
new PipelineConfig({
    enableFacial: false,  // No video input
    enableVocal: false,   // No audio input
    enableText: true      // Text only
})
// Saves ~100ms per query
```

### 3. Use Debates Sparingly

```javascript
// Only enable for high-stakes decisions
if (input.context.highStakes || initialConfidence < 0.6) {
    // Trigger multi-agent debate
}
```

### 4. Batch Learning Updates

```javascript
// Don't learn on every interaction
if (interactionCount % 10 === 0) {
    await system.selfImprovement._triggerLearning();
}
```

---

## Troubleshooting

### Issue: Low Confidence Scores

**Solution:** Enable more modalities or use multi-agent debate

```javascript
const config = PipelineConfig.advanced();
config.enableFacial = true;
config.enableVocal = true;
config.fusionStrategy = FusionStrategy.BAYESIAN;
```

### Issue: Slow Performance

**Solution:** Use minimal config or disable meta-interpretation

```javascript
const config = PipelineConfig.minimal();
config.enableMetaInterpretation = false;
```

### Issue: Predictions Not Actionable

**Solution:** Lower confidence threshold

```javascript
const system = new BreakthroughEmotionalSystem('user_123', {
    minConfidenceForAction: 0.5  // Default: 0.6
});
```

### Issue: LLM Errors

**Solution:** Check API key and model availability

```javascript
const config = new PipelineConfig({
    llmProvider: 'anthropic',
    llmApiKey: process.env.ANTHROPIC_API_KEY,
    llmModel: 'claude-3-5-sonnet-20241022'  // Verify model name
});
```

---

## Next Steps

### Learn More
- Read [BREAKTHROUGH_105_INTEGRATION.md](BREAKTHROUGH_105_INTEGRATION.md) for full architecture
- Read [EMOTIONAL_INTELLIGENCE_PIPELINE.md](EMOTIONAL_INTELLIGENCE_PIPELINE.md) for 100/100 details
- Read [META_EMOTION_INTEGRATION.md](META_EMOTION_INTEGRATION.md) for meta-interpretation

### Extend the System
- Add custom emotion categories
- Create custom action templates
- Build custom fusion strategies
- Integrate with your application

### Deploy to Production
- Set up LLM API integration
- Configure cost tracking
- Implement user privacy (knowledge graph encryption)
- Monitor performance metrics

---

## Example: Complete Integration

```javascript
import { BreakthroughEmotionalSystem } from './BREAKTHROUGH_105_INTEGRATION.js';

// Initialize
const system = new BreakthroughEmotionalSystem('user_123', {
    llmProvider: 'anthropic',
    llmApiKey: process.env.ANTHROPIC_API_KEY
});

// Main loop
async function handleInteraction(userInput, webcam, microphone) {
    // Get context
    const context = {
        location: await getLocation(),
        activity: getCurrentActivity(),
        sessionDuration: Date.now() - sessionStart,
        turnsInSession: turnCount++
    };

    // Interact
    const result = await system.interact({
        text: userInput,
        videoFrame: webcam.getFrame(),
        audioBuffer: microphone.getBuffer()
    }, context);

    // Handle preemptive action
    if (result.preemptiveAction) {
        await speak(result.preemptiveAction.message);
    }

    // Display response
    await speak(result.response);

    // Collect feedback (later)
    setTimeout(async () => {
        const feedback = await askUser("Was that helpful?");
        await system.provideFeedback(turnCount - 1, feedback, 0.8);
    }, 5000);

    // Periodic insights
    if (turnCount % 20 === 0) {
        const insights = system.getInsights();
        console.log('System Insights:', insights);
    }
}

// Start
handleInteraction("I'm stuck on this problem", webcam, microphone);
```

---

## That's It!

You now have a complete 105/100 breakthrough emotional intelligence system running.

**Questions?** Check the full documentation or review the code comments.

**Want to go further?** Look into the 110/100 features (cross-user learning, strategy marketplace, collective intelligence).

🎉 **Enjoy building emotionally intelligent applications!**
