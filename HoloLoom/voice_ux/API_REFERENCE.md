# Voice UX Emotional Intelligence System - Complete API Reference

**Version:** 110/100 (Meta-Meta-Learning Edition)
**Last Updated:** 2025-11-22

Complete API documentation for all classes, methods, parameters, and return types in the Voice UX Emotional Intelligence System.

---

## Table of Contents

1. [Core Emotion Pipeline](#core-emotion-pipeline)
2. [Breakthrough Features (105/100)](#breakthrough-features-105100)
3. [Cross-User Learning (110/100)](#cross-user-learning-110100)
4. [LLM Integration](#llm-integration)
5. [Configuration](#configuration)
6. [Types & Interfaces](#types--interfaces)

---

## Core Emotion Pipeline

### EmotionalIntelligencePipeline

Main pipeline for processing multimodal emotional inputs.

#### Constructor

```javascript
new EmotionalIntelligencePipeline(config: PipelineConfig)
```

**Parameters:**
- `config` (PipelineConfig): Configuration object

**Example:**
```javascript
const pipeline = new EmotionalIntelligencePipeline(
    PipelineConfig.standard()
);
```

#### Methods

##### `process(input)`

Process multimodal emotional input.

**Parameters:**
- `input` (object):
  - `text` (string): Text input
  - `videoFrame` (Buffer, optional): Video frame for facial detection
  - `audioBuffer` (Buffer, optional): Audio buffer for vocal analysis
  - `context` (object, optional): Contextual information

**Returns:**
- Promise\<EmotionalResult\>:
  - `emotion` (string): Detected emotion
  - `confidence` (number): Confidence score (0-1)
  - `multimodal` (object): Results from each modality
  - `response` (string): Generated response
  - `metadata` (object): Additional processing information

**Example:**
```javascript
const result = await pipeline.process({
    text: "I'm feeling frustrated",
    videoFrame: frameBuffer,
    audioBuffer: audioBuffer,
    context: { activity: 'coding' }
});
```

---

## Breakthrough Features (105/100)

### RecursiveSelfImprovement

Self-improving system that learns from every interaction.

#### Constructor

```javascript
new RecursiveSelfImprovement(config?)
```

**Parameters:**
- `config` (object, optional):
  - `learningRate` (number): Learning rate for weight updates (default: 0.01)
  - `calibrationWindow` (number): Number of interactions for calibration (default: 50)

#### Methods

##### `recordInteraction(interaction)`

Record an interaction for learning.

**Parameters:**
- `interaction` (object):
  - `timestamp` (number): Unix timestamp
  - `emotion` (string): Detected emotion
  - `confidence` (number): Detection confidence (0-1)
  - `modalities` (object): Results from each modality
    - `facial` (object): Facial detection results
    - `vocal` (object): Vocal analysis results
    - `physiological` (object, optional): Physiological signals

**Returns:**
- void

**Example:**
```javascript
selfImprovement.recordInteraction({
    timestamp: Date.now(),
    emotion: 'happy',
    confidence: 0.85,
    modalities: {
        facial: { confidence: 0.8, features: {...} },
        vocal: { confidence: 0.9, features: {...} }
    }
});
```

##### `updateOutcome(outcome, actualEmotion?)`

Update the system with interaction outcome.

**Parameters:**
- `outcome` (string): 'success' | 'failure'
- `actualEmotion` (string, optional): Ground truth emotion

**Returns:**
- object:
  - `successRate` (number): Updated success rate
  - `totalInteractions` (number): Total interactions

**Example:**
```javascript
const stats = selfImprovement.updateOutcome('success', 'happy');
console.log(`Success rate: ${stats.successRate}`);
```

##### `calibrateModalities()`

Calibrate modality weights based on recent performance.

**Returns:**
- object: Updated modality weights
  - `facial` (number): Weight for facial modality
  - `vocal` (number): Weight for vocal modality
  - `physiological` (number): Weight for physiological modality

**Example:**
```javascript
const weights = selfImprovement.calibrateModalities();
console.log('Updated weights:', weights);
```

##### `getLearningMetrics()`

Get comprehensive learning metrics.

**Returns:**
- object:
  - `totalInteractions` (number)
  - `successes` (number)
  - `failures` (number)
  - `successRate` (number)
  - `modalityWeights` (object)
  - `learningCurve` (Array\<number\>)

---

### ChainOfEmotionReasoner

Tracks emotional state transitions over time.

#### Constructor

```javascript
new ChainOfEmotionReasoner(config?)
```

**Parameters:**
- `config` (object, optional):
  - `maxChainLength` (number): Maximum chain length per user (default: 20)
  - `escalationThreshold` (number): Threshold for escalation detection (default: 0.3)

#### Methods

##### `addState(userId, emotionalState)`

Add an emotional state to the user's chain.

**Parameters:**
- `userId` (string): User identifier
- `emotionalState` (object):
  - `emotion` (string): Current emotion
  - `intensity` (number): Emotion intensity (0-1)
  - `valence` (number): Emotional valence (-1 to 1)
  - `arousal` (number): Arousal level (0-1)
  - `dominance` (number): Dominance level (0-1)

**Returns:**
- object: Chain summary

**Example:**
```javascript
chain.addState('user_123', {
    emotion: 'frustrated',
    intensity: 0.7,
    valence: -0.5,
    arousal: 0.8,
    dominance: 0.6
});
```

##### `detectEscalation(userId)`

Detect if emotions are escalating.

**Parameters:**
- `userId` (string): User identifier

**Returns:**
- object:
  - `isEscalating` (boolean)
  - `pattern` (string): 'increasing' | 'stable' | 'decreasing'
  - `delta` (number): Change in intensity

**Example:**
```javascript
const escalation = chain.detectEscalation('user_123');
if (escalation.isEscalating) {
    console.log('Alert: Emotions escalating!');
}
```

##### `getTrajectory(userId)`

Get the emotional trajectory for a user.

**Parameters:**
- `userId` (string): User identifier

**Returns:**
- object:
  - `trajectory` (Array): Sequence of emotional states
  - `dominantEmotion` (string): Most frequent emotion
  - `averageValence` (number): Average valence
  - `volatility` (number): Emotional volatility

**Example:**
```javascript
const trajectory = chain.getTrajectory('user_123');
console.log('Dominant emotion:', trajectory.dominantEmotion);
```

---

### MultiAgentDebateOrchestrator

Multi-agent system for resolving emotional ambiguity.

#### Constructor

```javascript
new MultiAgentDebateOrchestrator(config)
```

**Parameters:**
- `config` (object):
  - `llmClient` (LLMClient): LLM client for agents
  - `maxRounds` (number, optional): Maximum debate rounds (default: 3)
  - `consensusThreshold` (number, optional): Agreement threshold (default: 0.8)

#### Methods

##### `startDebate(emotionalData)`

Start a multi-agent debate to resolve emotional ambiguity.

**Parameters:**
- `emotionalData` (object):
  - `emotion` (string): Initial emotion hypothesis
  - `confidence` (number): Initial confidence
  - `context` (object): Contextual information
  - `multimodalData` (object, optional): Data from multiple modalities

**Returns:**
- Promise\<DebateResult\>:
  - `consensus` (string): Consensus emotion
  - `confidence` (number): Consensus confidence
  - `rounds` (number): Number of debate rounds
  - `agentVotes` (Array): Individual agent conclusions
  - `reasoning` (string): Debate reasoning

**Example:**
```javascript
const result = await debate.startDebate({
    emotion: 'frustrated',
    confidence: 0.6,
    context: { activity: 'debugging' }
});

console.log('Consensus:', result.consensus);
console.log('Confidence:', result.confidence);
```

---

### EmotionalKnowledgeGraph

Per-user knowledge graph of emotional patterns and triggers.

#### Constructor

```javascript
new EmotionalKnowledgeGraph(userId, config?)
```

**Parameters:**
- `userId` (string): User identifier
- `config` (object, optional):
  - `minOccurrences` (number): Minimum pattern occurrences (default: 3)
  - `decayFactor` (number): Pattern decay rate (default: 0.95)

#### Methods

##### `recordExperience(experience)`

Record an emotional experience.

**Parameters:**
- `experience` (object):
  - `emotion` (string): Emotion experienced
  - `trigger` (string): What triggered the emotion
  - `intensity` (number): Intensity (0-1)
  - `context` (object): Additional context
  - `resolution` (string, optional): How it was resolved

**Returns:**
- void

**Example:**
```javascript
kg.recordExperience({
    emotion: 'frustrated',
    trigger: 'compilation_error',
    intensity: 0.7,
    context: { language: 'python', time: 'late_night' },
    resolution: 'took_a_break'
});
```

##### `getTriggers(emotion, minOccurrences?)`

Get triggers for a specific emotion.

**Parameters:**
- `emotion` (string): Emotion to query
- `minOccurrences` (number, optional): Minimum occurrences to include

**Returns:**
- Array\<Trigger\>:
  - `trigger` (string)
  - `count` (number): Number of occurrences
  - `averageIntensity` (number)
  - `contexts` (Array): Associated contexts

**Example:**
```javascript
const triggers = kg.getTriggers('frustrated', 3);
triggers.forEach(t => {
    console.log(`${t.trigger}: ${t.count} times`);
});
```

##### `detectPatterns()`

Detect recurring emotional patterns.

**Returns:**
- Array\<Pattern\>:
  - `pattern` (string): Pattern description
  - `frequency` (number): How often it occurs
  - `sequence` (Array): Emotional sequence
  - `confidence` (number): Pattern confidence

**Example:**
```javascript
const patterns = kg.detectPatterns();
patterns.forEach(p => {
    console.log(`Pattern: ${p.pattern} (${p.frequency}x)`);
});
```

##### `getStatistics()`

Get knowledge graph statistics.

**Returns:**
- object:
  - `totalInteractions` (number)
  - `uniqueEmotions` (number)
  - `uniqueTriggers` (number)
  - `patternsDetected` (number)
  - `graphDensity` (number)

---

### PredictiveEmotionalAnticipation

Predicts future emotional states based on context.

#### Constructor

```javascript
new PredictiveEmotionalAnticipation(config)
```

**Parameters:**
- `config` (object):
  - `userId` (string): User identifier
  - `knowledgeGraph` (EmotionalKnowledgeGraph): User's knowledge graph
  - `chainReasoner` (ChainOfEmotionReasoner, optional): Chain reasoner
  - `predictionWindow` (number, optional): Prediction window in minutes (default: 30)

#### Methods

##### `predict(context, currentEmotion?)`

Predict future emotional state.

**Parameters:**
- `context` (object): Current context
  - `activity` (string, optional)
  - `location` (string, optional)
  - `timeOfDay` (string, optional)
  - `recentEvents` (Array, optional)
- `currentEmotion` (string, optional): Current emotional state

**Returns:**
- Promise\<Prediction\>:
  - `predictedEmotion` (string)
  - `confidence` (number)
  - `timeframe` (string): When prediction applies
  - `triggers` (Array): Likely triggers
  - `preventativeActions` (Array): Suggested interventions

**Example:**
```javascript
const prediction = await predictor.predict({
    activity: 'debugging',
    timeOfDay: 'late_night',
    recentEvents: ['compilation_error', 'compilation_error']
});

console.log(`Prediction: ${prediction.predictedEmotion}`);
console.log('Preventative:', prediction.preventativeActions);
```

##### `validatePrediction(predictionId, actualEmotion)`

Validate a previous prediction.

**Parameters:**
- `predictionId` (number): Prediction ID
- `actualEmotion` (string): Actual emotion that occurred

**Returns:**
- object:
  - `correct` (boolean)
  - `accuracy` (number): Overall prediction accuracy

**Example:**
```javascript
const validation = predictor.validatePrediction(0, 'frustrated');
console.log(`Prediction accuracy: ${validation.accuracy}`);
```

---

## Cross-User Learning (110/100)

### CrossUserLearningSystem

Privacy-preserving cross-user learning system.

#### Constructor

```javascript
new CrossUserLearningSystem(userId, config?)
```

**Parameters:**
- `userId` (string): User identifier
- `config` (object, optional):
  - `privacyLevel` (PrivacyLevel): Privacy level (STRICT | MODERATE | PERMISSIVE)
  - `enableFederatedLearning` (boolean): Enable federated learning (default: true)
  - `enableStrategyMarketplace` (boolean): Enable strategy sharing (default: true)
  - `enableCollectiveIntelligence` (boolean): Enable collective predictions (default: true)
  - `enableMetaMetaLearning` (boolean): Enable meta-meta-learning (default: true)

#### Methods

##### `contributeModelUpdate(localModel, sampleCount, averageConfidence)`

Contribute a local model update to the global model.

**Parameters:**
- `localModel` (object):
  - `modalityWeights` (object): Modality weights
  - `strategyPreferences` (object): Strategy preferences
- `sampleCount` (number): Number of samples in local model
- `averageConfidence` (number): Average confidence of local model

**Returns:**
- Promise\<GlobalModel\>: Updated global model

**Example:**
```javascript
const globalModel = await system.contributeModelUpdate(
    {
        modalityWeights: { facial: 0.7, vocal: 0.8 },
        strategyPreferences: { calm: 0.9, redirect: 0.6 }
    },
    100,
    0.85
);
```

##### `shareStrategy(strategy)`

Share a successful strategy with other users.

**Parameters:**
- `strategy` (StrategyPattern): Strategy to share

**Returns:**
- object:
  - `success` (boolean)
  - `patternId` (string, optional)
  - `reason` (string, optional)

**Example:**
```javascript
const result = system.shareStrategy({
    id: 'strat_001',
    name: 'Break Time Strategy',
    category: 'frustration_management',
    trigger: { emotion: 'frustrated', context: 'coding' },
    action: 'suggest_break',
    quality: 0.9,
    usageCount: 15
});
```

##### `findStrategies(conditions, category?)`

Find strategies for the current situation.

**Parameters:**
- `conditions` (object): Current conditions to match
- `category` (string, optional): Strategy category filter

**Returns:**
- Array\<StrategyPattern\>: Matching strategies

**Example:**
```javascript
const strategies = system.findStrategies(
    { emotion: 'frustrated', context: 'debugging' },
    'frustration_management'
);
```

##### `contributePrediction(predictionId, emotion, confidence)`

Contribute to a collective prediction.

**Parameters:**
- `predictionId` (string): Prediction identifier
- `emotion` (string): Predicted emotion
- `confidence` (number): Prediction confidence

**Returns:**
- CollectivePrediction: Updated collective prediction

**Example:**
```javascript
const collective = system.contributePrediction(
    'pred_001',
    'focused',
    0.8
);
```

##### `recordLearningOutcome(strategyName, context, outcome)`

Record learning outcome for meta-meta-learning.

**Parameters:**
- `strategyName` (string): Strategy used
- `context` (object): Context where used
- `outcome` (object):
  - `success` (boolean)
  - `quality` (number, optional)

**Returns:**
- void

**Example:**
```javascript
system.recordLearningOutcome(
    'break_time_strategy',
    { emotion: 'frustrated', activity: 'coding' },
    { success: true, quality: 0.9 }
);
```

##### `getMetaLearningInsights()`

Get insights from meta-meta-learning system.

**Returns:**
- object:
  - `topStrategies` (Array): Top performing strategies
  - `currentHyperparameters` (object): Optimized hyperparameters
  - `learningProgress` (object): Learning metrics
  - `contextualPatterns` (Array): Discovered patterns

**Example:**
```javascript
const insights = system.getMetaLearningInsights();
console.log('Top strategies:', insights.topStrategies);
console.log('Optimized hyperparameters:', insights.currentHyperparameters);
```

---

### MetaMetaLearningEngine

Learns how to optimize the learning process itself.

#### Constructor

```javascript
new MetaMetaLearningEngine(config?)
```

**Parameters:**
- `config` (object, optional):
  - `minSamplesForOptimization` (number): Minimum samples before optimizing (default: 50)
  - `optimizationInterval` (number): Iterations between optimizations (default: 100)
  - `explorationRate` (number): Exploration rate (default: 0.1)

#### Methods

##### `recordStrategyOutcome(strategyName, context, outcome)`

Record the outcome of using a learning strategy.

**Parameters:**
- `strategyName` (string): Name of strategy
- `context` (object): Context where used
- `outcome` (object):
  - `success` (boolean)
  - `quality` (number, optional)

**Returns:**
- void

##### `recommendStrategy(context)`

Get recommended strategy for current context.

**Parameters:**
- `context` (object): Current context

**Returns:**
- string: Recommended strategy name

**Example:**
```javascript
const recommended = metaLearner.recommendStrategy({
    emotion: 'frustrated',
    activity: 'coding'
});
```

##### `getOptimizedHyperparameters()`

Get current optimized hyperparameters.

**Returns:**
- object:
  - `privacyEpsilon` (number): Privacy budget
  - `aggregationStrategy` (string): Aggregation method
  - `learningRate` (number): Learning rate
  - `minQualityThreshold` (number): Quality threshold

---

## LLM Integration

### LLMClientFactory

Factory for creating LLM clients with automatic fallback.

#### Static Methods

##### `create(config)`

Create an LLM client with automatic provider selection.

**Parameters:**
- `config` (object):
  - `providers` (Array\<string\>): Ordered list of providers to try
  - `apiKeys` (object, optional): API keys for each provider
  - `defaultModel` (object, optional): Default models per provider

**Returns:**
- LLMClient: Configured LLM client

**Example:**
```javascript
const llm = LLMClientFactory.create({
    providers: ['anthropic', 'openai', 'ollama'],
    apiKeys: {
        anthropic: process.env.ANTHROPIC_API_KEY,
        openai: process.env.OPENAI_API_KEY
    },
    defaultModel: {
        anthropic: 'claude-3-5-sonnet-20241022',
        openai: 'gpt-4-turbo-preview'
    }
});
```

---

### AnthropicClient

Anthropic Claude API client.

#### Constructor

```javascript
new AnthropicClient(apiKey, options?)
```

**Parameters:**
- `apiKey` (string): Anthropic API key
- `options` (object, optional):
  - `model` (string): Model name (default: 'claude-3-5-sonnet-20241022')
  - `maxRetries` (number): Max retry attempts (default: 3)

#### Methods

##### `complete(prompt, systemPrompt?, options?)`

Generate a completion.

**Parameters:**
- `prompt` (string): User prompt
- `systemPrompt` (string, optional): System prompt
- `options` (object, optional):
  - `maxTokens` (number): Max tokens to generate
  - `temperature` (number): Sampling temperature (0-1)

**Returns:**
- Promise\<string\>: Generated text

**Example:**
```javascript
const response = await claude.complete(
    "Explain Thompson Sampling",
    "You are a helpful AI assistant",
    { maxTokens: 500, temperature: 0.7 }
);
```

---

### CostTracker

Tracks LLM API costs across providers.

#### Methods

##### `trackRequest(provider, model, inputTokens, outputTokens)`

Track the cost of an API request.

**Parameters:**
- `provider` (string): Provider name
- `model` (string): Model name
- `inputTokens` (number): Number of input tokens
- `outputTokens` (number): Number of output tokens

**Returns:**
- number: Cost in USD

##### `getTotalCost()`

Get total costs.

**Returns:**
- object:
  - `total` (number): Total cost
  - `byProvider` (object): Cost per provider
  - `byModel` (object): Cost per model
  - `requestCount` (number): Total requests

**Example:**
```javascript
const costs = tracker.getTotalCost();
console.log(`Total spent: $${costs.total.toFixed(4)}`);
console.log(`GPT-4 costs: $${costs.byModel['gpt-4'].toFixed(4)}`);
```

---

## Configuration

### PipelineConfig

Configuration presets for the emotion pipeline.

#### Static Methods

##### `minimal()`

Minimal configuration (100/100 baseline).

**Returns:**
- PipelineConfig

##### `standard()`

Standard configuration (105/100 with breakthrough features).

**Returns:**
- PipelineConfig

##### `advanced()`

Advanced configuration (110/100 with cross-user learning).

**Returns:**
- PipelineConfig

**Example:**
```javascript
const config = PipelineConfig.advanced();
const pipeline = new EmotionalIntelligencePipeline(config);
```

---

## Types & Interfaces

### EmotionalResult

```typescript
interface EmotionalResult {
    emotion: string;
    confidence: number;
    multimodal: {
        facial?: FacialResult;
        vocal?: VocalResult;
        physiological?: PhysiologicalResult;
    };
    response: string;
    metadata: {
        processingTime: number;
        modelsUsed: string[];
        [key: string]: any;
    };
}
```

### StrategyPattern

```typescript
interface StrategyPattern {
    id: string;
    name: string;
    category: string;
    trigger: {
        emotion: string;
        context?: object;
    };
    action: string;
    quality: number;
    usageCount: number;
    successRate?: number;
}
```

### PrivacyLevel

```typescript
enum PrivacyLevel {
    STRICT = 'strict',      // Only encrypted gradients, anonymous
    MODERATE = 'moderate',  // Anonymized patterns shared
    PERMISSIVE = 'permissive' // Full collective intelligence
}
```

### CollectivePrediction

```typescript
interface CollectivePrediction {
    emotion: string;
    confidence: number;
    agreement: number;
    contributors: number;
}
```

---

## Usage Patterns

### Basic Emotion Detection

```javascript
const pipeline = new EmotionalIntelligencePipeline(
    PipelineConfig.minimal()
);

const result = await pipeline.process({
    text: "I'm feeling great today!",
    context: { activity: 'morning_coffee' }
});

console.log(`Emotion: ${result.emotion} (${result.confidence})`);
```

### Self-Improving System

```javascript
const selfImprovement = new RecursiveSelfImprovement();
const pipeline = new EmotionalIntelligencePipeline(
    PipelineConfig.standard()
);

// Process interaction
const result = await pipeline.process(input);

// Record for learning
selfImprovement.recordInteraction({
    timestamp: Date.now(),
    emotion: result.emotion,
    confidence: result.confidence,
    modalities: result.multimodal
});

// Update outcome
selfImprovement.updateOutcome('success');

// Calibrate periodically
if (selfImprovement.getInteractions().length % 50 === 0) {
    const weights = selfImprovement.calibrateModalities();
    console.log('Updated weights:', weights);
}
```

### Cross-User Learning

```javascript
const system = new CrossUserLearningSystem('user_123', {
    privacyLevel: PrivacyLevel.MODERATE,
    enableMetaMetaLearning: true
});

// Contribute local model
await system.contributeModelUpdate(
    localModel,
    sampleCount,
    averageConfidence
);

// Find and use shared strategies
const strategies = system.findStrategies(
    { emotion: 'frustrated', activity: 'coding' }
);

// Record outcome for meta-learning
system.recordLearningOutcome(
    strategies[0].name,
    context,
    { success: true, quality: 0.9 }
);
```

---

## Error Handling

All asynchronous methods may throw errors. Recommended error handling:

```javascript
try {
    const result = await pipeline.process(input);
} catch (error) {
    if (error.code === 'RATE_LIMIT_EXCEEDED') {
        // Handle rate limiting
        await delay(1000);
        return retry();
    } else if (error.code === 'INVALID_INPUT') {
        // Handle validation error
        console.error('Invalid input:', error.message);
    } else {
        // Handle unexpected errors
        console.error('Unexpected error:', error);
    }
}
```

---

## Performance Considerations

### Latency Expectations

| Configuration | Avg Latency | Best For |
|---------------|-------------|----------|
| Minimal (100/100) | ~50ms | Real-time applications |
| Standard (105/100) | ~150ms | Balanced performance |
| Advanced (110/100) | ~300ms | Maximum intelligence |

### Memory Usage

| Configuration | Peak Memory |
|---------------|-------------|
| Minimal | ~50 MB |
| Standard | ~120 MB |
| Advanced | ~200 MB |

### Cost Optimization

Use local models (Ollama) for:
- Development and testing
- High-volume, low-complexity queries
- Privacy-critical applications

Use cloud APIs (Anthropic/OpenAI) for:
- Production deployments
- Complex reasoning tasks
- Best accuracy requirements

---

## See Also

- [Quick Start Guide](QUICK_START.md)
- [Breakthrough Features Documentation](BREAKTHROUGH_105_INTEGRATION.md)
- [Performance Benchmarks](benchmarks/README.md)
- [Examples](examples/)

---

**Questions or Issues?**
File an issue or check the examples directory for complete working code.
