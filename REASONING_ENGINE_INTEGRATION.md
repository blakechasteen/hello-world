# Reasoning Engine Integration Guide

**Elegant Integration with Every HoloLoom Layer**

---

## Philosophy

> **"Great systems integrate seamlessly. Great code reads like poetry."**

The Reasoning Engine is designed to integrate gracefully with every layer of HoloLoom, enhancing rather than complicating. This guide shows you how to weave reasoning into your system with elegance and purpose.

---

## Table of Contents

1. [Integration Patterns](#integration-patterns)
2. [Component Integration](#component-integration)
3. [Architectural Patterns](#architectural-patterns)
4. [Best Practices](#best-practices)

---

## Integration Patterns

### Pattern 1: Minimal Integration

**Start simple. Complexity earns its place.**

```python
from HoloLoom.reasoning import auto_reason

result = await auto_reason(query, features, context)
```

That's it. Auto mode selection, verification, provenance.

**When to use**: Prototyping, simple applications, "just works" integrations.

---

### Pattern 2: WeavingOrchestrator Integration

**Full pipeline integration with zero configuration.**

```python
from HoloLoom.config import Config
from HoloLoom.weaving_orchestrator_reasoning import ReasoningOrchestrator
from HoloLoom.documentation.types import Query

config = Config.fused()
config.enable_reasoning = True

async with ReasoningOrchestrator(cfg=config, shards=shards) as orchestrator:
    spacetime = await orchestrator.weave(Query(text="Your question here"))

    chain = spacetime.metadata['reasoning_chain']
    mode = spacetime.metadata['reasoning_mode']
    confidence = spacetime.metadata['reasoning_confidence']
```

**What happens**: Reasoning layer inserted at step 5 of 10-step weaving cycle. Scratchpad provenance tracked automatically. Reasoning chain attached to Spacetime.

**When to use**: Production applications, full HoloLoom integration.

---

### Pattern 3: Middleware Integration

**Compose behaviors like LEGO blocks.**

```python
from HoloLoom.reasoning import ReasoningEngine
from HoloLoom.reasoning.types import ReasoningMode

class MyCustomPipeline:
    def __init__(self):
        self.reasoning = ReasoningEngine(
            mode=ReasoningMode.STANDARD,
            max_thinking_steps=7,
            verification_threshold=0.8
        )

    async def process(self, query, features, context):
        enriched_features = await self.enrich(features)
        reasoning_result = await self.reasoning.reason(query, enriched_features, context)
        final_decision = await self.decide(reasoning_result.chain, reasoning_result.total_confidence)
        return final_decision
```

**When to use**: Custom pipelines, specialized workflows, research.

---

## Component Integration

### Integration 1: Recursive Learning + Reasoning

**Concept**: Reasoning chains become learning signals.

```python
from HoloLoom.recursive import FullLearningEngine
from HoloLoom.reasoning import ReasoningEngine, ReasoningMode

class ReasoningLearningPipeline:
    def __init__(self, cfg, shards):
        self.learning_engine = FullLearningEngine(
            cfg=cfg,
            shards=shards,
            enable_background_learning=True
        )
        self.reasoning_engine = ReasoningEngine(mode=ReasoningMode.STANDARD)

    async def weave_and_learn(self, query, features, context):
        reasoning_result = await self.reasoning_engine.reason(query, features, context)
        spacetime = await self.learning_engine.weave(query)

        if reasoning_result.total_confidence >= 0.8:
            pattern = self._extract_pattern(reasoning_result)
            await self.learning_engine.pattern_learner.learn_pattern(pattern)

        spacetime.metadata['reasoning_chain'] = reasoning_result.chain
        spacetime.metadata['reasoning_confidence'] = reasoning_result.total_confidence

        return spacetime

    def _extract_pattern(self, reasoning_result):
        return {
            'motifs': [s.thought for s in reasoning_result.chain],
            'confidence': reasoning_result.total_confidence,
            'mode': reasoning_result.mode.value,
            'step_types': [s.step_type.value for s in reasoning_result.chain]
        }
```

**Key insight**: High-confidence reasoning chains become training data. Feedback loop between reasoning and learning.

---

### Integration 2: Thompson Sampling Mode Selection

**Concept**: Let the system learn which modes work best.

```python
from HoloLoom.reasoning.bandit import ReasoningModeBandit
from HoloLoom.reasoning import ReasoningEngine

class AdaptiveReasoningPipeline:
    def __init__(self):
        self.engine = ReasoningEngine()
        self.bandit = ReasoningModeBandit()
        self.bandit.load_state("./reasoning_bandit.json")

    async def process(self, query, features, context):
        complexity = self._estimate_complexity(query, features)
        selected_mode = self.bandit.select_mode(complexity)
        result = await self.engine.reason(query, features, context, mode=selected_mode)

        success = result.total_confidence >= 0.75
        self.bandit.update(selected_mode, success, result.total_confidence)

        if self.bandit.total_selections % 100 == 0:
            self.bandit.save_state("./reasoning_bandit.json")

        return result

    def _estimate_complexity(self, query, features):
        return min(1.0, (len(query.text.split()) / 20.0 + len(features.motifs) / 10.0))
```

**Key insight**: Learns optimal mode selection over time. Adapts to your specific use cases. Persists learning across sessions.

---

### Integration 3: Scratchpad Provenance

**Concept**: Every reasoning step becomes traceable.

```python
from HoloLoom.recursive.reasoning_provenance import ReasoningProvenanceTracker
from HoloLoom.recursive import Scratchpad
from HoloLoom.reasoning import ReasoningEngine

class ProvenanceIntegration:
    def __init__(self):
        self.engine = ReasoningEngine()
        self.scratchpad = Scratchpad()
        self.tracker = ReasoningProvenanceTracker()

    async def weave_with_provenance(self, query, features, context):
        result = await self.engine.reason(query, features, context)
        scratchpad_entries = self.tracker.extract_reasoning_provenance(result)

        for entry in scratchpad_entries:
            self.scratchpad.add_entry(entry)

        history = self.scratchpad.get_history()
        return result, history
```

**Scratchpad entry format**:
```python
ScratchpadEntry(
    thought="Query type: factual, requires: definition",
    action="reasoning_step_1",
    observation="Key concepts: thompson, sampling, bayesian",
    score=0.90,
    iteration=1,
    metadata={'mode': 'standard', 'step_type': 'understanding'}
)
```

**Key insight**: Complete audit trail for every decision. Enables debugging and meta-learning.

---

### Integration 4: Memory System Integration

**Concept**: Reasoning informs retrieval; retrieval informs reasoning.

```python
from HoloLoom.memory.backend_factory import create_memory_backend
from HoloLoom.reasoning import ReasoningEngine
from HoloLoom.config import Config, MemoryBackend

class ReasoningMemoryPipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        self.reasoning = ReasoningEngine()

    async def __aenter__(self):
        self.memory = await create_memory_backend(self.cfg)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self.memory, 'close'):
            await self.memory.close()

    async def reason_with_memory(self, query):
        context = await self.memory.retrieve(query.text, top_k=10)
        result = await self.reasoning.reason(query, extract_features(context), context)

        if result.total_confidence < 0.7:
            expanded_query = self._expand_from_reasoning(result)
            context = await self.memory.retrieve(expanded_query, top_k=20)
            result = await self.reasoning.reason(query, extract_features(context), context)

        await self.memory.store_reasoning_chain(query.text, result.chain)
        return result

    def _expand_from_reasoning(self, result):
        key_concepts = [step.thought for step in result.chain if step.step_type == StepType.UNDERSTANDING]
        return " ".join(key_concepts)
```

**Key insight**: Low confidence triggers expanded retrieval. Reasoning chains stored in memory for future use. Bidirectional feedback loop.

---

## Architectural Patterns

### Pattern 1: Layered Reasoning

**Stack reasoning layers for progressive refinement.**

```python
class LayeredReasoning:
    def __init__(self):
        self.fast = ReasoningEngine(mode=ReasoningMode.FAST)
        self.standard = ReasoningEngine(mode=ReasoningMode.STANDARD)
        self.deep = ReasoningEngine(mode=ReasoningMode.DEEP)

    async def reason_layered(self, query, features, context):
        fast_result = await self.fast.reason(query, features, context)

        if fast_result.total_confidence >= 0.9:
            return fast_result

        standard_result = await self.standard.reason(
            query, features, context,
            prior_chain=fast_result.chain
        )

        if standard_result.total_confidence >= 0.85:
            return standard_result

        deep_result = await self.deep.reason(
            query, features, context,
            prior_chain=standard_result.chain
        )

        return deep_result
```

**When to use**: When you need both speed and thoroughness. Fast path for simple queries, deep reasoning only when needed.

---

### Pattern 2: Ensemble Reasoning

**Combine multiple reasoning strategies.**

```python
class EnsembleReasoning:
    def __init__(self):
        self.engines = [
            ReasoningEngine(mode=ReasoningMode.FAST),
            ReasoningEngine(mode=ReasoningMode.STANDARD),
            ReasoningEngine(mode=ReasoningMode.DEEP)
        ]

    async def reason_ensemble(self, query, features, context):
        results = await asyncio.gather(*[
            engine.reason(query, features, context) for engine in self.engines
        ])

        best = max(results, key=lambda r: r.total_confidence)
        return best, results
```

**When to use**: When accuracy matters more than latency. Get multiple perspectives and choose the best.

---

### Pattern 3: Streaming Reasoning

**Stream reasoning steps as they're generated.**

```python
class StreamingReasoning:
    def __init__(self):
        self.engine = ReasoningEngine()

    async def reason_streaming(self, query, features, context):
        async for step in self.engine.reason_stream(query, features, context):
            yield step
            await self.on_step(step)

    async def on_step(self, step):
        print(f"[{step.confidence:.2f}] {step.thought}")
```

**When to use**: Interactive applications, UI feedback, debugging.

---

## Best Practices

### 1. Start Simple, Scale Complexity

```python
# Start with auto_reason
result = await auto_reason(query, features, context)

# Add mode selection when needed
engine = ReasoningEngine(mode=ReasoningMode.STANDARD)
result = await engine.reason(query, features, context)

# Add adaptive selection when patterns emerge
bandit = ReasoningModeBandit()
mode = bandit.select_mode(complexity)
result = await engine.reason(query, features, context, mode=mode)
```

**Principle**: Don't optimize prematurely. Let usage patterns guide complexity.

---

### 2. Use Async Context Managers

```python
# Good: Automatic cleanup
async with ReasoningOrchestrator(cfg=config, shards=shards) as orch:
    spacetime = await orch.weave(query)

# Also good: Explicit cleanup
pipeline = ReasoningMemoryPipeline(config)
try:
    async with pipeline:
        result = await pipeline.reason_with_memory(query)
finally:
    # Resources cleaned up automatically
    pass
```

**Principle**: Lifecycle management prevents resource leaks.

---

### 3. Monitor From Day One

```python
from HoloLoom.performance.reasoning_metrics import track_reasoning, get_reasoning_metrics

with track_reasoning(mode="standard") as tracker:
    result = await engine.reason(query, features, context)
    tracker.set_result(result)

if query_count % 100 == 0:
    metrics = get_reasoning_metrics()
    print(metrics.get_summary())
```

**Principle**: Metrics enable optimization. Track early, optimize later.

---

### 4. Visualize Early, Visualize Often

```python
from HoloLoom.visualization.reasoning_chain import render_from_reasoning_result

result = await engine.reason(query, features, context)

if result.total_confidence < 0.7:
    html = render_from_reasoning_result(result, show_evidence=True)
    Path(f'debug_{query.id}.html').write_text(html)
```

**Principle**: Visual debugging is faster than print debugging.

---

### 5. Degrade Gracefully

```python
async def robust_reasoning(query, features, context):
    try:
        return await auto_reason(query, features, context)
    except Exception as e:
        logger.warning(f"Reasoning failed: {e}, using fallback")
        return FallbackResult(confidence=0.5, chain=[])
```

**Principle**: Never let reasoning failure break the pipeline.

---

## Integration Checklist

**Before Production**:

- [ ] Reasoning enabled in config
- [ ] Mode selection strategy chosen (auto, adaptive, or fixed)
- [ ] Metrics collection integrated
- [ ] Visualization setup for debugging
- [ ] Async context managers used for lifecycle management
- [ ] Fallback strategy defined
- [ ] Performance profiling completed
- [ ] Memory backend integrated (if using dynamic memory)
- [ ] Scratchpad provenance enabled (if needed)
- [ ] Learning integration tested (if using recursive learning)

**Performance Targets**:

| Metric | Target | Action if Missed |
|--------|--------|------------------|
| Avg confidence | >0.80 | Review context quality, try DEEP mode |
| Avg duration | <250ms | Use FAST mode more, check complexity estimation |
| P95 duration | <500ms | Set timeout, enable adaptive mode selection |
| Mode distribution | 60% STANDARD | Review complexity thresholds |

---

## Example: Complete Production Integration

```python
from HoloLoom.config import Config
from HoloLoom.weaving_orchestrator_reasoning import ReasoningOrchestrator
from HoloLoom.memory.backend_factory import create_memory_backend
from HoloLoom.performance.reasoning_metrics import track_reasoning, get_reasoning_metrics
from HoloLoom.documentation.types import Query
import logging

logger = logging.getLogger(__name__)

async def production_pipeline():
    config = Config.fused()
    config.enable_reasoning = True
    config.enable_adaptive_reasoning = True
    config.max_reasoning_time_ms = 500.0

    memory = await create_memory_backend(config)

    async with ReasoningOrchestrator(cfg=config, shards=shards, memory=memory) as orch:
        for query in queries:
            try:
                with track_reasoning() as tracker:
                    spacetime = await orch.weave(query)
                    tracker.set_result(spacetime.metadata['reasoning_result'])

                    if spacetime.confidence < 0.7:
                        logger.warning(f"Low confidence: {spacetime.confidence:.2f} for query: {query.text[:50]}")

            except TimeoutError:
                logger.error(f"Timeout for query: {query.text[:50]}")
            except Exception as e:
                logger.error(f"Error processing query: {e}")

            if query_count % 100 == 0:
                metrics = get_reasoning_metrics()
                logger.info(f"Metrics: {metrics.get_summary()}")

    await memory.close()
```

---

## Next Steps

1. **Try the patterns**: Start with Pattern 1 (minimal), then explore others
2. **Read EXTENSIBILITY.md**: Learn how to build custom reasoning components
3. **Read ARCHITECTURE.md**: Understand the system internals
4. **Experiment**: The best integration is the one that fits your needs

---

**"Great systems integrate seamlessly. Great code reads like poetry."**
