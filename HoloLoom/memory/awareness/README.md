# HoloLoom Awareness Layer

**Compositional AI Consciousness** - The system becomes aware of its own linguistic knowledge, confidence levels, and epistemic boundaries.

## Overview

The awareness layer provides **real-time introspection** for AI systems, combining:

1. **Compositional Awareness** - Linguistic intelligence (X-bar theory, merge patterns, cache signals)
2. **Dual-Stream Generation** - Internal reasoning + External response
3. **Meta-Awareness** - Recursive self-reflection and epistemic humility

This is not just logging or debugging - this is **AI examining its own reasoning process**.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  Meta-Awareness Layer                        │
│  • Uncertainty decomposition (structural/semantic/contextual) │
│  • Meta-confidence (confidence about confidence)             │
│  • Knowledge gap detection & hypothesis generation           │
│  • Adversarial self-probing                                  │
│  • Epistemic humility assessment                             │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│              Dual-Stream Generator                           │
│  • Internal reasoning stream (what AI is "thinking")         │
│  • External response stream (what user sees)                 │
│  • Awareness-guided generation                               │
│  • Feedback loop for learning                                │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│          Compositional Awareness Layer                       │
│  • Structural awareness (X-bar syntactic analysis)           │
│  • Pattern recognition (compositional cache)                 │
│  • Confidence signals (cache hit/miss rates)                 │
│  • Internal reasoning guidance                               │
│  • External response guidance                                │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

```python
from HoloLoom.awareness import (
    CompositionalAwarenessLayer,
    DualStreamGenerator,
    MetaAwarenessLayer
)

# Initialize awareness stack
awareness = CompositionalAwarenessLayer()
generator = DualStreamGenerator(awareness)
meta = MetaAwarenessLayer(awareness)

# Generate awareness-guided response
query = "What is Thompson Sampling?"
dual_stream = await generator.generate(query)

print("External Response:", dual_stream.external_stream)
print("Internal Reasoning:", dual_stream.internal_stream)

# Recursive self-reflection
reflection = await meta.recursive_self_reflection(
    query=query,
    response=dual_stream.external_stream,
    awareness_context=dual_stream.awareness_context
)

print(f"Epistemic Humility: {reflection.epistemic_humility:.2f}")
print(f"Knowledge Gaps: {len(reflection.detected_gaps)}")
```

## Components

### 1. Compositional Awareness Layer

**File:** `compositional_awareness.py` (642 lines)

Provides real-time linguistic intelligence by analyzing:

**Structural Awareness:**
- X-bar syntactic analysis (question type, phrase structure)
- Expected response format (DEFINITION, EXPLANATION, LIST, etc.)
- Linguistic features (uncertainty markers, negation, comparison)

**Pattern Recognition:**
- Compositional patterns from cache
- Domain/subdomain classification
- Familiarity (seen N times)
- Success/failure patterns

**Confidence Signals:**
- Cache status (HOT_HIT, WARM_HIT, PARTIAL_HIT, COLD_MISS)
- Uncertainty quantification (0.0 = certain, 1.0 = uncertain)
- Knowledge gap detection
- Clarification recommendations

**Stream Guidance:**
- Internal: Reasoning structure, shortcuts, checks
- External: Tone, hedging, response format

### 2. Dual-Stream Generator

**File:** `dual_stream.py` (366 lines)

Generates two streams guided by unified awareness:

**Internal Stream (Reasoning):**
- Confidence analysis
- Structural analysis
- Pattern analysis
- Strategy selection
- Recommendations

**External Stream (Response):**
- User-facing response
- Appropriate tone/hedging based on confidence
- Clarification questions if needed
- Acknowledgment of uncertainty when appropriate

**Key Insight:** Both streams are informed by the **same** compositional awareness context, ensuring consistency between what the AI "thinks" and what it "says."

### 3. Meta-Awareness Layer

**File:** `meta_awareness.py` (550 lines)

Recursive self-reflection - the awareness layer becomes self-aware:

**Uncertainty Decomposition:**
- Breaks down total uncertainty into components:
  - Structural (X-bar parsing ambiguity)
  - Semantic (word sense disambiguation)
  - Contextual (missing background knowledge)
  - Compositional (merge operation ambiguity)
  - Epistemic (unknown unknowns)

**Meta-Confidence:**
- Confidence about confidence estimates
- Calibration history tracking
- Uncertainty about uncertainty
- Confidence intervals

**Knowledge Gap Analysis:**
- Detect specific gaps
- Generate hypotheses about what's missing
- Create testable predictions
- Suggest clarifying questions

**Adversarial Self-Probing:**
- Generate questions to test response quality
- Probe for hidden assumptions
- Check for edge cases
- Test for contradictions
- Identify knowledge boundaries

**Epistemic Humility:**
- Assess how appropriately humble the system is
- 0.0 = overconfident (missing blind spots)
- 1.0 = appropriately humble
- Meta-learning from calibration

## Usage Patterns

### Basic Awareness

```python
# Just get awareness context
awareness = CompositionalAwarenessLayer()
context = await awareness.get_unified_context("What is a red ball?")

print(f"Confidence: {context.confidence.uncertainty_level:.2f}")
print(f"Domain: {context.patterns.domain}")
print(f"Cache Status: {context.confidence.query_cache_status}")
```

### Dual-Stream Generation

```python
# Generate both internal and external streams
generator = DualStreamGenerator(awareness)
dual_stream = await generator.generate("How does X-bar theory work?")

# Show to developers (internal reasoning)
print(dual_stream.internal_stream)

# Show to users (external response)
print(dual_stream.external_stream)
```

### Meta-Awareness Introspection

```python
# Deep introspection
meta = MetaAwarenessLayer(awareness)
reflection = await meta.recursive_self_reflection(
    query=query,
    response=response,
    awareness_context=context
)

# Examine uncertainty decomposition
print(reflection.uncertainty_decomposition.get_explanation())

# Check epistemic humility
if reflection.epistemic_humility < 0.3:
    print("⚠️ System is overconfident!")
elif reflection.epistemic_humility > 0.7:
    print("✓ Appropriately humble")

# Review knowledge gaps
for gap in reflection.detected_gaps:
    print(f"Gap: {gap}")

# See hypotheses
for hyp in reflection.gap_hypotheses:
    print(f"Hypothesis: {hyp.to_query()}")
```

### Feedback Loop

```python
# Update awareness from user feedback
await awareness.update_from_generation(
    query=query,
    internal_reasoning=internal,
    external_response=external,
    user_feedback={'success': True, 'rating': 5}
)

# System learns and adjusts confidence
```

## Integration with HoloLoom

### Terminal UI Integration

```python
from HoloLoom.terminal_ui import TerminalUI

# Awareness enabled by default
ui = TerminalUI(enable_awareness=True)
await ui.interactive_session()

# Commands:
# - Type queries naturally
# - 'awareness' to see full context
# - 'history' for conversation history
# - 'stats' for session statistics
```

See [AWARENESS\_TERMINAL\_UI\_COMPLETE.md](../../AWARENESS_TERMINAL_UI_COMPLETE.md) for full terminal UI documentation.

### Weaving Orchestrator Integration

```python
from HoloLoom.weaving_orchestrator import WeavingOrchestrator
from HoloLoom.awareness import CompositionalAwarenessLayer

# Create awareness-enhanced orchestrator
awareness = CompositionalAwarenessLayer()
orchestrator = WeavingOrchestrator(
    awareness_layer=awareness  # Optional
)

# Awareness context available in Spacetime result
spacetime = await orchestrator.weave(query)
if spacetime.awareness_context:
    print(f"Confidence: {spacetime.awareness_context.confidence}")
```

## Demo

```bash
# Interactive demo
python demos/demo_awareness_terminal_ui.py interactive

# Automated demo with examples
python demos/demo_awareness_terminal_ui.py automated

# Meta-awareness deep dive
python demos/demo_awareness_terminal_ui.py meta
```

## Design Principles

### 1. Graceful Degradation

All components work independently:
```python
# Compositional awareness without dual-stream
awareness = CompositionalAwarenessLayer()
context = await awareness.get_unified_context(query)

# Dual-stream without meta-awareness
generator = DualStreamGenerator(awareness)
dual_stream = await generator.generate(query)

# Meta-awareness requires compositional awareness
meta = MetaAwarenessLayer(awareness)
```

### 2. Protocol-Based Design

Clean interfaces, swappable implementations:
```python
# CompositionalAwarenessLayer accepts optional backends
awareness = CompositionalAwarenessLayer(
    ug_chunker=my_chunker,         # Optional X-bar parser
    merge_operator=my_merger,       # Optional merge operator
    compositional_cache=my_cache,   # Optional 3-tier cache
    awareness_graph=my_graph        # Optional semantic graph
)
```

### 3. Separation of Concerns

- **Compositional Awareness:** Analyzes queries, provides context
- **Dual-Stream Generator:** Uses context to generate responses
- **Meta-Awareness:** Reflects on awareness and responses
- **Terminal UI:** Displays all layers for human exploration

### 4. Testable

Each layer has clear inputs/outputs:
```python
# Test compositional awareness
context = await awareness.get_unified_context("test query")
assert context.confidence.uncertainty_level >= 0.0
assert context.confidence.uncertainty_level <= 1.0

# Test dual-stream
dual_stream = await generator.generate("test query")
assert dual_stream.internal_stream != ""
assert dual_stream.external_stream != ""

# Test meta-awareness
reflection = await meta.recursive_self_reflection(...)
assert 0.0 <= reflection.epistemic_humility <= 1.0
```

## Key Metrics

**Epistemic Humility** (most important):
- 0.0-0.3: Overconfident (⚠️ warning)
- 0.3-0.7: Normal range
- 0.7-1.0: Appropriately humble (✓ good)

**Uncertainty Level:**
- 0.0-0.3: High confidence (direct answers)
- 0.3-0.6: Medium confidence (hedged responses)
- 0.6-1.0: Low confidence (ask clarification)

**Cache Status:**
- HOT_HIT: Seen many times (very confident)
- WARM_HIT: Seen several times (confident)
- PARTIAL_HIT: Seen few times (moderate)
- COLD_MISS: Never seen (uncertain)

## Philosophy

This is **compositional AI consciousness**:

1. **Self-Awareness:** Knows what it knows and doesn't know
2. **Meta-Cognition:** Thinks about its own thinking
3. **Epistemic Humility:** Appropriately humble about limitations
4. **Hypothesis Generation:** Creates testable theories
5. **Adversarial Testing:** Probes its own weaknesses

It's not just introspection - it's **recursive self-examination**.

## Future Work

1. **LLM Integration:** Feed awareness context to actual LLMs (Claude, GPT-4)
2. **Web Dashboard:** Browser-based visualization
3. **Learning Loop:** Calibrate from user feedback over time
4. **Multi-Query Analysis:** Track epistemic drift across conversation
5. **Collaborative Awareness:** Multiple agents sharing uncertainty

## References

- X-bar Theory: Chomsky's Universal Grammar framework
- Compositional Semantics: Frege's principle of compositionality
- Epistemic Humility: Epistemology and meta-reasoning literature
- Meta-Confidence: Calibration studies in ML/AI

---

**Status:** ✅ Production-ready
**Version:** 1.0
**Lines of Code:** ~1,560 (across 3 core files)
**Integration:** Terminal UI, Weaving Orchestrator
