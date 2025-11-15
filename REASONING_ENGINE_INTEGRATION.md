# Reasoning Engine Integration Guide

**HoloLoom Layer 6: Elegant Integration Patterns**

*Focus: Integration • Extensibility • Elegance*

---

## Philosophy

> **"Great systems integrate seamlessly. Great code reads like poetry."**

The Reasoning Engine is designed to integrate gracefully with every layer of HoloLoom, enhancing rather than complicating. This guide shows you how to weave reasoning into your system with elegance and purpose.

---

## Table of Contents

1. [Integration Patterns](#integration-patterns)
2. [Component Integration](#component-integration)
3. [Extensibility](#extensibility)
4. [Architectural Patterns](#architectural-patterns)
5. [Best Practices](#best-practices)

---

## Integration Patterns

### Pattern 1: Minimal Integration (The 3-Line Integration)

**Philosophy**: *Start simple. Complexity earns its place.*

```python
from HoloLoom.reasoning import auto_reason

result = await auto_reason(query, features, context)
# That's it. Auto mode selection, verification, provenance.
```

**What you get**:
- ✅ Automatic mode selection (FAST/STANDARD/DEEP)
- ✅ Multi-step reasoning
- ✅ Self-verification
- ✅ Complete provenance

**When to use**: Prototyping, simple applications, "just works" integrations.

---

### Pattern 2: WeavingOrchestrator Integration (Production Ready)

**Philosophy**: *Full pipeline integration with zero configuration.*

```python
from HoloLoom.config import Config
from HoloLoom.weaving_orchestrator_reasoning import ReasoningOrchestrator
from HoloLoom.documentation.types import Query, MemoryShard

# 1. Configure once
config = Config.fused()
config.enable_reasoning = True
config.reasoning_mode = ReasoningMode.STANDARD

# 2. Create orchestrator
shards = [MemoryShard(...), MemoryShard(...)]

async with ReasoningOrchestrator(cfg=config, shards=shards) as orchestrator:
    # 3. Weave
    spacetime = await orchestrator.weave(Query(text="Your question here"))

    # 4. Access reasoning chain
    chain = spacetime.metadata['reasoning_chain']
    mode = spacetime.metadata['reasoning_mode']
    confidence = spacetime.metadata['reasoning_confidence']
```

**What you get**:
- ✅ Complete 10-step weaving cycle (reasoning inserted at step 5)
- ✅ Scratchpad provenance automatically tracked
- ✅ Reasoning chain attached to Spacetime
- ✅ Async context manager lifecycle
- ✅ Graceful cleanup

**When to use**: Production applications, full HoloLoom integration.

---

### Pattern 3: Middleware Integration (Maximum Flexibility)

**Philosophy**: *Compose behaviors like LEGO blocks.*

```python
from HoloLoom.reasoning import ReasoningEngine
from HoloLoom.reasoning.types import ReasoningMode

class MyCustomPipeline:
    def __init__(self):
        # Reasoning as middleware
        self.reasoning = ReasoningEngine(
            mode=ReasoningMode.STANDARD,
            max_thinking_steps=7,  # Custom config
            verification_threshold=0.8
        )

    async def process(self, query, features, context):
        # Pre-processing
        enriched_features = await self.enrich(features)

        # Reasoning layer
        reasoning_result = await self.reasoning.reason(
            query, enriched_features, context
        )

        # Post-processing with reasoning insights
        final_decision = await self.decide(
            reasoning_result.chain,
            reasoning_result.total_confidence
        )

        return final_decision
```

**What you get**:
- ✅ Complete control over pipeline
- ✅ Custom pre/post processing
- ✅ Reasoning as composable middleware
- ✅ Integration with existing systems

**When to use**: Custom pipelines, specialized workflows, research.

---

## Component Integration

### Integration 1: Recursive Learning + Reasoning

**Concept**: *Reasoning chains become learning signals.*

```python
from HoloLoom.recursive import FullLearningEngine
from HoloLoom.reasoning import ReasoningEngine, ReasoningMode

class ReasobningLearningPipeline:
    """Combines reasoning with recursive learning."""

    def __init__(self, cfg, shards):
        # Learning engine tracks patterns
        self.learning_engine = FullLearningEngine(
            cfg=cfg,
            shards=shards,
            enable_background_learning=True
        )

        # Reasoning engine generates chains
        self.reasoning_engine = ReasoningEngine(
            mode=ReasoningMode.STANDARD
        )

    async def weave_and_learn(self, query, features, context):
        # 1. Reason about query
        reasoning_result = await self.reasoning_engine.reason(
            query, features, context
        )

        # 2. Process with learning engine
        spacetime = await self.learning_engine.weave(query)

        # 3. Extract learning signals from reasoning chain
        if reasoning_result.total_confidence >= 0.8:
            # High-confidence reasoning becomes training data
            pattern = self._extract_pattern(reasoning_result)
            await self.learning_engine.pattern_learner.learn_pattern(pattern)

        # 4. Attach reasoning to spacetime
        spacetime.metadata['reasoning_chain'] = reasoning_result.chain
        spacetime.metadata['reasoning_confidence'] = reasoning_result.total_confidence

        return spacetime

    def _extract_pattern(self, reasoning_result):
        """Convert reasoning chain to learned pattern."""
        return {
            'motifs': [s.thought for s in reasoning_result.chain],
            'confidence': reasoning_result.total_confidence,
            'mode': reasoning_result.mode.value,
            'step_types': [s.step_type.value for s in reasoning_result.chain]
        }
```

**Benefits**:
- Reasoning chains improve over time
- Pattern extraction from successful reasoning
- Feedback loop between reasoning and learning

---

### Integration 2: Thompson Sampling Mode Selection

**Concept**: *Let the system learn which modes work best.*

```python
from HoloLoom.reasoning.bandit import ReasoningModeBandit
from HoloLoom.reasoning import ReasoningEngine

class AdaptiveReasoningPipeline:
    """Automatically learns optimal mode selection."""

    def __init__(self):
        self.engine = ReasoningEngine()
        self.bandit = ReasoningModeBandit()
        self.bandit.load_state("./reasoning_bandit.json")  # Resume learning

    async def process(self, query, features, context):
        # 1. Estimate query complexity
        complexity = self._estimate_complexity(query, features)

        # 2. Thompson Sampling selects mode
        selected_mode = self.bandit.select_mode(complexity)

        # 3. Reason with selected mode
        result = await self.engine.reason(
            query, features, context, mode=selected_mode
        )

        # 4. Update bandit based on outcome
        success = result.total_confidence >= 0.75
        self.bandit.update(selected_mode, success, result.total_confidence)

        # 5. Periodically save learning state
        if self.bandit.total_selections % 100 == 0:
            self.bandit.save_state("./reasoning_bandit.json")

        return result

    def _estimate_complexity(self, query, features):
        """Estimate query complexity [0.0, 1.0]."""
        # Simple heuristic
        return min(1.0, (len(query.text.split()) / 20.0 + len(features.motifs) / 10.0))
```

**Benefits**:
- Learns optimal mode selection over time
- Adapts to your specific use cases
- Persists learning across sessions

---

### Integration 3: Scratchpad Provenance

**Concept**: *Every reasoning step becomes traceable.*

```python
from HoloLoom.recursive.reasoning_provenance import ReasoningProvenanceTracker
from Promptly.promptly.recursive_loops import Scratchpad
from HoloLoom.reasoning import ReasoningEngine

class ProvenanceIntegration:
    """Full provenance tracking for reasoning."""

    def __init__(self):
        self.engine = ReasoningEngine()
        self.scratchpad = Scratchpad()
        self.tracker = ReasoningProvenanceTracker()

    async def weave_with_provenance(self, query, features, context):
        # 1. Reason
        result = await self.engine.reason(query, features, context)

        # 2. Extract provenance
        scratchpad_entries = self.tracker.extract_reasoning_provenance(result)

        # 3. Add to scratchpad
        for entry in scratchpad_entries:
            self.scratchpad.add_entry(entry)

        # 4. View complete history
        history = self.scratchpad.get_history()

        return result, history
```

**Scratchpad Entry Format**:
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

---

### Integration 4: Memory System Integration

**Concept**: *Reasoning informs retrieval; retrieval informs reasoning.*

```python
from HoloLoom.memory.backend_factory import create_memory_backend
from HoloLoom.reasoning import ReasoningEngine
from HoloLoom.config import Config, MemoryBackend

class ReasoningMemoryPipeline:
    """Bidirectional reasoning-memory integration."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.reasoning = ReasoningEngine()

    async def __aenter__(self):
        # Create persistent memory backend
        self.memory = await create_memory_backend(self.cfg)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Cleanup
        if hasattr(self.memory, 'close'):
            await self.memory.close()

    async def weave(self, query):
        # 1. Retrieve initial context from memory
        context = await self.memory.retrieve(query.text, top_k=10)

        # 2. Reason about query
        reasoning_result = await self.reasoning.reason(
            query,
            features=self._extract_features(context),
            context=context
        )

        # 3. Use reasoning insights to refine retrieval
        if reasoning_result.total_confidence < 0.7:
            # Low confidence → expand search
            expanded_query = self._expand_from_reasoning(reasoning_result)
            context = await self.memory.retrieve(expanded_query, top_k=20)

            # Re-reason with expanded context
            reasoning_result = await self.reasoning.reason(
                query,
                features=self._extract_features(context),
                context=context
            )

        # 4. Store reasoning chain in memory for future use
        await self.memory.store_reasoning_chain(
            query=query.text,
            chain=reasoning_result.chain,
            confidence=reasoning_result.total_confidence
        )

        return reasoning_result

    def _expand_from_reasoning(self, reasoning_result):
        """Extract expansion terms from reasoning chain."""
        # Collect key concepts from high-confidence steps
        expansion_terms = []
        for step in reasoning_result.chain:
            if step.confidence >= 0.8:
                # Extract key terms from evidence
                expansion_terms.extend(step.evidence.split()[:5])

        return " ".join(expansion_terms)
```

---

## Extensibility

### Custom Component 1: Custom Verifier

**Use Case**: *Domain-specific verification logic.*

```python
from HoloLoom.reasoning.verifier import SelfVerifier
from HoloLoom.reasoning.types import VerificationResult, VerificationSeverity
from typing import List

class MedicalVerifier(SelfVerifier):
    """Custom verifier for medical domain."""

    def __init__(self, threshold: float = 0.75):
        super().__init__(threshold)

        # Domain-specific requirements
        self.required_evidence_types = ['clinical_study', 'peer_review']
        self.forbidden_keywords = ['cure', 'guaranteed', 'miracle']

    async def verify(self, chain, context) -> VerificationResult:
        # 1. Run base verification
        base_result = await super().verify(chain, context)

        if not base_result.passed:
            return base_result

        # 2. Domain-specific checks
        domain_result = self._verify_medical_standards(chain, context)

        return domain_result

    def _verify_medical_standards(self, chain, context):
        """Medical domain verification."""
        # Check 1: Evidence quality
        has_clinical_evidence = any(
            'clinical' in step.evidence.lower() or 'study' in step.evidence.lower()
            for step in chain
        )

        if not has_clinical_evidence:
            return VerificationResult(
                passed=False,
                issue="No clinical evidence found in chain",
                correction="Add references to peer-reviewed studies",
                severity=VerificationSeverity.CRITICAL,
                confidence=0.9
            )

        # Check 2: Forbidden keywords
        forbidden_found = []
        for step in chain:
            for keyword in self.forbidden_keywords:
                if keyword in step.thought.lower():
                    forbidden_found.append(keyword)

        if forbidden_found:
            return VerificationResult(
                passed=False,
                issue=f"Forbidden medical keywords found: {forbidden_found}",
                correction="Remove unsubstantiated claims",
                severity=VerificationSeverity.CRITICAL,
                confidence=0.95
            )

        # All checks passed
        return VerificationResult(passed=True, confidence=0.95)
```

**Usage**:
```python
from HoloLoom.reasoning import ReasoningEngine

engine = ReasoningEngine()
engine.verifier = MedicalVerifier(threshold=0.8)  # Inject custom verifier
```

---

### Custom Component 2: Custom Query Planner

**Use Case**: *Domain-specific query decomposition.*

```python
from HoloLoom.reasoning.planner import QueryPlanner
from HoloLoom.reasoning.types import QueryPlan, PlanStep, QueryType

class ResearchQueryPlanner(QueryPlanner):
    """Custom planner for academic research queries."""

    def create_plan(self, query, features, context):
        intent = self.analyze_intent(query, features)

        # Research-specific plan
        if self._is_research_query(query.text):
            return self._create_research_plan(query, intent)

        # Fallback to base planner
        return super().create_plan(query, features, context)

    def _is_research_query(self, text):
        """Detect research queries."""
        research_keywords = ['literature review', 'state of the art', 'survey', 'meta-analysis']
        return any(keyword in text.lower() for keyword in research_keywords)

    def _create_research_plan(self, query, intent):
        """Create multi-stage research plan."""
        steps = [
            PlanStep(
                question="What is the research question?",
                required_for="Scope definition",
                complexity=0.3
            ),
            PlanStep(
                question="What are the key papers in this area?",
                required_for="Literature foundation",
                complexity=0.6
            ),
            PlanStep(
                question="What methodologies are used?",
                required_for="Methodological understanding",
                complexity=0.7
            ),
            PlanStep(
                question="What are the research gaps?",
                required_for="Identifying open questions",
                complexity=0.8
            ),
            PlanStep(
                question="How do findings relate?",
                required_for="Synthesis and connections",
                complexity=0.9
            ),
            PlanStep(
                question="What is the current consensus?",
                required_for="Final synthesis",
                complexity=0.5
            ),
        ]

        return QueryPlan(
            steps=steps,
            estimated_complexity=0.8,
            dependencies={
                1: [0],  # Key papers depend on research question
                2: [1],  # Methodologies depend on papers
                3: [1, 2],  # Gaps depend on papers and methods
                4: [1, 2, 3],  # Relations depend on everything
                5: [4],  # Consensus depends on relations
            }
        )
```

---

### Custom Component 3: Custom Chain-of-Thought Generator

**Use Case**: *Specialized reasoning for specific domains.*

```python
from HoloLoom.reasoning.chain_of_thought import ChainOfThought
from HoloLoom.reasoning.types import ReasoningStep, StepType, Synthesis

class MathematicalChainOfThought(ChainOfThought):
    """Chain-of-thought for mathematical reasoning."""

    def generate_standard_chain(self, query, intent, context):
        """Generate math-specific reasoning chain."""
        chain = []

        # Step 1: Problem understanding (with equation extraction)
        equations = self._extract_equations(query.text)
        chain.append(ReasoningStep(
            thought=f"Mathematical problem: {len(equations)} equations found",
            evidence=f"Equations: {', '.join(equations[:3])}",
            confidence=0.95,
            step_type=StepType.UNDERSTANDING
        ))

        # Step 2: Strategy selection
        strategy = self._select_strategy(equations)
        chain.append(ReasoningStep(
            thought=f"Solving strategy: {strategy}",
            evidence=f"Best approach for {len(equations)} equations",
            confidence=0.9,
            step_type=StepType.PLANNING
        ))

        # Step 3: Step-by-step solving
        solution_steps = self._solve(equations, strategy)
        for i, step in enumerate(solution_steps):
            chain.append(ReasoningStep(
                thought=f"Step {i+1}: {step['description']}",
                evidence=step['work'],
                confidence=step['confidence'],
                step_type=StepType.SYNTHESIS
            ))

        # Step 4: Verification (check answer)
        verification = self._verify_solution(equations, solution_steps)
        chain.append(ReasoningStep(
            thought=f"Verification: {verification['status']}",
            evidence=verification['check'],
            confidence=verification['confidence'],
            step_type=StepType.VERIFICATION
        ))

        return chain

    def _extract_equations(self, text):
        """Extract mathematical equations from text."""
        import re
        # Simple regex for equations (extend as needed)
        return re.findall(r'[\d\w\+\-\*/\(\)=\s]+', text)

    def _select_strategy(self, equations):
        """Select solving strategy."""
        if len(equations) == 1:
            return "direct_solving"
        elif len(equations) == 2:
            return "substitution"
        else:
            return "matrix_methods"

    def _solve(self, equations, strategy):
        """Solve equations (placeholder)."""
        return [
            {
                'description': 'Isolate variable',
                'work': 'x = (b - c) / a',
                'confidence': 0.95
            }
        ]

    def _verify_solution(self, equations, steps):
        """Verify solution."""
        return {
            'status': 'correct',
            'check': 'Substitution back into original equation',
            'confidence': 0.98
        }
```

---

## Architectural Patterns

### Pattern: Layered Reasoning (Progressive Enhancement)

**Concept**: *Stack reasoning layers for progressive refinement.*

```
Query
  ↓
[FAST Reasoning] ────────→ High confidence? → Done ✓
  ↓ Low confidence
[STANDARD Reasoning] ────→ Acceptable? → Done ✓
  ↓ Still low
[DEEP Reasoning] ────────→ Final answer
```

**Implementation**:
```python
class LayeredReasoningPipeline:
    """Progressive reasoning with fallback layers."""

    def __init__(self):
        self.fast = ReasoningEngine(mode=ReasoningMode.FAST)
        self.standard = ReasoningEngine(mode=ReasoningMode.STANDARD)
        self.deep = ReasoningEngine(mode=ReasoningMode.DEEP)

    async def reason(self, query, features, context):
        # Layer 1: FAST
        result = await self.fast.reason(query, features, context)
        if result.total_confidence >= 0.85:
            return result

        # Layer 2: STANDARD
        result = await self.standard.reason(query, features, context)
        if result.total_confidence >= 0.75:
            return result

        # Layer 3: DEEP (last resort)
        return await self.deep.reason(query, features, context)
```

---

### Pattern: Ensemble Reasoning (Multi-Model Voting)

**Concept**: *Run multiple reasoning strategies and vote.*

```python
class EnsembleReasoning:
    """Ensemble of multiple reasoning strategies."""

    def __init__(self):
        self.engines = [
            ReasoningEngine(mode=ReasoningMode.STANDARD),
            ReasoningEngine(mode=ReasoningMode.STANDARD, max_thinking_steps=7),
            ReasoningEngine(mode=ReasoningMode.DEEP),
        ]

    async def reason(self, query, features, context):
        # 1. Run all engines in parallel
        results = await asyncio.gather(*[
            engine.reason(query, features, context)
            for engine in self.engines
        ])

        # 2. Weighted voting (confidence-weighted)
        weighted_confidence = sum(
            r.total_confidence for r in results
        ) / len(results)

        # 3. Select best reasoning chain
        best_result = max(results, key=lambda r: r.total_confidence)

        # 4. Attach ensemble metadata
        best_result.metadata['ensemble_results'] = [
            {'mode': r.mode.value, 'confidence': r.total_confidence}
            for r in results
        ]
        best_result.metadata['ensemble_confidence'] = weighted_confidence

        return best_result
```

---

## Best Practices

### 1. Start Simple, Scale Complexity

```python
# ❌ Don't start with complexity
engine = ReasoningEngine(
    mode=ReasoningMode.DEEP,
    max_thinking_steps=20,
    verification_threshold=0.95,
    enable_backtracking=True,
    enable_multi_pass=True,
    # ... 15 more parameters
)

# ✅ Do start simple
result = await auto_reason(query, features, context)
# Let the system decide. Add complexity only when needed.
```

---

### 2. Use Context Managers for Lifecycle

```python
# ❌ Manual cleanup (error-prone)
orchestrator = ReasoningOrchestrator(cfg=config, shards=shards)
try:
    result = await orchestrator.weave(query)
finally:
    await orchestrator.close()  # Easy to forget

# ✅ Context manager (automatic cleanup)
async with ReasoningOrchestrator(cfg=config, shards=shards) as orchestrator:
    result = await orchestrator.weave(query)
    # Automatic cleanup on exit
```

---

### 3. Visualize Early, Debug Faster

```python
# ✅ Always visualize during development
from HoloLoom.visualization.reasoning_chain import render_from_reasoning_result

result = await engine.reason(query, features, context)

# Save visualization
html = render_from_reasoning_result(result, title="Debug: Low Confidence")
Path(f"debug_chain_{result.total_confidence:.2f}.html").write_text(html)
```

---

### 4. Monitor in Production

```python
# ✅ Track metrics from day one
from HoloLoom.performance.reasoning_metrics import track_reasoning, get_reasoning_metrics

with track_reasoning(mode="standard") as tracker:
    result = await engine.reason(query, features, context)
    tracker.set_result(result)

# Periodic reporting
if query_count % 100 == 0:
    metrics = get_reasoning_metrics()
    summary = metrics.get_summary()
    logger.info(f"Reasoning metrics: {summary}")
```

---

### 5. Graceful Degradation

```python
# ✅ Always have fallbacks
from HoloLoom.reasoning import ReasoningEngine, ReasoningMode

async def robust_reasoning(query, features, context):
    try:
        # Try DEEP mode
        engine = ReasoningEngine(mode=ReasoningMode.DEEP)
        return await engine.reason(query, features, context)

    except Exception as e:
        logger.warning(f"DEEP mode failed: {e}, falling back to STANDARD")

        try:
            # Fall back to STANDARD
            engine = ReasoningEngine(mode=ReasoningMode.STANDARD)
            return await engine.reason(query, features, context)

        except Exception as e2:
            logger.error(f"STANDARD failed: {e2}, falling back to FAST")

            # Last resort: FAST mode
            engine = ReasoningEngine(mode=ReasoningMode.FAST)
            return await engine.reason(query, features, context)
```

---

## Summary

**Integration**:
- 3 integration patterns (minimal, orchestrator, middleware)
- 4 component integrations (learning, Thompson, scratchpad, memory)

**Extensibility**:
- Custom verifiers for domain logic
- Custom planners for query decomposition
- Custom chain-of-thought generators

**Architecture**:
- Layered reasoning (progressive enhancement)
- Ensemble reasoning (multi-model voting)

**Best Practices**:
- Start simple
- Use context managers
- Visualize early
- Monitor always
- Degrade gracefully

---

**Next**: See `REASONING_ENGINE_EXTENSIBILITY.md` for advanced customization patterns.
