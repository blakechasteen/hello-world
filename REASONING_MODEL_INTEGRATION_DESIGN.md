# Reasoning Model Integration - HoloLoom 1.1 Design

**Author**: Claude Code
**Date**: 2025-11-15
**Branch**: `claude/reasoning-model-research-011CUedjHRfzNcMWgtsznvQ3`
**Status**: Design Phase

---

## Executive Summary

This document proposes integrating reasoning model capabilities into HoloLoom 1.1 by adding **Layer 6: Reasoning Engine** to the existing 9-layer weaving architecture. The design leverages existing infrastructure (scratchpad provenance, multi-strategy refinement, Thompson Sampling) to create a self-improving reasoning system with minimal architectural disruption.

**Key Principle**: *Think before you act, verify before you commit.*

---

## Background: What is a Reasoning Model?

Reasoning models perform **multi-step logical thinking** before generating answers:

1. **Chain-of-Thought**: Break problems into steps
2. **Planning**: Decompose complex queries into sub-tasks
3. **Verification**: Self-check for consistency and accuracy
4. **Backtracking**: Revise earlier steps if contradictions found
5. **Self-Critique**: Evaluate quality before committing

Examples: OpenAI o1/o3, Google Gemini reasoning mode, Claude's thinking blocks

---

## HoloLoom's Current Foundation

### Existing Infrastructure (Phases 1-5)

| Component | Capability | Relevance to Reasoning |
|-----------|-----------|----------------------|
| **Scratchpad** (Phase 1) | Provenance tracking (thought → action → observation → score) | ✅ Perfect for reasoning chains |
| **AdvancedRefiner** (Phase 4) | Multi-strategy refinement (VERIFY, ELEGANCE, HOFSTADTER) | ✅ Multi-pass verification |
| **Full Learning Loop** (Phase 5) | Thompson Sampling, background learning | ✅ Explore reasoning strategies |
| **WeavingOrchestrator** | 9-step cycle with mythRL protocols | ✅ Pipeline integration point |

### Current Weaving Cycle (9 Steps)

```
1. Loom Command → Pattern Card (BARE/FAST/FUSED)
2. Chrono Trigger → Temporal Window
3. Yarn Graph → Thread Selection
4. Resonance Shed → Feature Extraction (DotPlasma)
5. Warp Space → Continuous Manifold
6. Convergence Engine → Tool Selection
7. Tool Execution → Results
8. Spacetime Fabric → Provenance
9. Reflection Buffer → Learning
```

**Gap**: No explicit reasoning step between feature extraction (4) and decision (6).

---

## Proposed Architecture: Layer 6 - Reasoning Engine

### Integration Point

Insert **Reasoning Engine** between existing layers:

```
OLD FLOW:
4. Resonance Shed (features) → 6. Convergence Engine (decision)

NEW FLOW:
4. Resonance Shed (features) →
   6. REASONING ENGINE (thinking) →
   7. Convergence Engine (decision)
```

*Note: Renumber subsequent layers in implementation*

### Reasoning Engine Components

```python
# HoloLoom/reasoning/engine.py

class ReasoningEngine:
    """
    Multi-step reasoning layer for explicit chain-of-thought generation.

    Modes:
    - FAST: Single-pass reasoning (<50ms overhead)
    - STANDARD: Multi-step CoT (3-5 steps, ~200ms)
    - DEEP: Planning + verification + backtracking (~500ms+)
    """

    def __init__(
        self,
        mode: ReasoningMode = ReasoningMode.STANDARD,
        scratchpad: Optional[Scratchpad] = None,
        max_thinking_steps: int = 5,
        verification_threshold: float = 0.75
    ):
        self.mode = mode
        self.scratchpad = scratchpad
        self.max_steps = max_thinking_steps
        self.verification_threshold = verification_threshold

        # Strategy components
        self.planner = QueryPlanner()           # Decompose queries
        self.cot_generator = ChainOfThought()   # Generate reasoning steps
        self.verifier = SelfVerifier()          # Check consistency
        self.backtracker = Backtracker()        # Revise earlier steps
```

### Three Reasoning Modes

#### 1. FAST Mode (Minimal Overhead)

**When**: Simple factual queries, high-confidence retrieval
**Overhead**: <50ms
**Steps**: Single reasoning step + confidence check

```python
async def reason_fast(self, features: Features, context: Context) -> ReasoningResult:
    """
    Fast reasoning: Quick sanity check before decision.

    Steps:
    1. Check if context directly answers query
    2. Estimate confidence
    3. If high confidence (>0.85), proceed directly
    4. If low confidence (<0.7), escalate to STANDARD mode
    """
    confidence = self._estimate_confidence(features, context)

    if confidence >= 0.85:
        # Direct answer mode
        reasoning_chain = [
            ReasoningStep(
                thought="Context directly answers query",
                evidence=f"{len(context.shards)} relevant shards found",
                confidence=confidence
            )
        ]
        return ReasoningResult(chain=reasoning_chain, mode=ReasoningMode.FAST)

    # Escalate to STANDARD
    return await self.reason_standard(features, context)
```

#### 2. STANDARD Mode (Chain-of-Thought)

**When**: Most queries
**Overhead**: ~200ms
**Steps**: 3-5 reasoning steps with verification

```python
async def reason_standard(self, features: Features, context: Context) -> ReasoningResult:
    """
    Standard reasoning: Multi-step chain-of-thought.

    Steps:
    1. Analyze query intent (what are they really asking?)
    2. Identify key evidence from context
    3. Synthesize reasoning chain
    4. Self-verification check
    5. If verification fails, add corrective step
    """
    chain = []

    # Step 1: Understand intent
    intent = self.planner.analyze_intent(query, features)
    chain.append(ReasoningStep(
        thought=f"Query type: {intent.type}, requires: {intent.requirements}",
        evidence=f"Motifs: {features.motifs[:3]}",
        confidence=intent.confidence
    ))

    # Step 2: Identify evidence
    evidence = self._extract_evidence(context, intent)
    chain.append(ReasoningStep(
        thought=f"Found {len(evidence)} relevant pieces of evidence",
        evidence="; ".join(evidence[:3]),
        confidence=self._evidence_confidence(evidence)
    ))

    # Step 3: Synthesize reasoning
    synthesis = self._synthesize(intent, evidence)
    chain.append(ReasoningStep(
        thought=synthesis.reasoning,
        evidence=synthesis.key_points,
        confidence=synthesis.confidence
    ))

    # Step 4: Self-verification
    verification = await self.verifier.verify(chain, context)
    if not verification.passed:
        # Add corrective step
        chain.append(ReasoningStep(
            thought=f"Verification issue: {verification.issue}",
            evidence=verification.correction,
            confidence=0.5
        ))

        # Optionally backtrack if critical issue
        if verification.severity == VerificationSeverity.CRITICAL:
            chain = await self.backtracker.revise(chain, verification)

    return ReasoningResult(chain=chain, mode=ReasoningMode.STANDARD)
```

#### 3. DEEP Mode (Planning + Verification + Backtracking)

**When**: Complex multi-part queries, research mode, contradictions detected
**Overhead**: ~500ms+
**Steps**: Planning → execution → verification → backtracking (if needed)

```python
async def reason_deep(self, features: Features, context: Context) -> ReasoningResult:
    """
    Deep reasoning: Planning, multi-pass verification, backtracking.

    Steps:
    1. Create query plan (decompose into sub-questions)
    2. Execute plan steps sequentially
    3. Multi-pass verification (VERIFY strategy from AdvancedRefiner)
    4. Detect contradictions and backtrack if needed
    5. Synthesize final reasoning chain
    """
    chain = []

    # Step 1: Planning
    plan = await self.planner.create_plan(query, features, context)
    chain.append(ReasoningStep(
        thought=f"Plan: {len(plan.steps)} sub-questions",
        evidence="; ".join([s.question for s in plan.steps]),
        confidence=1.0
    ))

    # Step 2: Execute plan
    for i, step in enumerate(plan.steps):
        result = await self._execute_substep(step, context)
        chain.append(ReasoningStep(
            thought=f"Sub-question {i+1}: {result.answer}",
            evidence=result.evidence,
            confidence=result.confidence
        ))

    # Step 3: Multi-pass verification
    verification = await self.verifier.verify_multipass(chain, context)

    for pass_num, verify_result in enumerate(verification.passes):
        if not verify_result.passed:
            chain.append(ReasoningStep(
                thought=f"Verification pass {pass_num+1} issue: {verify_result.issue}",
                evidence=verify_result.correction,
                confidence=0.5
            ))

    # Step 4: Backtracking if contradictions
    if verification.has_contradictions:
        chain = await self.backtracker.resolve_contradictions(
            chain,
            verification.contradictions
        )

    # Step 5: Synthesize
    synthesis = self._synthesize_deep(chain, plan)
    chain.append(ReasoningStep(
        thought=f"Final synthesis: {synthesis.conclusion}",
        evidence=synthesis.supporting_evidence,
        confidence=synthesis.confidence
    ))

    return ReasoningResult(chain=chain, mode=ReasoningMode.DEEP)
```

---

## Integration with Existing Components

### 1. Scratchpad Integration

**Reasoning chains → Scratchpad entries**:

```python
class ReasoningProvenanceTracker(ProvenanceTracker):
    """Extended provenance tracker for reasoning chains."""

    def extract_reasoning_provenance(
        self,
        reasoning_result: ReasoningResult
    ) -> List[ScratchpadEntry]:
        """
        Convert reasoning chain to scratchpad entries.

        Each reasoning step becomes an entry:
        - Thought: The reasoning step's thought
        - Action: "reasoning_step_{i}"
        - Observation: Evidence collected
        - Score: Confidence of this step
        """
        entries = []

        for i, step in enumerate(reasoning_result.chain):
            entry = ScratchpadEntry(
                thought=step.thought,
                action=f"reasoning_step_{i+1}",
                observation=step.evidence,
                score=step.confidence,
                iteration=i + 1,
                metadata={
                    "mode": reasoning_result.mode.value,
                    "step_type": step.step_type.value,
                    "timestamp": step.timestamp
                }
            )
            entries.append(entry)

        return entries
```

### 2. AdvancedRefiner Integration

**Trigger deep reasoning when refinement needed**:

```python
class ReasoningAwareRefiner(AdvancedRefiner):
    """Enhanced refiner that triggers deep reasoning for low confidence."""

    async def refine(
        self,
        query: Query,
        initial_spacetime: Spacetime,
        **kwargs
    ) -> RefinementResult:
        """
        Refine with reasoning-aware strategy selection.

        Strategy mapping:
        - Low confidence (<0.5): DEEP reasoning mode
        - Medium confidence (0.5-0.75): STANDARD reasoning mode
        - High confidence (>0.75): FAST reasoning mode (if enabled)
        """
        confidence = initial_spacetime.trace.tool_confidence

        if confidence < 0.5:
            # Trigger DEEP reasoning mode
            kwargs['reasoning_mode'] = ReasoningMode.DEEP
        elif confidence < 0.75:
            # Trigger STANDARD reasoning mode
            kwargs['reasoning_mode'] = ReasoningMode.STANDARD
        else:
            # Fast path
            kwargs['reasoning_mode'] = ReasoningMode.FAST

        return await super().refine(query, initial_spacetime, **kwargs)
```

### 3. WeavingOrchestrator Integration

**Add reasoning layer to weaving cycle**:

```python
class ReasoningOrchestrator(WeavingOrchestrator):
    """Enhanced orchestrator with reasoning layer."""

    def __init__(
        self,
        cfg: Config,
        shards: List[MemoryShard],
        enable_reasoning: bool = True,
        reasoning_mode: ReasoningMode = ReasoningMode.STANDARD,
        **kwargs
    ):
        super().__init__(cfg, shards, **kwargs)

        self.enable_reasoning = enable_reasoning

        if enable_reasoning:
            self.reasoning_engine = ReasoningEngine(
                mode=reasoning_mode,
                scratchpad=self.scratchpad,
                max_thinking_steps=cfg.max_reasoning_steps,
                verification_threshold=cfg.reasoning_verification_threshold
            )

    async def weave(self, query: Query) -> Spacetime:
        """
        Enhanced weaving cycle with reasoning layer.

        Modified cycle:
        1-4. [Same as before] → DotPlasma features
        5. REASONING ENGINE → Generate reasoning chain
        6. Warp Space → Continuous manifold
        7. Convergence Engine → Tool selection (informed by reasoning)
        8-9. [Same as before] → Spacetime + reflection
        """
        # Steps 1-4: Feature extraction (unchanged)
        features = await self._extract_features(query)
        context = await self._retrieve_context(query, features)

        # Step 5: NEW - Reasoning layer
        reasoning_result = None
        if self.enable_reasoning:
            reasoning_result = await self.reasoning_engine.reason(
                query, features, context
            )

            # Add reasoning to scratchpad
            if self.scratchpad:
                reasoning_entries = self.reasoning_tracker.extract_reasoning_provenance(
                    reasoning_result
                )
                for entry in reasoning_entries:
                    self.scratchpad.add_entry(entry)

        # Steps 6-7: Decision (informed by reasoning)
        decision = await self._make_decision(
            features,
            context,
            reasoning_result=reasoning_result  # NEW parameter
        )

        # Steps 8-9: Execution + reflection (unchanged)
        spacetime = await self._execute_and_reflect(query, decision, context)

        # Attach reasoning chain to spacetime metadata
        if reasoning_result:
            spacetime.metadata['reasoning_chain'] = reasoning_result.chain
            spacetime.metadata['reasoning_mode'] = reasoning_result.mode.value

        return spacetime
```

### 4. Thompson Sampling Integration

**Learn which reasoning modes work best**:

```python
class ReasoningModeBandit:
    """Thompson Sampling for reasoning mode selection."""

    def __init__(self):
        # Priors for each reasoning mode
        self.priors = {
            ReasoningMode.FAST: ThompsonPrior(alpha=10, beta=2),     # Initially favor FAST
            ReasoningMode.STANDARD: ThompsonPrior(alpha=15, beta=5), # Balanced
            ReasoningMode.DEEP: ThompsonPrior(alpha=5, beta=5),      # Expensive, use cautiously
        }

    def select_mode(self, query_complexity: float) -> ReasoningMode:
        """
        Select reasoning mode using Thompson Sampling + query complexity.

        Strategy:
        - Sample from each mode's beta distribution
        - Weight by query complexity (complex queries bias toward DEEP)
        - Select mode with highest weighted sample
        """
        samples = {}
        for mode, prior in self.priors.items():
            sample = prior.sample()

            # Complexity weighting
            if mode == ReasoningMode.DEEP and query_complexity > 0.7:
                sample *= 1.5  # Boost DEEP for complex queries
            elif mode == ReasoningMode.FAST and query_complexity < 0.3:
                sample *= 1.3  # Boost FAST for simple queries

            samples[mode] = sample

        return max(samples.items(), key=lambda x: x[1])[0]

    def update(self, mode: ReasoningMode, success: bool, confidence: float):
        """Update priors based on outcome."""
        if success and confidence >= 0.75:
            self.priors[mode].alpha += confidence
        else:
            self.priors[mode].beta += (1 - confidence)
```

---

## Implementation Plan

### Phase 1: Foundation (Week 1)

**Goal**: Basic reasoning engine with FAST and STANDARD modes

**Tasks**:
1. Create `HoloLoom/reasoning/` directory
2. Implement core types:
   - `ReasoningMode`, `ReasoningStep`, `ReasoningResult`
   - `QueryPlanner`, `ChainOfThought`, `SelfVerifier`
3. Implement FAST mode (single-step reasoning)
4. Implement STANDARD mode (3-5 step CoT)
5. Unit tests for reasoning components

**Deliverables**:
- `HoloLoom/reasoning/engine.py` (~400 lines)
- `HoloLoom/reasoning/types.py` (~150 lines)
- `HoloLoom/tests/unit/test_reasoning_engine.py` (~300 lines)

### Phase 2: Integration (Week 2)

**Goal**: Integrate reasoning engine into WeavingOrchestrator

**Tasks**:
1. Create `ReasoningOrchestrator` (extends `WeavingOrchestrator`)
2. Implement `ReasoningProvenanceTracker` (scratchpad integration)
3. Add reasoning chain to Spacetime metadata
4. Integration tests with full pipeline

**Deliverables**:
- `HoloLoom/weaving_orchestrator_reasoning.py` (~300 lines)
- `HoloLoom/recursive/reasoning_provenance.py` (~200 lines)
- `HoloLoom/tests/integration/test_reasoning_integration.py` (~400 lines)

### Phase 3: Advanced Features (Week 3)

**Goal**: DEEP mode, backtracking, Thompson Sampling

**Tasks**:
1. Implement DEEP mode (planning + verification)
2. Implement `Backtracker` (contradiction resolution)
3. Implement `ReasoningModeBandit` (Thompson Sampling)
4. Connect to `AdvancedRefiner` for automatic escalation

**Deliverables**:
- `HoloLoom/reasoning/planner.py` (~350 lines)
- `HoloLoom/reasoning/backtracker.py` (~250 lines)
- `HoloLoom/reasoning/bandit.py` (~200 lines)
- `HoloLoom/recursive/reasoning_refiner.py` (~150 lines)

### Phase 4: Visualization & Tooling (Week 4)

**Goal**: Make reasoning chains visible and debuggable

**Tasks**:
1. Create Tufte-style reasoning chain visualizer
2. Add reasoning metrics to Prometheus
3. Create interactive reasoning playground demo
4. Documentation and examples

**Deliverables**:
- `HoloLoom/visualization/reasoning_chain.py` (~400 lines)
- `demos/reasoning_playground.py` (~500 lines)
- `REASONING_MODEL_GUIDE.md` (~1500 lines)
- Updated `CLAUDE.md` with reasoning features

---

## Configuration

### Config Options

```python
# HoloLoom/config.py

class Config:
    # Reasoning model settings
    enable_reasoning: bool = True
    reasoning_mode: ReasoningMode = ReasoningMode.STANDARD
    max_reasoning_steps: int = 5
    reasoning_verification_threshold: float = 0.75

    # Adaptive mode selection
    enable_adaptive_reasoning: bool = True
    reasoning_complexity_threshold: float = 0.5

    # Performance limits
    max_reasoning_time_ms: float = 500.0
    reasoning_timeout_fallback: ReasoningMode = ReasoningMode.FAST
```

### Usage Examples

```python
from HoloLoom.config import Config, ReasoningMode
from HoloLoom.weaving_orchestrator_reasoning import ReasoningOrchestrator
from HoloLoom.documentation.types import Query

# Basic usage - STANDARD mode
config = Config.fused()
config.enable_reasoning = True
config.reasoning_mode = ReasoningMode.STANDARD

async with ReasoningOrchestrator(cfg=config, shards=shards) as orchestrator:
    spacetime = await orchestrator.weave(Query(text="Explain how transformers work"))

    # View reasoning chain
    for i, step in enumerate(spacetime.metadata['reasoning_chain']):
        print(f"Step {i+1}: {step.thought}")
        print(f"  Evidence: {step.evidence}")
        print(f"  Confidence: {step.confidence:.2f}\n")

# Adaptive mode - let Thompson Sampling choose
config.enable_adaptive_reasoning = True
async with ReasoningOrchestrator(cfg=config, shards=shards) as orchestrator:
    # Simple query → likely FAST mode
    result1 = await orchestrator.weave(Query(text="What is 2+2?"))

    # Complex query → likely DEEP mode
    result2 = await orchestrator.weave(Query(
        text="Compare and contrast the philosophical implications of "
             "determinism vs free will in the context of AI decision-making"
    ))

# Force DEEP mode for research
config.reasoning_mode = ReasoningMode.DEEP
async with ReasoningOrchestrator(cfg=config, shards=shards) as orchestrator:
    spacetime = await orchestrator.weave(Query(text="Prove P != NP"))
    # Will use planning, multi-pass verification, backtracking
```

---

## Performance Characteristics

### Overhead Estimates

| Mode | Overhead | When to Use | Example Queries |
|------|----------|-------------|-----------------|
| **FAST** | <50ms | Simple factual queries, high confidence | "What is X?", "Define Y" |
| **STANDARD** | ~200ms | Most queries, moderate complexity | "How does X work?", "Compare A and B" |
| **DEEP** | ~500ms+ | Complex multi-part, research mode | "Prove X", "Design system for Y" |

### Accuracy vs Speed Tradeoff

```
Accuracy: DEEP (95%) > STANDARD (85%) > FAST (75%) > NO_REASONING (70%)
Speed:    FAST (50ms) > STANDARD (200ms) > DEEP (500ms+)

Sweet spot: STANDARD mode with adaptive escalation
```

### Thompson Sampling Learning Curve

```
Initial (cold start):
- FAST: 80% selection rate (low accuracy, but fast)
- STANDARD: 15% selection rate
- DEEP: 5% selection rate

After 1000 queries (warm):
- FAST: 30% selection rate (learned when appropriate)
- STANDARD: 60% selection rate (default for most)
- DEEP: 10% selection rate (reserved for complex)
```

---

## Testing Strategy

### Unit Tests

```python
# HoloLoom/tests/unit/test_reasoning_engine.py

async def test_fast_mode_simple_query():
    """FAST mode should complete in <50ms for simple queries."""
    engine = ReasoningEngine(mode=ReasoningMode.FAST)

    start = time.time()
    result = await engine.reason(simple_query, features, context)
    duration = (time.time() - start) * 1000

    assert duration < 50, f"FAST mode took {duration}ms"
    assert len(result.chain) == 1, "FAST mode should have 1 step"
    assert result.chain[0].confidence > 0.7

async def test_standard_mode_chain_of_thought():
    """STANDARD mode should generate 3-5 reasoning steps."""
    engine = ReasoningEngine(mode=ReasoningMode.STANDARD)

    result = await engine.reason(moderate_query, features, context)

    assert 3 <= len(result.chain) <= 5
    assert all(step.confidence > 0.5 for step in result.chain)
    assert result.chain[-1].step_type == StepType.SYNTHESIS

async def test_deep_mode_backtracking():
    """DEEP mode should detect and resolve contradictions."""
    engine = ReasoningEngine(mode=ReasoningMode.DEEP)

    # Query with contradictory evidence
    result = await engine.reason(contradictory_query, features, context)

    # Should have backtracking steps
    backtrack_steps = [s for s in result.chain if s.step_type == StepType.BACKTRACK]
    assert len(backtrack_steps) > 0

    # Final synthesis should resolve contradiction
    assert result.chain[-1].confidence > 0.7
```

### Integration Tests

```python
# HoloLoom/tests/integration/test_reasoning_integration.py

async def test_reasoning_scratchpad_integration():
    """Reasoning chains should be recorded in scratchpad."""
    config = Config.fast()
    config.enable_reasoning = True

    scratchpad = Scratchpad()
    async with ReasoningOrchestrator(
        cfg=config,
        shards=shards,
        scratchpad=scratchpad
    ) as orchestrator:
        spacetime = await orchestrator.weave(query)

        # Scratchpad should have reasoning entries
        entries = scratchpad.get_history()
        reasoning_entries = [e for e in entries if 'reasoning_step' in e.action]

        assert len(reasoning_entries) >= 3  # STANDARD mode
        assert all(e.score > 0 for e in reasoning_entries)

async def test_adaptive_mode_selection():
    """Adaptive mode should select appropriate reasoning mode."""
    config = Config.fast()
    config.enable_adaptive_reasoning = True

    async with ReasoningOrchestrator(cfg=config, shards=shards) as orchestrator:
        # Simple query → FAST mode
        simple = await orchestrator.weave(Query(text="What is 2+2?"))
        assert simple.metadata['reasoning_mode'] == ReasoningMode.FAST.value

        # Complex query → DEEP mode
        complex = await orchestrator.weave(Query(
            text="Explain the relationship between Gödel's incompleteness "
                 "theorems and the halting problem"
        ))
        assert complex.metadata['reasoning_mode'] in [
            ReasoningMode.STANDARD.value,
            ReasoningMode.DEEP.value
        ]
```

### End-to-End Tests

```python
# HoloLoom/tests/e2e/test_reasoning_pipeline.py

async def test_full_reasoning_pipeline():
    """End-to-end test: query → reasoning → decision → response."""
    config = Config.fused()
    config.enable_reasoning = True
    config.reasoning_mode = ReasoningMode.STANDARD

    async with ReasoningOrchestrator(cfg=config, shards=shards) as orchestrator:
        spacetime = await orchestrator.weave(Query(
            text="How does Thompson Sampling balance exploration and exploitation?"
        ))

        # Should have reasoning chain
        assert 'reasoning_chain' in spacetime.metadata
        chain = spacetime.metadata['reasoning_chain']
        assert len(chain) >= 3

        # Should have high confidence
        assert spacetime.trace.tool_confidence > 0.75

        # Should have relevant response
        assert 'thompson' in spacetime.response.lower()
        assert 'exploration' in spacetime.response.lower()
```

---

## Visualization: Reasoning Chain Display

### Tufte-Style Reasoning Chain Visualizer

```python
# HoloLoom/visualization/reasoning_chain.py

def render_reasoning_chain(
    chain: List[ReasoningStep],
    mode: ReasoningMode,
    title: str = "Reasoning Chain"
) -> str:
    """
    Render reasoning chain in Tufte style.

    Features:
    - Sequential step flow (top to bottom)
    - Confidence sparklines
    - Evidence tooltips
    - Backtracking indicators
    - Critical step highlighting
    """
    html = f"<div class='reasoning-chain'>"
    html += f"<h3>{title} <span class='mode-badge'>{mode.value}</span></h3>"

    for i, step in enumerate(chain):
        # Step indicator
        step_class = 'critical' if step.confidence < 0.5 else 'normal'
        if step.step_type == StepType.BACKTRACK:
            step_class = 'backtrack'

        html += f"<div class='reasoning-step {step_class}'>"

        # Step number + type icon
        icon = STEP_TYPE_ICONS[step.step_type]
        html += f"<div class='step-header'>"
        html += f"  <span class='step-num'>{i+1}</span> {icon}"
        html += f"  <span class='confidence'>{step.confidence:.2f}</span>"
        html += f"</div>"

        # Thought
        html += f"<div class='thought'>{step.thought}</div>"

        # Evidence (collapsible)
        if step.evidence:
            html += f"<details class='evidence'>"
            html += f"  <summary>Evidence</summary>"
            html += f"  <p>{step.evidence}</p>"
            html += f"</details>"

        html += "</div>"

    html += "</div>"
    return html
```

**Example Output**:

```
Reasoning Chain [STANDARD]
━━━━━━━━━━━━━━━━━━━━━━━━

1 🧠 [0.92]
  Query type: comparative, requires: multi-source evidence
  ▸ Evidence: Motifs: thompson, sampling, exploration

2 🔍 [0.88]
  Found 7 relevant pieces of evidence
  ▸ Evidence: Beta distribution sampling; UCB comparison; Multi-armed bandits

3 🔗 [0.85]
  Thompson Sampling uses Bayesian priors to balance exploration/exploitation
  ▸ Evidence: Alpha/beta updates; Sample from posteriors; Automatic decay

4 ✓ [0.90]
  Verification passed: Consistent with all sources
  ▸ Evidence: Cross-checked 3 sources

5 💎 [0.93]
  Final synthesis: Thompson Sampling is a probabilistic algorithm that...
  ▸ Evidence: [Full synthesis with citations]
```

---

## Success Metrics

### Quality Metrics

- **Accuracy Improvement**: +15-25% on complex queries (DEEP mode)
- **Consistency**: <5% contradiction rate (with verification)
- **Confidence Calibration**: <10% error between predicted and actual accuracy

### Performance Metrics

- **FAST Mode**: <50ms overhead, 75%+ accuracy
- **STANDARD Mode**: <200ms overhead, 85%+ accuracy
- **DEEP Mode**: <500ms overhead, 95%+ accuracy

### Learning Metrics (Thompson Sampling)

- **Mode Selection Accuracy**: >80% after 1000 queries
- **Exploration Rate**: 10-15% (healthy exploration)
- **Regret Bound**: Logarithmic regret growth

---

## Risks & Mitigations

### Risk 1: Performance Overhead

**Risk**: Reasoning adds latency, hurts user experience
**Mitigation**:
- Default to FAST mode for simple queries
- Adaptive mode selection based on query complexity
- Hard timeout with fallback (max 500ms → fallback to FAST)
- Cache reasoning chains for repeated queries

### Risk 2: Infinite Loops

**Risk**: Backtracking could loop indefinitely
**Mitigation**:
- Hard limit on reasoning steps (default: 5)
- Detect cycles in reasoning chain
- Timeout mechanism with graceful degradation

### Risk 3: Quality Regression

**Risk**: Reasoning might hurt simple queries
**Mitigation**:
- A/B testing: measure accuracy with/without reasoning
- Gradual rollout: start with DEEP mode only for complex queries
- Thompson Sampling learns when reasoning helps

### Risk 4: Complexity Explosion

**Risk**: System becomes too complex to debug
**Mitigation**:
- Full scratchpad provenance for every reasoning step
- Tufte-style visualization of reasoning chains
- Unit/integration/e2e tests at every phase

---

## Future Extensions

### Phase 5: Multi-Agent Reasoning (v1.2)

Multiple reasoning engines debate and vote:
- Ensemble of FAST + STANDARD + DEEP modes
- Majority voting for final decision
- Adversarial verification (one agent challenges another)

### Phase 6: Learned Reasoning Strategies (v1.3)

Train a meta-learner to discover new reasoning strategies:
- Reinforcement learning on reasoning chain quality
- Evolutionary search over reasoning templates
- Transfer learning from successful patterns

### Phase 7: Interactive Reasoning (v1.4)

User can steer reasoning process in real-time:
- "Show me your thinking"
- "Try a different approach"
- "Focus more on X"

---

## Conclusion

This design integrates reasoning model capabilities into HoloLoom 1.1 by:

1. **Minimal disruption**: Adds Layer 6, existing layers unchanged
2. **Leverages existing infrastructure**: Scratchpad, refinement, Thompson Sampling
3. **Progressive complexity**: FAST → STANDARD → DEEP modes
4. **Self-improving**: Thompson Sampling learns optimal mode selection
5. **Observable**: Tufte-style visualization, full provenance

**Next Steps**:
1. Review this design document
2. Get approval for architectural approach
3. Begin Phase 1 implementation (foundation)
4. Iterate based on testing and feedback

**Estimated Timeline**: 4 weeks to production-ready v1.1

---

**Questions for Review**:
1. Is the 3-mode approach (FAST/STANDARD/DEEP) the right granularity?
2. Should reasoning be opt-in or opt-out by default?
3. What performance overhead is acceptable for STANDARD mode?
4. Any other integration points we should consider?
