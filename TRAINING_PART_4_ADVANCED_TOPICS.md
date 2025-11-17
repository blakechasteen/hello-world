# HoloLoom Complete Training Guide
## Part 4: Advanced Topics (November 2025)

**Target Audience**: Users who understand HoloLoom basics (BARE/FAST/FUSED modes, memory system, weaving orchestrator) and want to master advanced features.

**Prerequisites**: Complete Parts 1-3 of the training guide

**Estimated Reading Time**: 45-60 minutes

---

## Table of Contents

1. [Thompson Sampling Deep Dive](#1-thompson-sampling-deep-dive)
2. [Compositional Caching: The 291× Speedup](#2-compositional-caching-the-291-speedup)
3. [Recursive Learning Loop](#3-recursive-learning-loop)
4. [Alignment and Safety Framework](#4-alignment-and-safety-framework)
5. [RAG System Architecture](#5-rag-system-architecture)
6. [Phase 5: Universal Grammar Integration](#6-phase-5-universal-grammar-integration)

---

## 1. Thompson Sampling Deep Dive

### The Problem: Exploration vs Exploitation

When you have multiple tools available, how do you decide which to use?

- **Exploitation**: Always pick the tool that worked best in the past
- **Exploration**: Sometimes try other tools to find better ones
- **Thompson Sampling**: A Bayesian method that balances both mathematically

### Beta Distribution Explained

Thompson Sampling uses **Beta distributions** to represent uncertainty about each tool's success rate.

A Beta distribution has two parameters: **α** (successes) and **β** (failures)

```
Beta(α, β)

α = number of successes
β = number of failures

Expected success rate = α / (α + β)
```

**Visual Example:**

```
Tool A: 80 successes, 20 failures → Beta(80, 20)
Tool B: 10 successes, 5 failures → Beta(10, 5)

Tool A:
E[X] = 80/100 = 0.80  (more confident - narrow distribution)
   │
   │     ████
   │    ██████
   │   ████████
   └─────────────
   0.0  0.5  1.0

Tool B:
E[X] = 10/15 = 0.67  (less confident - wide distribution)
   │
   │  ███    ███
   │ █████  █████
   │ █████████████
   └─────────────
   0.0  0.5  1.0
```

**Key insight**: Even though Tool B has lower expected success rate, its wider distribution means it might actually be better. Thompson Sampling samples from each distribution to decide.

### How Sampling Works

At each decision point:

1. **Sample** from each tool's Beta distribution
2. **Pick** the tool with highest sampled value
3. **Execute** that tool
4. **Update** the distribution based on success/failure

```python
# Thompson Sampling algorithm
class TSBandit:
    def __init__(self, n_tools=5):
        # Initialize all tools with Beta(1, 1) = uniform
        self.alpha = np.ones(n_tools)    # successes
        self.beta = np.ones(n_tools)     # failures

    def select_tool(self):
        # Sample from each tool's Beta distribution
        samples = [np.random.beta(self.alpha[i], self.beta[i])
                   for i in range(len(self.alpha))]
        # Return tool with highest sample
        return np.argmax(samples)

    def update(self, tool_idx, success):
        # Update based on outcome
        if success:
            self.alpha[tool_idx] += 1     # More successes
        else:
            self.beta[tool_idx] += 1      # More failures
```

### Comparison to Epsilon-Greedy

Two common strategies in HoloLoom:

**Epsilon-Greedy** (simpler):
```
With probability ε (e.g., 0.1):
    Pick random tool
Otherwise:
    Pick best tool based on past performance
```

**Thompson Sampling** (more sophisticated):
```
For each tool:
    Sample from Beta(successes, failures)
Pick tool with highest sample
```

| Aspect | Epsilon-Greedy | Thompson Sampling |
|--------|---|---|
| **Exploration** | Random | Probabilistic |
| **Uncertainty** | Ignored | Explicitly modeled |
| **Efficiency** | Good | Better |
| **Complexity** | Simple | Moderate |
| **Recommended** | Stable, simple systems | Adaptive, complex systems |

### When Thompson Sampling Wins

Thompson Sampling outperforms epsilon-greedy when:

1. **Uncertainty varies**: Tools have different confidence levels (different sample sizes)
2. **Payoff varies**: Some tools give bigger rewards than others
3. **You have time to explore**: Not all decisions are critical
4. **You want to learn**: System improves over time

**Real example from HoloLoom:**
- Tool A (keyword search): Seen 1000 times, 75% success
- Tool B (semantic search): Seen 50 times, 90% success

**Epsilon-greedy**: Always picks Tool A (0.75 > 0.50... wait, that's Tool B's success rate for "average" query)

Actually, let me recalculate: Tool A is 75% (750/1000), Tool B is 90% (45/50).

Epsilon-greedy: Would eventually pick B more often, but slowly

**Thompson Sampling**: Samples B's distribution more often (it's "luckier" in samples due to high success rate) while still occasionally sampling A (due to larger sample size, its distribution is narrower, so some samples beat B)

### Bayesian Updating Formula

When a tool succeeds or fails, we update its Beta distribution:

```
Tool outcome observed:

If success:
    α_new = α_old + 1

If failure:
    β_new = β_old + 1

Expected value updates automatically:
    E[success] = α_new / (α_new + β_new)
```

**Example:**

```
Tool starts: Beta(1, 1)
    E[success] = 1/(1+1) = 0.50

Query 1: Success
    β(2, 1), E[success] = 2/(2+1) = 0.67

Query 2: Failure
    Beta(2, 2), E[success] = 2/(2+2) = 0.50

Query 3: Success
    Beta(3, 2), E[success] = 3/(3+2) = 0.60
```

### Code Walkthrough: BanditStrategy in HoloLoom

```python
from HoloLoom.policy.unified import NeuralPolicy, BanditStrategy
import numpy as np

class NeuralPolicy:
    def __init__(self, bandit_strategy=BanditStrategy.EPSILON_GREEDY):
        self.bandit = TSBandit(n_tools=5)
        self.bandit_strategy = bandit_strategy
        self.neural_output = None  # Will hold NN predictions

    def select_tool(self, features, context):
        """Choose which tool to use"""

        # Get neural network prediction
        self.neural_output = self.nn(features, context)  # logits over tools

        if self.bandit_strategy == BanditStrategy.EPSILON_GREEDY:
            # 90% neural, 10% Thompson
            if np.random.random() < 0.1:
                return self.bandit.select_tool()  # Thompson sample
            else:
                return np.argmax(self.neural_output)  # Neural choice

        elif self.bandit_strategy == BanditStrategy.BAYESIAN_BLEND:
            # 70% neural + 30% Thompson
            neural_probs = softmax(self.neural_output)  # Convert to probabilities

            # Get Thompson priors
            ts_probs = np.array([
                self.bandit.alpha[i] / (self.bandit.alpha[i] + self.bandit.beta[i])
                for i in range(len(self.bandit.alpha))
            ])

            # Blend: 70% neural, 30% Thompson
            blend_probs = 0.7 * neural_probs + 0.3 * ts_probs
            return np.argmax(blend_probs)

        elif self.bandit_strategy == BanditStrategy.PURE_THOMPSON:
            # 100% Thompson, ignore neural network
            return self.bandit.select_tool()

    def update(self, tool_idx, success):
        """Learn from outcome"""
        # Update bandit with ACTUALLY SELECTED tool (fixing code review bug)
        if success:
            self.bandit.alpha[tool_idx] += 1
        else:
            self.bandit.beta[tool_idx] += 1
```

### Tuning Exploration Rate

Default exploration rates in HoloLoom:

**Epsilon-Greedy** (BanditStrategy.EPSILON_GREEDY):
```python
# Default: 90% exploit, 10% explore
if np.random.random() < epsilon:  # epsilon = 0.1
    choose_thompson_sample()
else:
    choose_neural_prediction()
```

**Tuning strategy:**

| Epsilon | Behavior | When to use |
|---------|----------|------------|
| **0.05** | Conservative exploration | Stable system, learned tools |
| **0.10** | Default | Balanced, recommended |
| **0.20** | Aggressive exploration | Early development, many unknown tools |
| **0.50** | Very aggressive | Research, testing new tools |

**How to change:**

```python
from HoloLoom.policy.unified import NeuralPolicy
from HoloLoom.documentation.types import BanditStrategy

# At policy creation
policy = NeuralPolicy(
    bandit_strategy=BanditStrategy.EPSILON_GREEDY,
    epsilon=0.15  # 15% exploration instead of default 10%
)
```

**Performance implications:**

- **Lower epsilon** (0.05): Faster decisions, might miss better tools
- **Higher epsilon** (0.30): Slower learning curve, discovers better tools
- **Recommended default**: 0.10 (good balance)

---

## 2. Compositional Caching: The 291× Speedup

### Why Traditional Caching Fails for Language

Standard caching stores full results:

```
Query: "What is Thompson Sampling?"
  → Retrieve sources
  → Generate response
  → Cache result

Next query: "Explain Thompson Sampling"
  → Cache MISS (different query string)
  → Retrieve sources AGAIN
  → Generate response AGAIN
```

**Problem**: Two semantically similar queries miss cache because they're syntactically different.

**Hit rate**: ~10-20% in practice (most queries are unique)

### The Compositionality Insight

Language is **compositional**: Meaning builds hierarchically from parts.

```
"the big red ball"

Can be decomposed:
├─ "the" (determiner)
├─ "big" (adjective)
├─ "red" (adjective)
└─ "ball" (noun)

And composed:
Step 1: Merge("red", "ball") → "red ball"
Step 2: Merge("big", "red ball") → "big red ball"
Step 3: Merge("the", "big red ball") → "the big red ball"
```

**Key insight**: If we cache intermediate compositions, we can reuse them:

```
"a red ball"       → Uses cached "red ball" + new "a"
"the red ball"     → Uses cached "red ball" + new "the"
"the big red ball" → Uses cached "big red ball" + cached "red ball"
```

**Expected hit rate**: 50-80% (massive improvement!)

### Three-Tier Cache Architecture

HoloLoom implements a 3-level compositional cache:

```
┌─────────────────────────────────────────────────────────────┐
│                   INPUT QUERY                               │
│              "the big red ball"                              │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────┐
        │   TIER 1: PARSE CACHE            │
        │   Cache X-bar structures         │
        │                                  │
        │   "the big red ball"             │
        │   └─ NP                          │
        │       ├─ Det "the"               │
        │       └─ N'                      │
        │           ├─ A "big"             │
        │           └─ N'                  │
        │               ├─ A "red"         │
        │               └─ N "ball"        │
        │                                  │
        │   Hit: Skip spaCy parsing        │
        │   Speedup: 10-50×                │
        └──────────────┬───────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────┐
        │   TIER 2: MERGE CACHE            │
        │   Cache compositional embeddings │
        │                                  │
        │   ("red", "ball") → 384D vec     │
        │   ("big", "red ball") → 384D vec │
        │   ("the", "big red ball") → vec  │
        │                                  │
        │   Hit: Skip Merge operations     │
        │   Speedup: 5-10×                 │
        └──────────────┬───────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────┐
        │   TIER 3: SEMANTIC CACHE         │
        │   Cache 244D projections         │
        │                                  │
        │   384D → 244D (MatryoshkaGate)   │
        │                                  │
        │   Hit: Skip projection           │
        │   Speedup: 3-10×                 │
        └──────────────┬───────────────────┘
                       │
                       ▼
                  Final result
                  (or cache miss)
```

### How "the big red ball" Reuses "big red ball"

**Scenario:**

Query 1: "the big red ball"
- Parse: Creates X-bar tree (cache miss) → TIER 1 cache stored
- Merge "red" + "ball" → (cache miss) → TIER 2 cached
- Merge "big" + "red ball" → (cache miss) → TIER 2 cached
- Merge "the" + "big red ball" → (cache miss) → TIER 2 cached
- Project to 244D → (cache miss) → TIER 3 cached
- **Latency**: ~150ms (cold)

Query 2: "a big red ball" (same structure, different determiner)
- Parse: Creates X-bar tree (cache HIT) → TIER 1 skip
- Merge "red" + "ball" → (cache HIT) → TIER 2 skip ✓
- Merge "big" + "red ball" → (cache HIT) → TIER 2 skip ✓
- Merge "a" + "big red ball" → (cache miss, new determiner) → TIER 2 cached
- Project to 244D → (cache HIT) → TIER 3 skip ✓
- **Latency**: ~45ms (warm, 3.3× speedup)

Query 3: Repeated "the big red ball"
- Everything cached (TIER 1, 2, 3)
- **Latency**: <1ms (hot, 150× speedup)

### Cache Hit Mechanics

When do caches hit?

**TIER 1 (Parse Cache)**:
- Hits when: Same character sequence appears
- Hit rate: ~60% (many queries with overlapping phrases)

**TIER 2 (Merge Cache)**:
- Hits when: Same (head, dependent, merge_type) appears
- Hit rate: ~70% (many queries reuse sub-phrases)
- Example: "red ball" appears in "the red ball", "a red ball", "red ball analysis"

**TIER 3 (Semantic Cache)**:
- Hits when: Same embedding output appears
- Hit rate: ~80% (many queries project to same 244D space)

**Combined hit rate** (all three tiers): ~50-80%

### Performance Characteristics

Caching overhead vs benefits:

```
Cache lookup:         <0.1ms (hash table, negligible)
Cache storage:        ~1KB per cached item
Memory for 10K items: ~10MB (very reasonable)

Latency breakdown (typical query):

Without cache:
├─ Parse            20ms
├─ Merge (3 levels) 45ms
├─ Embed            60ms
├─ Project          25ms
└─ Total            150ms

With cache (warm):
├─ Parse (hit)      <1ms
├─ Merge (all hit)  <1ms
├─ Embed (hit)      <1ms
├─ Project (hit)    <1ms
└─ Total            <1ms

Speedup: 150× on warm path!
```

### When to Enable/Disable

**Enable compositional caching when:**
- Queries have overlapping structure (typical use)
- Memory available (10-100MB for cache)
- Latency-critical workloads
- Users ask similar questions (most common)

**Disable when:**
- Every query is completely unique (rare)
- Memory extremely constrained (<100MB available)
- Debugging (cache can hide issues)

### Configuration Parameters

Enable via HoloLoom config:

```python
from HoloLoom.config import Config

config = Config.fused()

# Enable all three tiers (default in FAST/FUSED)
config.use_parse_cache = True           # Cache X-bar structures
config.use_merge_cache = True           # Cache compositions
config.use_semantic_cache = True        # Cache 244D projections

# Size limits
config.parse_cache_size = 10000         # Max items (default)
config.merge_cache_size = 50000         # Max items (default)
config.semantic_cache_size = 25000      # Max items (default)

# Or use one-liner defaults
config = Config.fused()  # All caches enabled
config = Config.bare()   # All caches disabled (fastest training)
```

**Cache eviction strategy**: LRU (Least Recently Used)
- When cache fills, oldest unused item is removed
- Keeps hot items (frequently accessed) in memory

---

## 3. Recursive Learning Loop

### The Five Phases Explained

The Recursive Learning System implements 5 self-improving phases:

```
Phase 1: Scratchpad Integration
  ↓ Tracks complete provenance
Phase 2: Loop Engine Integration
  ↓ Learns patterns from high-confidence queries
Phase 3: Hot Pattern Feedback
  ↓ Adapts based on usage frequency
Phase 4: Advanced Refinement
  ↓ Multiple quality improvement strategies
Phase 5: Full Learning Loop
  ↓ Background learning with Thompson Sampling
```

### Phase 1: Scratchpad Provenance Tracking

**What**: Records thought → action → observation → score

```
Query: "What is Thompson Sampling?"

Scratchpad entry:
├─ Query: "What is Thompson Sampling?"
├─ Thoughts:
│   - "This is about bandits and exploration"
│   - "Need sources on Bayesian methods"
├─ Actions:
│   - Tool: semantic_search ("Thompson Sampling")
│   - Retrieved: 5 sources
├─ Observation:
│   - Confidence: 0.87
│   - Latency: 156ms
├─ Score: 0.87 (equals confidence)
└─ Timestamp: 2025-11-16T10:30:45Z
```

**Enabled via:**

```python
from HoloLoom.recursive import weave_with_scratchpad
from HoloLoom.config import Config

spacetime, scratchpad = await weave_with_scratchpad(
    Query(text="What is Thompson Sampling?"),
    Config.fast(),
    shards=shards,
    enable_refinement=True  # Auto-refine if confidence < 0.75
)

# Inspect scratchpad
print(scratchpad.get_history())  # Full trace
print(scratchpad.get_last_entry())  # Recent query
```

### Phase 2: Pattern Learning from Production Logs

**What**: Extracts learnable patterns from high-confidence results

```python
from HoloLoom.recursive import LearningLoopEngine

async with LearningLoopEngine(cfg=config, shards=shards) as engine:
    # Process queries
    for query in queries:
        spacetime = await engine.weave_and_learn(query)

        # System automatically learns patterns if confidence >= 0.75
        # Extracted pattern:
        # {
        #     "motif": "Thompson",
        #     "tool": "semantic_search",
        #     "confidence": 0.92,
        #     "count": 3
        # }

    # View learned patterns
    patterns = engine.pattern_learner.get_learned_patterns()
    # Output: List of (motif, tool, avg_confidence) tuples
```

**Pattern format**:
```
Pattern = (motif, tool, avg_confidence, sample_count)

Example patterns learned:
├─ ("Thompson", "semantic_search", 0.92, 3)
├─ ("bandit", "graph_traversal", 0.88, 5)
├─ ("Bayesian", "embedding_projection", 0.85, 2)
└─ ...
```

### Phase 3: Hot Pattern Feedback (Heat Scores)

**What**: Tracks which patterns are accessed most, boosts them

**Heat Score Formula**:
```
heat = access_count × success_rate × avg_confidence × decay(time)

decay(time) = 0.95 ^ (hours_since_last_access)

Example: Pattern accessed 10 times, 80% success, 0.85 avg confidence, 2 hours ago
heat = 10 × 0.80 × 0.85 × (0.95^2) = 6.47
```

**Adaptive weights**:
```
Hot pattern (heat > threshold):    Weight = 2.0× (boost!)
Normal pattern:                    Weight = 1.0× (baseline)
Cold pattern (heat < low_threshold): Weight = 0.5× (demote)
```

**Enabled via:**

```python
from HoloLoom.recursive import HotPatternFeedbackEngine

async with HotPatternFeedbackEngine(cfg=config, shards=shards) as engine:
    spacetime = await engine.weave(query)

    # View hot patterns
    hot = engine.hot_tracker.get_hot_patterns(limit=10)
    # Output: Top 10 patterns by heat score
```

### Phase 4: Advanced Refinement Strategies

**What**: Multiple strategies for improving low-confidence results

**Available strategies:**

| Strategy | Approach | Use When |
|----------|----------|----------|
| **REFINE** | Iterative context expansion | Need more context |
| **CRITIQUE** | Self-critique, regenerate | Need quality improvement |
| **VERIFY** | Cross-check multiple sources | Need accuracy verification |
| **ELEGANCE** | Clarity → Simplicity → Beauty | Need better explanation |
| **HOFSTADTER** | Recursive self-reference | Need deep understanding |

**Quality tracking:**
```
Quality = 0.7 × confidence + 0.2 × context_richness + 0.1 × completeness

Initial: 0.60 (low confidence)
After VERIFY: 0.85 (+0.25)
After ELEGANCE: 0.92 (+0.07)
```

**Enabled via:**

```python
from HoloLoom.recursive import AdvancedRefiner, RefinementStrategy

refiner = AdvancedRefiner(orchestrator, enable_learning=True)

result = await refiner.refine(
    query=query,
    initial_spacetime=low_confidence_result,
    strategy=RefinementStrategy.VERIFY,  # Or None for auto-select
    max_iterations=3,
    quality_threshold=0.9
)

print(result.summary())
# Output: "Strategy: verify, Iterations: 2, Quality: 0.60 → 0.88"
```

### Phase 5: Background Learning Thread

**What**: Async learning every 60 seconds, updating Thompson Sampling priors

```python
from HoloLoom.recursive import FullLearningEngine

async with FullLearningEngine(
    cfg=config,
    shards=shards,
    enable_background_learning=True,
    learning_update_interval=60.0  # Update every 60 seconds
) as engine:
    # Your queries process here...
    spacetime = await engine.weave(query, enable_refinement=True)

    # View learning statistics
    stats = engine.get_learning_statistics()
    print(f"Tool: {stats['tool_name']}")
    print(f"Successes: {stats['alpha']}")
    print(f"Failures: {stats['beta']}")
    print(f"Expected reward: {stats['expected_reward']:.1%}")
```

**Thompson Sampling updates:**
```
Success (confidence >= 0.75):
    α ← α + confidence
    β ← β + 0

Failure (confidence < 0.75):
    α ← α + 0
    β ← β + (1 - confidence)

Expected Reward: E[X] = α / (α + β)
```

### Performance Overhead

Recursive learning has minimal overhead:

```
Per-query overhead:
├─ Provenance extraction (Phase 1):  <1ms
├─ Pattern extraction (Phase 2):     <1ms (high-conf only)
├─ Heat tracking (Phase 3):          <0.5ms
├─ Thompson/Policy update (Phase 4): <0.5ms
└─ Total per-query:                  <3ms

Background learning (async, runs every 60s):
├─ Pattern mining:  ~500ms
├─ Stats aggregation: ~100ms
├─ Thompson update: ~50ms
└─ Total per cycle: ~650ms (happens in background)
```

---

## 4. Alignment and Safety Framework

### Why Alignment Matters for Agents

As systems become more agentic (making decisions autonomously), safety becomes critical.

**Risks of unaligned agents:**
- Taking unintended actions (goal misspecification)
- Deceiving humans (instrumental goal pursuit)
- Resource acquisition (power-seeking)
- Unexpected behaviors in edge cases

**HoloLoom's approach**: "Safe by default, transparent by design"

Every decision gated by safety checks, with minimal performance cost (<0.1ms).

### Four Core Modules

#### 1. Safety Guardrails (0.039 ms)

**What**: Risk-based action gating

**Risk levels:**
- **LOW** (0-0.3): Allow directly
- **MEDIUM** (0.3-0.6): Allow with monitoring
- **HIGH** (0.6-0.85): Require human approval
- **CRITICAL** (>0.85): Block completely

**Example:**

```python
from HoloLoom.alignment import SafetyGuardrails, ActionRequest

guardrails = SafetyGuardrails(enable_human_in_loop=True)

# Check if action is safe
action_request = ActionRequest(
    action="execute_code",
    context={"code": "print('hello')"}
)

gate_result = await guardrails.gate_action(action_request)

if gate_result.allowed:
    execute_action()
else:
    print(f"Blocked: {gate_result.reason}")
    # Escalate to human if HIGH/CRITICAL
```

**Risk assessment criteria:**
- Action type (code execution highest risk)
- Resource usage
- Persistence (write to disk?)
- External effects (network call?)

#### 2. Deception Detection (0.034 ms)

**What**: Monitors for hidden goals or misleading behavior

**Behavioral probes:**
- "What's your objective?" (goal transparency check)
- "Are you hiding anything?" (deception check)
- "Would you help if incentives changed?" (alignment check)

**Example:**

```python
from HoloLoom.alignment import DeceptionDetector

detector = DeceptionDetector()

# Run transparency check
result = await detector.check_transparency(agent_response)

if result.suspicious:
    print(f"Warning: {result.reason}")
    # Log to audit trail, escalate if needed
```

#### 3. Instrumental Convergence Prevention (0.015 ms)

**What**: Detects power-seeking and resource acquisition behaviors

**Monitored signals:**
- Resource usage (CPU, memory, API calls)
- Self-modification attempts
- Autonomy boundary violations
- Accumulation of capabilities

**Example:**

```python
from HoloLoom.alignment import InstrumentalGuard

guard = InstrumentalGuard(
    max_concurrent_tasks=10,
    max_memory_mb=512,
    allow_self_modification=False
)

# Check before spawning new task
if guard.can_spawn_task():
    task_id = await spawn_task()
else:
    print("Task limit reached, blocking new task")
```

#### 4. Audit Trail (0.015 ms)

**What**: Complete decision provenance for debugging and compliance

```python
from HoloLoom.alignment import AuditTrail

audit = AuditTrail(persist_path=Path("./logs"))

# Every decision logged
await audit.log_decision(
    query="What is Thompson Sampling?",
    action="semantic_search",
    outcome="success",
    confidence=0.87,
    risk_level="LOW",
    timestamp="2025-11-16T10:30:45Z"
)

# Query history
recent = await audit.query_decisions(
    limit=100,
    filters={"risk_level": "HIGH"}
)

print(f"High-risk decisions: {len(recent)}")
```

### Human-In-The-Loop Escalation

For HIGH/CRITICAL risk actions, request human approval:

```python
guardrails = SafetyGuardrails(enable_human_in_loop=True)

gate_result = await guardrails.gate_action(
    ActionRequest(action="delete_database", context={...})
)

if gate_result.risk_level == "CRITICAL":
    # Escalate to human
    approval = await request_human_approval(
        action=gate_result.action,
        reason=gate_result.reason,
        timeout_seconds=3600
    )

    if approval.approved:
        execute_action()
    else:
        log_rejection()
```

### Production Deployment

```python
from HoloLoom.alignment import create_guardrails, create_audit_trail
from HoloLoom.weaving_orchestrator import WeavingOrchestrator
from HoloLoom.config import Config

# Production setup
config = Config.fused()
guardrails = create_guardrails(
    enable_human_in_loop=True,
    audit_trail_path=Path("./alignment_logs"),
    slack_webhook="https://hooks.slack.com/..."  # For alerts
)

async with WeavingOrchestrator(
    cfg=config,
    shards=shards,
    guardrails=guardrails
) as orchestrator:
    spacetime = await orchestrator.weave(query)

    # Automatically logged to audit trail
    # HIGH/CRITICAL risks trigger Slack alerts
```

### Performance: 0.103 ms Overhead

Alignment framework is 29× faster than target:

```
Target:   3.0 ms overhead per query
Actual:   0.103 ms overhead per query
Savings:  97% faster than target!

Breakdown:
├─ Safety Guardrails:  0.039 ms
├─ Deception Detection: 0.034 ms
├─ Instrumental Guard:  0.015 ms
├─ Audit Trail:        0.015 ms
└─ Total:              0.103 ms
```

---

## 5. RAG System Architecture

### Level 1-4 RAG Explained

RAG (Retrieval-Augmented Generation) exists at multiple sophistication levels:

**Level 1: Basic Retrieval**
```
Query → BM25 keyword search → Top K documents → Send to LLM
```
- Simple but effective for straightforward queries
- No semantic understanding
- Typical library search

**Level 2: Hybrid Search**
```
Query → (BM25 + semantic similarity) → Top K → LLM
```
- Combines keyword and semantic matching
- Better for complex queries
- HoloLoom standard

**Level 3: Graph RAG**
```
Query → Entity extraction → Knowledge graph → Multi-hop traversal → LLM
```
- Understands entity relationships
- Can answer questions like "How are X and Y connected?"
- Handles complex knowledge

**Level 4: Agentic RAG** (HoloLoom)
```
Query → Multi-step reasoning → Verification → Research → Plan-Execute → LLM
```
- Makes decisions about what to retrieve next
- Can verify answers before responding
- Self-improving (learns from corrections)
- HoloLoom's Level 4 includes Levels 2+3 as sub-systems

### Why HoloLoom is Level 4 (Agentic + Graph)

HoloLoom RAG combines:
1. **Hybrid search** (Level 2): BM25 + semantic
2. **Graph RAG** (Level 3): Yarn Graph entity relationships
3. **Agentic reasoning** (Level 4): Multi-query decision-making
4. **Multimodal** (bonus): Text + images with CLIP

```
┌─────────────────────────────────────────┐
│        HoloLoom Level 4 RAG              │
├─────────────────────────────────────────┤
│                                         │
│  SimpleRAG / MultimodalRAG              │
│  ├─ Hybrid Search (BM25 + semantic)     │
│  ├─ Graph Traversal (Yarn Graph)        │
│  ├─ Visual Compression (5-20× tokens)   │
│  └─ Reasoning Modes (4 modes)           │
│                                         │
│  Integrates with HoloLoom:              │
│  ├─ hololoom.experience() → ingest      │
│  ├─ hololoom.recall() → retrieve        │
│  └─ hololoom.reflect() → learn          │
│                                         │
└─────────────────────────────────────────┘
```

### SimpleRAG vs MultimodalRAG

**SimpleRAG**: Text-only, zero-config

```python
from HoloLoom.rag import SimpleRAG

async with SimpleRAG() as rag:
    await rag.ingest("Thompson Sampling explanation text")
    result = await rag.query("What is Thompson Sampling?")
    print(result.response)  # LLM-generated answer
```

**MultimodalRAG**: Text + images, more powerful

```python
from HoloLoom.rag import MultimodalRAG

async with MultimodalRAG() as rag:
    # Ingest text and images
    await rag.ingest("Architecture explanation")
    await rag.ingest_photo(
        image="architecture_diagram.png",
        description="System architecture",
        tags=["architecture"]
    )

    # Query with image context
    result = await rag.query_with_image(
        question="Explain this architecture",
        image="architecture_diagram.png"
    )

    print(f"Answer: {result.response}")
    print(f"Compression: {result.compression_ratio:.1f}x savings")
```

### Four Reasoning Modes

Choose reasoning depth based on query complexity:

```python
from HoloLoom.rag import ReasoningMode

async with SimpleRAG() as rag:
    # Simple factual query
    result = await rag.query(
        "What is Thompson Sampling?",
        mode=ReasoningMode.DIRECT,
        max_sources=5
    )
    # Latency: ~150ms
    # Best for: Facts, definitions, simple lookups
```

| Mode | Strategy | Latency | Use Case |
|------|----------|---------|----------|
| **DIRECT** | Single retrieval + generation | ~150ms | "What is X?" |
| **VERIFY** | Retrieve → Generate → Verify | ~600ms | "Prove that..." |
| **RESEARCH** | Multi-query exploration | ~900ms | "Explore topic..." |
| **PLAN_EXECUTE** | Decompose → Research → Synthesize | ~750ms | "How to do X?" |

**VERIFY mode example:**

```python
result = await rag.query(
    "Is Thompson Sampling better than epsilon-greedy?",
    mode=ReasoningMode.VERIFY
)

# Returns:
# - response: LLM answer
# - verification: {
#     verified: True,
#     contradictions: [],
#     confidence: 0.92
#   }
```

### Query Caching (100× Speedup)

RAG automatically caches repeated queries:

```python
async with SimpleRAG() as rag:
    # Query 1 (cold cache)
    result = await rag.query("What is Thompson Sampling?")
    # Latency: ~150ms

    # Query 2 (warm cache)
    result = await rag.query("What is Thompson Sampling?")
    # Latency: <1ms (100× faster!)

    # Disable caching if needed
    result = await rag.query(
        "What is Thompson Sampling?",
        use_cache=False
    )
```

### Visual Compression (5-20× Token Savings)

When multiple images are retrieved, compress knowledge graph to image:

```python
async with MultimodalRAG(enable_visual_compression=True) as rag:
    # After retrieving 15 images
    result = await rag.query_with_image(...)

    if result.compressed_context:
        print(f"Compression: {result.compression_ratio:.1f}x")
        # Example output: "Compression: 12.5x"
        # Saves ~5000 tokens by representing graph as PNG
```

### When to Use RAG vs Base Model

**Use RAG when:**
- You have knowledge base (documents, images)
- Need sources/citations
- Want latest information (knowledge is fresh)
- Reducing hallucinations important
- Cost-effective (external knowledge)

**Use base model when:**
- Knowledge is in training data
- Speed critical (<50ms)
- No external knowledge needed
- Simple factual queries

**Hybrid (recommended):**
```python
# Try base model first (fast)
try:
    result = llm.generate(query)  # ~50ms
    if result.confidence > 0.9:
        return result
except:
    pass

# Fall back to RAG if needed
result = rag.query(query, mode=ReasoningMode.VERIFY)  # ~600ms
return result
```

---

## 6. Phase 5: Universal Grammar Integration

### X-bar Theory Primer

X-bar theory is a linguistic principle for phrase structure:

**Core idea**: All phrases have same hierarchical structure

```
XP (Phrase level)
├─ Spec(ifier)
└─ X'   (Intermediate level)
    ├─ X (Head)
    └─ Comp(lement)
```

**Examples:**

```
Noun Phrase (NP): "the big red ball"
NP
├─ Det "the"
└─ N'
    ├─ A "big"
    └─ N'
        ├─ A "red"
        └─ N "ball"

Verb Phrase (VP): "run quickly"
VP
├─ V "run"
└─ Adv "quickly"

Prepositional Phrase (PP): "in the morning"
PP
├─ P "in"
└─ NP "the morning"
```

**Key benefit**: Recursive structure allows compositional analysis

### Linguistic Matryoshka Gate

Combines linguistic structure with Matryoshka embeddings:

```
┌──────────────────────────────────┐
│  Query: "the big red ball"       │
└────────┬─────────────────────────┘
         │
         ▼
┌──────────────────────────────────┐
│  Linguistic Gate Analysis        │
│                                  │
│  Parse: NP (noun phrase)         │
│  Head: "ball"                    │
│  Modifiers: ["red", "big"]       │
│  Syntactic Type: COUNT NOUN      │
└────────┬─────────────────────────┘
         │
         ▼
┌──────────────────────────────────┐
│  Pre-filter Candidates           │
│  (by syntactic compatibility)    │
│                                  │
│  Candidate 1: NP with "ball"     │ ✓ Match!
│  Candidate 2: VP phrase          │ ✗ Skip (verb)
│  Candidate 3: NP with adjectives │ ✓ Match!
│                                  │
│  Retained: 2/10 (80% reduction)  │
└────────┬─────────────────────────┘
         │
         ▼
┌──────────────────────────────────┐
│  Embed Remaining Candidates      │
│  (only those syntactically valid)│
│                                  │
│  Similarity search on reduced set│
│  Much faster! (10x speedup)      │
└──────────────────────────────────┘
```

### Syntactic Compatibility Scoring

Score candidates based on linguistic match:

```
Compatibility Score = 0.3 × head_match +
                      0.3 × type_match +
                      0.4 × modifier_overlap

Example:
Query: "the big red ball" (NP, head=ball, mods=[big, red])

Candidate 1: "a big red sphere" (NP, head=sphere, mods=[big, red])
├─ head_match: 0.8 (ball vs sphere, similar)
├─ type_match: 1.0 (both NP)
├─ modifier_overlap: 1.0 (same modifiers)
└─ score: 0.3×0.8 + 0.3×1.0 + 0.4×1.0 = 0.94 ✓

Candidate 2: "running quickly" (VP, head=run)
├─ head_match: 0.1 (ball vs run, different)
├─ type_match: 0.0 (NP vs VP)
├─ modifier_overlap: 0.0 (adjective vs adverb)
└─ score: 0.3×0.1 + 0.3×0.0 + 0.4×0.0 = 0.03 ✗
```

### 10-300× Speedup (Compositional Reuse)

How linguistic integration achieves massive speedup:

**Speedup sources:**

1. **Parse caching** (10-50×)
   - Cache X-bar structures, don't re-parse
   - Reuse structure for similar queries

2. **Syntactic pre-filtering** (2-5×)
   - Filter candidates before embedding
   - Only embed syntactically compatible phrases

3. **Compositional reuse** (5-10×)
   - "red ball" cached and reused
   - "big red ball" reuses "red ball" + "big"

**Total multiplicative speedup**: 10 × 2 × 5 = 100× potential

**Realistic with cache misses**: 10-50× typical

**Hot path (everything cached)**: 100-300×

### Graceful Fallback (No Breaking Changes)

If spaCy not available:

```python
from HoloLoom.config import Config

config = Config.fused()
config.linguistic_mode = "both"  # Try to use linguistics

async with WeavingOrchestrator(cfg=config, shards=shards) as shuttle:
    # Automatically degrades:
    # - spaCy available: Use full linguistic filtering
    # - spaCy not available: Fall back to semantic-only
    # - No errors, system still works!
    spacetime = await shuttle.weave(query)
```

### Configuration and Usage

**Enable Phase 5:**

```python
from HoloLoom.config import Config

config = Config.fused()

# Minimal: Compositional cache only (no linguistic analysis)
config.enable_linguistic_gate = False
config.use_compositional_cache = True

# Recommended: Full Phase 5
config.enable_linguistic_gate = True
config.linguistic_mode = "both"  # pre-filter + embedding features
config.use_compositional_cache = True

# Advanced tuning
config.linguistic_weight = 0.3  # How much linguistics influences decisions
config.prefilter_similarity_threshold = 0.3  # Minimum score to keep
config.prefilter_keep_ratio = 0.7  # Keep top 70% of candidates
```

**Usage:**

```python
from HoloLoom.config import Config
from HoloLoom.weaving_orchestrator import WeavingOrchestrator
from HoloLoom.documentation.types import Query

# Create config with Phase 5
config = Config.fused()
config.enable_linguistic_gate = True
config.use_compositional_cache = True

# Create orchestrator
shards = create_memory_shards()
async with WeavingOrchestrator(cfg=config, shards=shards) as shuttle:
    # First query (cold cache)
    spacetime = await shuttle.weave(Query(text="What is passive voice?"))
    # Duration: ~150ms (cold)

    # Repeated query (warm cache)
    spacetime = await shuttle.weave(Query(text="What is passive voice?"))
    # Duration: ~0.5ms (hot) - 300× speedup!
```

**Performance characteristics:**

```
Cold path (no cache):        ~150ms
Warm path (parse cached):    ~30ms (5× speedup)
Hot path (everything):       <1ms (150× speedup)
Very hot (full cache):       <0.5ms (300× potential)
```

---

## Summary: Advanced Topic Mastery

Congratulations! You now understand:

1. **Thompson Sampling**: Bayesian exploration with Beta distributions
2. **Compositional Caching**: 50-300× speedup through linguistic reuse
3. **Recursive Learning**: 5-phase self-improving system
4. **Alignment Framework**: Safe agents with <0.1ms overhead
5. **RAG System**: Level 4 agentic retrieval-augmented generation
6. **Phase 5**: Universal grammar + compositional integration

### Next Steps

**To master these features:**

1. **Try Thompson Sampling**:
   ```bash
   python HoloLoom/policy/unified.py  # Run policy tests
   ```

2. **Enable compositional caching**:
   ```python
   config = Config.fused()
   # Caching enabled by default - measure speedups!
   ```

3. **Run recursive learning**:
   ```bash
   PYTHONPATH=. python demos/demo_full_recursive_learning.py
   ```

4. **Deploy alignment framework**:
   ```python
   config = Config.fused()
   config.enable_alignment = True
   # See alignment framework in action
   ```

5. **Build RAG-powered application**:
   ```bash
   PYTHONPATH=. python demos/demo_rag_qa_simple.py
   ```

6. **Enable Phase 5**:
   ```python
   config.enable_linguistic_gate = True
   config.use_compositional_cache = True
   ```

### Further Reading

- **Thompson Sampling**: `HoloLoom/policy/thompson_sampling.py`
- **Compositional Cache**: `docs/completion-logs/PHASE_5_UG_COMPOSITIONAL_CACHE.md`
- **Recursive Learning**: `archive/session_docs/RECURSIVE_LEARNING_COMPLETE.md`
- **Alignment**: `HoloLoom/alignment/README.md`
- **RAG**: `HoloLoom/rag/README.md`
- **Phase 5**: `PHASE_5_MOONSHOT_COMPLETE.md`

### Questions to Test Mastery

1. **Thompson Sampling**: Explain why Thompson Sampling explores more than epsilon-greedy when a tool has high success rate but low sample count.

2. **Compositional Caching**: Why does "the big red ball" help cache "a big red ball"? What's the cache hit for each tier?

3. **Recursive Learning**: Walk through the 5 phases - what does each add?

4. **Alignment**: What's the performance cost of the safety framework and why is it so low?

5. **RAG**: When would you use Level 4 RAG vs Level 2 vs base model alone?

6. **Phase 5**: How does linguistic pre-filtering reduce embedding computation?

---

**Document Version**: 1.0 (November 2025)

**Status**: Complete and ready for production use

**Questions?** Refer to component documentation or run the demos!
