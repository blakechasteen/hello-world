# Reasoning Engine Architecture

**Visual Guide to HoloLoom Layer 6**

*A picture is worth a thousand words. A good diagram is worth a thousand lines of code.*

---

## System Overview

```
┌───────────────────────────────────────────────────────────────────────┐
│                                                                       │
│                     HoloLoom Weaving Architecture                     │
│                          (10 Layers with Reasoning)                   │
│                                                                       │
├───────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  1. Loom Command        → Pattern Card Selection (BARE/FAST/FUSED)   │
│  2. Chrono Trigger      → Temporal Window Creation                   │
│  3. Yarn Graph          → Thread Selection from Memory               │
│  4. Resonance Shed      → Feature Extraction (DotPlasma)             │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │                                                                 │ │
│  │  5. REASONING ENGINE  → Multi-Step Chain-of-Thought (NEW)      │ │
│  │                                                                 │ │
│  │     Input:  Features + Context                                 │ │
│  │     Output: Reasoning Chain + Confidence                       │ │
│  │                                                                 │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│  6. Warp Space          → Continuous Manifold Tensioning             │
│  7. Convergence Engine  → Tool Selection (Informed by Reasoning)     │
│  8. Tool Execution      → Action with Results                        │
│  9. Spacetime Fabric    → Provenance + Trace (with Reasoning Chain) │
│  10. Reflection Buffer  → Learning from Outcome                      │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘
```

---

## Reasoning Engine Internal Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│                          ReasoningEngine                                 │
│                                                                          │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────┐               │
│  │   Query     │    │   Features   │    │   Context    │               │
│  │   Planner   │    │  Extractor   │    │  Retriever   │               │
│  └──────┬──────┘    └──────┬───────┘    └──────┬───────┘               │
│         │                  │                   │                        │
│         └──────────────────┴───────────────────┘                        │
│                            │                                            │
│                            ▼                                            │
│                   ┌─────────────────┐                                   │
│                   │  Mode Selector  │                                   │
│                   │  (Thompson      │                                   │
│                   │   Sampling)     │                                   │
│                   └────────┬────────┘                                   │
│                            │                                            │
│              ┌─────────────┼─────────────┐                              │
│              │             │             │                              │
│              ▼             ▼             ▼                              │
│        ┌─────────┐   ┌─────────┐   ┌─────────┐                         │
│        │  FAST   │   │STANDARD │   │  DEEP   │                         │
│        │  Mode   │   │  Mode   │   │  Mode   │                         │
│        │         │   │         │   │         │                         │
│        │  <50ms  │   │ ~200ms  │   │ ~500ms  │                         │
│        │ 1 step  │   │ 3-5steps│   │ 5-12step│                         │
│        └────┬────┘   └────┬────┘   └────┬────┘                         │
│             │             │             │                               │
│             └─────────────┼─────────────┘                               │
│                           │                                             │
│                           ▼                                             │
│                  ┌────────────────┐                                     │
│                  │ Self-Verifier  │                                     │
│                  │ + Backtracker  │                                     │
│                  └────────┬───────┘                                     │
│                           │                                             │
│                           ▼                                             │
│                  ┌────────────────┐                                     │
│                  │  Reasoning     │                                     │
│                  │  Result        │                                     │
│                  │  + Chain       │                                     │
│                  └────────────────┘                                     │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Reasoning Flow: STANDARD Mode

```
Query: "What is Thompson Sampling?"
  │
  ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 1: UNDERSTANDING                                       │
│ ─────────────────────────────────────────────────────────── │
│                                                             │
│  QueryPlanner analyzes intent:                              │
│    • Type: FACTUAL                                          │
│    • Complexity: 0.35                                       │
│    • Requirements: [definition, basic facts]                │
│    • Key concepts: [thompson, sampling, bayesian]           │
│                                                             │
│  Confidence: 0.90 🟢                                         │
└─────────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 2: EVIDENCE                                            │
│ ─────────────────────────────────────────────────────────── │
│                                                             │
│  ChainOfThought extracts evidence from context:             │
│    • Found 7 relevant shards                                │
│    • Key evidence:                                          │
│      - "Bayesian approach to bandits"                       │
│      - "Samples from posterior distribution"                │
│      - "Natural exploration/exploitation balance"           │
│                                                             │
│  Confidence: 0.85 🟢                                         │
└─────────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 3: SYNTHESIS                                           │
│ ─────────────────────────────────────────────────────────── │
│                                                             │
│  Reasoning synthesis:                                       │
│    "Thompson Sampling is a Bayesian approach to the        │
│     multi-armed bandit problem. It samples from the        │
│     posterior distribution of each arm's reward and        │
│     naturally balances exploration and exploitation."      │
│                                                             │
│  Confidence: 0.88 🟢                                         │
└─────────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 4: VERIFICATION                                        │
│ ─────────────────────────────────────────────────────────── │
│                                                             │
│  SelfVerifier checks:                                       │
│    ✓ Confidence degradation check                          │
│    ✓ Evidence consistency check                            │
│    ✓ Completeness check                                    │
│                                                             │
│  Result: PASSED ✓                                           │
│  Confidence: 0.90 🟢                                         │
└─────────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────────┐
│ Final Result                                                │
│ ─────────────────────────────────────────────────────────── │
│                                                             │
│  Mode: STANDARD                                             │
│  Steps: 4                                                   │
│  Total Confidence: 0.88                                     │
│  Duration: 185ms                                            │
│                                                             │
│  Reasoning Chain → Spacetime Metadata                       │
└─────────────────────────────────────────────────────────────┘
```

---

## Reasoning Flow: DEEP Mode

```
Complex Query: "Compare Thompson Sampling and UCB algorithms"
  │
  ▼
┌────────────────────────────────────────────────────────────────┐
│ Phase 1: PLANNING                                              │
│ ────────────────────────────────────────────────────────────── │
│                                                                │
│  QueryPlanner creates hierarchical plan:                       │
│                                                                │
│  1. Understand Thompson Sampling                              │
│  2. Gather Thompson evidence                                  │
│  3. Understand UCB algorithm                                  │
│  4. Gather UCB evidence                                       │
│  5. Identify key differences                                  │
│  6. Identify similarities                                     │
│  7. Synthesize comparison                                     │
│                                                                │
│  Dependencies: 2→1, 4→3, 5→[2,4], 6→[2,4], 7→[5,6]           │
└────────────────────────────────────────────────────────────────┘
  │
  ▼
┌────────────────────────────────────────────────────────────────┐
│ Phase 2: EXECUTION                                             │
│ ────────────────────────────────────────────────────────────── │
│                                                                │
│  Execute each plan step:                                       │
│                                                                │
│  Step 1: TS understanding → Bayesian approach                 │
│  Step 2: TS evidence → 5 sources                              │
│  Step 3: UCB understanding → Optimistic selection             │
│  Step 4: UCB evidence → 4 sources                             │
│  Step 5: Differences → TS probabilistic, UCB deterministic    │
│  Step 6: Similarities → Both balance explore/exploit          │
│  Step 7: Synthesis → Comprehensive comparison                 │
│                                                                │
└────────────────────────────────────────────────────────────────┘
  │
  ▼
┌────────────────────────────────────────────────────────────────┐
│ Phase 3: MULTI-PASS VERIFICATION                               │
│ ────────────────────────────────────────────────────────────── │
│                                                                │
│  Pass 1: Accuracy Check                                        │
│    ✓ All confidences ≥ 0.7                                     │
│    ✓ No severe degradation                                     │
│                                                                │
│  Pass 2: Completeness Check                                    │
│    ✓ All step types present                                    │
│    ✓ Chain length ≥ 3                                          │
│                                                                │
│  Pass 3: Consistency Check                                     │
│    ⚠ Potential contradiction detected:                         │
│       "Thompson always better" vs "UCB has advantages"         │
│                                                                │
└────────────────────────────────────────────────────────────────┘
  │
  ▼
┌────────────────────────────────────────────────────────────────┐
│ Phase 4: BACKTRACKING                                          │
│ ────────────────────────────────────────────────────────────── │
│                                                                │
│  Backtracker revises step 5:                                   │
│                                                                │
│  OLD: "Thompson Sampling is always better"                     │
│  NEW: "Thompson Sampling often performs better in practice,    │
│        but UCB has theoretical guarantees"                     │
│                                                                │
│  Contradiction resolved ✓                                      │
│                                                                │
└────────────────────────────────────────────────────────────────┘
  │
  ▼
┌────────────────────────────────────────────────────────────────┐
│ Phase 5: SYNTHESIS                                             │
│ ────────────────────────────────────────────────────────────── │
│                                                                │
│  Weighted synthesis (recent steps weighted 1.5-2.0×):          │
│                                                                │
│  "Thompson Sampling and UCB are both effective algorithms      │
│   for the multi-armed bandit problem. Thompson Sampling uses   │
│   a Bayesian approach and often performs better empirically,   │
│   while UCB provides theoretical regret bounds. The choice     │
│   depends on whether theoretical guarantees or empirical       │
│   performance is more important."                              │
│                                                                │
│  Final Confidence: 0.92                                        │
│  Duration: 520ms                                               │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

## Mode Selection Flow

```
                    Query + Features + Context
                              │
                              ▼
                    ┌──────────────────┐
                    │ Estimate         │
                    │ Complexity       │
                    └────────┬─────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
        Complexity    Complexity      Complexity
          < 0.4        0.4 - 0.7         > 0.7
              │              │              │
              ▼              ▼              ▼
        ┌─────────┐    ┌─────────┐    ┌─────────┐
        │Confidence│   │Confidence│   │Confidence│
        │  Check   │   │  Check   │   │  Check   │
        └────┬─────┘   └────┬─────┘   └────┬─────┘
             │              │              │
    ┌────────┼────────┐     │     ┌────────┼────────┐
    │        │        │     │     │        │        │
    ▼        ▼        ▼     ▼     ▼        ▼        ▼
  ≥0.85   <0.85     ≥0.75 <0.75  ≥0.7    <0.7
    │        │        │     │     │        │
    ▼        ▼        ▼     ▼     ▼        ▼
  FAST   STANDARD  STANDARD DEEP  DEEP    DEEP
  Mode     Mode      Mode   Mode  Mode    Mode
    │        │        │     │     │        │
    └────────┴────────┴─────┴─────┴────────┘
                     │
                     ▼
            ┌─────────────────┐
            │ Thompson Sampling│
            │ Updates Priors   │
            │ Based on Outcome │
            └─────────────────┘
```

---

## Thompson Sampling Learning Loop

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│                    Thompson Sampling Bandit                     │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Initial Priors:                                                │
│  ┌──────────┬───────┬───────┐                                  │
│  │  Mode    │   α   │   β   │  Expected Reward                 │
│  ├──────────┼───────┼───────┤  E[X] = α / (α + β)              │
│  │  FAST    │  10   │   2   │  0.83  (favor initially)         │
│  │  STANDARD│  15   │   5   │  0.75  (balanced)                │
│  │  DEEP    │   5   │   5   │  0.50  (cautious)                │
│  └──────────┴───────┴───────┘                                  │
│                                                                 │
│  After 1000 Queries:                                            │
│  ┌──────────┬───────┬───────┐                                  │
│  │  Mode    │   α   │   β   │  Expected Reward                 │
│  ├──────────┼───────┼───────┤                                  │
│  │  FAST    │  85   │  25   │  0.77  (learned limits)          │
│  │  STANDARD│ 520   │  80   │  0.87  (best performer)          │
│  │  DEEP    │  95   │  15   │  0.86  (complex queries)         │
│  └──────────┴───────┴───────┘                                  │
│                                                                 │
│  Update Rules (per query):                                      │
│  ─────────────────────────────                                 │
│  Success (confidence ≥ 0.75):  α ← α + confidence              │
│  Failure (confidence < 0.75):  β ← β + (1 - confidence)        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Integration with WeavingOrchestrator

```
                     WeavingOrchestrator
                            │
         ┌──────────────────┼──────────────────┐
         │                  │                  │
         ▼                  ▼                  ▼
    Config.fused()    MemoryShards    ReasoningConfig
         │                  │                  │
         │                  │                  │
         └──────────────────┴──────────────────┘
                            │
                            ▼
                  ┌──────────────────┐
                  │  Initialization  │
                  │  ──────────────  │
                  │  • Config loaded │
                  │  • Shards loaded │
                  │  • Reasoning ON  │
                  └────────┬─────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                      Weaving Cycle                           │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  1. LoomCommand   → Pattern: FUSED                          │
│  2. ChronoTrigger → TemporalWindow created                  │
│  3. YarnGraph     → 15 threads selected                     │
│  4. ResonanceShed → DotPlasma extracted                     │
│     ├─ Motifs: [thompson, sampling, bayesian]               │
│     ├─ Embeddings: [96d, 192d, 384d]                        │
│     └─ Spectral: [eigenvalues, SVD]                         │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ 5. ReasoningEngine (NEW)                               │ │
│  │    ───────────────────────────────────────────         │ │
│  │    Input:  DotPlasma features + Context                │ │
│  │    Mode:   STANDARD (auto-selected)                    │
│  │    Steps:  4 reasoning steps                           │ │
│  │    Output: Reasoning chain [0.88 confidence]           │ │
│  │    Duration: 185ms                                      │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  6. WarpSpace      → Continuous manifold                    │
│  7. ConvergenceEng → Tool: "answer" (informed by reasoning) │
│  8. ToolExecution  → Execute with reasoning insights        │
│  9. Spacetime      → Attach reasoning chain to metadata:    │
│     ├─ reasoning_chain: [4 steps]                           │
│     ├─ reasoning_mode: "standard"                           │
│     ├─ reasoning_confidence: 0.88                           │
│     └─ reasoning_duration_ms: 185                           │
│  10. ReflectionBuf → Learn from reasoning outcome           │
│                                                              │
└──────────────────────────────────────────────────────────────┘
                           │
                           ▼
                 ┌──────────────────┐
                 │  Return Result   │
                 │  ──────────────  │
                 │  • Spacetime     │
                 │  • Reasoning     │
                 │  • Provenance    │
                 └──────────────────┘
```

---

## Data Flow Diagram

```
┌─────────┐
│  Query  │
│  "What  │
│   is X?"│
└────┬────┘
     │
     ▼
┌──────────────────────────────────────────────────────────┐
│ Feature Extraction                                       │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  Motifs:     [x, definition, concept]                    │
│  Embeddings: [[0.1, 0.2, ...], [0.3, 0.4, ...]]          │
│  Spectral:   {eigenvalues: [0.5, 0.3, 0.2]}              │
│                                                          │
└──────────────┬───────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────┐
│ Context Retrieval                                        │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  Shard 1: "X is a concept in field Y..."                 │
│  Shard 2: "The definition of X includes..."              │
│  Shard 3: "X relates to Z because..."                    │
│  ...                                                     │
│                                                          │
└──────────────┬───────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────┐
│ Reasoning Engine                                         │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  ┌─────────────────────────────────────────────┐         │
│  │ Step 1: Understanding [0.90]                │         │
│  │   → Query type: factual                     │         │
│  │   → Requirements: definition                │         │
│  └─────────────────────────────────────────────┘         │
│          │                                               │
│          ▼                                               │
│  ┌─────────────────────────────────────────────┐         │
│  │ Step 2: Evidence [0.85]                     │         │
│  │   → Found 3 relevant shards                 │         │
│  │   → Key evidence extracted                  │         │
│  └─────────────────────────────────────────────┘         │
│          │                                               │
│          ▼                                               │
│  ┌─────────────────────────────────────────────┐         │
│  │ Step 3: Synthesis [0.88]                    │         │
│  │   → "X is defined as..."                    │         │
│  │   → Reasoning complete                      │         │
│  └─────────────────────────────────────────────┘         │
│          │                                               │
│          ▼                                               │
│  ┌─────────────────────────────────────────────┐         │
│  │ Step 4: Verification [0.90]                 │         │
│  │   → Consistency check ✓                     │         │
│  │   → Completeness check ✓                    │         │
│  └─────────────────────────────────────────────┘         │
│                                                          │
└──────────────┬───────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────┐
│ Output: ReasoningResult                                  │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  chain: [Step1, Step2, Step3, Step4]                     │
│  mode: STANDARD                                          │
│  total_confidence: 0.88                                  │
│  duration_ms: 185                                        │
│  metadata: {intent: "factual", ...}                      │
│                                                          │
└──────────────┬───────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────┐
│ Integration Points                                       │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  → Scratchpad: thought → action → observation → score   │
│  → Spacetime: Attach chain to metadata                  │
│  → Memory: Store reasoning for future retrieval          │
│  → Metrics: Track performance (Prometheus)               │
│  → Visualization: Render chain as HTML                   │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

---

## Component Interaction Diagram

```
                     ┌──────────────┐
                     │   Query      │
                     │   Features   │
                     │   Context    │
                     └──────┬───────┘
                            │
                            ▼
              ┌─────────────────────────┐
              │   QueryPlanner          │
              │   ─────────────────     │
              │   • Analyze intent      │
              │   • Classify type       │
              │   • Estimate complexity │
              └──────────┬──────────────┘
                         │
                         ▼
              ┌─────────────────────────┐
              │   ModeBandit (Thompson) │
              │   ─────────────────────  │
              │   • Sample from priors  │
              │   • Select mode         │
              └──────────┬──────────────┘
                         │
            ┌────────────┼────────────┐
            │            │            │
            ▼            ▼            ▼
      ┌─────────┐  ┌─────────┐  ┌─────────┐
      │  FAST   │  │STANDARD │  │  DEEP   │
      │  Mode   │  │  Mode   │  │  Mode   │
      └────┬────┘  └────┬────┘  └────┬────┘
           │            │            │
           └────────────┼────────────┘
                        │
                        ▼
              ┌─────────────────────────┐
              │  ChainOfThought         │
              │  ─────────────────────  │
              │  • Extract evidence     │
              │  • Synthesize reasoning │
              │  • Generate steps       │
              └──────────┬──────────────┘
                         │
                         ▼
              ┌─────────────────────────┐
              │   SelfVerifier          │
              │   ─────────────────     │
              │   • Check confidence    │
              │   • Check consistency   │
              │   • Check completeness  │
              └──────────┬──────────────┘
                         │
                    ┌────┴────┐
                    │         │
                 Pass      Fail
                    │         │
                    ▼         ▼
              ┌─────────┐  ┌──────────┐
              │  Done   │  │Backtracker│
              │         │  │   OR     │
              │         │  │Correction│
              └─────────┘  └─────┬────┘
                                 │
                                 ▼
                          ┌─────────────┐
                          │  Revised    │
                          │  Chain      │
                          └──────┬──────┘
                                 │
                                 ▼
                          ┌─────────────┐
                          │  Result     │
                          │  +          │
                          │  Provenance │
                          └─────────────┘
```

---

## State Transitions

```
ReasoningEngine State Machine
────────────────────────────

     ┌─────────┐
     │  IDLE   │
     └────┬────┘
          │ reason(query, features, context)
          ▼
     ┌─────────┐
     │PLANNING │ ← Query analysis, mode selection
     └────┬────┘
          │
          ▼
     ┌─────────┐
     │REASONING│ ← Generate chain (FAST/STANDARD/DEEP)
     └────┬────┘
          │
          ▼
     ┌─────────┐
     │VERIFYING│ ← Self-verification
     └────┬────┘
          │
     ┌────┴────┐
     │         │
  Pass      Fail
     │         │
     ▼         ▼
┌─────────┐  ┌──────────┐
│COMPLETE │  │CORRECTING│
└─────────┘  └─────┬────┘
                   │
                   ▼
              ┌──────────┐
              │BACKTRACK │
              │   OR     │
              │  RETRY   │
              └─────┬────┘
                    │
                    └───────→ VERIFYING
```

---

## Confidence Flow

```
Confidence Progression Through Reasoning Chain

High ┐
     │     ●────────●
0.9  │    /          \
     │   /            ●
0.8  │  ●              \
     │                  ●
0.7  │
     │
0.5  │
     └────────────────────────────→ Time
      1    2    3    4    5

     Understanding → Evidence → Synthesis → Verification → Final

     Ideal pattern: Slight variations around high confidence
     Warning: Large drops indicate issues
     Critical: Steps < 0.5 require correction
```

---

## Summary

This document provides visual architecture diagrams for:

1. **System Overview** - 10-layer weaving architecture
2. **Internal Architecture** - Reasoning engine components
3. **Mode Flows** - STANDARD and DEEP mode execution
4. **Mode Selection** - Decision tree for mode selection
5. **Thompson Sampling** - Learning loop visualization
6. **WeavingOrchestrator Integration** - Full pipeline
7. **Data Flow** - Query → Features → Reasoning → Output
8. **Component Interaction** - How components communicate
9. **State Transitions** - Engine state machine
10. **Confidence Flow** - Confidence progression patterns

---

**Next**: See implementation details in `REASONING_ENGINE_GUIDE.md`
