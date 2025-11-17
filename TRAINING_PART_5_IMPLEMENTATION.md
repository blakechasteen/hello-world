# HoloLoom Complete Training Guide - Part 5: Implementation Walkthroughs

**Date**: November 2025
**Level**: Advanced (Requires Parts 1-4)
**Duration**: 120+ minutes reading
**Focus**: Line-by-line code walkthroughs of actual HoloLoom implementation

---

## Table of Contents

1. [Complete Query Lifecycle Walkthrough](#1-complete-query-lifecycle-walkthrough)
2. [Policy Engine Decision Making](#2-policy-engine-decision-making)
3. [Embedding Computation and Caching](#3-embedding-computation-and-caching)
4. [Knowledge Graph Traversal](#4-knowledge-graph-traversal)
5. [Spacetime Construction and Provenance](#5-spacetime-construction-and-provenance)
6. [Lifecycle Management and Cleanup](#6-lifecycle-management-and-cleanup)

---

## 1. Complete Query Lifecycle Walkthrough

### Overview

This section traces a single query from entry point to final `Spacetime` output. We'll follow the actual code path through all 9 steps of the weaving cycle.

### Entry Point: The Query

```python
# User code (e.g., in demo or application)
from HoloLoom.Documentation.types import Query
from HoloLoom.weaving_orchestrator import WeavingOrchestrator
from HoloLoom.config import Config

# Step 1: Create a query
query = Query(text="What is Thompson Sampling?")
# - text: "What is Thompson Sampling?"
# - metadata: {} (empty by default)

# Step 2: Create orchestrator instance
config = Config.fused()
shards = create_memory_shards()  # Pre-loaded knowledge
orchestrator = WeavingOrchestrator(cfg=config, shards=shards)

# Step 3: Initiate weaving (the main pipeline)
spacetime = await orchestrator.weave(query)
```

### Step-by-Step Lifecycle

#### **Step 1: Loom Command Pattern Selection**

**Location**: `HoloLoom/weaving_orchestrator.py` → `weave()` method

```python
async def weave(self, query: Query, **kwargs) -> Spacetime:
    """
    Main weaving cycle - dispatches to mythRL protocol layers.

    Input:
        - query: Query(text="What is Thompson Sampling?")

    Output:
        - spacetime: Complete woven result with trace
    """
    # Line 1: Start timer
    start_time = datetime.now()
    self.logger.info(f"Weaving started: {query.text[:50]}")

    # Line 2-5: Pattern card selection via Loom Command
    # The Loom Command determines which "pattern card" to use
    loom_cmd = self.loom_command
    pattern_card: PatternCard = loom_cmd.select_pattern(query)

    # pattern_card is one of:
    # - BARE: Minimal (regex motifs, 1 scale)
    # - FAST: Balanced (hybrid motifs, 2 scales)
    # - FUSED: Full (all features, 3 scales)

    self.logger.debug(f"Selected pattern: {pattern_card.mode.name}")

    # For our example, assume FUSED mode was selected:
    # pattern_card = PatternCard(
    #     mode=ExecutionMode.FUSED,
    #     scales=[96, 192, 384],  # All embedding scales
    #     feature_types=['motif', 'embedding', 'spectral'],
    #     timeout=5.0
    # )
```

**Key Data Structure After Step 1**:
```python
pattern_card = {
    'mode': 'FUSED',                      # Full processing mode
    'scales': [96, 192, 384],            # Multi-scale embeddings
    'feature_types': ['motif', 'embedding', 'spectral'],
    'timeout': 5.0,                      # Max execution time
    'pattern_name': 'universal_3scale'   # Which pattern card to apply
}
```

#### **Step 2: Chrono Trigger - Temporal Window Creation**

**Location**: `HoloLoom/chrono/trigger.py` → `fire()` method

```python
    # Line 6-10: Fire Chrono Trigger for temporal control
    chrono_trigger = self.chrono_trigger
    temporal_window: TemporalWindow = chrono_trigger.fire(
        pattern_card=pattern_card,
        query=query
    )

    # TemporalWindow contains time bounds for memory selection
    # Typical FUSED mode window:
    # - activation_window: Last 30 days
    # - decay_factor: Exponential decay (0.95^hours)
    # - heartbeat_interval: 100ms
    # - halt_condition: Query confidence OR timeout

    self.logger.debug(f"Temporal window: {temporal_window}")

    # temporal_window = TemporalWindow(
    #     start_time=datetime.now() - timedelta(days=30),
    #     end_time=datetime.now(),
    #     decay_factor=0.95,
    #     heartbeat_interval=0.1,  # 100ms
    #     halt_on_confidence=0.9,
    #     halt_on_timeout=5.0
    # )
```

**Key Data Structure After Step 2**:
```python
temporal_window = {
    'start_time': datetime(2025, 10, 17),  # 30 days ago
    'end_time': datetime(2025, 11, 16),    # Now
    'decay_factor': 0.95,
    'activation_window_days': 30,
    'halt_on_confidence': 0.9,
    'halt_on_timeout': 5.0
}
```

#### **Step 3: Yarn Graph Thread Selection**

**Location**: `HoloLoom/weaving_orchestrator.py` → Memory retrieval

```python
    # Line 11-20: Select threads from Yarn Graph (memory)
    yarn_graph = self.yarn_graph  # Knowledge graph of all memories

    # Select relevant threads based on temporal window
    # In simple implementation, returns all shards
    # In production, would filter by:
    # - Temporal bounds (start_time → end_time)
    # - Recency weight (newer memories > older memories)
    # - Episode filter (semantic clustering)
    # - Query relevance (BM25 + semantic similarity)

    selected_threads: List[MemoryShard] = yarn_graph.select_threads(
        temporal_window=temporal_window,
        query=query
    )

    # For Thompson Sampling example, might select:
    # MemoryShard(
    #     id="ts_001",
    #     text="Thompson Sampling is a Bayesian approach to exploration...",
    #     entities=["Thompson Sampling", "Bayesian", "exploration"],
    #     motifs=["definition", "probabilistic_model"]
    # ),
    # MemoryShard(
    #     id="ts_002",
    #     text="The key insight is using Beta distributions for each arm...",
    #     entities=["Beta distribution", "multi-armed bandit"],
    #     motifs=["technical_detail"]
    # ),
    # ... more relevant shards

    self.logger.debug(f"Selected {len(selected_threads)} threads")
```

**Key Data Structure After Step 3**:
```python
selected_threads = [
    {
        'id': 'ts_001',
        'text': 'Thompson Sampling is a Bayesian approach...',
        'entities': ['Thompson Sampling', 'Bayesian', 'exploration'],
        'motifs': ['definition']
    },
    {
        'id': 'ts_002',
        'text': 'The key insight is using Beta distributions...',
        'entities': ['Beta distribution'],
        'motifs': ['technical_detail']
    }
]
# Total: 3-5 shards selected based on relevance
```

#### **Step 4: Resonance Shed - Feature Extraction**

**Location**: `HoloLoom/resonance/shed.py` → `lift_features()` method

```python
    # Line 21-40: Resonate - extract features
    resonance_shed = self.resonance_shed  # Feature extraction component

    # The Resonance Shed "lifts" three types of feature threads:
    # 1. MOTIFS: Linguistic patterns (keywords, entities)
    # 2. EMBEDDINGS: Multi-scale vector representations (Ψ)
    # 3. SPECTRAL: Graph-based structural features

    features: Features = await resonance_shed.lift_features(
        query=query,
        context_threads=selected_threads,
        pattern_card=pattern_card
    )

    # Internal: Motif Detection
    # Extract linguistic patterns from query text
    # "What is Thompson Sampling?"
    #  - Motif 1: "What is" (definition question)
    #  - Motif 2: "Thompson Sampling" (entity reference)
    motifs = await self.motif_detector.detect(query.text)

    # Internal: Embedding Generation (Multi-Scale)
    # Generate embeddings at all scales: [96, 192, 384]
    # Each scale provides a different "resolution"
    embeddings = await self.embedder.encode_multiscale(
        query.text,
        scales=[96, 192, 384]
    )
    # embeddings = {
    #     96: [0.21, -0.45, ..., 0.67],   # Coarse scale
    #     192: [...],                       # Medium scale
    #     384: [...]                        # Fine scale
    # }

    # Internal: Spectral Features
    # Extract topological features from knowledge graph
    spectral_features = self.compute_spectral_features(
        subgraph=yarn_graph.get_subgraph(selected_threads)
    )
    # Returns: Laplacian eigenvalues, connected components, centrality

    # Final Features object (DotPlasma - flowing representation)
    features = Features(
        psi=embeddings[384],              # Main 384D embedding (Ψ)
        motifs=motifs,                    # [Motif(...), Motif(...)]
        metrics={
            'coherence': 0.87,            # How coherent is the feature set?
            'density': 0.42,              # Feature space density
            'spectral_gap': 2.34          # Graph connectivity
        },
        confidence=0.92,                  # Feature extraction confidence
        metadata={'embedding_scales': [96, 192, 384]}
    )

    self.logger.debug(f"Features extracted: {len(features.motifs)} motifs, "
                      f"psi shape: {len(features.psi)}")
```

**Key Data Structure After Step 4**:
```python
features = {
    'psi': [0.21, -0.45, ..., 0.67],           # 384D embedding vector
    'motifs': [
        Motif(pattern='definition_question', span=(0, 7), score=0.95),
        Motif(pattern='thompson_sampling', span=(12, 30), score=0.98)
    ],
    'metrics': {
        'coherence': 0.87,
        'density': 0.42,
        'spectral_gap': 2.34
    },
    'confidence': 0.92
}
```

#### **Step 5: Warp Space Tensioning**

**Location**: `HoloLoom/warp/space.py` → `tension()` method

```python
    # Line 41-60: Tension threads into continuous manifold
    warp_space = self.warp_space

    # Warp Space is a temporary manifold where continuous mathematics happens
    # It "tensions" discrete threads (memories, features) into a continuous field

    tensioned_field = await warp_space.tension(
        features=features,
        context_threads=selected_threads,
        temporal_window=temporal_window
    )

    # Internal: Create tensor representation
    # Combine:
    # 1. Query embedding (384D)
    # 2. Context embeddings (selected_threads compressed to 384D each)
    # 3. Spectral features (graph topology)
    # into a single tensor field

    # In FUSED mode:
    # - Query tensor: [1, 384]
    # - Context tensors: [len(selected_threads), 384] = [4, 384]
    # - Combined field: [5, 384] (query + contexts)
    # - With attention: [5, 5, 384] (all-pairs relationships)

    # The field now exists as a continuous tensor manifold
    # enabling mathematical operations like:
    # - Attention (what parts of context are relevant?)
    # - Gradient flow (how to refine embeddings?)
    # - Geodesic distance (similarity in curved space)

    self.logger.debug(f"Warp Space tensioned: field shape {tensioned_field.shape}")

    # tensioned_field = {
    #     'query_tensor': [0.21, -0.45, ..., 0.67],  # Query in manifold
    #     'context_tensors': [[...], [...], [...], [...]],  # 4 shards
    #     'attention_matrix': [[1.0, 0.2, 0.3, 0.1],
    #                          [0.2, 1.0, 0.6, 0.4],
    #                          [0.3, 0.6, 1.0, 0.5],
    #                          [0.1, 0.4, 0.5, 1.0]],  # Pairwise relevance
    #     'manifold_type': 'hyperbolic'  # Or euclidean
    # }
```

**Key Data Structure After Step 5**:
```python
tensioned_field = {
    'query_embedding': [0.21, -0.45, ..., 0.67],  # Query on manifold
    'context_embeddings': [[...], [...], [...], [...]],  # Contexts on manifold
    'attention_weights': [[1.0, 0.2, 0.3, 0.1], ...],   # What's relevant?
    'context_relevance_scores': [0.95, 0.72, 0.81, 0.58],  # Per-shard
    'combined_context_embedding': [...]  # Weighted fusion
}
```

#### **Step 6: Convergence Engine - Policy Decision**

**Location**: `HoloLoom/convergence/engine.py` → `collapse()` method

```python
    # Line 61-80: Decision collapse - continuous → discrete
    convergence_engine = self.convergence_engine

    # The Convergence Engine makes the critical decision:
    # "What tool should we use?" (answer, search, write, calc, etc.)

    # Input: Continuous tensor field (from Warp Space)
    # Output: Discrete tool selection with confidence

    collapse_result: CollapseResult = await convergence_engine.collapse(
        warp_field=tensioned_field,
        features=features,
        context=context,
        available_tools=["answer", "search", "notion_write", "calc"]
    )

    # Internal: Policy network makes prediction
    # The unified policy engine (neural + Thompson Sampling) decides:

    # 1. Neural path: Feed tensioned field through policy network
    policy_logits = policy_network(
        query_embedding=tensioned_field['query_embedding'],
        context_embeddings=tensioned_field['context_embeddings'],
        attention_weights=tensioned_field['attention_weights'],
        motif_gates=extract_motif_gates(features.motifs)
    )
    # policy_logits = [0.1, 0.85, 0.02, 0.03]  # Confidence for each tool
    #                   answer=0.85 (highest)

    # 2. Thompson Sampling path: Bandit priors for exploration
    ts_samples = thompson_sampler.sample(
        tools=["answer", "search", "notion_write", "calc"],
        prior_stats={
            "answer": {"alpha": 50, "beta": 10},      # Successful before
            "search": {"alpha": 20, "beta": 5},
            "notion_write": {"alpha": 15, "beta": 12},
            "calc": {"alpha": 8, "beta": 20}
        }
    )
    # ts_samples ≈ [0.78, 0.25, 0.35, 0.15]  # Sampled rewards

    # 3. Blend: Strategy depends on bandit_strategy config
    # For EPSILON_GREEDY (default):
    # - 90% of time: Use neural prediction (0.85 → "answer")
    # - 10% of time: Use Thompson sample (explore)

    # For our example, neural wins:
    selected_tool = "answer"
    tool_confidence = 0.85

    collapse_result = CollapseResult(
        tool=selected_tool,
        confidence=tool_confidence,
        adapter="fused_mode",  # Which LoRA adapter to use
        strategy="epsilon_greedy",
        reasoning={
            'neural_logits': [0.1, 0.85, 0.02, 0.03],
            'exploration_sample': [0.78, 0.25, 0.35, 0.15],
            'strategy_used': 'exploit (neural)'
        }
    )

    self.logger.info(f"Collapsed to tool: {selected_tool} (confidence: {tool_confidence:.2%})")

    # Update Thompson bandit statistics
    # We'll update this again after execution completes
    await policy.bandit.update(
        tool=selected_tool,
        success=True,  # Optimistic - will refine after execution
        confidence=tool_confidence
    )
```

**Key Data Structure After Step 6**:
```python
collapse_result = {
    'tool': 'answer',                       # Selected tool
    'confidence': 0.85,                     # Selection confidence
    'adapter': 'fused_mode',                # Which LoRA adapter
    'strategy': 'epsilon_greedy',
    'reasoning': {
        'neural_prediction': 'answer (0.85)',
        'exploration_sample': 'search (0.78)',
        'chose': 'neural (exploit)',
        'bandit_stats_before': {
            'answer': {'successes': 50, 'failures': 10},
            'search': {'successes': 20, 'failures': 5}
        }
    }
}
```

#### **Step 7: Tool Execution**

**Location**: `HoloLoom/weaving_orchestrator.py` → `ToolExecutor.execute()` method

```python
    # Line 81-100: Execute the selected tool
    tool_executor = self.tool_executor

    # Call the selected tool with query and context
    tool_result = await tool_executor.execute(
        tool=collapse_result.tool,      # "answer"
        query=query,                     # Original query
        context=context                  # Retrieved context
    )

    # Internal: Handler dispatch
    # For tool="answer":
    async def _handle_answer(self, query: Query, context: Context) -> Dict:
        """Generate an answer based on context."""

        # Step 1: Prepare LLM context
        shard_texts = [shard.text for shard in context.shards[:5]]
        llm_context = "\n\n".join(shard_texts)

        # Step 2: Build prompt
        system_prompt = (
            "You are a helpful AI assistant. "
            "Answer based on the provided context. "
            "Be concise and accurate."
        )

        user_prompt = f"""Context:
{llm_context}

Question: {query.text}

Answer:"""

        # Step 3: Call LLM (Ollama)
        # This is where the actual text generation happens!
        if self.llm:
            response = await self.llm.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                max_tokens=500,
                temperature=0.7
            )
            return {
                "tool": "answer",
                "result": response.content,  # LLM-generated answer!
                "confidence": 0.85,
                "sources": len(context.shards),
                "context_tokens": len(llm_context) // 4
            }

        # Fallback (no LLM)
        return {
            "tool": "answer",
            "result": f"[Fallback] {query.text}\n\nContext: {llm_context[:300]}",
            "confidence": 0.5
        }

    # Result from LLM
    tool_result = {
        "tool": "answer",
        "result": "Thompson Sampling is a Bayesian approach to the multi-armed bandit "
                  "problem that maintains Beta distributions for each option's reward. "
                  "At each step, it samples from these distributions and selects the "
                  "option with the highest sample. This naturally balances exploration "
                  "(trying uncertain options) with exploitation (using known good options).",
        "confidence": 0.85,
        "sources": 4,
        "context_tokens": 487
    }

    self.logger.info(f"Tool execution complete: {tool_result['confidence']:.2%} confidence")
```

**Key Data Structure After Step 7**:
```python
tool_result = {
    'tool': 'answer',
    'result': 'Thompson Sampling is a Bayesian approach...',
    'confidence': 0.85,
    'sources': 4,
    'context_tokens': 487,
    'llm_model': 'llama3.2:3b'
}
```

#### **Step 8: Spacetime Construction - Building Provenance**

**Location**: `HoloLoom/fabric/spacetime.py` → Constructor

```python
    # Line 101-130: Construct Spacetime with complete trace

    # Create comprehensive trace capturing entire journey
    trace = WeavingTrace(
        start_time=start_time,
        end_time=datetime.now(),
        duration_ms=(datetime.now() - start_time).total_seconds() * 1000,

        # Stage timings
        stage_durations={
            'pattern_selection': 2.3,      # Loom Command
            'temporal_window': 1.1,        # Chrono Trigger
            'thread_selection': 5.2,       # Yarn Graph
            'feature_extraction': 45.7,    # Resonance Shed
            'warp_tensioning': 23.4,       # Warp Space
            'policy_decision': 8.9,        # Convergence Engine
            'tool_execution': 78.5,        # LLM generation (slowest!)
            'trace_building': 2.1          # Spacetime construction
        },

        # Feature extraction details
        motifs_detected=['definition_question', 'thompson_sampling'],
        embedding_scales_used=[96, 192, 384],
        spectral_features={
            'laplacian_eigenvalues': [0.0, 0.23, 1.45, 3.67],
            'connected_components': 1,
            'graph_density': 0.42
        },

        # Memory retrieval details
        threads_activated=[
            'ts_001', 'ts_002', 'bandit_001', 'stats_005'
        ],
        context_shards_count=4,
        retrieval_mode='hybrid_bm25_semantic',

        # Decision details
        policy_adapter='fused_mode',
        tool_selected='answer',
        tool_confidence=0.85,
        bandit_statistics={
            'answer': {'successes': 50, 'failures': 10, 'prior': 0.833},
            'search': {'successes': 20, 'failures': 5, 'prior': 0.800},
            'total_samples': 90
        },

        # Error tracking
        errors=[],  # None in success case
        warnings=[]
    )

    # Build final Spacetime object
    spacetime = Spacetime(
        # Output content
        response_text=tool_result['result'],
        response_confidence=tool_result['confidence'],

        # Complete trace
        trace=trace,

        # Semantic representation
        query_embedding=features.psi,        # Query's 384D embedding
        context_embeddings=context.shards,   # Retrieved memories
        response_embedding=None,             # (Optional) Embedding of response

        # Overall metrics
        confidence=tool_result['confidence'],
        quality_score=0.92,  # Calculated from multiple signals

        # Metadata
        metadata={
            'query_text': query.text,
            'sources': [s.id for s in context.shards],
            'model_used': 'llama3.2:3b',
            'execution_mode': 'FUSED',
            'timestamp': datetime.now().isoformat()
        }
    )

    self.logger.info(f"Spacetime constructed in {trace.duration_ms:.1f}ms")
```

**Key Data Structure After Step 8** (Spacetime Object):
```python
spacetime = {
    'response_text': 'Thompson Sampling is a Bayesian approach...',
    'response_confidence': 0.85,
    'confidence': 0.85,
    'quality_score': 0.92,
    'trace': {
        'start_time': '2025-11-16T14:23:15.234',
        'end_time': '2025-11-16T14:23:15.345',
        'duration_ms': 111.0,
        'stage_durations': {
            'pattern_selection': 2.3,
            'feature_extraction': 45.7,
            'tool_execution': 78.5
        },
        'tool_selected': 'answer',
        'tool_confidence': 0.85,
        'threads_activated': ['ts_001', 'ts_002', 'bandit_001', 'stats_005']
    },
    'metadata': {
        'query_text': 'What is Thompson Sampling?',
        'sources': ['ts_001', 'ts_002', 'bandit_001', 'stats_005'],
        'execution_mode': 'FUSED',
        'timestamp': '2025-11-16T14:23:15.345'
    }
}
```

#### **Step 9: Reflection Buffer Update**

**Location**: `HoloLoom/reflection/buffer.py` → `store()` method

```python
    # Line 131-145: Store in Reflection Buffer for learning
    if self.enable_reflection:
        learning_signal = LearningSignal(
            spacetime=spacetime,
            reward=tool_result['confidence'],  # Quality signal
            feedback={
                'user_satisfaction': None,     # (Optional) Will be set by user
                'automatic_quality': spacetime.quality_score,
                'tool_used': collapse_result.tool,
                'execution_time_ms': trace.duration_ms
            }
        )

        # Store episodic memory for future learning
        await self.reflection_buffer.store(learning_signal)

        # This enables:
        # 1. Pattern learning: What worked well?
        # 2. Tool statistics: Which tools are most successful?
        # 3. Query type clustering: Similar queries need similar approaches
        # 4. Bandit prior updates: Refine exploration priors

        self.logger.debug(f"Stored learning signal: reward={learning_signal.reward:.2%}")

    # Return complete Spacetime to caller
    return spacetime
```

### Complete Query Lifecycle Summary

```
Query Input
    ↓
1. Loom Command
    ├─ Select pattern card (BARE/FAST/FUSED)
    └─ Output: PatternCard
    ↓
2. Chrono Trigger
    ├─ Create temporal window
    └─ Output: TemporalWindow
    ↓
3. Yarn Graph
    ├─ Select relevant threads (memories)
    └─ Output: List[MemoryShard]
    ↓
4. Resonance Shed
    ├─ Extract motifs
    ├─ Generate embeddings (multi-scale)
    ├─ Compute spectral features
    └─ Output: Features (DotPlasma)
    ↓
5. Warp Space
    ├─ Tension threads into tensor field
    ├─ Compute attention relationships
    └─ Output: TensionedField
    ↓
6. Convergence Engine
    ├─ Neural prediction + Thompson Sampling
    ├─ Select tool (answer, search, write, calc)
    └─ Output: CollapseResult
    ↓
7. Tool Execution
    ├─ Call selected tool (e.g., LLM for "answer")
    └─ Output: ToolResult
    ↓
8. Spacetime Construction
    ├─ Build complete trace
    ├─ Package response with provenance
    └─ Output: Spacetime
    ↓
9. Reflection Buffer
    ├─ Store learning signal
    ├─ Update bandit priors
    └─ Update tool statistics
    ↓
Spacetime Output (with complete trace)
```

---

## 2. Policy Engine Decision Making

### Overview

The Policy Engine is the "shuttle" - it decides which tool to use based on extracted features and context. This section walks through the actual neural network computation and Thompson Sampling integration.

### Entry Point: Policy Decision

```python
# From HoloLoom/policy/unified.py
from HoloLoom.policy.unified import create_policy, NeuralCore
from HoloLoom.Documentation.types import Features, Context

# Create policy instance
policy = create_policy(
    mem_dim=384,                    # Embedding dimension
    emb=embedder,                   # Embedding model
    scales=[96, 192, 384],          # Multi-scale embeddings
    n_tools=4,                      # Number of available tools
    bandit_strategy="epsilon_greedy",  # Exploration strategy
    epsilon=0.10                    # 10% exploration rate
)

# Make a decision
action_plan = await policy.decide(
    features=features,              # Extracted from query
    context=context                 # Retrieved memories
)
```

### Neural Network Architecture Walkthrough

#### **Layer 1: Input Encoding**

```python
# File: HoloLoom/policy/unified.py, NeuralCore.__init__()
class NeuralCore(nn.Module):
    """
    Neural decision network for tool selection.

    Architecture (FUSED mode):
    Input (384D) → Embedding Fusion → Attention → LoRA → Output (4 tools)
    """

    def __init__(self, d_model: int, n_tools: int, n_motifs: int = 8):
        super().__init__()
        self.d_model = d_model  # 384
        self.n_tools = n_tools  # 4

        # Input fusion layer
        # Combines multiple input signals:
        # - Query embedding (Ψ)
        # - Context embeddings
        # - Spectral features
        self.input_fusion = nn.Linear(d_model * 2, d_model)
        # Takes: [query_emb (384) + context_mean (384)] = 768
        # Outputs: Fused representation (384)

        # Transformer-style processing
        self.attention_layer = MotifGatedMHA(d_model, n_heads=4, n_motifs=n_motifs)
        # Adds dynamic attention gating based on detected motifs
        # Allows different attention heads to activate for different query types

        # LoRA-style adapters for different execution modes
        self.adapters = LoRALikeFFN(d_model, d_ff=1024, r=8, n_adapters=4)
        # 4 adapters for: BARE, FAST, FUSED, RESEARCH modes

        # Tool selection head
        self.tool_logits = nn.Linear(d_model, n_tools)
        # Maps: d_model (384) → n_tools (4 logits)
```

#### **Layer 2: Input Preparation**

```python
# Inside NeuralCore.forward()
def forward(self,
            query_embedding: torch.Tensor,      # [1, 384]
            context_embeddings: torch.Tensor,   # [N, 384] where N = num contexts
            motif_vector: torch.Tensor) -> torch.Tensor:  # [1, 8]
    """
    Execute neural decision network.

    Args:
        query_embedding: Query's multi-scale embedding (Ψ)
        context_embeddings: Retrieved memory embeddings
        motif_vector: Binary/soft gates for detected motifs

    Returns:
        logits: [1, 4] - unnormalized confidence for each tool
    """

    # Step 1: Aggregate context embeddings
    # Combine all retrieved memories into single context representation
    context_mean = context_embeddings.mean(dim=0, keepdim=True)  # [1, 384]
    # This is a simple average. More sophisticated: use attention-weighted sum

    context_max = context_embeddings.max(dim=0, keepdim=True)[0]  # [1, 384]
    # Also track maximum for diversity signal

    # Step 2: Fuse query and context
    combined = torch.cat([query_embedding, context_mean], dim=1)  # [1, 768]
    # Concatenate: query (384) + context_mean (384) = 768 total

    fused = self.input_fusion(combined)  # [1, 384]
    # Project back to 384 dimensions
    # This is a learnable transformation that learns how to combine signals

    # Activation function
    fused = F.relu(fused)  # ReLU to introduce non-linearity
```

#### **Layer 3: Motif-Gated Attention**

```python
    # Step 3: Attention with motif gating
    # Custom attention mechanism that's modulated by detected linguistic patterns

    attn_output, attn_weights = self.attention_layer(
        x=fused.unsqueeze(0),           # [1, 1, 384] - add sequence dimension
        motif_ctrl=motif_vector         # [1, 8] - gate control vector
    )
    # attn_output: [1, 1, 384]
    # attn_weights: [1, H, 1, 1] where H=4 (num heads)

    # What's happening:
    # 1. Motif vector controls attention gates
    #    - If "definition_question" detected → activate definition-answering head
    #    - If "comparison_question" detected → activate comparison head
    # 2. Each head can specialize on different question types
    # 3. Gates are soft (continuous), not hard (discrete)

    # Remove sequence dimension
    attn_output = attn_output.squeeze(0)  # [1, 384]
```

#### **Layer 4: Adapter Selection**

```python
    # Step 4: Select execution adapter based on mode
    # Adapters allow same base network to work for different complexity modes

    adapter_idx = self.get_adapter_for_mode(execution_mode)
    # execution_mode could be: BARE (0), FAST (1), FUSED (2), RESEARCH (3)

    # LoRA-style residual adapter
    adapted = self.adapters(attn_output, adapter_idx=adapter_idx)
    # [1, 384] → [1, 384]

    # What the adapter does:
    # base_ff = F.relu(fc1(adapted))  # Project up to 1024
    # base_ff = fc2(base_ff)           # Project back to 384
    # adapter = adapter_network(adapted)  # Low-rank transformation
    # output = base_ff + adapter       # Residual connection

    # Different adapters → Different tool selection preferences
    # - BARE adapter: Fast & simple tools (answer, search)
    # - FUSED adapter: All tools available, complex reasoning
```

#### **Layer 5: Tool Logits**

```python
    # Step 5: Generate tool selection logits
    logits = self.tool_logits(adapted)  # [1, 4]
    # logits = [0.1, 0.85, 0.02, 0.03]  for [answer, search, write, calc]

    return logits
```

### Thompson Sampling Integration

#### **Bandit Statistics Tracking**

```python
# From HoloLoom/policy/thompson_sampling.py
from HoloLoom.policy.thompson_sampling import TSBandit

class TSBandit:
    """Thompson Sampling bandit for exploration/exploitation."""

    def __init__(self, tools: List[str]):
        self.tools = tools  # ["answer", "search", "notion_write", "calc"]

        # Track Beta distribution parameters for each tool
        self.stats = {
            tool: {
                'alpha': 1,  # Prior successes
                'beta': 1,   # Prior failures
                'samples': 0,  # Times sampled
                'successes': 0,  # Actual successes
            }
            for tool in tools
        }

    def sample(self) -> Dict[str, float]:
        """
        Sample from Beta distributions.

        Returns: Dict mapping tool → sampled reward (0.0 to 1.0)
        """
        samples = {}

        for tool in self.tools:
            stats = self.stats[tool]

            # Sample from Beta(alpha, beta)
            # Beta distribution is conjugate prior for Bernoulli (success/failure)
            sampled_reward = np.random.beta(
                a=stats['alpha'],  # Successes + 1
                b=stats['beta']    # Failures + 1
            )
            # sampled_reward ≈ 0.75 for "answer" (50 successes, 10 failures)
            # sampled_reward ≈ 0.20 for "calc" (8 successes, 20 failures)

            samples[tool] = sampled_reward

        return samples

    def update(self, tool: str, success: bool, confidence: float):
        """
        Update bandit statistics after tool execution.

        Args:
            tool: Tool that was used
            success: Did it succeed?
            confidence: How confident was the LLM?
        """
        stats = self.stats[tool]

        # Update counts
        stats['samples'] += 1
        if success:
            stats['successes'] += 1
            stats['alpha'] += confidence  # Weight by confidence
        else:
            stats['beta'] += (1 - confidence)

        # Calculate expected reward
        expected = stats['alpha'] / (stats['alpha'] + stats['beta'])

        self.logger.info(
            f"Updated {tool}: "
            f"successes={stats['successes']}, "
            f"samples={stats['samples']}, "
            f"expected_reward={expected:.2%}"
        )
```

### Policy Decision Strategy

#### **Strategy 1: Epsilon-Greedy (Default)**

```python
# 90% exploitation (neural), 10% exploration (Thompson)
async def decide_epsilon_greedy(
    self,
    neural_logits: np.ndarray,  # [0.1, 0.85, 0.02, 0.03]
    epsilon: float = 0.1
) -> str:
    """
    Epsilon-greedy strategy for exploration.

    Args:
        neural_logits: Neural network's confidence for each tool
        epsilon: Exploration probability (default 10%)

    Returns:
        Selected tool name
    """

    # Random coin flip
    if np.random.random() < epsilon:  # 10% chance
        # EXPLORE: Sample from Thompson bandit
        ts_samples = self.bandit.sample()  # {"answer": 0.78, "search": 0.25, ...}
        selected_tool = max(ts_samples, key=ts_samples.get)
        # Might select unexpected tool for exploration
        self.logger.info(f"Exploring: selected {selected_tool}")

    else:  # 90% chance
        # EXPLOIT: Use neural prediction
        tool_idx = np.argmax(neural_logits)  # argmax([0.1, 0.85, 0.02, 0.03]) = 1
        selected_tool = self.tools[tool_idx]  # self.tools[1] = "search"
        self.logger.info(f"Exploiting: selected {selected_tool}")

    return selected_tool
```

#### **Strategy 2: Bayesian Blend**

```python
# 70% neural + 30% bandit priors
async def decide_bayesian_blend(
    self,
    neural_logits: np.ndarray,  # [0.1, 0.85, 0.02, 0.03]
    neural_weight: float = 0.7,
    bandit_weight: float = 0.3
) -> str:
    """
    Blend neural predictions with Thompson Sampling priors.

    Args:
        neural_logits: Neural network logits
        neural_weight: Weight for neural (default 70%)
        bandit_weight: Weight for bandit (default 30%)

    Returns:
        Selected tool name
    """

    # Normalize neural logits to probabilities
    neural_probs = softmax(neural_logits)  # [0.10, 0.72, 0.10, 0.08]

    # Get bandit priors
    bandit_priors = {}
    for tool in self.tools:
        stats = self.bandit.stats[tool]
        # Prior success probability
        prior = (stats['alpha'] - 1) / (stats['alpha'] + stats['beta'] - 2)
        bandit_priors[tool] = max(0, prior)  # Clamp to [0, 1]
    # bandit_priors = {"answer": 0.83, "search": 0.80, "write": 0.55, "calc": 0.29}

    # Blend
    blended = {}
    for i, tool in enumerate(self.tools):
        blended[tool] = (
            neural_weight * neural_probs[i] +
            bandit_weight * bandit_priors[tool]
        )
    # blended = {
    #     "answer": 0.7*0.72 + 0.3*0.83 = 0.504 + 0.249 = 0.753,
    #     "search": 0.7*0.10 + 0.3*0.80 = 0.070 + 0.240 = 0.310,
    #     ...
    # }

    # Select highest blend
    selected_tool = max(blended, key=blended.get)
    return selected_tool
```

#### **Strategy 3: Pure Thompson**

```python
# 100% Thompson Sampling (ignore neural)
async def decide_pure_thompson(self) -> str:
    """
    Pure Thompson Sampling without neural network.

    Useful for:
    - Maximum exploration in uncertain environments
    - Testing Thompson Sampling independently
    - Research on bandit algorithms

    Returns:
        Selected tool name
    """

    ts_samples = self.bandit.sample()
    selected_tool = max(ts_samples, key=ts_samples.get)
    return selected_tool
```

### Complete Decision Example

```python
# Putting it all together

# Query: "What is Thompson Sampling?"
query = Query(text="What is Thompson Sampling?")

# Feature extraction produces
features = Features(
    psi=[0.21, -0.45, ..., 0.67],  # 384D embedding
    motifs=[
        Motif(pattern='definition_question', score=0.95),
        Motif(pattern='thompson_sampling', score=0.98)
    ],
    metrics={'coherence': 0.87}
)

# Neural network forward pass
neural_logits = policy.neural_core.forward(
    query_embedding=features.psi,
    context_embeddings=context_tensors,
    motif_vector=extract_motif_gates(features.motifs)
)
# neural_logits = torch.tensor([0.1, 0.85, 0.02, 0.03])

# Apply bandit strategy
if policy.bandit_strategy == BanditStrategy.EPSILON_GREEDY:
    selected_tool = await policy.decide_epsilon_greedy(
        neural_logits=neural_logits.cpu().numpy(),
        epsilon=0.10
    )
    # 90% chance: "search" (argmax)
    # 10% chance: Random Thompson sample

# Update bandit after execution
await policy.bandit.update(
    tool=selected_tool,
    success=True,  # Assuming execution succeeded
    confidence=0.85  # From LLM generation
)

# Result
print(f"Selected: {selected_tool}")
# Output: "Selected: search"
```

---

## 3. Embedding Computation and Caching

### Overview

This section explains how queries are converted into multi-scale embeddings and retrieved from cache.

### Matryoshka Multi-Scale Embeddings

#### **Architecture**

```python
# From HoloLoom/embedding/spectral.py
from HoloLoom.embedding.spectral import MatryoshkaEmbeddings

class MatryoshkaEmbeddings:
    """
    Multi-scale Matryoshka embeddings.

    A Matryoshka is a Russian nesting doll. Similarly, our embeddings
    nest different scales inside each other:

    - 384D: Fine-grained (highest detail)
      ├─ 192D: Medium scale (coarse detail)
      │  └─ 96D: Coarse scale (broad concepts)

    This enables:
    1. Coarse-to-fine search (start with 96D for speed)
    2. Progressive refinement (96D → 192D → 384D)
    3. Multi-scale retrieval (use appropriate scale for context)
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # Use sentence-transformers for embeddings
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = 384  # MiniLM outputs 384D vectors

        # Cache for computed embeddings
        self.cache = {}  # text → embedding dict
```

#### **Encoding Process**

```python
async def encode(self, text: str) -> Dict[int, np.ndarray]:
    """
    Encode text into multi-scale embeddings.

    Args:
        text: Input text to encode

    Returns:
        Dictionary mapping scale → embedding vector:
        {
            96: [0.21, -0.45, ..., 0.67],    # Coarse scale
            192: [...],                       # Medium scale
            384: [...]                        # Fine scale
        }
    """

    # Step 1: Check cache
    if text in self.cache:
        return self.cache[text]

    # Step 2: Generate 384D embedding (full scale)
    # This is what SentenceTransformer produces
    embedding_384 = self.model.encode(text)  # np.ndarray of shape (384,)
    # embedding_384 = [0.21, -0.45, ..., 0.67] (384 values)

    # Step 3: Extract multi-scale views
    # This is where Matryoshka prefix property comes in!
    # Key insight: First k dimensions contain k-dimensional representation

    embeddings = {
        96: embedding_384[:96],      # Take first 96 dimensions
        192: embedding_384[:192],    # Take first 192 dimensions
        384: embedding_384           # Full 384 dimensions
    }
    # This is essentially "free" - just slicing, no recomputation!

    # Step 4: Cache result
    self.cache[text] = embeddings

    # Step 5: Return
    return embeddings
```

#### **Multi-Scale Retrieval**

```python
async def retrieve_similar(
    self,
    query_text: str,
    candidate_texts: List[str],
    scale: int = 384,
    top_k: int = 5
) -> List[Tuple[str, float]]:
    """
    Retrieve similar texts using specific embedding scale.

    Args:
        query_text: Query string
        candidate_texts: Texts to search
        scale: Embedding scale (96, 192, or 384)
        top_k: Number of results to return

    Returns:
        List of (text, similarity_score) tuples
    """

    # Encode query at requested scale
    query_embeddings = await self.encode(query_text)
    query_vec = query_embeddings[scale]  # Get vector at scale
    # query_vec shape: (scale,) e.g., (192,) for medium scale

    # Encode candidates
    similarities = []
    for candidate in candidate_texts:
        candidate_embeddings = await self.encode(candidate)
        candidate_vec = candidate_embeddings[scale]

        # Compute cosine similarity
        sim = cosine_similarity(query_vec, candidate_vec)
        # cosine_similarity: dot product / (norm1 * norm2)
        # Range: [-1, 1], but typically [0, 1] for normalized embeddings

        similarities.append((candidate, sim))

    # Sort by similarity (descending) and take top_k
    results = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]
    return results

    # Example:
    # Query: "What is Thompson Sampling?"
    #
    # Step 1: Encode at scale=192 (medium)
    # query_vec = [0.21, -0.45, ..., ...]  (192 dims)
    #
    # Step 2: Compare against candidates
    # Candidate 1: "Thompson Sampling balances exploration..."
    #   similarity = 0.92
    # Candidate 2: "Beta distributions are used for..."
    #   similarity = 0.78
    # Candidate 3: "The policy network decides which tool to use..."
    #   similarity = 0.34
    #
    # Step 3: Return top 5
    # [
    #   ("Thompson Sampling balances...", 0.92),
    #   ("Beta distributions are...", 0.78),
    #   ...
    # ]
```

### Zero-Copy Optimization

#### **Prefix Property Exploitation**

```python
# From HoloLoom/embedding/zero_copy.py
import mmap
import numpy as np

class ZeroCopyEmbeddings:
    """
    Memory-mapped zero-copy embeddings.

    Key insight: Matryoshka embeddings have prefix property:
    - First 96 dimensions contain 96D representation
    - First 192 dimensions contain 192D representation
    - All 384 dimensions contain full representation

    Instead of extracting with slicing (which copies), we use
    memory-mapped views (zero-copy).
    """

    def __init__(self, cache_path: str, cache_size: int = 10000):
        self.cache_path = cache_path
        self.cache_size = cache_size

        # Create memory-mapped array
        # Stores full 384D embeddings on disk
        self.embeddings_mmap = np.memmap(
            cache_path,
            dtype=np.float32,
            mode='r+',
            shape=(cache_size, 384)
        )
        # embeddings_mmap[i] = 384D embedding for text i

        # Index: text → row number
        self.index = {}  # "text" → row_id

    def get_view(self, text: str, scale: int) -> np.ndarray:
        """
        Get zero-copy view at requested scale.

        Args:
            text: Text to retrieve
            scale: Scale (96, 192, or 384)

        Returns:
            View (no copying) of embedding at scale
        """

        # Step 1: Get row index
        row_id = self.index[text]

        # Step 2: Get zero-copy view (NO COPY!)
        # Instead of: embedding = embeddings_mmap[row_id, :scale].copy()
        # We do:
        full_embedding = self.embeddings_mmap[row_id]  # Memory-mapped row
        view = full_embedding[:scale]                   # Slice (view, not copy!)

        return view
        # This is essentially free from a memory perspective!
        # We're just pointing to different memory locations
```

#### **Performance Characteristics**

```
# Benchmark: Extract embeddings at different scales

# Without zero-copy (traditional approach)
embedding_384 = np.array([...])  # Load full 384D from disk
embedding_96 = embedding_384[:96].copy()   # Copy + slice
embedding_192 = embedding_384[:192].copy() # Copy + slice
# Total time: ~10ms per query

# With zero-copy (Matryoshka views)
embedding_384 = mmap[row][:]    # Memory-mapped view (free!)
embedding_96 = mmap[row][:96]   # Slice (view, no copy!)
embedding_192 = mmap[row][:192] # Slice (view, no copy!)
# Total time: ~0.26ms per query
# Speedup: 37.7x!

# Memory usage
# Traditional: 3 copies × 384 × 4 bytes = 4.6KB per query
# Zero-copy: 1 copy × 384 × 4 bytes = 1.5KB per query
# Savings: 50%
```

### Query Cache

#### **Three-Tier Caching**

```python
# From HoloLoom/performance/cache.py
from HoloLoom.performance.cache import QueryCache

class QueryCache:
    """
    Three-tier cache for query results.

    Tier 1: PARSE cache
        Cache: Motif detection results
        Hit rate: 15-20% (reuses detected patterns)
        Speedup: 5x (avoid regex matching)

    Tier 2: MERGE cache
        Cache: Composition results (multi-shard synthesis)
        Hit rate: 5-10% (similar context combinations)
        Speedup: 3x (avoid context fusion)

    Tier 3: SEMANTIC cache
        Cache: Full query results (query → embedding → retrieval)
        Hit rate: 30-50% (semantic similarity matches)
        Speedup: 100x (avoid LLM generation)
    """

    def __init__(self, max_size: int = 1000, ttl_seconds: float = 3600):
        # Tier 1: Motif patterns
        self.parse_cache = {}  # query_text → [Motif, ...]

        # Tier 2: Composition results
        self.merge_cache = {}  # frozenset(shard_ids) → merged_result

        # Tier 3: Full results
        self.semantic_cache = {}  # query_embedding_hash → spacetime

        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.access_times = {}  # For LRU eviction
```

#### **Cache Operations**

```python
async def get(self, query: Query) -> Optional[Spacetime]:
    """
    Retrieve query result from cache if available.

    Args:
        query: Query to look up

    Returns:
        Cached Spacetime if found, None otherwise
    """

    # Compute cache key
    # Use embedding hash for semantic matching
    query_embedding = await embedder.encode(query.text)
    cache_key = hash_embedding(query_embedding[384])  # Hash 384D embedding

    # Tier 3: Check semantic cache
    if cache_key in self.semantic_cache:
        cached_result = self.semantic_cache[cache_key]

        # Check TTL
        age = time.time() - self.access_times[cache_key]
        if age < self.ttl_seconds:
            self.logger.info(f"Cache HIT (semantic): {query.text[:30]}")
            return cached_result
        else:
            # Expired
            del self.semantic_cache[cache_key]

    # Cache miss
    self.logger.debug(f"Cache MISS: {query.text[:30]}")
    return None

async def put(self, query: Query, spacetime: Spacetime) -> None:
    """
    Store query result in cache.

    Args:
        query: Original query
        spacetime: Result to cache
    """

    # Compute cache key
    query_embedding = await embedder.encode(query.text)
    cache_key = hash_embedding(query_embedding[384])

    # Check size
    if len(self.semantic_cache) >= self.max_size:
        # Evict oldest (LRU)
        oldest_key = min(self.access_times, key=self.access_times.get)
        del self.semantic_cache[oldest_key]
        del self.access_times[oldest_key]

    # Store
    self.semantic_cache[cache_key] = spacetime
    self.access_times[cache_key] = time.time()

    self.logger.debug(f"Cache PUT: {query.text[:30]} "
                      f"(cache size: {len(self.semantic_cache)}/{self.max_size})")
```

#### **Cache Example**

```
Query 1: "What is Thompson Sampling?"
→ Parse cache MISS: detect motifs (5ms)
→ Merge cache MISS: retrieve + fuse context (20ms)
→ Semantic cache MISS: LLM generation (80ms)
→ Total: 105ms
→ Store in cache ✓

Query 2: "Tell me about Thompson Sampling" (very similar)
→ Embedding is similar to Query 1
→ Semantic cache HIT! ✓
→ Return cached result immediately
→ Total: <1ms
→ Speedup: 105x! ✓

Query 3: "What is Thompson Sampling?" (exact match)
→ Exact same embedding as Query 1
→ Semantic cache HIT! ✓
→ Return cached result
→ Total: <1ms
```

---

## 4. Knowledge Graph Traversal

### Overview

The Yarn Graph (KG) is the persistent memory of relationships. This section explains how entities and relationships are stored and traversed.

### Knowledge Graph Storage

#### **Data Structures**

```python
# From HoloLoom/memory/graph.py
from HoloLoom.memory.graph import KG, KGEdge
import networkx as nx

class KG:
    """Knowledge Graph - persistent symbolic memory."""

    def __init__(self):
        # Use NetworkX MultiDiGraph
        # - Multi: Multiple edges between same entities
        # - Di: Directed (can have A→B and B→A separately)
        # - Graph: Nodes and edges
        self.G = nx.MultiDiGraph()

        # Fast neighbor lookup index
        self._entity_index = {}  # entity_name → set(neighbor_names)
```

#### **Edge Representation**

```python
@dataclass
class KGEdge:
    """
    A directed edge in the knowledge graph.

    Features:
    - Typed relationships (IS_A, USES, MENTIONS, etc.)
    - Weighted confidence
    - Bi-temporal tracking (event_time, ingestion_time)
    - Metadata linking to source
    """
    src: str                    # Source entity ("Thompson Sampling")
    dst: str                    # Destination entity ("Bayesian approach")
    type: str                   # Relationship type ("IS_A")
    weight: float = 1.0         # Confidence (0.0 to 1.0)
    span_id: Optional[str] = None  # Link to source text span
    metadata: Dict = field(default_factory=dict)

    # Bi-temporal fields
    event_time: Optional[datetime] = None      # When event occurred
    ingestion_time: Optional[datetime] = None  # When we learned about it
    valid_from: Optional[datetime] = None      # When edge became valid
    valid_to: Optional[datetime] = None        # When edge was invalidated

# Example edges
edges = [
    KGEdge("Thompson Sampling", "multi-armed bandit", "IS_A", weight=0.95),
    KGEdge("Thompson Sampling", "Beta distribution", "USES", weight=0.92),
    KGEdge("Beta distribution", "Bayesian statistics", "IS_A", weight=0.98),
    KGEdge("multi-armed bandit", "exploration-exploitation", "MENTIONS", weight=0.85),
]
```

### Entity Extraction and Graph Building

#### **Adding Edges**

```python
async def add_edges(self, edges: List[KGEdge]) -> None:
    """
    Add edges to the knowledge graph.

    Args:
        edges: List of KGEdge objects
    """

    for edge in edges:
        # Ensure nodes exist
        if edge.src not in self.G:
            self.G.add_node(edge.src)
        if edge.dst not in self.G:
            self.G.add_node(edge.dst)

        # Add edge with metadata
        self.G.add_edge(
            edge.src,
            edge.dst,
            type=edge.type,
            weight=edge.weight,
            span_id=edge.span_id,
            metadata=edge.metadata
        )

        # Update fast index
        if edge.src not in self._entity_index:
            self._entity_index[edge.src] = set()
        self._entity_index[edge.src].add(edge.dst)

        self.logger.debug(
            f"Added edge: {edge.src} -[{edge.type}]-> {edge.dst} "
            f"(weight={edge.weight})"
        )

# Example: Building KG from memory shard
shard = MemoryShard(
    id="ts_001",
    text="Thompson Sampling uses Beta distributions for each option. "
         "It balances exploration and exploitation.",
    entities=["Thompson Sampling", "Beta distribution", "exploration"],
    motifs=["definition", "probabilistic_model"]
)

# Extract edges (simplified - would normally use NER)
edges_from_shard = [
    KGEdge("Thompson Sampling", "Beta distribution", "USES", weight=0.9, span_id=shard.id),
    KGEdge("Thompson Sampling", "exploration-exploitation", "HANDLES", weight=0.85, span_id=shard.id),
]

kg.add_edges(edges_from_shard)
```

### Subgraph Extraction

#### **Entity-Centric Retrieval**

```python
def get_subgraph(
    self,
    seed_entities: List[str],
    max_hops: int = 2,
    max_nodes: int = 50
) -> nx.MultiDiGraph:
    """
    Extract subgraph around seed entities.

    Args:
        seed_entities: Starting point entities
        max_hops: Maximum relationship hops to follow
        max_nodes: Maximum nodes to include

    Returns:
        NetworkX subgraph

    Use case:
        Query: "What is Thompson Sampling?"
        Extracted entities: ["Thompson Sampling"]
        Subgraph: All entities related to TS within 2 hops
    """

    # BFS from seed entities
    visited = set()
    to_visit = [(entity, 0) for entity in seed_entities]  # (entity, hops)

    while to_visit and len(visited) < max_nodes:
        entity, hops = to_visit.pop(0)

        if entity in visited:
            continue
        visited.add(entity)

        # Explore neighbors if hops remaining
        if hops < max_hops:
            # Get neighbors in both directions
            for neighbor in self.G.successors(entity):
                if neighbor not in visited:
                    to_visit.append((neighbor, hops + 1))

            for neighbor in self.G.predecessors(entity):
                if neighbor not in visited:
                    to_visit.append((neighbor, hops + 1))

    # Build subgraph
    subgraph = self.G.subgraph(visited).copy()

    self.logger.debug(
        f"Extracted subgraph: {len(subgraph.nodes)} nodes, "
        f"{len(subgraph.edges)} edges"
    )

    return subgraph

# Example: Extract subgraph for Thompson Sampling query
subgraph = kg.get_subgraph(
    seed_entities=["Thompson Sampling"],
    max_hops=2,
    max_nodes=50
)

# Result includes:
# Nodes: "Thompson Sampling", "Beta distribution", "multi-armed bandit",
#        "exploration-exploitation", "Bayesian statistics", ...
# Edges: All relationships between these entities
```

### Spectral Graph Analysis

#### **Topological Features**

```python
def compute_spectral_features(self, subgraph: nx.MultiDiGraph) -> Dict:
    """
    Compute spectral (eigenvalue-based) features of subgraph.

    Returns:
        Dictionary of graph topology metrics

    Use case:
        These features feed into policy network
        - Laplacian eigenvalues: Graph connectivity structure
        - Connected components: How fragmented is knowledge?
        - Centrality: Which entities are most important?
    """

    features = {}

    # 1. Laplacian eigenvalues
    # Graph Laplacian: L = D - A
    #   D = degree matrix
    #   A = adjacency matrix
    # Eigenvalues encode graph structure
    try:
        L = nx.laplacian_matrix(subgraph).todense()
        eigenvalues = np.linalg.eigvals(L)
        eigenvalues = np.sort(np.real(eigenvalues))[:10]  # Top 10
        features['laplacian_eigenvalues'] = eigenvalues.tolist()

        # Spectral gap = second-smallest eigenvalue
        # High gap = well-connected graph
        if len(eigenvalues) > 1:
            features['spectral_gap'] = float(eigenvalues[1])
    except Exception as e:
        self.logger.warning(f"Laplacian computation failed: {e}")
        features['laplacian_eigenvalues'] = []

    # 2. Connected components
    # How many separate clusters?
    n_components = nx.number_weakly_connected_components(subgraph)
    features['connected_components'] = n_components

    # 3. Density
    # How many edges relative to possible edges?
    density = nx.density(subgraph)
    features['density'] = float(density)

    # 4. Degree centrality
    # Which entities are most connected?
    centrality = nx.degree_centrality(subgraph)
    top_entities = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]
    features['top_central_entities'] = top_entities

    return features

# Example: Thompson Sampling subgraph features
features = kg.compute_spectral_features(subgraph)
# Result:
# {
#     'laplacian_eigenvalues': [0.0, 0.23, 1.45, 3.67, ...],
#     'spectral_gap': 0.23,                    # Well-connected
#     'connected_components': 1,               # Single cluster
#     'density': 0.42,                         # Moderately dense
#     'top_central_entities': [
#         ('Thompson Sampling', 0.95),         # Most central
#         ('Bayesian statistics', 0.87),
#         ('Beta distribution', 0.84),
#         ...
#     ]
# }
```

### Path Finding in Knowledge Graphs

#### **Entity Relationship Paths**

```python
def find_paths(
    self,
    src: str,
    dst: str,
    max_length: int = 3
) -> List[List[str]]:
    """
    Find relationship paths between entities.

    Args:
        src: Source entity
        dst: Destination entity
        max_length: Maximum path length

    Returns:
        List of paths (each path is list of entities)

    Use case:
        Query reasoning: "How does Thompson Sampling relate to Bandit problems?"
        Find path: Thompson Sampling → IS_A → multi-armed bandit
    """

    try:
        paths = list(nx.all_simple_paths(
            self.G,
            src,
            dst,
            cutoff=max_length
        ))
        return paths
    except nx.NetworkXNoPath:
        return []

# Example: Path finding for Thompson Sampling
paths = kg.find_paths(
    src="Thompson Sampling",
    dst="Bayesian statistics",
    max_length=3
)

# Results:
# Path 1: Thompson Sampling → Beta distribution → Bayesian statistics
# Path 2: Thompson Sampling → Bayesian approach → Bayesian statistics
# These paths explain the relationship!
```

---

## 5. Spacetime Construction and Provenance

### Overview

Spacetime is the final woven fabric - the response with complete computational lineage.

### Spacetime Data Structure

```python
# From HoloLoom/fabric/spacetime.py
@dataclass
class WeavingTrace:
    """
    Complete computational trace - the "how" of weaving.

    Records every stage of the shuttle's journey, enabling:
    - Debugging (what went wrong?)
    - Analysis (which stages are bottlenecks?)
    - Learning (what patterns led to success?)
    - Reproducibility (replicate exact computation)
    """

    # Temporal markers
    start_time: datetime        # When weaving started
    end_time: datetime          # When weaving ended
    duration_ms: float          # Total time

    # Stage timings (9 stages)
    stage_durations: Dict[str, float]  # {stage_name: duration_ms}

    # Feature extraction details
    motifs_detected: List[str]           # ["definition_question", "thompson_sampling"]
    embedding_scales_used: List[int]     # [96, 192, 384]
    spectral_features: Optional[Dict]    # Graph topology features

    # Memory retrieval details
    threads_activated: List[str]         # ["ts_001", "ts_002", "bandit_001"]
    context_shards_count: int            # How many memories retrieved?
    retrieval_mode: str                  # "hybrid_bm25_semantic"

    # Decision details
    policy_adapter: str                  # "fused_mode"
    tool_selected: str                   # "answer"
    tool_confidence: float               # 0.85
    bandit_statistics: Optional[Dict]    # Bandit state before/after

    # Warp space trace
    warp_operations: List[tuple]         # [(op_name, params), ...]
    tensor_field_stats: Optional[Dict]   # Tensor shape, dtype, etc.

    # Error tracking
    errors: List[Dict]                   # Error list (empty on success)
    warnings: List[str]                  # Warnings
```

### Construction Process

```python
# Building trace from execution
async def weave(self, query: Query) -> Spacetime:
    """Main weaving cycle with trace building."""

    # Record start
    start_time = datetime.now()
    stage_durations = {}

    # Step 1: Pattern selection
    t1 = time.time()
    pattern_card = loom_cmd.select_pattern(query)
    stage_durations['pattern_selection'] = (time.time() - t1) * 1000

    # Step 2: Temporal window
    t2 = time.time()
    temporal_window = chrono_trigger.fire(pattern_card, query)
    stage_durations['temporal_window'] = (time.time() - t2) * 1000

    # ... (Steps 3-8) ...

    # Step 8: Build trace
    t8_start = time.time()

    trace = WeavingTrace(
        start_time=start_time,
        end_time=datetime.now(),
        duration_ms=(datetime.now() - start_time).total_seconds() * 1000,

        # Collect all stage durations
        stage_durations=stage_durations,

        # Feature details
        motifs_detected=[m.pattern for m in features.motifs],
        embedding_scales_used=[96, 192, 384],
        spectral_features=features.metrics,

        # Memory details
        threads_activated=[s.id for s in selected_threads],
        context_shards_count=len(context.shards),
        retrieval_mode=context.metadata.get('retrieval_mode', 'unknown'),

        # Decision details
        policy_adapter=collapse_result.adapter,
        tool_selected=collapse_result.tool,
        tool_confidence=collapse_result.confidence,
        bandit_statistics=policy.bandit.get_stats(),

        # Errors
        errors=[],  # Empty if no errors
        warnings=[]
    )

    stage_durations['trace_building'] = (time.time() - t8_start) * 1000

    # Build Spacetime
    spacetime = Spacetime(
        response_text=tool_result['result'],
        response_confidence=tool_result['confidence'],
        trace=trace,

        # Semantic content
        query_embedding=features.psi,
        context_embeddings=context_tensors,

        # Metrics
        confidence=tool_result['confidence'],
        quality_score=calculate_quality(spacetime),

        # Metadata
        metadata={
            'query_text': query.text,
            'sources': [s.id for s in context.shards],
            'execution_mode': str(self.cfg.execution_mode),
            'timestamp': datetime.now().isoformat()
        }
    )

    return spacetime
```

### Using Trace for Debugging

```python
# Analyzing a Spacetime result
spacetime = await orchestrator.weave(query)

# Get timing breakdown
print("Stage Timings:")
for stage, duration in spacetime.trace.stage_durations.items():
    print(f"  {stage:25} {duration:7.1f}ms")

# Output:
# Stage Timings:
#   pattern_selection            2.3ms
#   temporal_window              1.1ms
#   thread_selection             5.2ms
#   feature_extraction          45.7ms
#   warp_tensioning             23.4ms
#   policy_decision              8.9ms
#   tool_execution              78.5ms ← SLOWEST (LLM generation)
#   trace_building               2.1ms

# Identify bottleneck
slowest_stage = max(
    spacetime.trace.stage_durations.items(),
    key=lambda x: x[1]
)
print(f"Bottleneck: {slowest_stage[0]} ({slowest_stage[1]:.1f}ms)")
# Output: Bottleneck: tool_execution (78.5ms)

# Get memory usage
print(f"Memory Retrieved: {spacetime.trace.context_shards_count} shards")
print(f"Threads Activated: {spacetime.trace.threads_activated}")

# Get decision details
print(f"Tool Selected: {spacetime.trace.tool_selected} "
      f"(confidence: {spacetime.trace.tool_confidence:.1%})")
print(f"Adapter Used: {spacetime.trace.policy_adapter}")

# Check for errors
if spacetime.trace.errors:
    print("ERRORS:")
    for error in spacetime.trace.errors:
        print(f"  {error}")
else:
    print("No errors")
```

---

## 6. Lifecycle Management and Cleanup

### Overview

HoloLoom uses async context managers for proper resource cleanup.

### Initialization and Teardown

#### **Context Manager Protocol**

```python
# Usage pattern
async with WeavingOrchestrator(cfg=config, shards=shards) as orchestrator:
    # Work here
    spacetime = await orchestrator.weave(query)

    # Background tasks run
    # Resources managed

# Automatic cleanup on exit
# - Background tasks cancelled
# - Database connections closed
# - Reflection buffer metrics flushed
# - Memory freed
```

#### **Implementation**

```python
# From HoloLoom/weaving_orchestrator.py
class WeavingOrchestrator:
    """Weaving Orchestrator with lifecycle management."""

    def __init__(self, cfg: Config, shards: List[MemoryShard], ...):
        """Initialize orchestrator."""
        self.cfg = cfg
        self.shards = shards

        # Component initialization
        self.loom_command = LoomCommand()
        self.chrono_trigger = ChronoTrigger()
        self.yarn_graph = KG()
        self.resonance_shed = ResonanceShed()
        self.warp_space = WarpSpace()
        self.convergence_engine = ConvergenceEngine()
        self.reflection_buffer = ReflectionBuffer(capacity=1000)

        # Background task tracking
        self._background_tasks = set()
        self._shutdown_event = asyncio.Event()

        self.logger = logging.getLogger(__name__)

    async def __aenter__(self):
        """Enter context manager."""
        self.logger.info("Initializing WeavingOrchestrator")

        # Start background learning (if enabled)
        if self.enable_reflection:
            task = asyncio.create_task(self._background_learning_loop())
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager - cleanup."""
        self.logger.info("Shutting down WeavingOrchestrator")

        try:
            # Step 1: Signal shutdown to background tasks
            self._shutdown_event.set()

            # Step 2: Cancel all background tasks
            for task in self._background_tasks:
                if not task.done():
                    task.cancel()

            # Step 3: Wait for cancellation (with timeout)
            if self._background_tasks:
                await asyncio.wait(
                    self._background_tasks,
                    timeout=5.0  # 5 second grace period
                )

            # Step 4: Flush reflection buffer metrics
            if hasattr(self, 'reflection_buffer'):
                await self.reflection_buffer.flush()
                self.logger.info("Reflection buffer flushed")

            # Step 5: Close database connections (if any)
            if hasattr(self, 'memory_backend') and self.memory_backend:
                await self.memory_backend.close()
                self.logger.info("Memory backend closed")

            # Step 6: Report any errors
            if exc_type is not None:
                self.logger.error(
                    f"Shutdown with exception: {exc_type.__name__}: {exc_val}"
                )
                return False  # Re-raise exception

            self.logger.info("Shutdown complete")
            return True  # Suppress any exceptions

        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}", exc_info=True)
            raise
```

### Background Learning Loop

```python
async def _background_learning_loop(self):
    """
    Background task that learns from queries.

    Runs continuously (every 60 seconds) and:
    1. Analyzes recent spacetime results
    2. Updates bandit priors
    3. Extracts successful patterns
    4. Adjusts retrieval weights
    """

    self.logger.info("Starting background learning loop")

    try:
        while not self._shutdown_event.is_set():
            try:
                # Wait for next learning cycle (60s or shutdown signal)
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=60.0  # 60 second cycle
                )
                # If we get here, shutdown was signaled
                break

            except asyncio.TimeoutError:
                # Time for learning cycle
                self.logger.debug("Running background learning cycle")

                try:
                    # Step 1: Get recent spacetime results
                    recent = await self.reflection_buffer.get_recent(n=100)

                    if not recent:
                        continue

                    # Step 2: Analyze patterns
                    successful_tools = {}
                    successful_retrieval_modes = {}

                    for spacetime in recent:
                        tool = spacetime.trace.tool_selected
                        successful_tools[tool] = successful_tools.get(tool, 0) + 1

                        mode = spacetime.trace.retrieval_mode
                        successful_retrieval_modes[mode] = (
                            successful_retrieval_modes.get(mode, 0) + 1
                        )

                    # Step 3: Update weights based on success
                    for tool, count in successful_tools.items():
                        win_rate = count / len(recent)
                        self.logger.debug(
                            f"Tool {tool}: {win_rate:.1%} win rate "
                            f"({count}/{len(recent)} queries)"
                        )

                    # Step 4: Log summary
                    self.logger.info(
                        f"Learning cycle complete: "
                        f"analyzed {len(recent)} recent queries"
                    )

                except Exception as e:
                    self.logger.error(f"Learning cycle failed: {e}", exc_info=True)
                    # Continue despite errors (non-critical background task)

    except asyncio.CancelledError:
        self.logger.info("Background learning loop cancelled")
        raise

    finally:
        self.logger.info("Background learning loop stopped")
```

### Resource Cleanup Sequence

```python
# Cleanup sequence on context exit

# Step 1: Signal shutdown
_shutdown_event.set()

# Step 2: Cancel background tasks
for task in _background_tasks:
    task.cancel()

# Step 3: Wait for cancellation (timeout after 5s)
await asyncio.wait(_background_tasks, timeout=5.0)

# Step 4: Flush reflection buffer
# - Write metrics to disk
# - Close file handles
# - Flush outstanding writes
await reflection_buffer.flush()

# Step 5: Close database connections
# - Neo4j driver close
# - Qdrant connection close
# - Any other DB connections
await memory_backend.close()

# Step 6: Clean up temporary files
# - Delete cache files (if configured)
# - Clean mmap files
# - Remove temp directories
```

### Error Handling During Shutdown

```python
async def __aexit__(self, exc_type, exc_val, exc_tb):
    """Handle exceptions during shutdown."""

    # If an exception occurred in the with block
    if exc_type is not None:
        self.logger.error(
            f"Context exiting with exception: {exc_type.__name__}: {exc_val}",
            exc_info=(exc_type, exc_val, exc_tb)
        )

        # Still try to cleanup
        try:
            # Aggressive cleanup
            # Cancel tasks immediately
            for task in self._background_tasks:
                task.cancel()

            # Wait briefly
            try:
                await asyncio.wait(self._background_tasks, timeout=2.0)
            except:
                pass

            # Force flush (ignore errors)
            try:
                await self.reflection_buffer.flush(force=True)
            except:
                pass

        except Exception as cleanup_error:
            self.logger.error(f"Cleanup failed: {cleanup_error}")

        # Re-raise original exception
        return False

    # No exception in with block
    return True
```

---

## Summary

This Part 5 has covered the complete implementation details of HoloLoom:

1. **Query Lifecycle**: All 9 steps from Query input to Spacetime output
2. **Policy Engine**: Neural networks + Thompson Sampling decision making
3. **Embeddings**: Multi-scale Matryoshka embeddings with zero-copy optimization
4. **Knowledge Graph**: Entity storage, relationships, and spectral analysis
5. **Spacetime**: Complete provenance tracking and trace building
6. **Lifecycle**: Async context managers and proper resource cleanup

Each section included:
- Actual source code from the repository
- Line-by-line explanations
- Data structure examples
- Performance characteristics
- Complete usage examples

For deeper understanding, review the actual source files referenced throughout this guide.

---

## Recommended Reading Order

1. Start with **Query Lifecycle** to understand the overall flow
2. Deep dive into **Policy Engine** for decision making details
3. Study **Embeddings** for semantic representation
4. Explore **Knowledge Graph** for relationship tracking
5. Review **Spacetime** for output and provenance
6. Understand **Lifecycle Management** for production robustness

---

**Date Created**: 2025-11-16
**Last Updated**: 2025-11-16
**Based on**: HoloLoom source code (Phase 5 complete)
