# DreamEngine: Memory Synthesis Architecture

**"HoloLoom doesn't just remember - it thinks about what it knows"**

**Status:** Week 3 MVP Design
**Date:** 2025-11-17
**Author:** HoloLoom Team

---

## Vision

Most memory systems are passive repositories: you store data, you retrieve data. **DreamEngine** makes HoloLoom **active and intelligent**:

- **Pattern Synthesis**: Discovers emergent patterns and creates summary memories
- **Contradiction Detection**: Identifies conflicting information across time
- **Gap Identification**: Finds missing knowledge to suggest learning
- **Background Processing**: Runs during idle time (like REM sleep in humans)

**Analogy:** If memory storage is like note-taking, DreamEngine is like **reviewing your notes, finding themes, and questioning inconsistencies**.

---

## Core Philosophy

### 1. **Memories as Living Knowledge**

Traditional systems: `store(X)` → `retrieve(X)`

DreamEngine: `store(X)` → `synthesize(X, Y, Z)` → `create(Summary)` → `detect_conflicts(X, X')` → `suggest_gaps()`

Memories aren't static - they evolve, combine, and reveal insights.

### 2. **Inspired by Human Memory Consolidation**

Humans don't just store experiences - we:
- **Consolidate**: Similar experiences merge into patterns
- **Reconcile**: Contradictions trigger re-evaluation
- **Prune**: Unimportant details fade, essence remains
- **Connect**: Related memories strengthen associations

DreamEngine mimics this during background "dream" cycles.

### 3. **Provenance Everything**

All synthetic memories tagged with:
- `provenance="synthesized_from_pattern"` or `"contradiction_detection"`
- `source_memories=[id1, id2, ...]` - what memories led to this insight
- `confidence=0.0-1.0` - how confident is this synthesis
- `created_at=timestamp` - when was this created

**Transparency**: Users can trace every synthetic insight back to source memories.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    DreamEngine                          │
│                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │   Pattern    │  │Contradiction │  │     Gap      │ │
│  │  Synthesis   │  │  Detection   │  │Identification│ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
│         │                  │                  │        │
│         └──────────────────┴──────────────────┘        │
│                            │                           │
│                  ┌─────────▼─────────┐                 │
│                  │Background Scheduler│                 │
│                  └───────────────────┘                 │
└─────────────────────────────────────────────────────────┘
                           │
                ┌──────────▼──────────┐
                │   Knowledge Graph   │
                │  (Yarn Graph / KG)  │
                └─────────────────────┘
```

### Component Responsibilities

| Component | Input | Output | Trigger |
|-----------|-------|--------|---------|
| **Pattern Synthesis** | High-confidence query patterns | Summary memories | Every 100 queries OR idle time |
| **Contradiction Detection** | Temporal memory pairs | Conflict alerts | Every 1000 queries OR daily |
| **Gap Identification** | Memory graph structure | Missing knowledge suggestions | Weekly OR on-demand |
| **Background Scheduler** | System idle detection | Triggers synthesis cycles | Configurable (default: nightly) |

---

## Component 1: Pattern Synthesis

### What It Does

Finds recurring patterns in high-confidence interactions and creates **summary memories**.

**Example:**
```
User queries:
1. "What is Thompson Sampling?" (confidence: 0.92)
2. "How does Thompson Sampling work?" (confidence: 0.89)
3. "Thompson Sampling vs epsilon-greedy" (confidence: 0.91)

Pattern detected: User learning about Thompson Sampling

Synthetic memory created:
"Thompson Sampling: A Bayesian approach to exploration/exploitation in
multi-armed bandits. Balances trying new actions (exploration) vs
exploiting known good actions. Uses beta distributions to sample arm
probabilities. Compared to epsilon-greedy, Thompson Sampling is more
principled and often converges faster."

Provenance:
- source_queries: [q1, q2, q3]
- confidence: 0.88 (avg of source confidences)
- created_at: 2025-11-17T10:30:00Z
- type: "pattern_synthesis"
```

### Algorithm

```python
def synthesize_patterns(memory_window: TimeWindow, threshold: float = 0.85):
    """
    Find patterns in recent high-confidence interactions.

    Args:
        memory_window: Time window to analyze (e.g., last 100 queries)
        threshold: Minimum confidence to consider (default: 0.85)

    Returns:
        List of synthetic summary memories
    """
    # 1. Extract high-confidence queries
    high_conf_queries = [
        q for q in memory_window.queries
        if q.confidence >= threshold
    ]

    # 2. Cluster by semantic similarity (using embeddings)
    clusters = cluster_by_similarity(
        high_conf_queries,
        min_cluster_size=3,  # Need at least 3 related queries
        similarity_threshold=0.7
    )

    # 3. For each cluster, extract common theme
    synthetic_memories = []
    for cluster in clusters:
        # Find common entities, motifs
        common_entities = extract_common_entities(cluster)
        common_motifs = extract_common_motifs(cluster)

        # Generate summary
        summary = generate_summary(
            queries=cluster,
            entities=common_entities,
            motifs=common_motifs
        )

        # Create synthetic memory
        synthetic_mem = Memory(
            content=summary,
            provenance="pattern_synthesis",
            source_memories=[q.id for q in cluster],
            confidence=np.mean([q.confidence for q in cluster]),
            metadata={
                "cluster_size": len(cluster),
                "common_entities": common_entities,
                "common_motifs": common_motifs
            }
        )

        synthetic_memories.append(synthetic_mem)

    return synthetic_memories
```

### Configuration

```python
@dataclass
class PatternSynthesisConfig:
    """Configuration for pattern synthesis."""
    enabled: bool = True
    confidence_threshold: float = 0.85  # Minimum confidence
    min_cluster_size: int = 3  # Min queries to form pattern
    similarity_threshold: float = 0.7  # Semantic similarity
    window_size: int = 100  # Look at last N queries
    trigger_interval: int = 100  # Run every N queries
```

---

## Component 2: Contradiction Detection

### What It Does

Identifies **conflicting information** across memories and alerts users to reconcile.

**Example:**
```
Memory 1 (Jan 2025):
"TypeScript is better than JavaScript for large projects"

Memory 2 (March 2025):
"JavaScript's flexibility makes it better for rapid prototyping than TypeScript"

Contradiction detected:
- Conflict type: "preference_reversal"
- Semantic similarity: 0.82 (same topic)
- Sentiment conflict: positive(TS) vs positive(JS)
- Temporal gap: 2 months

Alert to user:
"You expressed different views on TypeScript vs JavaScript.
 In January, you preferred TypeScript for large projects.
 In March, you preferred JavaScript for rapid prototyping.
 These aren't necessarily contradictory, but which reflects your current view?"

Actions:
1. Mark one as outdated
2. Create clarification memory ("TS for production, JS for prototypes")
3. Keep both (context-dependent)
```

### Algorithm

```python
def detect_contradictions(memory_graph: KG, lookback_days: int = 90):
    """
    Detect contradictory memories within a time window.

    Args:
        memory_graph: Knowledge graph with temporal edges
        lookback_days: How far back to check (default: 90 days)

    Returns:
        List of detected contradictions
    """
    # 1. Get memories from time window
    cutoff_time = now() - timedelta(days=lookback_days)
    recent_memories = memory_graph.get_memories_since(cutoff_time)

    contradictions = []

    # 2. Compare pairs with high semantic similarity
    for i, mem1 in enumerate(recent_memories):
        for mem2 in recent_memories[i+1:]:
            # Skip if same memory
            if mem1.id == mem2.id:
                continue

            # Check semantic similarity (same topic?)
            similarity = semantic_similarity(mem1.embedding, mem2.embedding)
            if similarity < 0.75:  # Different topics, skip
                continue

            # Check for contradiction signals
            contradiction_score = 0.0

            # Signal 1: Opposite sentiment on same entity
            if has_opposite_sentiment(mem1, mem2):
                contradiction_score += 0.3

            # Signal 2: Mutually exclusive statements
            if are_mutually_exclusive(mem1, mem2):
                contradiction_score += 0.5

            # Signal 3: Updated facts (time-sensitive)
            if is_fact_update(mem1, mem2):
                contradiction_score += 0.4

            # If high contradiction score, flag it
            if contradiction_score >= 0.5:
                contradictions.append(Contradiction(
                    memory1=mem1,
                    memory2=mem2,
                    score=contradiction_score,
                    type=classify_contradiction_type(mem1, mem2),
                    temporal_gap=mem2.timestamp - mem1.timestamp
                ))

    return contradictions
```

### Contradiction Types

| Type | Description | Example |
|------|-------------|---------|
| **fact_update** | Later memory supersedes earlier fact | "Paris population: 2M" → "Paris population: 2.1M" |
| **preference_reversal** | User preference changes | "I love Python" → "Python is too slow" |
| **belief_change** | Worldview evolution | "AI will never match humans" → "AI surpassed expectations" |
| **context_dependent** | Both valid in different contexts | "TypeScript for production" + "JavaScript for prototypes" |
| **logical_conflict** | Mutually exclusive claims | "Earth is flat" + "Earth is spherical" |

### Configuration

```python
@dataclass
class ContradictionConfig:
    """Configuration for contradiction detection."""
    enabled: bool = True
    lookback_days: int = 90  # Check last 90 days
    similarity_threshold: float = 0.75  # Topic similarity
    contradiction_threshold: float = 0.5  # Minimum score
    trigger_interval: int = 1000  # Run every N queries
```

---

## Component 3: Gap Identification

### What It Does

Analyzes the **knowledge graph structure** to find missing connections and suggest learning.

**Example:**
```
Knowledge graph analysis:
- Strong knowledge: "Machine Learning", "Python", "Neural Networks"
- Weak connection: ML → "Data Preprocessing"
- Missing link: "Feature Engineering" (mentioned but not explained)

Gap identified:
"You know about Machine Learning and Neural Networks, but there's limited
information about Feature Engineering. This is a critical step between
data and model training. Would you like to learn more about feature
scaling, encoding, and dimensionality reduction?"

Suggested queries:
1. "What is feature engineering?"
2. "How to normalize data for ML?"
3. "When to use PCA for dimensionality reduction?"
```

### Algorithm

```python
def identify_gaps(memory_graph: KG, min_importance: float = 0.5):
    """
    Identify knowledge gaps using graph structure.

    Args:
        memory_graph: Knowledge graph
        min_importance: Minimum importance score for gaps

    Returns:
        List of identified gaps with suggested learning
    """
    gaps = []

    # 1. Find strongly connected subgraphs (knowledge clusters)
    clusters = memory_graph.strongly_connected_components()

    for cluster in clusters:
        # 2. Find bridge nodes (mentioned but underdeveloped)
        bridge_nodes = find_bridge_nodes(cluster)

        for node in bridge_nodes:
            # Calculate importance (PageRank-style)
            importance = calculate_node_importance(node, cluster)

            if importance >= min_importance:
                # 3. Identify what's missing
                missing_concepts = get_related_concepts(node)  # From ontology
                existing_concepts = set(cluster.nodes)
                gap_concepts = missing_concepts - existing_concepts

                if gap_concepts:
                    gaps.append(KnowledgeGap(
                        bridge_concept=node,
                        missing_concepts=list(gap_concepts),
                        importance=importance,
                        suggested_queries=generate_learning_queries(
                            node, gap_concepts
                        ),
                        related_cluster=cluster.id
                    ))

    return gaps
```

### Gap Types

| Type | Description | Detection Method |
|------|-------------|------------------|
| **bridge_gap** | Mentioned concept but not explained | Low out-degree, high in-degree |
| **missing_prerequisite** | Know advanced topic, missing basics | Topological sort violation |
| **incomplete_category** | Know some items, missing others | Category completeness check |
| **stale_knowledge** | Old information, new developments exist | Temporal analysis + external sources |

### Configuration

```python
@dataclass
class GapIdentificationConfig:
    """Configuration for gap identification."""
    enabled: bool = True
    min_importance: float = 0.5  # Minimum importance score
    max_gaps_per_run: int = 5  # Limit to avoid overload
    trigger_frequency: str = "weekly"  # weekly, daily, manual
```

---

## Component 4: Background Scheduler

### What It Does

Orchestrates when synthesis cycles run - preferably during **idle time** to avoid impacting user queries.

**Scheduling Modes:**
1. **Idle-based**: Run when no queries for N seconds
2. **Time-based**: Run at specific times (e.g., 3 AM daily)
3. **Threshold-based**: Run after N new queries accumulated
4. **Manual**: User-triggered synthesis

### Algorithm

```python
class BackgroundScheduler:
    """
    Schedules background synthesis cycles.

    Inspired by human REM sleep cycles - runs during "downtime" to
    consolidate memories without interfering with active use.
    """

    def __init__(self, config: SchedulerConfig):
        self.config = config
        self.last_synthesis = None
        self.query_count_since_synthesis = 0
        self.idle_start_time = None

    async def should_run_synthesis(self) -> bool:
        """
        Determine if synthesis should run now.

        Returns:
            True if synthesis should run, False otherwise
        """
        now = datetime.now()

        # Mode 1: Idle-based
        if self.config.mode == "idle":
            if self.idle_start_time:
                idle_duration = (now - self.idle_start_time).total_seconds()
                if idle_duration >= self.config.idle_threshold_seconds:
                    return True

        # Mode 2: Time-based
        elif self.config.mode == "time":
            if now.hour == self.config.scheduled_hour:
                if not self.last_synthesis or \
                   (now - self.last_synthesis).days >= 1:
                    return True

        # Mode 3: Threshold-based
        elif self.config.mode == "threshold":
            if self.query_count_since_synthesis >= \
               self.config.query_threshold:
                return True

        return False

    async def run_synthesis_cycle(self, engine: DreamEngine):
        """
        Execute complete synthesis cycle.

        Runs all enabled synthesis components in sequence.
        """
        print("🌙 DreamEngine: Starting synthesis cycle...")

        results = {}

        # Phase 1: Pattern Synthesis
        if engine.pattern_synthesis.enabled:
            patterns = await engine.synthesize_patterns()
            results['patterns'] = len(patterns)
            print(f"  ✓ Patterns synthesized: {len(patterns)}")

        # Phase 2: Contradiction Detection
        if engine.contradiction_detection.enabled:
            contradictions = await engine.detect_contradictions()
            results['contradictions'] = len(contradictions)
            print(f"  ⚠ Contradictions found: {len(contradictions)}")

        # Phase 3: Gap Identification
        if engine.gap_identification.enabled:
            gaps = await engine.identify_gaps()
            results['gaps'] = len(gaps)
            print(f"  📖 Knowledge gaps identified: {len(gaps)}")

        # Update tracking
        self.last_synthesis = datetime.now()
        self.query_count_since_synthesis = 0

        print("🌙 DreamEngine: Synthesis cycle complete\n")

        return results

    def on_query_processed(self):
        """
        Callback after each query processed.

        Increments query counter and resets idle timer.
        """
        self.query_count_since_synthesis += 1
        self.idle_start_time = None  # Reset idle detection

    def on_idle_start(self):
        """
        Callback when system becomes idle.

        Starts idle timer for synthesis triggering.
        """
        if not self.idle_start_time:
            self.idle_start_time = datetime.now()
```

### Configuration

```python
@dataclass
class SchedulerConfig:
    """Configuration for background scheduler."""
    mode: str = "threshold"  # idle, time, threshold, manual
    idle_threshold_seconds: int = 300  # 5 minutes idle
    scheduled_hour: int = 3  # 3 AM for time-based
    query_threshold: int = 100  # Run after 100 queries
    max_cycle_duration_seconds: int = 60  # Max 1 minute
```

---

## Complete DreamEngine API

### Initialization

```python
from HoloLoom.synthesis import DreamEngine, DreamEngineConfig

config = DreamEngineConfig(
    pattern_synthesis=PatternSynthesisConfig(enabled=True),
    contradiction_detection=ContradictionConfig(enabled=True),
    gap_identification=GapIdentificationConfig(enabled=True),
    scheduler=SchedulerConfig(mode="threshold", query_threshold=100)
)

engine = DreamEngine(
    knowledge_graph=kg,
    config=config
)

# Start background scheduler
await engine.start_background_synthesis()
```

### Manual Synthesis

```python
# Run synthesis on-demand
results = await engine.run_synthesis_cycle()

print(f"Patterns: {len(results['patterns'])}")
print(f"Contradictions: {len(results['contradictions'])}")
print(f"Gaps: {len(results['gaps'])}")
```

### Integration with HoloLoom

```python
from HoloLoom import HoloLoom
from HoloLoom.synthesis import DreamEngine

loom = HoloLoom()
dream_engine = DreamEngine(knowledge_graph=loom.graph)

# HoloLoom automatically triggers synthesis
await loom.experience("New information...")
# ... after 100 queries, DreamEngine runs synthesis in background
```

### Querying Synthetic Memories

```python
# Synthetic memories are queryable like any other memory
results = await loom.recall("summarize what I learned about ML")

# Can filter by provenance
synthetic_patterns = [
    m for m in results
    if m.provenance == "pattern_synthesis"
]

# Can trace back to source memories
for pattern in synthetic_patterns:
    print(f"Pattern: {pattern.content}")
    print(f"Sources: {pattern.source_memories}")
    print(f"Confidence: {pattern.confidence}")
```

---

## Performance Characteristics

| Operation | Frequency | Latency | Impact |
|-----------|-----------|---------|--------|
| **Pattern Synthesis** | Every 100 queries | ~500ms | Background (async) |
| **Contradiction Detection** | Daily or per 1000 queries | ~2s | Background (async) |
| **Gap Identification** | Weekly | ~5s | Background (async) |
| **Background Scheduler** | Continuous monitoring | <1ms | Negligible |

**Key Point:** All synthesis runs **asynchronously** - zero impact on query latency.

---

## Safety & Ethics

### Synthetic Memory Labeling

All synthetic memories clearly labeled:
- `provenance` field always set
- `source_memories` always tracked
- `confidence` score included

**Users can:**
- Filter out synthetic memories if desired
- Trace back to original sources
- Delete synthetic memories without affecting sources

### Contradiction Handling

**Never auto-delete memories** - only alert users:
- Contradictions are suggestions, not actions
- Users decide which memory is correct/current
- System preserves both until user intervenes

### Gap Suggestions

**Non-intrusive**:
- Gaps identified but not forced
- Suggestions available on-demand
- No automatic content fetching (user privacy)

---

## Testing Strategy

### Unit Tests

```python
# Test pattern synthesis
def test_pattern_synthesis():
    queries = create_test_queries(theme="Thompson Sampling", count=5)
    patterns = synthesize_patterns(queries)

    assert len(patterns) == 1
    assert "Thompson Sampling" in patterns[0].content
    assert patterns[0].provenance == "pattern_synthesis"
    assert len(patterns[0].source_memories) == 5

# Test contradiction detection
def test_contradiction_detection():
    mem1 = Memory("I love Python", timestamp=jan_2025)
    mem2 = Memory("Python is too slow", timestamp=mar_2025)

    contradictions = detect_contradictions([mem1, mem2])

    assert len(contradictions) == 1
    assert contradictions[0].type == "preference_reversal"
    assert contradictions[0].score >= 0.5
```

### Integration Tests

```python
# Test full synthesis cycle
async def test_full_synthesis_cycle():
    engine = DreamEngine(config=test_config)

    # Simulate 100 queries
    for i in range(100):
        await engine.process_query(f"Query {i}")

    # Trigger synthesis
    results = await engine.run_synthesis_cycle()

    assert results['patterns'] > 0
    assert results['contradictions'] >= 0
    assert results['gaps'] >= 0
```

---

## Roadmap

### MVP (Week 3)
- ✅ Pattern synthesis (basic clustering)
- ✅ Contradiction detection (sentiment-based)
- ✅ Gap identification (bridge nodes)
- ✅ Background scheduler (threshold-based)

### Phase 2 (Future)
- Advanced pattern synthesis (use LLM for summaries)
- Temporal contradiction analysis (track belief evolution)
- External gap validation (check Wikipedia, arXiv for completeness)
- Adaptive scheduling (learn optimal synthesis times per user)

### Phase 3 (Research)
- Meta-learning from synthetic memories
- Cross-user pattern discovery (privacy-preserving)
- Predictive gap identification (anticipate future needs)
- Dream narrative generation (explain synthesis process to users)

---

## Conclusion

DreamEngine transforms HoloLoom from a **passive memory store** to an **active thinking system**:

- **Pattern Synthesis**: Discovers insights you didn't explicitly store
- **Contradiction Detection**: Helps you reconcile conflicting beliefs
- **Gap Identification**: Guides you toward complete understanding
- **Background Processing**: Works quietly, no user intervention needed

**Result:** A memory system that **learns, questions, and grows** alongside you.

---

**Next Steps:**
1. Implement core DreamEngine class
2. Build pattern synthesis module
3. Build contradiction detection module
4. Build gap identification module
5. Integrate background scheduler
6. Test on real HoloLoom deployment
7. Demo with Personal Research Assistant (Week 4)

**Files to create:**
- `HoloLoom/synthesis/dream_engine.py` - Main class
- `HoloLoom/synthesis/pattern_synthesis.py` - Pattern discovery
- `HoloLoom/synthesis/contradiction_detection.py` - Conflict finder
- `HoloLoom/synthesis/gap_identification.py` - Knowledge gap analysis
- `HoloLoom/synthesis/background_scheduler.py` - Async scheduler
- `HoloLoom/synthesis/types.py` - Data structures
- `HoloLoom/synthesis/tests/` - Test suite

**Target:** Make HoloLoom the first memory system that **thinks**.
