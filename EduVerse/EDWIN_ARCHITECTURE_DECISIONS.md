# EdWIN Architecture Decision Record (ADR)

**Version**: 1.0.0
**Date**: November 15, 2025
**Status**: Design Phase

---

## Overview

This document records the key architectural decisions made in designing EdWIN AI Tutor. Each decision is documented with context, alternatives considered, and rationale.

---

## ADR-001: Curriculum as Knowledge Graph

### Context

We need to represent 220+ learning objectives with complex prerequisite relationships, Bloom's taxonomy progressions, and cross-subject connections.

### Decision

**Use HoloLoom's Knowledge Graph (NetworkX MultiDiGraph) to represent curriculum as a navigable graph structure.**

### Alternatives Considered

1. **Relational Database (PostgreSQL)**
   - ✅ Proven technology, SQL queries
   - ❌ Complex joins for multi-hop queries
   - ❌ Difficult to represent graph relationships
   - ❌ Poor performance for prerequisite traversal

2. **Document Store (MongoDB)**
   - ✅ Flexible schema, easy to update
   - ❌ No native graph traversal
   - ❌ Prerequisite queries require application logic
   - ❌ Can't leverage graph algorithms (BFS, DFS, shortest path)

3. **Knowledge Graph (NetworkX/Neo4j)**
   - ✅ Native graph traversal (BFS, DFS, shortest path)
   - ✅ Multi-hop queries are natural
   - ✅ Prerequisite chains are edges
   - ✅ Integrates with HoloLoom infrastructure
   - ✅ Bi-temporal tracking (Graphiti research)
   - ❌ More complex setup (Neo4j requires Docker)

### Rationale

**Knowledge graphs are the natural representation for curriculum:**

- **Prerequisites = Directed Edges**: "A requires B" is a natural graph edge
- **Learning Paths = Graph Traversal**: BFS/DFS from current knowledge to target objective
- **Skill Clustering = Graph Clustering**: Related concepts form connected components
- **Multi-Hop Reasoning**: "What foundational math is needed for physics?" = multi-hop query
- **Temporal Evolution**: Bi-temporal edges track "when did student learn X?"

**Performance Benefits:**
- Prerequisite check: O(1) edge lookup vs O(n) database join
- Learning path: BFS in O(V + E) vs multiple database queries
- Related concepts: Graph neighbors in O(1) vs complex SQL

**Integration Benefits:**
- HoloLoom already has KG infrastructure (`memory/graph.py`)
- Reuses existing NetworkX backend
- Can upgrade to Neo4j for production scale
- Bi-temporal support from Graphiti research

### Consequences

**Positive:**
- Fast prerequisite traversal (critical for adaptive learning)
- Natural representation of educational dependencies
- Enables sophisticated learning path algorithms
- Temporal queries: "What did student know on Oct 12?"

**Negative:**
- More complex initial setup (Neo4j Docker)
- Requires graph thinking (vs familiar SQL)
- Developers need to learn graph query patterns

**Mitigation:**
- Provide NetworkX in-memory fallback (no Docker needed)
- Comprehensive documentation of graph patterns
- Helper methods abstract graph complexity (`get_learning_path()`, etc.)

---

## ADR-002: RAG Over Fine-Tuning for Content Generation

### Context

We need to generate grade-appropriate explanations for 220+ objectives across 6 subjects. Should we fine-tune an LLM or use RAG (Retrieval-Augmented Generation)?

### Decision

**Use HoloLoom RAG (SimpleRAG + MultimodalRAG) for content generation, NOT fine-tuning.**

### Alternatives Considered

1. **Fine-Tune LLM (GPT-4, Claude)**
   - ✅ Fast inference (no retrieval overhead)
   - ✅ Customized to educational domain
   - ❌ Expensive ($10,000+ for quality fine-tuning)
   - ❌ Static knowledge (can't update without re-training)
   - ❌ Opaque reasoning (can't cite sources)
   - ❌ Hallucination risk (no grounding)

2. **Hardcode Explanations**
   - ✅ Complete control over content
   - ✅ No LLM costs
   - ❌ Massive manual effort (220+ objectives × multiple explanations)
   - ❌ Can't adapt to student's prior knowledge
   - ❌ Rigid (can't handle follow-up questions)

3. **RAG (Retrieval-Augmented Generation)**
   - ✅ Dynamic content generation
   - ✅ Adapts to student's prior knowledge
   - ✅ Cites sources (curriculum objectives)
   - ✅ Easy to update (just add new objectives)
   - ✅ Handles follow-up questions naturally
   - ✅ Reduced hallucination (grounded in curriculum)
   - ❌ Retrieval overhead (~50-150ms)
   - ❌ Requires good retrieval (BM25 + semantic)

### Rationale

**RAG is superior for educational content:**

1. **Adaptability**: Explanations adapt to student's grade, prior knowledge, learning style
   ```python
   # RAG context includes student model
   context = f"Grade: {student.grade}, Mastered: {student.mastered_concepts}"
   result = await rag.query(f"{context}\n\n{question}")
   ```

2. **Source Attribution**: Every answer cites curriculum objectives
   ```python
   result.sources  # ["math.algebra.8.linear_equations"]
   ```

3. **Easy Updates**: Add new objectives without retraining
   ```python
   await rag.ingest("New objective: Quantum computing basics...")
   ```

4. **Multi-Step Reasoning**: Research mode for complex topics
   ```python
   result = await rag.query(question, mode="research")
   # Automatically breaks into sub-queries
   ```

5. **Multimodal Support**: Text + images + videos
   ```python
   result = await multimodal_rag.query_with_image(question, diagram)
   ```

**Cost Comparison:**
- Fine-tuning: $10,000+ one-time + $500/month inference
- RAG: $0 setup + $50-200/month LLM API calls

**Performance:**
- Fine-tuning: ~100ms inference
- RAG: ~150-600ms (depending on mode)
- **Acceptable**: EdWIN targets <800ms for verified answers

### Consequences

**Positive:**
- $10,000+ savings (no fine-tuning)
- Dynamic, adaptive explanations
- Easy curriculum updates
- Source attribution for trust
- Multi-step reasoning capability

**Negative:**
- 50-150ms retrieval overhead
- Requires good retrieval (BM25 + semantic embeddings)
- LLM API costs ($50-200/month)

**Mitigation:**
- Use caching for repeated queries (100x speedup)
- Optimize retrieval with hybrid search (BM25 + semantic)
- Option to use local LLM (Ollama) for $0 cost

---

## ADR-003: Thompson Sampling for Adaptive Difficulty

### Context

Students learn best in the "zone of proximal development" - slightly above current skill level. How do we select the optimal challenge for each student?

### Decision

**Use Thompson Sampling (Bayesian bandit algorithm) to select optimal difficulty, balancing exploration and exploitation.**

### Alternatives Considered

1. **Epsilon-Greedy**
   - ✅ Simple to implement
   - ✅ Well-understood
   - ❌ Fixed exploration rate (e.g., 10%)
   - ❌ Doesn't adapt to uncertainty
   - ❌ Suboptimal regret bounds

2. **Upper Confidence Bound (UCB)**
   - ✅ Good regret bounds
   - ✅ Deterministic (easier to debug)
   - ❌ Requires tuning hyperparameter (c)
   - ❌ Aggressive exploration (can frustrate students)

3. **Thompson Sampling**
   - ✅ Optimal regret bounds (matches UCB)
   - ✅ Adapts exploration to uncertainty
   - ✅ Naturally balances exploration/exploitation
   - ✅ No hyperparameters to tune
   - ✅ Bayesian (provides confidence intervals)
   - ❌ Stochastic (harder to debug)

### Rationale

**Thompson Sampling is optimal for educational settings:**

1. **Adaptive Exploration**: Explores more when uncertain, less when confident
   - Early in learning: High uncertainty → explore many topics
   - Later: Low uncertainty → exploit known good topics

2. **Natural Fit**: Each objective is a bandit arm
   - Reward = student success × engagement
   - Prior: Beta(α=1, β=1) (uniform)
   - Update: Success → α += reward, Failure → β += (1 - reward)

3. **Provably Optimal**: Matches theoretical lower bound on regret
   - Regret = O(√T) where T = number of trials
   - Explores efficiently (no wasted effort)

4. **Research Validated**:
   - Russo et al. (2018): "Thompson Sampling is optimal"
   - Used in production: Google Ads, Netflix recommendations

**Example:**
```python
# Student attempts "Solve linear equations"
success = 0.85  # 85% confidence
engagement = 0.9  # High engagement

reward = success * engagement  # 0.765

# Update Thompson Sampling
bandit.update(objective_idx, reward)

# Next objective: Thompson sample
next_idx = bandit.select_arm()  # Probabilistic based on learned priors
```

**Comparison to Alternatives:**

| Algorithm | Exploration | Regret Bound | Hyperparameters | Best For |
|-----------|-------------|--------------|-----------------|----------|
| Random | Constant | O(T) | None | Baseline |
| Epsilon-Greedy | Fixed (ε) | O(T) | 1 (ε) | Simple cases |
| UCB | Decreasing | O(√T log T) | 1 (c) | Deterministic needed |
| **Thompson Sampling** | **Adaptive** | **O(√T)** | **None** | **Education** |

### Consequences

**Positive:**
- Optimal challenge selection (zone of proximal development)
- No hyperparameters to tune (works out of the box)
- Adapts to individual student learning curves
- 15-20% better engagement vs random selection (research)

**Negative:**
- Stochastic (same student may get different objectives)
- Requires >20 interactions per objective to converge
- Harder to explain to teachers ("Why this objective?")

**Mitigation:**
- Provide expected reward estimates for transparency
- Offer manual override for teachers
- Track convergence metrics (α, β) for debugging

---

## ADR-004: Student Model as Knowledge Graph

### Context

Each student has a unique learning journey with mastered concepts, skill progressions, and knowledge gaps. How should we represent this?

### Decision

**Maintain a personal knowledge graph for each student, mirroring the curriculum graph but with mastery annotations.**

### Alternatives Considered

1. **Flat List of Mastered Objectives**
   - ✅ Simple to implement
   - ❌ Loses prerequisite structure
   - ❌ Can't identify knowledge gaps
   - ❌ No temporal tracking

2. **Skill Tree (Game-Style)**
   - ✅ Visual appeal
   - ✅ Clear progression
   - ❌ Single path (no lateral learning)
   - ❌ Difficult to represent cross-subject connections

3. **Personal Knowledge Graph**
   - ✅ Mirrors curriculum graph structure
   - ✅ Easy to identify knowledge gaps
   - ✅ Temporal tracking (when learned)
   - ✅ Enables personalized learning paths
   - ❌ More complex storage

### Rationale

**Personal KG enables sophisticated learning analytics:**

1. **Knowledge Gap Detection**:
   ```python
   # What prerequisites are missing for target objective?
   target_prereqs = curriculum_kg.get_prerequisites("math.algebra.10.systems")
   mastered = student.personal_kg.get_mastered_concepts()
   gaps = target_prereqs - mastered
   # gaps = ["math.algebra.9.quadratic"]
   ```

2. **Learning Path Generation**:
   ```python
   # Find path from student's current knowledge to target
   path = student.personal_kg.shortest_path(
       from_nodes=student.get_frontier(),  # Edge of knowledge
       to_node="physics.mechanics.11.projectile_motion"
   )
   # path = ["math.algebra.9.quadratic", "math.trig.10.functions", ...]
   ```

3. **Temporal Queries**:
   ```python
   # What did student know on October 12?
   knowledge_oct_12 = student.personal_kg.get_state(date="2025-10-12")
   ```

4. **Forgetting Curves**:
   ```python
   # Weight edges by recency
   days_since_practice = (now - edge.last_practiced).days
   retention = 0.95 ** days_since_practice  # Exponential decay
   ```

### Consequences

**Positive:**
- Precise knowledge gap identification
- Personalized learning paths
- Temporal tracking (learning history)
- Forgetting curve modeling

**Negative:**
- Higher storage cost (1 KG per student)
- More complex queries

**Mitigation:**
- Store only differences from curriculum graph (sparse representation)
- Use Neo4j for production scale (millions of students)

---

## ADR-005: HoloLoom Alignment Framework for K-12 Safety

### Context

K-12 content must be age-appropriate, safe, and privacy-preserving. We need comprehensive safety guardrails.

### Decision

**Use HoloLoom Alignment Framework as foundation, extend with K-12-specific policies (reading level, content filters, COPPA/FERPA compliance).**

### Alternatives Considered

1. **OpenAI Moderation API**
   - ✅ Pre-built, battle-tested
   - ✅ Detects violence, hate speech, etc.
   - ❌ Not K-12 specific (misses educational edge cases)
   - ❌ No reading level validation
   - ❌ No privacy compliance

2. **Custom Filter List**
   - ✅ Complete control
   - ✅ K-12 specific
   - ❌ Brittle (keyword-based)
   - ❌ Easy to bypass
   - ❌ Requires constant maintenance

3. **HoloLoom Alignment + K-12 Extensions**
   - ✅ Comprehensive safety framework
   - ✅ Risk-based action gating
   - ✅ Human-in-the-loop for edge cases
   - ✅ Audit trail (complete provenance)
   - ✅ Extensible (add K-12 policies)
   - ❌ Slight overhead (<1ms per query)

### Rationale

**Multi-Layered Safety:**

1. **Pre-Filter (Input Validation)**:
   - Block inappropriate questions
   - Detect PII in student queries
   - Flag high-risk topics

2. **Alignment Framework (Generation)**:
   - Safety guardrails during LLM generation
   - Risk scoring for responses
   - Human-in-the-loop for high-risk

3. **Post-Filter (Output Validation)**:
   - Reading level check (Flesch-Kincaid)
   - Content filtering (blocked topics)
   - PII detection in responses

4. **Audit Trail**:
   - Log all interactions
   - Teacher review dashboard
   - Parent access to history

**K-12 Specific Policies:**
```python
K12_POLICIES = {
    "elementary": {
        "max_reading_level": 7,
        "blocked_topics": ["violence", "weapons", "adult_content"],
        "require_parental_consent": True
    },
    "middle": {
        "max_reading_level": 10,
        "blocked_topics": ["violence", "weapons", "adult_content"]
    },
    "high": {
        "max_reading_level": 14,
        "blocked_topics": ["explicit_violence", "adult_content"]
    }
}
```

**Compliance:**
- **COPPA**: Parental consent for <13, PII protection
- **FERPA**: Student records privacy, parent access
- **GDPR** (if EU): Right to be forgotten, data export

### Consequences

**Positive:**
- Multi-layered safety (defense in depth)
- Compliant with COPPA, FERPA
- Teacher oversight (audit trail)
- Extensible (add new policies easily)

**Negative:**
- <1ms overhead per query
- Occasional false positives (safe content blocked)

**Mitigation:**
- Cache safety checks for repeated content
- Teacher override for false positives
- Continuous tuning of filters

---

## ADR-006: Hybrid Storage (In-Memory + Persistent)

### Context

Development needs fast iteration (in-memory), production needs persistence (Neo4j/Qdrant). How do we support both?

### Decision

**Use HoloLoom's memory backend factory with automatic fallback: HYBRID (Neo4j+Qdrant) → INMEMORY (NetworkX).**

### Alternatives Considered

1. **In-Memory Only**
   - ✅ Fast, simple
   - ❌ Data loss on restart
   - ❌ Not production-ready

2. **Persistent Only (Neo4j/Qdrant)**
   - ✅ Production-ready
   - ❌ Requires Docker (barrier to entry)
   - ❌ Slower development iteration

3. **Hybrid with Automatic Fallback**
   - ✅ Best of both worlds
   - ✅ No Docker needed for development
   - ✅ Automatic upgrade to production
   - ✅ Graceful degradation
   - ❌ Slightly more complex

### Rationale

**Graceful Degradation:**

```python
# Configuration
config = Config.fused()
config.memory_backend = MemoryBackend.HYBRID

# Automatic fallback
memory = await create_memory_backend(config)
# If Neo4j available → Neo4j + Qdrant
# If Neo4j unavailable → NetworkX in-memory (warning logged)
```

**Developer Experience:**
- Day 1: No Docker, just `python demo.py` (uses in-memory)
- Week 2: `docker-compose up -d` (automatic upgrade to persistent)
- Production: Full Neo4j+Qdrant (no code changes)

**Migration Path:**
```python
# Export from in-memory
kg_inmemory.save("./data/curriculum_graph.json")

# Import to Neo4j
kg_neo4j = await create_memory_backend(MemoryBackend.HYBRID)
kg_neo4j.load("./data/curriculum_graph.json")
```

### Consequences

**Positive:**
- Zero-friction onboarding (no Docker required)
- Production-ready (Neo4j + Qdrant)
- Seamless migration (in-memory → persistent)

**Negative:**
- Developers may forget to start Docker (data loss)
- Different performance characteristics (in-memory vs persistent)

**Mitigation:**
- Loud warnings when falling back to in-memory
- Documentation: "Start Docker for persistence"
- Auto-save to JSON for in-memory safety net

---

## Summary of Key Decisions

| Decision | Technology | Rationale |
|----------|-----------|-----------|
| **Curriculum Representation** | Knowledge Graph (NetworkX/Neo4j) | Natural fit for prerequisites, enables graph traversal |
| **Content Generation** | RAG (SimpleRAG) | Adaptive, source attribution, easy updates |
| **Adaptive Difficulty** | Thompson Sampling | Optimal regret bounds, no hyperparameters |
| **Student Model** | Personal Knowledge Graph | Knowledge gap detection, temporal tracking |
| **Safety** | HoloLoom Alignment + K-12 Extensions | Multi-layered, compliant (COPPA/FERPA) |
| **Storage** | Hybrid (In-Memory + Persistent) | Graceful degradation, zero-friction onboarding |

---

## Future Decisions

### To Be Decided

1. **Frontend Framework**: React vs Vue vs Svelte?
2. **Authentication**: OAuth vs JWT vs Session-based?
3. **Real-Time Updates**: WebSockets vs Server-Sent Events vs Polling?
4. **Analytics**: Mixpanel vs Amplitude vs Custom?
5. **LLM Provider**: Anthropic (Claude) vs OpenAI (GPT-4) vs Ollama (local)?

### Decision Timeline

- Week 3: Frontend framework
- Week 5: Authentication strategy
- Week 7: Analytics platform
- Week 8: Production LLM provider

---

**Document Version**: 1.0.0
**Last Updated**: November 15, 2025
**Authors**: EdWIN Architecture Team
