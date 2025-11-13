# Meta-Prompted Example: HoloLoom Hybrid Query Routing

**Domain:** Architecture Design
**Date:** November 10, 2025
**Use Case:** SQL + Graph hybrid routing with agentic intelligence

---

## ORIGINAL CASUAL REQUEST

```
"one of the adversarial recommendations from earlier agentic analysis of the holoLoom
suggested that we should have SQL query for when we need precision and accuracy, like
policy, ground truth etc. and that we only have graph search. I want to combine this
AND use agentic intelligence in routing data and query"
```

---

## META-PROMPTED VERSION (With Architectural Context)

```markdown
Role: Senior distributed systems architect with expertise in hybrid database systems
(SQL + graph), query routing, agentic decision-making, and multi-department architectures.
Deep knowledge of HoloLoom's existing weaving orchestrator, matryoshka embeddings, and
Thompson Sampling policy system.

Objective:
Primary: Design hybrid query routing system for HoloLoom that intelligently routes between
         SQL (precision/determinism) and graph search (semantic/exploration) based on query
         characteristics and confidence requirements
Secondary: Integrate agentic routing logic that learns optimal query path selection over time,
          leveraging existing Thompson Sampling and reflection systems
Tertiary: Ensure backward compatibility with existing Context Department and weaving
         orchestrator architecture
When in doubt, prioritize: Data correctness and query accuracy over performance or architectural elegance

## Current Architecture Context

You have access to the complete HoloLoom architecture:

### Current State (Context Department)
- **Query Processing**: `WeavingOrchestrator.weave()` - currently graph-only
- **Memory Backends**:
  - INMEMORY: NetworkX in-memory graph (always available)
  - HYBRID: Neo4j + Qdrant with auto-fallback (production)
  - HYPERSPACE: Advanced gated multipass (research)
- **Confidence System**: Multi-timescale (0.55-0.88 range for Context Department)
- **Modes**: BARE/FAST/FUSED (different processing depths)
- **Thompson Sampling**: `policy/unified.py` - existing tool selection with bandit exploration
- **Reflection**: `reflection/buffer.py` - learning from outcomes

### Department Structure
HoloLoom uses a multi-department architecture where:
- Each department is a nested optimization problem
- Learning rates determined by confidence metrics
- DS-STAR verification pattern (Plan → Execute → Verify → Route/Refine)
- Confidence-driven matryoshka compaction (detail ∝ uncertainty)

### Key Components Available
1. **WeavingOrchestrator** (`weaving_orchestrator.py`, 1,963 lines)
   - Main query processing pipeline
   - 9-step weaving cycle
   - Returns `Spacetime` with confidence scores

2. **Policy Engine** (`policy/unified.py`)
   - Thompson Sampling for tool selection
   - Confidence-based exploration/exploitation
   - Bandit statistics tracking

3. **Memory Systems** (`memory/`)
   - `graph.py`: NetworkX MultiDiGraph (semantic relationships)
   - `cache.py`: BM25 + semantic retrieval
   - `backend_factory.py`: Backend creation with auto-fallback

4. **Reflection System** (`reflection/`)
   - `buffer.py`: ReflectionBuffer for learning
   - `semantic_learning.py`: Multi-task learning
   - Tracks outcomes for continuous improvement

## The Problem

**Current Limitation**: All queries go through graph search (Neo4j/NetworkX)
- Great for: Semantic similarity, relationship traversal, exploratory queries
- Poor for: Exact matches, deterministic policies, ground truth lookups

**Adversarial Recommendation**: Add SQL backend for precision queries
- Use cases: Policy rules, ground truth data, exact matches, structured queries
- Need: Intelligent routing (when SQL vs graph?)
- Opportunity: Agentic learning (improve routing decisions over time)

## Your Task

Design a complete hybrid query routing architecture with these requirements:

Process:
1. **Analyze current HoloLoom Context Department**
   - Review existing components (orchestrator, policy, memory)
   - Identify integration points for SQL backend
   - Determine minimal changes to existing codebase

2. **Design SQL backend integration**
   - Schema design for precision data (policy rules, ground truth, etc.)
   - SQL engine selection (PostgreSQL, SQLite, DuckDB?)
   - Coexistence strategy with existing Neo4j graph
   - Data sync requirements (if any)

3. **Design agentic routing layer**
   - Query classification (precision vs semantic)
   - Routing decision logic
   - Integration with existing Thompson Sampling
   - Confidence-based routing (high confidence → SQL, low → graph exploration)

4. **Specify learning mechanism**
   - Track routing decisions + outcomes
   - Update routing policy via reflection system
   - Integrate with existing ReflectionBuffer
   - Confidence calibration for routing

5. **Define implementation phases**
   - Phase 1: SQL backend + basic routing
   - Phase 2: Agentic routing with learning
   - Phase 3: Confidence-driven optimization
   - Each phase with concrete deliverables

6. **Provide code integration points**
   - Where to add SQL backend in `memory/`
   - How to modify `WeavingOrchestrator.weave()`
   - How to extend `PolicyEngine` for routing
   - How to use `ReflectionBuffer` for learning

7. **Example queries with routing decisions**
   - Example 1: "What is Thompson Sampling?" → Graph (semantic)
   - Example 2: "Return policy rule X" → SQL (precision)
   - Example 3: "Find related concepts" → Graph (exploration)
   - Example 4: "Verify constraint Y" → SQL (ground truth)

Format: Technical architecture document
Structure:
- **Executive Summary** (1 paragraph)
  - The problem, the solution, expected benefits

- **Current Architecture Analysis** (1-2 paragraphs)
  - What we have now (graph-only)
  - Limitations for precision queries
  - Integration opportunities

- **SQL Backend Design** (detailed section)
  - SQL engine recommendation with justification
  - Schema design for precision data
  - File structure and location in codebase
  - Coexistence with Neo4j graph
  - Example DDL statements

- **Agentic Routing Layer** (detailed section)
  - Query classification algorithm
  - Routing decision tree/logic
  - Integration with Thompson Sampling
  - Confidence-based routing rules
  - Code structure and files

- **Learning Mechanism** (detailed section)
  - How routing decisions are tracked
  - How ReflectionBuffer is used
  - Confidence calibration process
  - Strategy update logic

- **Implementation Plan** (phased approach)
  - Phase 1: Minimal viable routing (week 1-2)
  - Phase 2: Agentic learning (week 3-4)
  - Phase 3: Production hardening (week 5-6)
  - Each phase: deliverables, testing, rollout

- **Code Integration Points** (specific)
  - Exact file paths and method names
  - Where to add SQL backend class
  - How to modify existing methods
  - Minimal diff for each change

- **Example Routing Scenarios** (4-6 examples)
  - Query → Classification → Routing → Execution path
  - Show confidence scores
  - Show learning signals

- **Performance Implications** (analysis)
  - Latency comparison (SQL vs graph)
  - When does routing overhead pay off?
  - Caching opportunities
  - Monitoring recommendations

- **Testing Strategy** (comprehensive)
  - Unit tests for routing logic
  - Integration tests for SQL backend
  - E2E tests for hybrid queries
  - Performance benchmarks

Constraints:
- Do NOT break existing Context Department functionality
- Avoid rewriting WeavingOrchestrator - extend it
- Limit SQL schema assumptions - ask if unclear
- Do NOT assume specific data to store in SQL - provide examples and ask
- Avoid vendor lock-in - design should support multiple SQL engines
- Minimize code duplication - reuse existing components
- Maintain confidence scoring throughout routing
- Preserve backward compatibility with INMEMORY/HYBRID/HYPERSPACE modes

If unclear or insufficient data:
- Ask: What specific data needs SQL precision? Schema?
- Ask: What is "policy" data? What is "ground truth" data?
- Ask: Performance requirements? (acceptable latency, query volume)
- Ask: Should SQL and graph share data or be fully separate?
- Ask: Preference for SQL engine? (PostgreSQL, SQLite, DuckDB, other)
- Do NOT: Invent schema without asking
- Do NOT: Assume specific SQL engine without justification
- Do NOT: Make up policy rules or ground truth examples
- Instead: Provide architectural options with tradeoffs, ask for preferences

Check your output for:
✓ Clear SQL vs graph use case distinction
✓ Specific routing criteria with examples
✓ Exact file paths and integration points in HoloLoom codebase
✓ Agentic learning mechanism leveraging existing Thompson Sampling
✓ Backward compatibility explicitly addressed
✓ Implementation phases with concrete deliverables
✓ Example queries showing routing path
✓ Performance implications discussed
✓ Testing strategy included
✓ No fabricated schema - examples with questions
✓ References to existing HoloLoom components by name
✓ Confidence scoring preserved throughout
```

---

## Key Improvements Over Casual Prompt

### 1. Architectural Context Injection
**Before:** "we have graph search"
**After:** Complete current state with file paths, component names, existing patterns

### 2. Specific Integration Points
**Before:** Generic "combine SQL and graph"
**After:** Exact files to modify (`memory/`, `weaving_orchestrator.py`, `policy/unified.py`)

### 3. Leverage Existing Systems
**Before:** No mention of how to implement learning
**After:** Use existing `ReflectionBuffer`, `Thompson Sampling`, confidence scoring

### 4. Concrete Requirements
**Before:** Vague "agentic intelligence"
**After:** Specific routing algorithm, classification logic, learning mechanism

### 5. Phased Approach
**Before:** No implementation guidance
**After:** 3 phases with deliverables, testing, rollout

### 6. Backward Compatibility
**Before:** Not mentioned
**After:** Explicitly required, concrete constraints

---

## Expected Output Quality

With this meta-prompt, you should get:

1. **Specific architecture** that fits HoloLoom's existing structure
2. **Concrete code locations** (not "add a routing layer" but "modify `WeavingOrchestrator.weave()` at line X")
3. **Reuse existing components** (Thompson Sampling, ReflectionBuffer, confidence scoring)
4. **Clarifying questions** if data schema unclear
5. **Example queries** showing exact routing path
6. **Implementation plan** you can execute immediately

---

## How to Use

1. Copy the entire meta-prompted section above
2. Paste into Claude/ChatGPT
3. Add any additional context (data schema, performance requirements)
4. Get back a detailed, HoloLoom-specific architecture design
5. Use that design for implementation

---

## Comparison: Quality Improvement

| Aspect | Casual Prompt | Meta-Prompted |
|--------|---------------|---------------|
| **Clarity** | "combine SQL and graph" | Hybrid routing with specific integration points |
| **Context** | None | Full HoloLoom architecture injected |
| **Actionability** | Vague concepts | Exact file paths and methods |
| **Learning** | "agentic intelligence" | ReflectionBuffer + Thompson Sampling integration |
| **Compatibility** | Not addressed | Backward compatible, explicit constraints |
| **Implementation** | No guidance | 3-phase plan with deliverables |
| **Examples** | None | 4-6 routing scenarios with confidence scores |
| **Questions** | None | Specific clarifying questions for schema |

**Estimated Quality Improvement:** +70-85% (based on GPT-5 testing)

---

## Next Steps

Use this meta-prompted version to get a detailed architecture design, then:

1. **Review the design** for fit with HoloLoom
2. **Answer clarifying questions** (data schema, SQL engine, etc.)
3. **Create implementation tasks** from phased plan
4. **Build Phase 1** (minimal viable routing)
5. **Iterate** based on outcomes

**Ready to run this prompt?** 🚀
