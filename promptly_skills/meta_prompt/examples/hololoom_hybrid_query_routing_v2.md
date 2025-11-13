# Meta-Prompted Example: HoloLoom Hybrid Query Routing (v2)

**Domain:** Departmental Agent Architecture Design
**Date:** November 10, 2025
**Use Case:** SQL + Graph + Vector hybrid routing with agentic intelligence

---

## ORIGINAL CASUAL REQUEST

```
"one of the adversarial recommendations from earlier agentic analysis of the holoLoom
suggested that we should have SQL query for when we need precision and accuracy, like
policy, ground truth etc. and that we only have graph search. I want to combine this
AND use agentic intelligence in routing data and query"
```

---

## CONTEXT-ENRICHED META-PROMPT (v2 - With Departmental Architecture)

```markdown
Role: Senior distributed systems architect with expertise in:
- Multi-department agent architectures (Conway's Law for AI)
- Hybrid database systems (SQL, graph, vector)
- Agentic query routing and confidence-driven optimization
- MCP-based federated communication
- Deep knowledge of HoloLoom's 6-department architecture

Objective:
Primary: Design hybrid query backend for Infrastructure Department that supports SQL
         (precision/determinism), graph (semantic relationships), and vector (similarity)
         with intelligent routing in Context Department
Secondary: Implement agentic routing logic in Context that learns optimal backend
          selection over time, using existing Thompson Sampling and reflection systems
Tertiary: Ensure clean departmental boundaries via MCP tools, maintain backward
         compatibility with existing Infrastructure → Context communication
When in doubt, prioritize: Data correctness and departmental separation of concerns
                           over performance or architectural complexity

## HoloLoom Departmental Architecture Context

You have complete access to the 6-department agent swarm architecture:

### Department Registry

**Orchestration** (Top-level coordinator)
- Role: Task routing, department coordination, session management
- Tools: `route_task`, `get_session_context`, `escalate_decision`
- Dependencies: None (top-level)
- Context Budget: 100k tokens

**Infrastructure** (Data systems - YOUR FOCUS)
- Role: Data systems, performance optimization, zero-copy queries
- Tools: `query_neo4j`, `query_qdrant`, `diagnose_performance`
- Dependencies: None (foundational)
- Context Budget: 20k tokens
- **NOTE**: This is where SQL backend needs to be added

**Context** (HoloLoom - Query processor)
- Role: Multi-pass context enrichment, missing context detection
- Tools: `enrich_context`, `detect_missing_context`
- Dependencies: Infrastructure, MasterWeaver
- Context Budget: 60k tokens
- **NOTE**: This is where routing logic will live

**MasterWeaver** (Entity extraction)
- Role: Entity extraction, domain understanding, knowledge structuring
- Tools: `extract_entities`, `validate_entity_consistency`, `query_domain_ontology`
- Dependencies: Infrastructure
- Context Budget: 50k tokens

**Verification** (Quality assurance)
- Role: Quality assurance, confidence validation, alignment checking
- Tools: `validate_confidence_claim`, `request_rerun`, `cross_check_departments`
- Dependencies: MasterWeaver, Infrastructure, Execution
- Context Budget: 30k tokens

**Execution** (Task runner)
- Role: Task execution, code running, deployment orchestration
- Tools: `run_claude_code_task`, `check_task_status`
- Dependencies: Infrastructure
- Context Budget: 40k tokens

### Dependency Graph
```
Orchestration (no dependencies)
    │
    ├─► MasterWeaver → Infrastructure
    ├─► Verification → MasterWeaver, Infrastructure, Execution
    ├─► Infrastructure (no dependencies)
    ├─► Execution → Infrastructure
    └─► Context → Infrastructure, MasterWeaver
```

### Communication Pattern (MCP-based)
```
Context Department
    ↓ (MCP call)
Infrastructure.query_neo4j(query, session_id)
    ↓ (MCP response)
Context Department (receives graph data)

NEW FLOW (with SQL):
Context Department
    ↓ (decides: SQL or graph?)
    ├─► Infrastructure.query_sql(query, session_id)     [PRECISION]
    └─► Infrastructure.query_neo4j(query, session_id)   [SEMANTIC]
```

### Key Architectural Constraints

1. **Departmental Boundaries** (Conway's Law)
   - Infrastructure owns data access
   - Context owns routing decisions
   - Clean MCP interfaces between departments

2. **Context Budgets**
   - Infrastructure: 20k tokens (must fit SQL schema + query results)
   - Context: 60k tokens (must fit routing logic + results)

3. **Tool Permissions** (from ARCHITECTURE_SUMMARY.txt)
   ```
   Infrastructure: Read ✓, Write ✓, Execute ✗, Deploy ✗, Admin ✓
   Context:        Read ✓, Write ✓, Execute ✗, Deploy ✗, Admin ✗
   ```

4. **Session Management**
   - Each department tracks current work
   - Hierarchical: shared CLAUDE.md + per-dept CHARTER.md
   - Orchestration maintains master roadmap

5. **Error Handling**
   - Infrastructure retries failed queries internally
   - Context escalates to Verification if confidence low
   - Orchestration coordinates recovery

### Current Implementation Status

From the architecture document:

**Phase 1: Foundation**
- [✓] Department registry defined
- [✓] Architecture documentation
- [ ] Department CHARTER.md templates
- [ ] Basic MCP server framework ← WE ARE HERE
- [ ] Test inter-department communication

**Your task extends Phase 1 → Phase 2**

## The Problem (Detailed)

### Current State: Graph-Only

**Infrastructure Department provides:**
```python
# HoloLoom/infrastructure/mcp_server.py (conceptual)
async def query_neo4j(query: str, session_id: str) -> Dict:
    """Query Neo4j graph database"""
    # Returns: nodes, relationships, confidence

async def query_qdrant(embedding: List[float], session_id: str) -> Dict:
    """Query Qdrant vector database"""
    # Returns: similar vectors, scores
```

**Context Department uses:**
```python
# HoloLoom/context/weaving_orchestrator.py
async def weave(query: Query) -> Spacetime:
    # Always calls Infrastructure.query_neo4j()
    # No routing logic - graph-only
```

**Limitations:**
- ❌ No exact matching (SQL WHERE clauses)
- ❌ No deterministic policy lookups
- ❌ No ground truth verification
- ❌ No structured query optimization

### Adversarial Recommendation

**From Verification Department analysis:**
> "Infrastructure should support SQL for precision queries (policy rules,
> ground truth data, exact matches). Current graph-only limits deterministic
> operations. Recommend hybrid backend with agentic routing."

**Use Cases Requiring SQL:**
1. **Policy rules**: "Return penalty calculation for rule X"
2. **Ground truth**: "Verify constraint Y is satisfied"
3. **Exact matches**: "Find record WHERE id = 'abc-123'"
4. **Aggregations**: "COUNT users WHERE status = 'active'"
5. **Transactional**: "BEGIN; UPDATE...; COMMIT;"

### Desired End State: Hybrid Routing

**Infrastructure provides 3 backends:**
```python
# SQL for precision
async def query_sql(sql: str, params: Dict, session_id: str) -> Dict:
    """Execute SQL query for deterministic operations"""

# Graph for semantics
async def query_neo4j(cypher: str, session_id: str) -> Dict:
    """Query graph for relationship traversal"""

# Vector for similarity
async def query_qdrant(embedding: List[float], session_id: str) -> Dict:
    """Query vector DB for semantic similarity"""
```

**Context routes intelligently:**
```python
async def weave(query: Query) -> Spacetime:
    # 1. Classify query characteristics
    # 2. Route to appropriate backend(s)
    # 3. Learn from outcomes
```

## Your Task

Design the complete hybrid query routing system with these 2 components:

### Component 1: Infrastructure Department - SQL Backend

Add SQL backend to Infrastructure Department while maintaining clean MCP boundaries.

**Requirements:**
1. **New MCP Tool**: `query_sql(sql, params, session_id)`
2. **SQL Engine Selection**: Choose PostgreSQL, SQLite, or DuckDB with justification
3. **Schema Design**: Define precision data schema (policy, ground truth, etc.)
4. **Coexistence**: SQL alongside Neo4j + Qdrant (how do they relate?)
5. **Permission Model**: Infrastructure has Read/Write, enforce query safety
6. **Error Handling**: Internal retries, escalation to Orchestration
7. **Context Budget**: Fit SQL schema + results in 20k token budget

**File Location**: `HoloLoom/infrastructure/sql_backend.py`

**MCP Tool Definition**:
```python
# Should integrate with existing infrastructure MCP server
# Follow pattern from query_neo4j and query_qdrant
```

### Component 2: Context Department - Routing Logic

Add intelligent routing to Context Department's weaving orchestrator.

**Requirements:**
1. **Query Classification**: Determine SQL vs graph vs vector
2. **Routing Algorithm**: Decision tree or learned model
3. **Thompson Sampling Integration**: Use existing bandit for backend selection
4. **Confidence-Based Routing**: High confidence → SQL, low → graph exploration
5. **Multi-Backend Queries**: When to query multiple backends and merge?
6. **Learning Mechanism**: Track routing decisions via ReflectionBuffer
7. **Context Budget**: Fit routing logic + results in 60k token budget

**File Location**: `HoloLoom/context/query_router.py`

**Integration Point**: Modify `WeavingOrchestrator.weave()` to use router

### Process (Step-by-Step)

1. **Analyze Current Infrastructure**
   - Review existing `query_neo4j` and `query_qdrant` MCP tools
   - Understand Infrastructure's role as foundational data provider
   - Identify where SQL backend fits in department responsibilities

2. **Design SQL Backend for Infrastructure**
   - SQL engine selection (PostgreSQL/SQLite/DuckDB) with justification
   - Schema design for precision data (provide examples, ask for specifics)
   - File structure: where does `sql_backend.py` go?
   - MCP tool definition: `query_sql` following existing patterns
   - Safety: How to prevent SQL injection, enforce permissions?
   - Coexistence: How does SQL relate to Neo4j and Qdrant?

3. **Design Routing Logic for Context**
   - Query classification algorithm (what makes a query "SQL-worthy"?)
   - Routing decision tree or model
   - Integration with existing Thompson Sampling (`policy/unified.py`)
   - When to use multiple backends (SQL + graph, graph + vector, all 3?)
   - File structure: where does `query_router.py` go in Context?

4. **Specify Learning Mechanism**
   - Track: query → backend selection → confidence → outcome
   - Store in ReflectionBuffer (existing medium-term memory)
   - Update Thompson Sampling statistics per backend
   - Confidence calibration: did SQL work better than expected?

5. **Define MCP Communication**
   - Context → Infrastructure request/response format
   - How does Context call `query_sql` vs `query_neo4j`?
   - Session ID propagation across departments
   - Error escalation path (Infrastructure → Context → Orchestration)

6. **Implementation Phases**
   - Phase 1: Infrastructure SQL backend (week 1-2)
   - Phase 2: Context basic routing (week 3)
   - Phase 3: Agentic learning integration (week 4)
   - Each phase: deliverables, tests, MCP contracts

7. **Example Routing Scenarios**
   - Query: "What is Thompson Sampling?" → Graph (semantic exploration)
   - Query: "Return policy rule 'penalty_calc_v2'" → SQL (exact match)
   - Query: "Find similar concepts" → Vector (similarity)
   - Query: "Verify constraint X satisfied" → SQL (ground truth)
   - Query: "How does X relate to Y?" → Graph (relationships)

Format: Technical architecture document
Structure:
- **Executive Summary** (1 paragraph)
  - Problem, solution, expected benefits, departmental impact

- **Infrastructure Department: SQL Backend Design**
  - SQL engine recommendation (PostgreSQL/SQLite/DuckDB) with justification
  - Schema design for precision data (examples + questions)
  - File location and structure in HoloLoom/infrastructure/
  - MCP tool definition (following existing query_neo4j pattern)
  - Permission model and SQL injection prevention
  - Coexistence with Neo4j + Qdrant
  - Error handling and escalation
  - Context budget management (fit in 20k tokens)
  - Example DDL statements

- **Context Department: Routing Logic Design**
  - Query classification algorithm (SQL vs graph vs vector)
  - Routing decision logic (tree/model)
  - Integration with Thompson Sampling (policy/unified.py)
  - Confidence-based routing rules
  - Multi-backend query patterns (when to combine?)
  - File location in HoloLoom/context/
  - Integration with WeavingOrchestrator.weave()
  - Context budget management (fit in 60k tokens)

- **MCP Communication Protocol**
  - Context → Infrastructure request format
  - Infrastructure → Context response format
  - Session ID propagation
  - Error escalation (Infrastructure → Context → Orchestration)
  - Departmental boundaries maintained

- **Agentic Learning Mechanism**
  - Tracking: query → backend → confidence → outcome
  - ReflectionBuffer integration (medium-term memory)
  - Thompson Sampling updates per backend
  - Confidence calibration process
  - Strategy update logic

- **Implementation Plan** (phased)
  - Phase 1: Infrastructure SQL backend (deliverables, tests, rollout)
  - Phase 2: Context basic routing (deliverables, tests, rollout)
  - Phase 3: Agentic learning (deliverables, tests, rollout)
  - Each phase: MCP contract definition, testing strategy

- **Code Integration Points** (specific file paths)
  - Infrastructure: `HoloLoom/infrastructure/sql_backend.py`
  - Context: `HoloLoom/context/query_router.py`
  - Modify: `HoloLoom/context/weaving_orchestrator.py` (add routing call)
  - Modify: `HoloLoom/infrastructure/mcp_server.py` (add query_sql tool)
  - Test: Unit, integration, MCP communication

- **Example Routing Scenarios** (6-8 examples)
  - Query text → Classification → Backend selection → Execution path
  - Show confidence scores, Thompson statistics
  - Show learning signals and feedback

- **Performance & Resource Management**
  - Latency comparison (SQL vs graph vs vector)
  - Context budget usage per query type
  - When routing overhead pays off
  - Monitoring recommendations per department

- **Departmental Impact Analysis**
  - Infrastructure: New responsibility (SQL backend management)
  - Context: New responsibility (routing intelligence)
  - MasterWeaver: No change (still depends on Infrastructure)
  - Verification: Could validate routing decisions
  - Orchestration: May need to handle escalations
  - Execution: No change

- **Testing Strategy**
  - Unit tests: SQL backend, routing logic
  - Integration tests: Context → Infrastructure MCP calls
  - E2E tests: Query → route → execute → learn
  - MCP contract tests: Request/response formats
  - Performance benchmarks: Latency, context budget usage

Constraints:
- Do NOT break existing Infrastructure → Context communication
- Avoid tight coupling - maintain clean MCP boundaries
- Limit Infrastructure responsibility to data access (not routing!)
- Limit Context responsibility to routing (not data management!)
- Do NOT assume specific SQL schema - provide examples and ask
- Avoid SQL injection - enforce parameterized queries
- Maintain departmental context budgets (Infrastructure 20k, Context 60k)
- Preserve backward compatibility with existing Neo4j/Qdrant tools
- Follow existing patterns (query_neo4j, query_qdrant) for query_sql
- Respect department permissions (Infrastructure: Read/Write, Context: Read/Write)
- Keep Orchestration informed of major routing decisions
- Enable Verification to validate routing choices

If unclear or insufficient data:
- Ask: What is your policy data schema? Ground truth schema?
- Ask: Which SQL engine do you prefer? (PostgreSQL/SQLite/DuckDB)
- Ask: Should SQL, graph, and vector share data or be fully separate?
- Ask: What are acceptable latencies per backend?
- Ask: How should conflicts be resolved (SQL says X, graph says Y)?
- Ask: When should multiple backends be queried (always, never, confidence-based)?
- Do NOT: Invent schemas or policy rules
- Do NOT: Assume SQL engine without justification
- Do NOT: Make up departmental responsibilities
- Instead: Provide architectural options with tradeoffs, request preferences

Check your output for:
✓ Clear Infrastructure vs Context separation of concerns
✓ SQL backend follows existing query_neo4j MCP pattern
✓ Routing logic in Context (not Infrastructure)
✓ Clean MCP boundaries maintained
✓ Departmental context budgets respected
✓ Thompson Sampling leveraged for learning
✓ ReflectionBuffer used for outcome tracking
✓ Backward compatible with existing tools
✓ No fabricated schemas - examples + questions
✓ Specific file paths in HoloLoom codebase
✓ Example queries showing full routing path
✓ Departmental impact explicitly analyzed
✓ MCP request/response formats defined
✓ Error escalation path clear (dept → orch)
✓ Performance implications per department
```

---

## Key Improvements in v2

### 1. Departmental Architecture Context
**v1:** Generic HoloLoom context
**v2:** Full 6-department architecture with dependency graph

### 2. Department Boundaries
**v1:** "Add SQL to HoloLoom"
**v2:** "Add SQL to Infrastructure, routing to Context" (clean separation)

### 3. MCP Communication
**v1:** Not mentioned
**v2:** Explicit MCP request/response patterns, session ID propagation

### 4. Context Budgets
**v1:** Not mentioned
**v2:** Infrastructure 20k tokens, Context 60k tokens (must fit within)

### 5. Permission Model
**v1:** Generic
**v2:** Specific Infrastructure permissions (Read/Write, no Execute/Deploy)

### 6. Error Escalation
**v1:** Generic error handling
**v2:** Departmental escalation path (Infrastructure → Context → Orchestration)

### 7. Departmental Impact
**v1:** Not analyzed
**v2:** Explicit impact on all 6 departments

---

## Expected Output Quality

With v2 meta-prompt, you should get:

1. ✅ **Infrastructure SQL backend** that follows `query_neo4j` MCP pattern
2. ✅ **Context routing logic** that maintains departmental boundaries
3. ✅ **Clean MCP interfaces** between departments
4. ✅ **Context budget compliance** (20k Infrastructure, 60k Context)
5. ✅ **Departmental impact analysis** (which departments affected?)
6. ✅ **Error escalation path** (Infrastructure → Context → Orchestration)
7. ✅ **Thompson Sampling integration** (reuse existing bandit)
8. ✅ **ReflectionBuffer integration** (track routing outcomes)
9. ✅ **Backward compatibility** (existing tools still work)
10. ✅ **MCP contract definitions** (request/response formats)

**Quality Improvement over v1:** +25% (departmental boundaries, MCP patterns, context budgets)

---

## How to Use v2

1. Copy the entire **CONTEXT-ENRICHED META-PROMPT (v2)** section
2. Paste into Claude/ChatGPT
3. Add any additional specifics (SQL engine preference, policy schema, etc.)
4. Get back a **departmentally-aware** architecture design
5. Implement following clean MCP boundaries

---

## Why v2 is Better

| Aspect | v1 (Basic) | v2 (Departmental) |
|--------|-----------|-------------------|
| **Architecture** | Generic HoloLoom | 6-department agent swarm |
| **Boundaries** | Vague | Clean MCP interfaces |
| **Responsibilities** | Unclear | Infrastructure=data, Context=routing |
| **Communication** | Not specified | MCP request/response patterns |
| **Context Budgets** | Not mentioned | 20k (Infra), 60k (Context) |
| **Permissions** | Generic | Specific per department |
| **Error Handling** | Generic | Dept → Orch escalation path |
| **Impact Analysis** | Not done | All 6 departments analyzed |

**Result:** Architecture that fits naturally into existing department structure, not bolted on!

---

**Ready to run v2?** This will give you a design that:
- Respects departmental boundaries (Conway's Law)
- Follows existing MCP patterns
- Fits context budgets
- Maintains clean interfaces
- Enables Orchestration oversight

🚀
