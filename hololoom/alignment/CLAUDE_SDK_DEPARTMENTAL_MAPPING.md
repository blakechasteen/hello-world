# Claude SDK → Departmental Architecture Mapping

**Status**: Design Document
**Date**: November 9, 2025
**Author**: Architectural analysis from user insights

This document maps Claude SDK capabilities to HoloLoom's departmentalized agent architecture, implementing **Conway's Law for AI agents**: your agent architecture mirrors your organizational structure.

## Core Thesis

The Claude SDK provides 5 key capabilities that naturally map to organizational structure:

1. **Context Management** → Distributed context budgets + compaction at boundaries
2. **Tool Permissions** → Role-based access control per department
3. **Error Handling** → Cross-department escalation protocols
4. **Session Management** → Institutional memory and project continuity
5. **MCP Extensibility** → Federated inter-department communication

## 1. Context Management → Distributed Context Budgets

### SDK Capability

- Automatic context compaction when approaching limits
- Subagents use isolated context windows
- Return summaries, not full transcripts
- Just-in-time context injection

### Departmental Implementation

Each department declares its context budget upfront:

| Department | Context Budget | Rationale |
|------------|---------------|-----------|
| Orchestration | 100k tokens | Global coordination, maintains master session |
| Context (HoloLoom) | 60k tokens | Multi-pass enrichment across graph |
| MasterWeaver | 50k tokens | Large dataset processing (beekeeping transcripts) |
| Execution | 40k tokens | Complex task specifications |
| Verification | 30k tokens | Focused validation logic |
| Infrastructure | 20k tokens | System diagnostics |

**Compaction Strategy**:
- **MasterWeaver**: Extracts entities from 10MB transcript → Returns 2KB entity summary
- **Verification**: Receives entity summary (not full transcript) → Validates structure
- **Orchestration**: Maintains compact state: "MasterWeaver: 47 entities extracted, 0.89 confidence"

**Example Flow**:
```python
# MasterWeaver processes 50k tokens of beekeeping audio transcript
result = masterweaver.extract_entities(transcript_50k_tokens)

# Returns COMPACT summary (2k tokens)
{
    "entities": [...],  # Structured, not verbose
    "confidence": 0.89,
    "reasoning": "Used queen behavior patterns from domain ontology. Cross-validated with hive inspection reports."
}

# Verification receives only 2k tokens, not 50k
verification.validate_confidence_claim(claim=result, evidence=evidence)
```

**Verification Answer**: Does Verification need full transcripts?
→ **No. Entities + reasoning only**. This enforces separation: MasterWeaver owns extraction, Verification owns quality.

## 2. Tool Permissions → Role-Based Access Control

### SDK Capability

- Fine-grained per-tool allow/deny
- Policy modes (hard block vs soft deny + log)
- Capability controls in production
- Secrets never in agent-visible context

### Departmental Implementation

Each department has a **permission charter**:

**MasterWeaver**:
- ✅ Read Neo4j expertise nodes
- ✅ Write new entity extractions
- ✅ Query beekeeping domain memory
- ❌ CANNOT modify Verification outcomes
- ❌ CANNOT modify Infrastructure configs

**Infrastructure**:
- ✅ Read/Write Neo4j and Qdrant
- ✅ Execute performance diagnostics
- ✅ Update zero-copy query definitions
- ❌ CANNOT execute Claude Code tasks (Execution owns that)

**Execution**:
- ✅ Read from all memory systems
- ✅ Write execution logs
- ✅ Call Claude Code tools
- ❌ CANNOT modify security policies
- ❌ CANNOT modify data architecture

**Verification**:
- ✅ Read all department decisions
- ✅ Call validation tools
- ✅ Log confidence assessments
- ❌ CANNOT directly execute code
- ❌ CANNOT modify memory without approval

**Context (HoloLoom)**:
- ✅ Read from all sources
- ✅ Perform multi-pass enrichment
- ✅ Write context annotations
- ❌ CANNOT initiate actions

**Orchestration**:
- ✅ Read global state
- ✅ Route tasks
- ✅ Update roadmap
- ✅ Escalate decisions
- ❌ CANNOT execute domain logic (departments own that)

**Permission Modes**:
- **HARD** (tool not visible): Delete operations, deploy to production, modify schemas
- **SOFT** (visible but logged): Cross-department queries, re-run requests

**Example**:
```python
# Verification requests MasterWeaver re-run with stricter params
# This is SOFT permission - logged, can be denied by Orchestration

result = verification.request_rerun(
    department="MasterWeaver",
    task_id="extract_queen_behavior_q4",
    new_params={"confidence_threshold": 0.9},
    reason="Detected 15% overconfidence in prior run"
)

# System logs: "Verification → MasterWeaver: rerun request"
# Orchestration monitors and can intervene if re-runs cascade
```

**Verification Answer**: Soft vs hard permissions?
→ **Hybrid**: Hard for dangerous ops, soft for cross-department queries (creates audit trail).

## 3. Error Handling → Cross-Department Escalation

### SDK Capability

- Built-in error recovery
- Retry mechanisms
- OTEL traces and logging
- Anomaly detection
- Rollback protocols

### Departmental Implementation

**Local Recovery** (within department):
```python
# MasterWeaver fails to extract entity
try:
    entities = extract_with_strategy_A(transcript)
except LowConfidenceError:
    # Retry with different strategy
    entities = extract_with_strategy_B(transcript)
    # Log confidence drop
    log_warning("Strategy A failed, used Strategy B. Confidence: 0.72")
```

**Cross-Department Escalation**:
```python
# Infrastructure detects malformed query from Execution
try:
    result = neo4j.execute(query)
except CypherSyntaxError as e:
    # Don't execute - alert Execution Department
    infrastructure.alert(
        target_department="Execution",
        error=f"Malformed Cypher query: {e}",
        severity="HIGH"
    )
    # Create department-level incident
    orchestration.log_incident(
        department="Execution",
        issue="Generated invalid Cypher syntax",
        affected_tasks=[task_id]
    )
```

**Verification Catches Errors**:
```python
# Verification detects confidence mismatch
validation = verification.validate_confidence_claim(
    claim=masterweaver_output,
    evidence=ground_truth_sample
)

if validation.actual_confidence < validation.claimed_confidence - 0.2:
    # Escalate: confidence claims don't match reality
    verification.escalate(
        issue="MasterWeaver overconfident by 20%",
        recommendation="Manual review or rerun with domain expert constraints"
    )
```

**Orchestration Coordinates Recovery**:
```python
# Errors cascade: bad data from Infrastructure → Execution fails
if orchestration.detect_cascade():
    # Initiate recovery protocol
    orchestration.recovery_protocol(
        root_cause="Infrastructure data corruption",
        affected_departments=["Execution", "Verification"],
        action="rollback" if critical else "remediation"
    )
```

**Verification Answer**: Should Verification have "request re-run" tool or route to Orchestration?
→ **Direct re-run tool** with logging. Reduces latency. Orchestration monitors the channel and intervenes if needed.

## 4. Session Management → Institutional Memory

### SDK Capability

- Session persistence
- /compact for context reduction
- /resume for session restoration
- Fork sessions for parallel exploration
- Session state saved to disk

### Departmental Implementation

**Active Sessions** (per department):
```python
# Each department maintains current work session
masterweaver_session = {
    "session_id": "q4_beekeeping_2025",
    "phase": "entity_extraction",
    "context": "Processing Q4 beekeeping inspections",
    "state": {
        "processed_transcripts": 47,
        "entities_extracted": 1203,
        "current_file": "inspection_20251015.txt"
    }
}
```

**Session Forking** (explore uncertainty):
```python
# Verification uncertain about confidence threshold
# Fork session to explore both paths in parallel

fork_a = verification.fork_session(
    parent_session="q4_beekeeping_2025",
    fork_id="assume_confidence_valid",
    assumption="MasterWeaver confidence is accurate"
)

fork_b = verification.fork_session(
    parent_session="q4_beekeeping_2025",
    fork_id="assume_20pct_overconfident",
    assumption="MasterWeaver overconfident by 20%"
)

# Run both, compare results, inform decision
```

**Session Compaction for Handoff**:
```python
# Infrastructure completes "zero-copy optimization" task
# Compact session for Execution Department to inherit

compact_summary = infrastructure.compact_session(
    session_id="zero_copy_optimization_q4",
    target_department="Execution"
)

# Result: compact artifact, not full transcript
{
    "summary": "Queries now execute in <100ms",
    "query_definitions": [attachment_1, attachment_2],
    "gotchas": ["X requires proper indexing", "Y needs cache warmup"],
    "next_steps": ["Execution can now deploy with confidence"]
}
```

**Global Session Orchestration**:
```python
# Orchestration maintains master session per roadmap milestone
orchestration.master_session = {
    "milestone": "Q4 Roadmap",
    "completed": {
        "MasterWeaver": "entity_extraction",
        "Infrastructure": "zero_copy_setup"
    },
    "in_progress": {
        "Execution": "agent_autonomy_v1"
    },
    "blocked": {
        "Verification": "awaiting_confidence_framework"
    }
}
```

**CLAUDE.md as Shared Charter**:
```markdown
# Shared CLAUDE.md (Root)

## Vision
Aligned agent swarms for domain-specific problem solving

## Roadmap Phases
Phase 1: Entity extraction (MasterWeaver)
Phase 2: Zero-copy queries (Infrastructure)
Phase 3: Agent autonomy (Execution)
...

## Department Registry
- MasterWeaver: Entity extraction, knowledge structuring
- Verification: Quality assurance, confidence validation
- Infrastructure: Data systems, performance optimization
...

## Inter-Department Protocols
- All requests use session_id for context continuity
- Compaction required when crossing department boundaries
- Escalation protocol: Department → Orchestration → Human (if needed)
```

**Department-Specific Charters**:
```markdown
# departments/MasterWeaver/CHARTER.md

## Role
Entity extraction, domain understanding, knowledge structuring

## Capabilities
- Extract entities from multimodal input
- Validate entity consistency
- Query domain ontologies

## Success Criteria
- Entity extraction confidence > 0.75
- Consistency validation pass rate > 90%
- Query response time < 200ms

## Permissions
- Read: Neo4j expertise nodes, domain ontologies
- Write: Entity extractions, annotations
- CANNOT: Modify Verification outcomes, Infrastructure configs
```

**Verification Answer**: One shared CLAUDE.md or per-department?
→ **Hierarchical**: Shared CLAUDE.md for vision/protocols, per-department CHARTER.md for specifics.

## 5. MCP Extensibility → Federated Communication

### SDK Capability

- Model Context Protocol (MCP)
- Standardized integrations
- OAuth/auth handled automatically
- Growing ecosystem of MCP servers

### Departmental Implementation

**Each department = MCP server** with standardized tool signatures.

**MasterWeaver MCP Server**:
```json
{
  "name": "MasterWeaver",
  "version": "1.0.0",
  "tools": [
    {
      "name": "extract_entities",
      "description": "Extract structured entities from raw input",
      "input_schema": {
        "type": "object",
        "properties": {
          "input_data": {"type": "string"},
          "domain": {"type": "string"},
          "session_id": {"type": "string"}
        }
      }
    },
    ...
  ]
}
```

**Infrastructure MCP Server**:
```json
{
  "name": "Infrastructure",
  "tools": [
    {
      "name": "query_neo4j",
      "input_schema": {
        "cypher_query": {"type": "string"},
        "session_id": {"type": "string"}
      }
    },
    ...
  ]
}
```

**Federated Architecture**:
```
                    ┌─────────────────┐
                    │  Orchestration  │
                    │   MCP Server    │
                    └────────┬────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
    ┌────▼────┐         ┌────▼────┐         ┌───▼────┐
    │ Master  │         │  Infra  │         │  Exec  │
    │ Weaver  │◄────────┤ struct  │────────►│ ution  │
    │   MCP   │         │   MCP   │         │   MCP  │
    └────┬────┘         └────┬────┘         └───┬────┘
         │                   │                   │
         └───────────────────┼───────────────────┘
                             │
                        ┌────▼────────┐
                        │Verification │
                        │     MCP     │
                        └─────────────┘
```

**Benefits of Federated MCP**:
1. **Independent Evolution**: MasterWeaver algorithm improves without touching Verification
2. **Loose Coupling**: Departments communicate via standard protocols
3. **Graceful Degradation**: One department fails, others continue
4. **External Teams**: Domain experts can run departments independently, just export MCP endpoints

**Security** (per SDK best practices):
- OAuth 2.1-style auth
- TLS for all inter-department communication
- Scoped permissions per MCP tool
- Signed manifests
- Comprehensive logging

**Verification Answer**: Should MCP servers be stateless or maintain session state?
→ **Stateless with session IDs**. Departments accept `session_id` parameter, load state from Orchestration or persistent storage.

---

## Implementation Architecture

### File Structure

```
HoloLoom/
├── alignment/
│   ├── mcp_department_registry.py         # ✅ Created - Department definitions
│   ├── CLAUDE_SDK_DEPARTMENTAL_MAPPING.md # ✅ This file
│   └── department_orchestrator.py         # TODO - Orchestration implementation
│
├── departments/                            # TODO
│   ├── masterweaver/
│   │   ├── CHARTER.md
│   │   ├── mcp_server.py
│   │   └── entity_extraction.py
│   ├── verification/
│   │   ├── CHARTER.md
│   │   ├── mcp_server.py
│   │   └── confidence_validation.py
│   ├── infrastructure/
│   │   ├── CHARTER.md
│   │   ├── mcp_server.py
│   │   └── query_optimization.py
│   └── ...
│
└── CLAUDE.md                              # Shared charter (exists)
```

### Department Registry

See `mcp_department_registry.py` for complete definitions:

| Department | Role | Tools | Context Budget |
|------------|------|-------|----------------|
| MasterWeaver | Entity extraction | 3 | 50k tokens |
| Verification | Quality assurance | 3 | 30k tokens |
| Infrastructure | Data systems | 3 | 20k tokens |
| Execution | Task execution | 2 | 40k tokens |
| Context | Multi-pass enrichment | 2 | 60k tokens |
| Orchestration | Coordination | 3 | 100k tokens |

**Total Context Budget**: 300k tokens distributed across 6 departments

### Permission Matrix

| Department | Read | Write | Execute | Deploy | Admin |
|------------|------|-------|---------|--------|-------|
| MasterWeaver | ✅ | ✅ | ❌ | ❌ | ❌ |
| Verification | ✅ | ❌ | ✅ | ❌ | ❌ |
| Infrastructure | ✅ | ✅ | ❌ | ❌ | ✅ |
| Execution | ✅ | ❌ | ✅ | ❌ | ❌ |
| Context | ✅ | ✅ | ❌ | ❌ | ❌ |
| Orchestration | ✅ | ❌ | ✅ | ❌ | ✅ |

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

---

## Next Steps

### Phase 1: Foundation (Week 1)
- [x] Define department registry
- [ ] Create department CHARTER.md templates
- [ ] Implement basic MCP server framework
- [ ] Test inter-department communication

### Phase 2: Core Departments (Week 2-3)
- [ ] Implement MasterWeaver MCP server
- [ ] Implement Infrastructure MCP server
- [ ] Implement Verification MCP server
- [ ] Test with sample beekeeping dataset

### Phase 3: Integration (Week 4)
- [ ] Implement Orchestration coordinator
- [ ] Add session management
- [ ] Add error escalation protocols
- [ ] End-to-end testing

### Phase 4: Production Hardening (Week 5-6)
- [ ] Add OAuth/auth to MCP servers
- [ ] Implement monitoring/observability
- [ ] Load testing and optimization
- [ ] Deploy to staging

---

## Verification Questions Answered

1. **Context Management**: Verification needs **entities + reasoning only**, not full transcripts
2. **Permissions**: **Hybrid** - hard for dangerous ops, soft for cross-department queries
3. **Error Handling**: **Direct re-run tool** with Orchestration monitoring
4. **Session Management**: **Hierarchical** - shared CLAUDE.md + per-department CHARTER.md
5. **MCP Extensibility**: **Stateless with session IDs** - state lives in Orchestration/storage

---

## The Natural Conclusion

This isn't just using Claude SDK features—it's applying **Conway's Law to AI agents**: your agent architecture mirrors your organizational structure.

The SDK provides the technical capabilities. This architecture provides the organizational structure. Together, they create a federated agent swarm where:

- **Context is managed like budgets** (each department declares needs upfront)
- **Permissions are organizational roles** (departments can't exceed their scope)
- **Errors escalate through channels** (department → orchestration → human)
- **Sessions are institutional memory** (decisions persist, context doesn't rot)
- **MCP is the connective tissue** (loose coupling, independent evolution)

**Result**: An agent swarm that operates like a well-structured organization, not a monolithic AI system.

---

*Generated during Claude Code session on November 9, 2025*
*Based on architectural insights from user analysis of Claude SDK capabilities*
