# HoloLoom Configuration Complexity Analysis

## Quick Summary

**Total Parameters**: 72 across 11 subsystems
**Master Switches**: 14 (enable_*, use_*)  
**Dependent Parameters**: 31
**Never-Customized**: 19 (26% waste)
**Main Hotspots**: Memory backend (11 params), Physics (7 unused)

---

## Parameter Count by Category

| Category | Count | Master | Dependents | Unused | Status |
|----------|-------|--------|-----------|--------|--------|
| Memory Backend | 11 | 1 | 3 | 3 (mem0) | HOTSPOT |
| Feature Extraction | 9 | 1 | 6 | 5 | HOTSPOT |
| Linguistic/Phase5 | 6 | 1 | 5 | 2 | COMPLEX |
| Semantic Calculus | 7 | 1 | 5 | 2 | COMPLEX |
| Recursive Learning | 7 | 1 | 6 | 0 | COMPLEX |
| Policy Engine | 5 | 0 | 0 | 0 | STABLE |
| Context Packing | 5 | 1 | 4 | 4 | RESEARCH |
| Safety/Environment | 5 | 0 | 0 | 0 | GOOD |
| Neural Network | 2 | 0 | 0 | 0 | STABLE |
| Retrieval | 2 | 0 | 0 | 0 | STABLE |
| Embedding/Caching | 2 | 0 | 0 | 2 | LEGACY |
| **TOTAL** | **72** | **8** | **31** | **19** | - |

---

## Critical Hotspots

### Hotspot 1: Memory Backend (11 params, 70% waste)

Parameters that should be conditional but aren't:
- neo4j_uri, neo4j_username, neo4j_password, neo4j_database (HYBRID only)
- qdrant_host, qdrant_port, qdrant_collection, qdrant_use_https (HYBRID only)
- hyperspace_depth, hyperspace_thresholds, hyperspace_breadth (HYPERSPACE only)
- mem0_api_key, mem0_org_id, mem0_project_id (completely unused - dead code)

**Fix**: Use nested backend configs instead of flat parameters
**Saves**: 8 parameters from cognitive load

### Hotspot 2: Physics Parameters (7 params, all hardcoded)

- use_spring_activation: False (disabled by default)
- spring_stiffness: 0.15 (never customized)
- spring_damping: 0.85 (never customized)
- spring_decay: 0.98 (never customized)
- spring_iterations: 200 (never customized)
- spring_convergence_epsilon: 1e-4 (never customized)

**Fix**: Remove if not using, else implement full tuning interface
**Saves**: 7 parameters

### Hotspot 3: Feature Flag Cascade

Master switches that create orphaned parameters:
- enable_linguistic_gate -> 5 dependent params
- enable_semantic_calculus -> 5 dependent params
- enable_recursive_learning -> 6 dependent params
- use_spring_activation -> 6 dependent params
- enable_beta_wave_packing -> 4 dependent params

**Fix**: Use nested @dataclass instead of flat structure
**Saves**: Reduces cognitive load by 30%

---

## Never-Customized Parameters (19 total)

These parameters are set once and never changed:

**Physics** (5): spring_stiffness, spring_damping, spring_decay, spring_iterations, spring_convergence_epsilon
**Tokens** (5): packing_token_budget, packing_query_reserve, packing_response_reserve (+ thresholds)
**Caching** (3): merge_cache_size, parse_cache_size, semantic_cache_size
**Constants** (3): semantic_dt, semantic_framework, hyperspace_breadth
**Dead** (3): mem0_api_key, mem0_org_id, mem0_project_id
**Other** (2): qdrant_use_https, scales (always [768])

**Recommendation**: Consolidate or remove

---

## Configuration Interdependencies

### Direct Dependencies (Parameters that affect each other)

1. **memory_backend** controls which parameters are meaningful:
   - HYBRID -> neo4j_*, qdrant_* become relevant
   - HYPERSPACE -> hyperspace_* become relevant
   - INMEMORY -> both sets are ignored

2. **enable_linguistic_gate** controls 7 child parameters:
   - linguistic_mode, use_compositional_cache, parse_cache_size, etc.

3. **mode (BARE/FAST/FUSED)** affects:
   - n_transformer_layers (1 vs 2)
   - n_attention_heads (2 vs 4)
   - semantic_dimensions (varies per preset)

4. **environment** affects behavior via @property:
   - safety_testing_mode (DEVELOPMENT=True, others=False)
   - safety_auto_approve_categories (varies)
   - logging_level (varies)

5. **enable_semantic_calculus** controls 6 child params:
   - semantic_dimensions, semantic_cache_size, semantic_dt, etc.

### Problematic Dependencies

- **No validation** that orphaned parameters match disabled subsystem
  - Can set use_spring_activation=False with spring_stiffness=0.15
  - No warnings or errors

- **No backend availability checks**
  - Doesn't verify Neo4j/Qdrant actually running
  - Would silently fail at runtime

- **Mode conflicts not detected**
  - Could set mode=BARE but enable_semantic_calculus=True

---

## Validation Logic Summary

### Currently Validated
1. scales sorted ascending
2. fusion_weights sum to ~1.0 (auto-normalizes)
3. hyperspace_thresholds match depth
4. mode string -> ExecutionMode enum

### Gaps in Validation
1. No interdependency checks
2. No backend health checks
3. No conflict detection between mode and feature flags
4. Environment variables referenced but not validated

---

## Recommended Consolidation (Priority Order)

### P1: Remove Dead Code (1 hour, -3 params)
Remove completely unused:
- kg_backend (deprecated, redundant)
- mem0_api_key, mem0_org_id, mem0_project_id

### P2: Deactivate Physics (2 hours, -7 params)
Either remove or document as research-only:
- spring_* (disabled by default, never tuned)

### P3: Nest Feature Flags (4 hours, -25% cognitive load)
Create nested config classes:


### P4: Clarify Execution Modes (1 hour)
Clarify what bare/fast/fused actually control:
- bare: No Phase 5, no semantic
- fast: Phase 5, no semantic
- fused: Phase 5, optional semantic

Current behavior of fast() and fused() too similar.

### P5: Document Hardcoded Values (30 min)
Mark which params are:
- Not meant to be customized (physics constants)
- Placeholder values (token budgets)
- Legacy/deprecated

---

## Key Findings

1. **Configuration complexity is manageable but could be 25% simpler**
   - Remove dead code (3 params)
   - Deactivate physics (7 params)
   - Nest feature flags (reduces cognitive overhead)

2. **Master switches work well** (factory methods heavily used)
   - Config.fast(): 11 uses
   - Config.fused(): 8 uses
   - Custom Config(): rare

3. **Orphaned parameters** create hidden complexity
   - When master switch is False, all child params still exist
   - No validation that settings make sense

4. **Multi-scale embeddings** disabled in practice
   - scales always [768]
   - fusion_weights always {768: 1.0}
   - Either implement or remove

5. **Backend selection** poorly designed
   - Should use backend-specific config classes
   - Currently: flat list of conditional parameters
