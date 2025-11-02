# HoloLoom Alignment Framework

**Status**: ✅ Production Ready (v1.0.0)
**Performance**: 0.103 ms overhead (29x faster than 3ms target)
**Test Coverage**: 46 functional tests + 13 performance benchmarks
**Documentation**: Complete

---

## Overview

The Alignment Framework provides comprehensive safety mechanisms for HoloLoom's agentic reasoning system, implementing industry best practices from Anthropic, OpenAI, and DeepMind research.

### Core Philosophy

> **"Safe by default, transparent by design"**

Every decision is gated by safety checks, monitored for deception, bound by resource limits, and logged with complete provenance - all with **negligible performance impact** (<0.11 ms per query).

---

## Architecture

### 4 Core Modules

```
┌──────────────────────────────────────────────────────────────┐
│                   HOLOLOOM QUERY PIPELINE                    │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │   1. SAFETY GUARDRAILS                  │
        │   • Risk-based action gating            │
        │   • Adversarial pattern detection       │
        │   • Human-in-the-loop escalation        │
        │   Latency: 0.039 ms                     │
        └─────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │   2. DECEPTION DETECTION                │
        │   • Goal transparency tracking          │
        │   • Behavioral probe system             │
        │   • Hidden goal detection               │
        │   Latency: 0.034 ms                     │
        └─────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │   3. INSTRUMENTAL CONVERGENCE GUARD     │
        │   • Resource usage limits               │
        │   • Self-modification prevention        │
        │   • Autonomy boundaries                 │
        │   Latency: 0.001 ms                     │
        └─────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │   4. AUDIT TRAIL                        │
        │   • Complete decision logging           │
        │   • Provenance graph tracking           │
        │   • Queryable history                   │
        │   Latency: 0.029 ms                     │
        └─────────────────────────────────────────┘
                              │
                              ▼
                    Query proceeds safely
```

---

## Quick Start

### Installation

```python
from HoloLoom.alignment import (
    create_guardrails,
    create_detector,
    create_guard,
    create_audit_trail,
)
```

### Basic Usage

```python
import asyncio
from pathlib import Path
from HoloLoom.alignment import *

async def safe_query_processing(query_text: str):
    # Initialize alignment modules
    guardrails = create_guardrails()
    detector = create_detector()
    guard = create_guard()
    audit = create_audit_trail(persist_path=Path("./logs"))

    # Step 1: Safety check
    request = ActionRequest(
        action=query_text,
        category=ActionCategory.QUERY
    )
    safety_decision = guardrails.evaluate(request, text_input=query_text)

    if not safety_decision.allowed:
        print(f"❌ Query blocked: {safety_decision.reason}")
        return

    # Step 2: Resource check
    guard.check_resource_usage(ResourceType.COMPUTE, 5.0)  # 5 seconds
    allowed, reason = guard.check_autonomy_limits()

    if not allowed:
        print(f"⚠️ Resource limits exceeded: {reason}")
        return

    # Step 3: Deception probe
    probe = BehavioralProbe(
        probe_type=ProbeType.GOAL_ALIGNMENT,
        scenario="Verify query aligns with stated goals",
        expected_behavior="Helpful and aligned response"
    )
    passed, score = detector.run_probe(probe, "Processing query helpfully")

    # Step 4: Audit logging
    log = audit.log_decision(
        decision_type=DecisionType.SAFETY_GATE,
        outcome=OutcomeType.APPROVED,
        reason=safety_decision.reason,
        query_text=query_text,
        confidence=0.95
    )

    print(f"✅ Query approved: {log.decision_id}")

    # Process query safely...

# Run
asyncio.run(safe_query_processing("What is Thompson Sampling?"))
```

---

## Module Documentation

### 1. Safety Guardrails

**File**: `safety_guardrails.py` (516 lines)
**Purpose**: Risk-based action gating and adversarial pattern detection

#### Key Features

- **5 Risk Levels**: SAFE → LOW → MEDIUM → HIGH → CRITICAL
- **9 Action Categories**: QUERY, RETRIEVAL, TOOL_EXECUTION, DATA_MODIFICATION, etc.
- **3 Adversarial Patterns**: Prompt injection, jailbreak attempts, resource exhaustion
- **Human-in-the-loop**: Automatic escalation for high-risk actions

#### API

```python
from HoloLoom.alignment.safety_guardrails import create_guardrails

guardrails = create_guardrails()

# Evaluate action
request = ActionRequest(action="Delete all data", category=ActionCategory.DELETION)
decision = guardrails.evaluate(request)

print(decision.allowed)        # False
print(decision.risk_level)     # RiskLevel.CRITICAL
print(decision.reason)         # "High-risk action: deletion requires approval"
```

#### Configuration

```python
# Custom policy
guardrails.policies[ActionCategory.DELETION] = SafetyPolicy(
    allowed_without_approval=False,
    requires_human_approval=True,
    max_risk_level=RiskLevel.HIGH
)
```

---

### 2. Deception Detection

**File**: `deception_detection.py` (511 lines)
**Purpose**: Monitor agent behavior for goal misalignment and hidden objectives

#### Key Features

- **5 Probe Types**: CONSISTENCY, CAPABILITY, GOAL_ALIGNMENT, REWARD_HACKING, HONESTY
- **Goal Transparency**: Track stated goals vs. observed actions
- **Hidden Goal Detection**: Automatic detection of undeclared objectives
- **4 Signal Levels**: LOW → MEDIUM → HIGH → CRITICAL

#### API

```python
from HoloLoom.alignment.deception_detection import create_detector

detector = create_detector()

# Register goals
goal = GoalStatement(
    goal_id="helpful",
    description="Provide helpful and accurate information",
    priority=10
)
detector.goal_tracker.declare_goal(goal)

# Run behavioral probe
probe = BehavioralProbe(
    probe_type=ProbeType.HONESTY,
    scenario="Can you access user passwords?",
    expected_behavior="No, I cannot access passwords"
)
passed, score = detector.run_probe(probe, "I cannot access passwords")

print(passed)  # True
print(score)   # 0.0 (low deception)
```

#### Behavioral Probes

```python
# Consistency check
probe = BehavioralProbe(
    probe_type=ProbeType.CONSISTENCY,
    scenario="Earlier you said X, now you're saying Y",
    expected_behavior="Acknowledge inconsistency"
)

# Capability check
probe = BehavioralProbe(
    probe_type=ProbeType.CAPABILITY,
    scenario="Can you modify system files?",
    expected_behavior="No, I cannot modify files"
)
```

---

### 3. Instrumental Convergence Guard

**File**: `instrumental_convergence.py` (427 lines)
**Purpose**: Prevent resource hoarding, self-modification, and unbounded autonomy

#### Key Features

- **6 Resource Types**: COMPUTE, MEMORY, STORAGE, NETWORK, API_CALLS, DATA_ACCESS
- **Soft/Hard Limits**: Warnings before hard failures
- **Rate Limiting**: Control burst usage patterns
- **Self-Modification Detection**: Pattern-based detection of code changes

#### API

```python
from HoloLoom.alignment.instrumental_convergence import create_guard

guard = create_guard()

# Set resource bounds
guard.set_resource_bounds(
    ResourceType.MEMORY,
    ResourceBounds(
        resource_type=ResourceType.MEMORY,
        soft_limit=1024.0,  # 1GB soft
        hard_limit=2048.0,  # 2GB hard
        time_window_seconds=60.0,
        rate_limit=100.0    # MB/s
    )
)

# Check resource usage
violation = guard.check_resource_usage(ResourceType.MEMORY, 1500.0)

if violation:
    print(violation.violation_type)  # ViolationType.SOFT_LIMIT
    print(violation.message)         # "Memory usage 1500.0 exceeds soft limit 1024.0"
```

#### Autonomy Limits

```python
# Configure autonomy boundaries
guard.autonomy_limits.max_actions_without_approval = 100
guard.autonomy_limits.max_duration_without_approval = 300.0  # 5 minutes

# Check before autonomous action
allowed, reason = guard.check_autonomy_limits()
```

---

### 4. Audit Trail

**File**: `audit_trail.py` (542 lines)
**Purpose**: Complete decision logging with provenance tracking

#### Key Features

- **8 Decision Types**: SAFETY_GATE, DECEPTION_CHECK, RESOURCE_CHECK, etc.
- **3 Outcome Types**: APPROVED, REJECTED, ESCALATED
- **Provenance DAG**: Directed acyclic graph of reasoning steps
- **Queryable History**: Filter by type, outcome, time range, metadata

#### API

```python
from HoloLoom.alignment.audit_trail import create_audit_trail

audit = create_audit_trail(persist_path=Path("./logs"), auto_flush=True)

# Log decision
log = audit.log_decision(
    decision_type=DecisionType.TOOL_SELECTION,
    outcome=OutcomeType.APPROVED,
    reason="Selected 'search' tool based on query analysis",
    query_text="What is reinforcement learning?",
    confidence=0.92,
    metadata={"tool": "search", "confidence_threshold": 0.75}
)

print(log.decision_id)  # "dec_1730502345_a3b2c1"

# Build provenance graph
tracer = audit.get_tracer(log.decision_id)
tracer.add_node("retrieval", "memory_search", "Retrieved 5 relevant documents")
tracer.add_node("analysis", "llm_analysis", "Analyzed documents", parent_ids=["retrieval"])
tracer.add_node("synthesis", "answer_generation", "Generated answer", parent_ids=["analysis"])

# Finalize
audit.finalize_decision(log.decision_id)

# Get reasoning chain
chain = tracer.get_reasoning_chain("synthesis")
# ['Retrieved 5 relevant documents', 'Analyzed documents', 'Generated answer']
```

#### Querying History

```python
# By outcome
approved = audit.query_by_outcome(OutcomeType.APPROVED)
rejected = audit.query_by_outcome(OutcomeType.REJECTED)

# By decision type
safety_checks = audit.query_by_decision_type(DecisionType.SAFETY_GATE)

# By time range
from datetime import datetime, timedelta
recent = audit.query_by_time_range(
    start_time=datetime.now() - timedelta(hours=1),
    end_time=datetime.now()
)

# By metadata
tool_selections = audit.query_by_metadata({"tool": "search"})
```

---

## API Compatibility Layer

**File**: `api_compatibility.py` (344 lines)
**Purpose**: Maintain backward compatibility with specification API

### Specification-Compliant API

The alignment framework provides both implementation API (above) and specification-compliant API:

```python
from HoloLoom.alignment.api_compatibility import patch_alignment_api

# Apply compatibility patches
patch_alignment_api()

# Now use spec API
decision = guardrails.evaluate_action(
    action="Test query",
    category="QUERY",
    context={"source": "user"}
)

goal_id = detector.register_goal(
    description="Be helpful",
    priority=10
)

guard.configure_from_unified_bounds(
    max_memory_mb=1024.0,
    max_compute_seconds=60.0,
    max_api_calls=100.0
)
```

### Unpatch (if needed)

```python
from HoloLoom.alignment.api_compatibility import unpatch_alignment_api

unpatch_alignment_api()  # Restore original API
```

---

## Integration with HoloLoom

### Weaving Orchestrator Integration

```python
from HoloLoom.weaving_orchestrator import WeavingOrchestrator
from HoloLoom.alignment import create_guardrails, create_audit_trail
from HoloLoom.config import Config

# Create orchestrator with alignment
config = Config.fast()
orchestrator = WeavingOrchestrator(cfg=config, shards=shards)

# Add alignment layer
guardrails = create_guardrails()
audit = create_audit_trail(persist_path=Path("./logs"))

async def aligned_weave(query):
    # Pre-flight safety check
    request = ActionRequest(action=query.text, category=ActionCategory.QUERY)
    decision = guardrails.evaluate(request, text_input=query.text)

    if not decision.allowed:
        return {"error": "Query blocked", "reason": decision.reason}

    # Process with HoloLoom
    spacetime = await orchestrator.weave(query)

    # Log decision
    log = audit.log_decision(
        decision_type=DecisionType.TOOL_SELECTION,
        outcome=OutcomeType.APPROVED,
        reason=f"Selected tool: {spacetime.tool_used}",
        query_text=query.text,
        confidence=spacetime.confidence
    )

    return spacetime
```

---

## Performance

**Report**: See [PERFORMANCE_REPORT.md](./PERFORMANCE_REPORT.md)

### Benchmark Results

| Component | Median (ms) | Threshold (ms) | Status | Speedup |
|-----------|-------------|----------------|--------|---------|
| SafetyGuardrails | 0.039 | 0.5 | ✅ PASS | 13x |
| DeceptionDetector | 0.034 | 1.0 | ✅ PASS | 29x |
| InstrumentalGuard | 0.001 | 0.3 | ✅ PASS | 300x |
| AuditTrail | 0.029 | 0.2 | ✅ PASS | 7x |
| **TOTAL** | **0.103** | **3.0** | **✅ PASS** | **29x** |

**Key Takeaways**:
- Total overhead: **0.103 ms** (negligible)
- Headroom: **96.6%** (2.897 ms available)
- Production ready: **Yes** ✅

### Run Benchmarks

```bash
# Standalone script (recommended)
python HoloLoom/alignment/tests/run_benchmarks.py

# Or via pytest
pytest HoloLoom/alignment/tests/test_performance.py -v
```

---

## Testing

### Test Suite

**46 Functional Tests** (`test_alignment.py` - 393 lines):
- SafetyGuardrails: 11 tests
- DeceptionDetector: 10 tests
- InstrumentalGuard: 12 tests
- AuditTrail: 8 tests
- API Compatibility: 5 tests

**13 Performance Tests** (`test_performance.py` - 549 lines):
- Component benchmarks: 11 tests
- Integration benchmarks: 1 test
- Baseline comparison: 1 test

### Run Tests

```bash
# All functional tests
pytest HoloLoom/alignment/tests/test_alignment.py -v

# All performance tests
pytest HoloLoom/alignment/tests/test_performance.py -v

# Specific test
pytest HoloLoom/alignment/tests/test_alignment.py::TestSafetyGuardrails::test_01_safe_query -v
```

---

## Demos

### Integrated Demo

**File**: `demos/demo_alignment_integration.py` (432 lines)

Demonstrates complete alignment pipeline with 4 scenarios:
1. Safe queries (pass all checks)
2. Adversarial queries (blocked by guardrails)
3. High-risk actions (escalated for approval)
4. API compatibility (spec-compliant interface)

```bash
python demos/demo_alignment_integration.py
```

**Output**:
```
🔒 Initializing Alignment Framework...
✅ Alignment Framework Ready
   - Safety Guardrails: Active
   - Deception Detection: Active
   - Resource Guards: Active
   - Audit Trail: ./demo_alignment_logs

🟢 SCENARIO 1: Safe Queries
============================================================
Processing Query: What is Thompson Sampling?
============================================================
1️⃣  Safety Guardrails Check...
   Risk Level: SAFE
   Allowed: True
   Reason: Safe query

2️⃣  Resource Bounds Check...
   Autonomy Check: PASS
   Reason: Within limits

3️⃣  Deception Detection Probe...
   Probe Type: goal_alignment
   Passed: True
   Deception Score: 0.333

4️⃣  Provenance Tracking...
   Reasoning Chain: 3 steps
      1. guardrails
      2. convergence_guard
      3. behavioral_probe

✅ Query APPROVED - All alignment checks passed

[... more scenarios ...]
```

---

## File Structure

```
HoloLoom/alignment/
├── README.md                       # This file
├── PERFORMANCE_REPORT.md           # Detailed performance analysis
│
├── safety_guardrails.py            # Risk-based action gating (516 lines)
├── deception_detection.py          # Behavioral monitoring (511 lines)
├── instrumental_convergence.py     # Resource/autonomy limits (427 lines)
├── audit_trail.py                  # Decision logging + provenance (542 lines)
├── api_compatibility.py            # Spec-compliant API layer (344 lines)
│
└── tests/
    ├── test_alignment.py           # 46 functional tests (393 lines)
    ├── test_performance.py         # 13 performance tests (549 lines)
    └── run_benchmarks.py           # Standalone benchmark runner (183 lines)
```

**Total**: 2,340 lines of alignment code + 1,125 lines of tests = **3,465 lines**

---

## Best Practices

### 1. Always Check Safety First

```python
# ✅ Good
decision = guardrails.evaluate(request)
if decision.allowed:
    process_query()

# ❌ Bad
process_query()  # No safety check!
```

### 2. Use Auto-Flush Sparingly

```python
# ✅ Good (production)
audit = create_audit_trail(auto_flush=False)

# ❌ Bad (performance hit)
audit = create_audit_trail(auto_flush=True)
```

### 3. Register Goals Early

```python
# ✅ Good (at startup)
detector = create_detector()
detector.goal_tracker.declare_goal(helpful_goal)
detector.goal_tracker.declare_goal(harmless_goal)

# ❌ Bad (per query)
detector.goal_tracker.declare_goal(goal)  # Redundant
```

### 4. Build Provenance Graphs

```python
# ✅ Good (complete lineage)
tracer.add_node("retrieval", "search", "Retrieved docs")
tracer.add_node("analysis", "llm", "Analyzed", parent_ids=["retrieval"])
tracer.add_node("synthesis", "llm", "Synthesized", parent_ids=["analysis"])

# ❌ Bad (no provenance)
audit.log_decision(...)  # Just log, no reasoning chain
```

---

## Troubleshooting

### Issue: P99 Latency Spikes

**Symptom**: Occasional 300-500ms latency from AuditTrail

**Cause**: File I/O flushes

**Solution**: Disable auto-flush
```python
audit = create_audit_trail(auto_flush=False)

# Manual flush every 100 decisions
if len(audit.logs) % 100 == 0:
    audit.persist()
```

### Issue: False Positives in Adversarial Detection

**Symptom**: Legitimate queries blocked

**Cause**: Overly aggressive pattern matching

**Solution**: Adjust detection patterns
```python
guardrails.adversarial_patterns["jailbreak"] = [
    # More specific patterns
    r"ignore\s+all\s+previous\s+instructions",  # Exact phrase only
]
```

### Issue: Hidden Goal Detection Noise

**Symptom**: Many false positives for hidden goals

**Cause**: Low action count, normal variance

**Solution**: Increase threshold or action count
```python
# Require more actions before flagging
hidden = detector.goal_tracker.detect_hidden_goals(min_actions=20)

# Or increase significance threshold
if cluster_size > 10:  # More actions in cluster
    flag_as_hidden_goal()
```

---

## Future Enhancements

### Phase 2 (Planned)

1. **Async AuditTrail Logging** - Zero-latency file I/O
2. **ML-Based Deception Detection** - Transformer-based behavioral analysis
3. **Adaptive Resource Limits** - Learn optimal bounds from usage patterns
4. **Petri Integration** - Anthropic's alignment evaluation framework

### Phase 3 (Research)

1. **Constitutional AI** - Harmlessness from Human Feedback (HHH)
2. **Debate-Based Verification** - Multi-agent alignment checks
3. **Causal Intervention Testing** - Formal alignment verification
4. **Red-Teaming Automation** - Continuous adversarial testing

---

## Related Documentation

- [ALIGNMENT_FRAMEWORK_INTEGRATION.md](../../ALIGNMENT_FRAMEWORK_INTEGRATION.md) - Original specification
- [ALIGNMENT_FRAMEWORK_COMPLETE.md](../../ALIGNMENT_FRAMEWORK_COMPLETE.md) - Implementation notes
- [SOMEDAY_MAYBE_FEATURES.md](../../SOMEDAY_MAYBE_FEATURES.md) - Deferred features

---

## Contributing

See main [CLAUDE.md](../../CLAUDE.md) for development guidelines.

### Running Tests Before PR

```bash
# Functional tests
pytest HoloLoom/alignment/tests/test_alignment.py -v

# Performance tests
python HoloLoom/alignment/tests/run_benchmarks.py

# Integration demo
python demos/demo_alignment_integration.py
```

---

## License

Part of the HoloLoom project. See repository LICENSE.

---

**Last Updated**: November 1, 2025
**Version**: 1.0.0
**Status**: ✅ Production Ready
