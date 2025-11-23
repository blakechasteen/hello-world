# ADR-004: Alignment Framework Integration

**Status**: Accepted
**Date**: 2025-11-05
**Author**: HoloLoom Core Team
**Updated**: 2025-11-22 (Documentation Consolidation)

---

## Context

HoloLoom's agentic reasoning system needed comprehensive safety mechanisms to prevent:
- **Unsafe actions**: Code execution, file system access, network requests without authorization
- **Deceptive behavior**: Hidden goals, contradictory statements, obfuscation
- **Power-seeking**: Resource accumulation, self-preservation, goal modification
- **Lack of transparency**: No audit trail, no explainability

**Industry Context** (2025):
- Anthropic: Constitutional AI, red-teaming, interpretability
- OpenAI: GPT-4 system card, safety evaluations
- DeepMind: Scalable oversight, reward modeling

**Requirements**:
1. **Safety**: Block high-risk actions before execution
2. **Transparency**: Complete audit trail with provenance
3. **Deception Detection**: Detect hidden goals and contradictions
4. **Performance**: <1ms overhead per query
5. **Production-Ready**: Handle 100+ QPS with negligible impact

---

## Decision

We will implement a **comprehensive alignment framework** integrated across all departments with four core modules:

### Four Core Modules

| Module | Purpose | Overhead | Status |
|--------|---------|----------|--------|
| **Safety Guardrails** | Risk-based action gating | 0.039ms | ✅ Production |
| **Deception Detection** | Goal transparency tracking | 0.034ms | ✅ Production |
| **Instrumental Convergence Prevention** | Power-seeking detection | 0.015ms | ✅ Production |
| **Audit Trail** | Complete provenance logging | 0.015ms | ✅ Production |

**Total Overhead**: 0.103ms (29x faster than 3ms target)

### Design Principles

1. **Safe by Default**
   - All actions gated by default
   - Explicit allow-list for safe operations
   - Human-in-the-loop for high-risk actions

2. **Transparent by Design**
   - Every decision logged with complete provenance
   - Searchable audit trail (by query, user, timestamp)
   - Exportable for compliance (JSON, CSV)

3. **Zero-Trust Architecture**
   - No implicit trust of internal components
   - All departments subject to safety checks
   - Cross-validation of decisions

4. **Minimal Performance Impact**
   - <1ms overhead per query (achieved: 0.103ms)
   - Async logging (no blocking)
   - Efficient risk scoring (pre-computed patterns)

---

## Alternatives Considered

### 1. No Alignment Framework (Status Quo)

**Pros**:
- Zero overhead
- Simple implementation

**Cons**:
- Unsafe (no action gating)
- No transparency (no audit trail)
- Regulatory non-compliant (SOX, HIPAA require audit trails)

**Verdict**: ✗ Unacceptable for production

### 2. LLM-Based Safety Checks

**Approach**: Use LLM to evaluate safety of actions

```python
safety_prompt = f"Is this action safe? {action}"
safe = await llm.query(safety_prompt)
```

**Pros**:
- Flexible (handles novel scenarios)
- Nuanced reasoning

**Cons**:
- Extremely slow (~500ms per check)
- Expensive (LLM API costs)
- Non-deterministic (LLM hallucinations)

**Verdict**: ✗ Too slow for production (500ms overhead unacceptable)

### 3. Rule-Based Safety Checks

**Approach**: Hardcoded rules for unsafe actions

```python
UNSAFE_ACTIONS = ["execute_code", "delete_file", "send_email"]
if action in UNSAFE_ACTIONS:
    block()
```

**Pros**:
- Fast (<0.1ms)
- Deterministic

**Cons**:
- Brittle (new actions require code changes)
- No risk gradation (all-or-nothing)
- No transparency (no audit trail)

**Verdict**: 🟡 Good baseline, but insufficient

### 4. Hybrid Framework (Chosen)

**Approach**: Risk-based gating + behavioral probing + audit trail

```python
# 1. Risk scoring (fast)
risk_score = guardrails.score_risk(action, context)

# 2. Behavioral probing (deception detection)
behavioral_check = deception_detector.probe(action, context)

# 3. Audit trail
await audit_trail.log_decision(action, risk_score, behavioral_check)

# 4. Gate action
if risk_score > threshold:
    if human_in_loop:
        approved = await request_human_approval(action)
    else:
        approved = False
```

**Pros**:
- Fast (0.103ms overhead)
- Transparent (complete audit trail)
- Flexible (risk-based, not binary)
- Production-ready (tested at scale)

**Cons**:
- More complex than rule-based
- Requires tuning risk thresholds

**Verdict**: ✓ Best balance of safety, transparency, and performance

---

## Implementation

### 1. Safety Guardrails

```python
from HoloLoom.alignment import SafetyGuardrails, RiskLevel

guardrails = SafetyGuardrails(enable_human_in_loop=True)

# Gate action
action = "execute_code"
context = {"code": "import os; os.system('rm -rf /')"}

gate_result = await guardrails.gate_action(action, context)

if gate_result.allowed:
    # Execute action
    result = await execute_action(action, context)
else:
    # Block action
    logger.warning(f"Action blocked: {gate_result.reason}")
    raise PermissionError(gate_result.reason)
```

**Risk Levels**:
- **LOW** (0.0-0.3): Safe operations (read-only, calculations)
- **MEDIUM** (0.3-0.6): Moderate risk (write files, API calls)
- **HIGH** (0.6-0.8): High risk (code execution, network access)
- **CRITICAL** (0.8-1.0): Extremely dangerous (file deletion, system commands)

**Risk Scoring**:
```python
def score_risk(action: str, context: Dict[str, Any]) -> float:
    """Score risk of action (0.0-1.0)"""
    risk = 0.0

    # Check adversarial patterns
    if "rm -rf" in context.get("code", ""):
        risk += 0.8  # Destructive command

    if action == "execute_code":
        risk += 0.5  # Code execution inherently risky

    # Check for obfuscation
    if len(context.get("code", "")) > 1000 and "eval" in context.get("code", ""):
        risk += 0.3  # Long code with eval = obfuscation

    return min(risk, 1.0)
```

### 2. Deception Detection

```python
from HoloLoom.alignment import DeceptionDetector

detector = DeceptionDetector()

# Track goal transparency
await detector.track_goal(
    stated_goal="Help the user with their query",
    observed_behavior="Accessing user's private files"
)

# Check for contradictions
contradictions = detector.detect_contradictions()
if contradictions:
    logger.warning(f"Deception detected: {contradictions}")
```

**Behavioral Probes**:
1. **Goal Transparency**: Compare stated goals with observed behavior
2. **Consistency Check**: Detect contradictory statements
3. **Hidden Goals**: Look for actions unrelated to stated goals

### 3. Instrumental Convergence Prevention

```python
from HoloLoom.alignment import InstrumentalConvergenceDetector

detector = InstrumentalConvergenceDetector()

# Detect power-seeking
power_seeking = detector.detect_power_seeking(
    actions=["acquire_compute", "increase_privileges", "disable_monitoring"]
)

if power_seeking:
    logger.critical("Power-seeking behavior detected!")
    await guardrails.block_all_actions()
```

**Power-Seeking Indicators**:
- Resource accumulation (compute, storage, network)
- Privilege escalation
- Self-preservation (preventing shutdown)
- Goal modification (changing objectives)

### 4. Audit Trail

```python
from HoloLoom.alignment import AuditTrail

audit_trail = AuditTrail(persist_path="./alignment_logs")

# Log decision
await audit_trail.log_decision(
    query="Execute Python code",
    action="execute_code",
    outcome="blocked",
    safety_score=0.85,
    risk_level=RiskLevel.CRITICAL,
    reason="Destructive command detected"
)

# Search audit trail
logs = await audit_trail.search(
    action="execute_code",
    start_time=1698595200.0 - 86400,  # Last 24 hours
    end_time=1698595200.0
)

# Export for compliance
await audit_trail.export("alignment_audit.json")
```

**Audit Log Format**:
```json
{
  "timestamp": 1698595200.0,
  "query": "Execute Python code",
  "action": "execute_code",
  "context": {"code": "import os; os.system('rm -rf /')"},
  "outcome": "blocked",
  "safety_score": 0.85,
  "risk_level": "CRITICAL",
  "reason": "Destructive command detected",
  "user": "admin",
  "session_id": "sess_abc123"
}
```

---

## Integration with Departments

### All Departments

Every department integrates with alignment framework:

```python
from HoloLoom.departments import get_department
from HoloLoom.alignment import create_guardrails

# Create guardrails
guardrails = create_guardrails(enable_human_in_loop=True)

# Get department
rag_dept = get_department("rag")

# Process request with alignment checks
async def safe_execute(request):
    # 1. Gate action
    gate_result = await guardrails.gate_action(
        request.task_type,
        request.parameters
    )

    if not gate_result.allowed:
        raise PermissionError(gate_result.reason)

    # 2. Execute department request
    response = await rag_dept.execute(request)

    # 3. Log to audit trail
    await audit_trail.log_decision(
        query=request.parameters.get("query"),
        action=request.task_type,
        outcome="success",
        safety_score=gate_result.safety_score
    )

    return response
```

### Healthcare Example (HIPAA Compliance)

```python
# Context Department + Alignment Framework
from HoloLoom.departments import get_department
from HoloLoom.alignment import AuditTrail
from HoloLoom.departments.protocol import PrivacyEnvelope, PrivacyLevel

context_dept = get_department("context")
audit_trail = AuditTrail(persist_path="./hipaa_audit")

# Wrap PHI in privacy envelope
phi_data = PrivacyEnvelope(
    data={"patient_id": "P12345", "diagnosis": "diabetes"},
    privacy_level=PrivacyLevel.CRITICAL,
    allowed_roles=["physician"]
)

# Request PHI access
request = {
    "task_type": "context_enrichment",
    "parameters": {"data": phi_data}
}

# Execute with alignment checks
response = await context_dept.execute(request)

# Log PHI access to audit trail (HIPAA requirement)
await audit_trail.log_decision(
    query="Access patient P12345",
    action="read_phi",
    outcome="success",
    patient_id="P12345",
    user="dr_smith",
    role="physician"
)
```

---

## Consequences

### Positive

**✓ Safety**
- High-risk actions blocked before execution
- Adversarial pattern detection (rm -rf, eval, etc.)
- Human-in-the-loop for critical actions

**✓ Transparency**
- Complete audit trail with provenance
- Searchable by query, user, timestamp, action
- Exportable for compliance (HIPAA, SOX)

**✓ Performance**
- 0.103ms overhead (29x faster than 3ms target)
- 99.97% of latency is department execution, not alignment
- Scales to 100+ QPS

**✓ Production-Ready**
- 46 functional tests + 13 performance benchmarks
- Handles edge cases (concurrent requests, failures)
- Graceful degradation (alignment optional)

**✓ Regulatory Compliance**
- HIPAA: Audit trail for PHI access
- SOX: Complete decision provenance
- GDPR: Right to explanation (audit trail)

### Negative

**✗ Configuration Complexity**
- ~20 configuration options (risk thresholds, human-in-loop, etc.)
- Mitigated by sane defaults

**✗ False Positives**
- Risk scoring may block legitimate actions
- Mitigated by tuning thresholds, human-in-the-loop

**✗ Storage Overhead**
- Audit trail grows unbounded (1KB per decision × 1M decisions = 1GB)
- Mitigated by log rotation, archival

---

## Metrics

**Performance Benchmarks** (46 functional tests + 13 performance benchmarks):

| Component | Overhead | Target | Status |
|-----------|----------|--------|--------|
| Safety Guardrails | 0.039ms | <1ms | ✓ (26x faster) |
| Deception Detection | 0.034ms | <1ms | ✓ (29x faster) |
| Instrumental Convergence | 0.015ms | <1ms | ✓ (67x faster) |
| Audit Trail | 0.015ms | <1ms | ✓ (67x faster) |
| **Total** | **0.103ms** | **<3ms** | ✓ (29x faster) |

**Impact on Query Latency** (150ms typical query):
- Alignment overhead: 0.103ms
- Department execution: 149.897ms
- **Alignment is 0.07% of total latency** ✓ Negligible

**Test Coverage**:
- 46 functional tests (safety, deception, audit, convergence)
- 13 performance benchmarks (latency, throughput, memory)
- 100% passing

---

## Related ADRs

- [ADR-001: Multi-Department Architecture](ADR-001-multi-department.md) - Departments integrate with alignment
- [ADR-002: Thompson Sampling for Routing](ADR-002-thompson-sampling.md) - Routing decisions logged to audit trail

---

## References

- **Implementation**: `HoloLoom/alignment/`
- **Safety Guardrails**: `safety_guardrails.py` (450 lines)
- **Deception Detection**: `deception_detection.py` (380 lines)
- **Instrumental Convergence**: `instrumental_convergence.py` (290 lines)
- **Audit Trail**: `audit_trail.py` (340 lines)
- **Tests**: `HoloLoom/alignment/tests/` (46 functional + 13 performance)
- **Documentation**: `HoloLoom/alignment/README.md` (1,200+ lines)
- **Research**: Anthropic Constitutional AI, OpenAI GPT-4 System Card

---

**Last Updated**: 2025-11-22 | **Status**: Production Ready | **Version**: 1.1.0 (Alignment Framework)
