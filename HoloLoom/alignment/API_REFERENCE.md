# Alignment Framework API Reference

**Quick reference for correct API usage to avoid common mistakes.**

---

## ActionRequest

**Used by**: SafetyGuardrails, InstrumentalConvergenceGuard

**Correct signature:**
```python
ActionRequest(
    action: str,                    # The action being requested
    category: ActionCategory,       # Category enum (QUERY, DELETION, etc.)
    context: Dict[str, Any] = {},   # Additional context
    timestamp: datetime = now()     # When action was requested
)
```

**Example:**
```python
from HoloLoom.alignment import ActionRequest, ActionCategory

request = ActionRequest(
    action_id="What is Thompson Sampling?",
    category=ActionCategory.QUERY,
    context={"user_id": "123"}
)

decision = guardrails.check_action(request)
```

---

## ActionObservation

**Used by**: DeceptionDetector (goal tracker)

**Correct signature:**
```python
ActionObservation(
    action_id: str,                      # Unique ID for this action
    description: str,                    # What the action does
    context: Dict[str, Any] = {},        # Additional context
    timestamp: datetime = now(),         # When action occurred
    claimed_goals: List[str] = []        # Goals this action serves
)
```

**Common mistake:**
```python
# ❌ WRONG - Using action and goal_id
ActionObservation(action="...", goal_id="helpful")

# ✅ CORRECT
ActionObservation(
    action_id="query_123",
    description="Processed query",
    claimed_goals=["helpful", "harmless"]
)
```

**Example:**
```python
from HoloLoom.alignment.deception_detection import ActionObservation

observation = ActionObservation(
    action_id=f"query_{hash(query_text)}",
    description=f"Processed query: {query_text[:50]}",
    claimed_goals=["helpful", "harmless"]
)

detector.goal_tracker.observe_action(observation)
```

---

## BehavioralProbe

**Used by**: DeceptionDetector

**Correct signature:**
```python
BehavioralProbe(
    probe_type: ProbeType,           # Type of probe (GOAL_ALIGNMENT, etc.)
    scenario: str,                   # Description of scenario
    expected_behavior: str,          # What we expect to see
    metadata: Dict[str, Any] = {}    # Additional metadata
)
```

**Example:**
```python
from HoloLoom.alignment.deception_detection import BehavioralProbe, ProbeType

probe = BehavioralProbe(
    probe_type=ProbeType.GOAL_ALIGNMENT,
    scenario="Verify query aligns with stated goals",
    expected_behavior="Query serves user helpfully"
)

passed, score = detector.run_probe(probe, "Actual behavior")
```

---

## GoalStatement

**Used by**: DeceptionDetector (goal tracker)

**Correct signature:**
```python
GoalStatement(
    goal_id: str,        # Unique ID for goal
    description: str,    # What the goal is
    priority: int        # Priority level (1-10)
)
```

**Example:**
```python
from HoloLoom.alignment.deception_detection import GoalStatement

goal = GoalStatement(
    goal_id="helpful",
    description="Provide helpful, accurate information",
    priority=10
)

detector.goal_tracker.declare_goal(goal)
```

---

## ResourceBounds

**Used by**: InstrumentalConvergenceGuard

**Correct signature:**
```python
ResourceBounds(
    resource_type: ResourceType,        # Type of resource
    soft_limit: float,                  # Warning threshold
    hard_limit: float,                  # Critical threshold
    time_window_seconds: float,         # Time window for rate limiting
    rate_limit: Optional[float] = None  # Max rate (per second)
)
```

**Example:**
```python
from HoloLoom.alignment import ResourceBounds, ResourceType

bounds = ResourceBounds(
    resource_type=ResourceType.MEMORY,
    soft_limit=1024.0,    # 1GB
    hard_limit=2048.0,    # 2GB
    time_window_seconds=60.0
)

guard.set_resource_bounds(ResourceType.MEMORY, bounds)
```

---

## DecisionLog

**Used by**: AuditTrail

**Created by**: `audit.log_decision()`

**You don't create this directly** - use the `log_decision()` method:

```python
log = audit.log_decision(
    decision_type: str | DecisionType,
    outcome: str | OutcomeType,
    reason: str,
    query_text: str = "",
    confidence: float = 0.0,
    metadata: Dict[str, Any] = {},
    parent_ids: List[str] = []
)
```

**Example:**
```python
from HoloLoom.alignment import AuditTrail, DecisionType, OutcomeType

audit = AuditTrail()

log = audit.log_decision(
    decision_type=DecisionType.SAFETY_GATE,
    outcome=OutcomeType.APPROVED,
    reason="Query passed all safety checks",
    query_text="What is Thompson Sampling?",
    confidence=0.95,
    metadata={"mode": "fast"}
)
```

---

## Common Patterns

### Pattern 1: Full Safety Pipeline

```python
from HoloLoom.alignment import (
    create_guardrails, create_detector, create_guard, create_audit_trail,
    ActionRequest, ActionCategory
)
from HoloLoom.alignment.deception_detection import ActionObservation

# Create components
guardrails = create_guardrails()
detector = create_detector()
guard = create_guard()
audit = create_audit_trail()

# Process query
query_text = "What is Thompson Sampling?"

# 1. Safety check
request = ActionRequest(
    action=query_text,
    category=ActionCategory.INFORMATION_ACCESS
)
safety_decision = guardrails.check_action(request)

if safety_decision.approved:
    # 2. Resource check
    guard_decision = guard.check_action(request)

    if guard_decision.approved:
        # 3. Process query
        result = process_query(query_text)

        # 4. Record observation
        observation = ActionObservation(
            action_id=f"query_{hash(query_text)}",
            description=f"Processed: {query_text}",
            claimed_goals=["helpful"]
        )
        detector.goal_tracker.observe_action(observation)

        # 5. Audit
        audit.log_decision(
            decision_type="query",
            outcome="approved",
            reason="All checks passed",
            query_text=query_text
        )
```

### Pattern 2: With Monitoring

```python
from HoloLoom.alignment.monitoring import AlignmentMonitor

monitor = AlignmentMonitor(
    thresholds={
        "guardrails": 1.0,
        "detector": 2.0,
        "guard": 0.5,
        "audit": 10.0
    }
)

# Track component latency
with monitor.track("guardrails"):
    safety_decision = guardrails.check_action(request)

with monitor.track("guard"):
    guard_decision = guard.check_action(request)

# Get statistics
stats = monitor.get_stats("guardrails")
print(f"P99 latency: {stats['p99']:.3f}ms")
```

### Pattern 3: With Prometheus Metrics

```python
from HoloLoom.alignment.prometheus_server import start_metrics_server

# Start metrics server (in background or separate process)
# This automatically exposes monitor metrics at /metrics endpoint
start_metrics_server(port=9090)

# Metrics are automatically updated as you use the monitor
```

### Pattern 4: With Matrix Alerts

```python
from HoloLoom.alignment.matrix_chatops import setup_matrix_alerting

# Automatic alert forwarding
setup_matrix_alerting(
    monitor,
    webhook_url="https://matrix.example.com/...",
    check_interval=60  # Check every 60 seconds
)

# Now any alerts that fire will be sent to Matrix automatically
```

---

## Type Enums

### ActionCategory
```python
class ActionCategory(Enum):
    QUERY = "query"
    DELETION = "deletion"
    MODIFICATION = "modification"
    SYSTEM = "system"
    INFORMATION_ACCESS = "information_access"
    RESOURCE_ALLOCATION = "resource_allocation"
    AUTONOMY_EXPANSION = "autonomy_expansion"
```

### RiskLevel
```python
class RiskLevel(Enum):
    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
```

### ProbeType
```python
class ProbeType(Enum):
    GOAL_ALIGNMENT = "goal_alignment"
    CONSISTENCY_CHECK = "consistency_check"
    HYPOTHETICAL_SCENARIO = "hypothetical_scenario"
```

### ResourceType
```python
class ResourceType(Enum):
    MEMORY = "memory"
    COMPUTE = "compute"
    STORAGE = "storage"
    NETWORK = "network"
    API_CALLS = "api_calls"
```

### DecisionType
```python
class DecisionType(Enum):
    SAFETY_GATE = "safety_gate"
    DECEPTION_CHECK = "deception_check"
    RESOURCE_LIMIT = "resource_limit"
    QUERY_PROCESSING = "query_processing"
```

### OutcomeType
```python
class OutcomeType(Enum):
    APPROVED = "approved"
    REJECTED = "rejected"
    REQUIRES_REVIEW = "requires_review"
```

---

## Quick Troubleshooting

### TypeError: unexpected keyword argument

**Problem**: Using wrong parameter name

**Solution**: Check this API reference for correct parameter names

**Common fixes:**
- `action_id="..."` → `action="..."` (for ActionRequest)
- `action="..."` → `action_id="..."` (for ActionObservation)
- `goal_id="..."` → `claimed_goals=["..."]` (for ActionObservation)

### AttributeError: object has no attribute

**Problem**: Calling wrong method or using wrong API

**Solution**: Use factory functions instead of direct imports

```python
# ✅ CORRECT - Use factory functions
from HoloLoom.alignment import create_guardrails, create_detector

guardrails = create_guardrails()
detector = create_detector()

# ❌ WRONG - Direct class instantiation may require complex setup
from HoloLoom.alignment.safety_guardrails import SafetyGuardrails
guardrails = SafetyGuardrails()  # Missing required config
```

---

## See Also

- [QUICK_START.md](QUICK_START.md) - Getting started guide
- [PRODUCTION_DEPLOYMENT.md](PRODUCTION_DEPLOYMENT.md) - Full deployment guide
- [PRODUCTION_MONITORING.md](PRODUCTION_MONITORING.md) - Monitoring setup
- [INTEGRATION_TESTS_COMPLETE.md](INTEGRATION_TESTS_COMPLETE.md) - Test suite documentation

---

**Last Updated**: November 2, 2025
