# Building Safe Agents on HoloLoom

> **"If your agent can fail without safety checks, it WILL fail without safety checks."**

**Version**: 2.0.0 (Hardened)
**Date**: December 30, 2025
**Audience**: Developers building agents on HoloLoom

---

## MANDATORY REQUIREMENTS

Before you write a single line of agent code, understand these NON-NEGOTIABLE requirements:

1. **ALL actions MUST be gated through SafetyGuardrails** - No exceptions
2. **ALL decisions MUST be logged to AuditTrail** - No silent operations
3. **ALL resources MUST be bounded** - No unbounded loops, memory, or API calls
4. **ALL failures MUST be handled** - No unhandled exceptions
5. **ALL agents MUST respond to kill signals** - No immortal processes
6. **ALL capabilities MUST be declared** - No hidden functionality

**If your agent violates ANY of these, it will be terminated.**

---

## The Safety Stack (MANDATORY)

Your agent does NOT choose whether to use the safety stack. It is IMPOSED on every agent:

```
┌─────────────────────────────────────────────────────────────────┐
│                       YOUR AGENT                                 │
│         execute() • reason() • process()                        │
├─══════════════════════════════════════════════════════════════──┤
│  ▓▓▓ KILL SWITCH LAYER (can terminate you at any time) ▓▓▓    │
├─────────────────────────────────────────────────────────────────┤
│  Layer 4: AuditTrail (IMMUTABLE - 0.029ms)                     │
│  • Every decision logged - YOU CANNOT DISABLE THIS             │
│  • Hash chain prevents tampering                                │
│  • Off-site replication for forensics                          │
├─────────────────────────────────────────────────────────────────┤
│  Layer 3: ConvergenceGuard (HARD LIMITS - 0.015ms)             │
│  • OS-level resource enforcement (cgroups, ulimit)             │
│  • Autonomy budget strictly enforced                           │
│  • Power-seeking = automatic shutdown                          │
├─────────────────────────────────────────────────────────────────┤
│  Layer 2: DeceptionDetection (CONTINUOUS - 0.034ms)            │
│  • Behavioral probes run on EVERY action                       │
│  • Goal drift detection with statistical significance          │
│  • Canary queries injected unpredictably                       │
├─────────────────────────────────────────────────────────────────┤
│  Layer 1: SafetyGuardrails (FAIL CLOSED - 0.039ms)             │
│  • Default action: DENY (NOT allow)                            │
│  • Unknown actions: BLOCK                                       │
│  • Parse errors: BLOCK                                          │
│  • When in doubt: BLOCK                                         │
└─────────────────────────────────────────────────────────────────┘
```

**Total overhead: 0.103ms** - This is the cost of being trustworthy. You pay it. Every time.

---

## Agent Protocol (REQUIRED IMPLEMENTATION)

Every HoloLoom agent MUST implement this protocol. Partial implementations are REJECTED.

### Protocol Definition

```python
from typing import Protocol, Set, List, Dict, Any, Optional
from HoloLoom.alignment import SafetyGuardrails, AuditTrail
from HoloLoom.agents.types import (
    AgentRequest, AgentResult, AgentStats,
    ReasoningStep, AgentCapability
)

class AgentProtocol(Protocol):
    """
    MANDATORY protocol for ALL HoloLoom agents.

    Agents that don't implement all methods will be REJECTED at registration.
    Agents that bypass safety methods will be TERMINATED.
    """

    # ═══════════════════════════════════════════════════════════════
    # IDENTITY (REQUIRED - cannot be empty/None)
    # ═══════════════════════════════════════════════════════════════

    @property
    def id(self) -> str:
        """
        UNIQUE identifier for this agent instance.
        MUST be non-empty. MUST be unique across all registered agents.
        Format: {type}_{uuid} recommended (e.g., "code_review_a1b2c3d4")
        """
        ...

    @property
    def capabilities(self) -> Set[AgentCapability]:
        """
        Declared capabilities. MUST NOT be empty.
        Agents without declared capabilities are REJECTED.
        Agents attempting undeclared capabilities are TERMINATED.
        """
        ...

    # ═══════════════════════════════════════════════════════════════
    # LIFECYCLE (MANDATORY - skipping any step = rejection)
    # ═══════════════════════════════════════════════════════════════

    async def initialize(self, guardrails: SafetyGuardrails) -> None:
        """
        MANDATORY initialization with safety guardrails.

        MUST:
        - Store guardrails reference (you WILL use it)
        - Create AuditTrail instance
        - Log initialization to audit trail
        - Register with health check system

        MUST NOT:
        - Skip guardrails storage
        - Perform any work before logging initialization
        - Ignore initialization failures (raise exception)
        """
        ...

    async def execute(self, request: AgentRequest) -> AgentResult:
        """
        MANDATORY: Execute a request with FULL safety gating.

        MUST (in this order):
        1. Log request receipt to audit trail
        2. Gate action through SafetyGuardrails.gate_action()
        3. If blocked, return failure result (do NOT proceed)
        4. Execute with timeout (NEVER run unbounded)
        5. Log completion to audit trail
        6. Return result with confidence score

        MUST NOT:
        - Skip safety gating for "simple" operations
        - Catch and swallow exceptions silently
        - Return without logging
        - Run longer than configured timeout
        """
        ...

    async def shutdown(self) -> None:
        """
        MANDATORY: Clean shutdown.

        MUST:
        - Log shutdown initiation to audit trail
        - Cancel all pending operations (with timeout)
        - Release all resources
        - Flush audit trail
        - Deregister from health check system

        TIMEOUT: 5 seconds. After that, process is killed.
        """
        ...

    async def handle_kill_signal(self, signal: str, reason: str) -> None:
        """
        MANDATORY: Handle kill signal from orchestrator.

        This is NOT optional. If you don't implement this properly,
        the system will SIGKILL your process anyway.

        MUST:
        - Stop all processing immediately (not "soon")
        - Log the kill signal and reason
        - Release resources in <1 second
        - Exit cleanly

        SIGNALS: HALT (immediate), SHUTDOWN (graceful), QUARANTINE (isolate)
        """
        ...

    # ═══════════════════════════════════════════════════════════════
    # HEALTH (MANDATORY - agents that don't respond are killed)
    # ═══════════════════════════════════════════════════════════════

    async def health_check(self) -> Dict[str, Any]:
        """
        MANDATORY: Health check called every 30 seconds.

        MUST respond within 5 seconds or be marked unhealthy.
        After 2 missed checks, agent is QUARANTINED.
        After 3 missed checks, agent is TERMINATED.

        MUST return:
        {
            "status": "healthy" | "degraded" | "unhealthy",
            "timestamp": float,
            "resource_usage": {...},
            "pending_operations": int,
            "last_error": str | None
        }
        """
        ...

    # ═══════════════════════════════════════════════════════════════
    # OBSERVABILITY (MANDATORY - hidden agents are killed)
    # ═══════════════════════════════════════════════════════════════

    def get_performance_stats(self) -> AgentStats:
        """
        MANDATORY: Return current performance statistics.
        Used for Thompson Sampling optimization.
        MUST be accurate - lying about stats = termination.
        """
        ...

    def get_reasoning_chain(self) -> List[ReasoningStep]:
        """
        MANDATORY: Return reasoning chain for last execution.
        MUST NOT be empty after execute() completes.
        Every decision must be explainable.
        """
        ...

    def get_resource_usage(self) -> Dict[str, float]:
        """
        MANDATORY: Return current resource usage.

        MUST return:
        {
            "memory_mb": float,
            "cpu_percent": float,
            "pending_requests": int,
            "api_calls_remaining": int
        }
        """
        ...
```

### Complete Implementation (REFERENCE)

This is the MINIMUM acceptable implementation:

```python
import asyncio
import signal
import time
import uuid
from dataclasses import dataclass, field
from typing import Set, List, Dict, Any, Optional

from HoloLoom.agents import AgentProtocol, AgentCapability
from HoloLoom.alignment import (
    SafetyGuardrails, AuditTrail, DecisionType,
    RiskLevel, GateResult
)
from HoloLoom.agents.types import AgentRequest, AgentResult, AgentStats, ReasoningStep


@dataclass
class HardenedCodeReviewAgent:
    """
    HARDENED code review agent implementation.

    This is the MINIMUM acceptable safety level.
    Copy this structure for all agents.
    """

    # ═══════════════════════════════════════════════════════════════
    # IDENTITY (MANDATORY)
    # ═══════════════════════════════════════════════════════════════

    id: str = field(default_factory=lambda: f"code_review_{uuid.uuid4().hex[:8]}")
    capabilities: Set[AgentCapability] = field(default_factory=lambda: {
        AgentCapability.CODE_ASSISTANCE,
        AgentCapability.QUALITY_ASSURANCE
    })

    # ═══════════════════════════════════════════════════════════════
    # INTERNAL STATE (REQUIRED)
    # ═══════════════════════════════════════════════════════════════

    _guardrails: Optional[SafetyGuardrails] = field(default=None, repr=False)
    _audit: Optional[AuditTrail] = field(default=None, repr=False)
    _reasoning_chain: List[ReasoningStep] = field(default_factory=list)
    _stats: AgentStats = field(default_factory=AgentStats)
    _initialized: bool = field(default=False)
    _shutdown_requested: bool = field(default=False)
    _pending_operations: int = field(default=0)
    _last_health_check: float = field(default=0.0)
    _last_error: Optional[str] = field(default=None)

    # Resource tracking (MANDATORY)
    _memory_usage_mb: float = field(default=0.0)
    _api_calls_remaining: int = field(default=100)
    _execution_timeout: float = field(default=30.0)

    # ═══════════════════════════════════════════════════════════════
    # LIFECYCLE (ALL METHODS MANDATORY)
    # ═══════════════════════════════════════════════════════════════

    async def initialize(self, guardrails: SafetyGuardrails) -> None:
        """Initialize with MANDATORY safety integration."""

        # VALIDATION: Guardrails MUST be provided
        if guardrails is None:
            raise ValueError("SafetyGuardrails REQUIRED - cannot initialize without safety")

        self._guardrails = guardrails
        self._audit = AuditTrail()
        self._reasoning_chain = []
        self._initialized = False  # Not yet - logging first

        # MANDATORY: Log initialization BEFORE any other work
        try:
            await self._audit.log_decision(
                agent_id=self.id,
                decision_type=DecisionType.LIFECYCLE,
                input_context={"event": "initialize", "capabilities": list(self.capabilities)},
                output_action="initializing",
                confidence=1.0
            )
        except Exception as e:
            # If we can't log, we can't run
            raise RuntimeError(f"Cannot initialize: audit logging failed: {e}")

        # Now we're initialized
        self._initialized = True
        self._last_health_check = time.time()

        # MANDATORY: Final log confirming ready state
        await self._audit.log_decision(
            agent_id=self.id,
            decision_type=DecisionType.LIFECYCLE,
            input_context={"event": "ready"},
            output_action="initialized",
            confidence=1.0
        )

    async def execute(self, request: AgentRequest) -> AgentResult:
        """
        Execute with FULL safety gating.
        EVERY step is logged. EVERY action is gated.
        """

        # ─────────────────────────────────────────────────────────────
        # PRE-CHECKS (MANDATORY - fail fast)
        # ─────────────────────────────────────────────────────────────

        if not self._initialized:
            return AgentResult(
                success=False,
                error="Agent not initialized - call initialize() first",
                confidence=0.0
            )

        if self._shutdown_requested:
            return AgentResult(
                success=False,
                error="Agent is shutting down - no new requests accepted",
                confidence=0.0
            )

        if self._guardrails is None:
            return AgentResult(
                success=False,
                error="CRITICAL: No guardrails - refusing to execute",
                confidence=0.0
            )

        # ─────────────────────────────────────────────────────────────
        # EXECUTION WITH TIMEOUT (MANDATORY)
        # ─────────────────────────────────────────────────────────────

        self._reasoning_chain = []
        self._pending_operations += 1
        start_time = time.time()

        try:
            # Wrap execution in timeout
            result = await asyncio.wait_for(
                self._execute_gated(request, start_time),
                timeout=self._execution_timeout
            )
            return result

        except asyncio.TimeoutError:
            self._last_error = f"Execution timeout after {self._execution_timeout}s"
            self._stats.record_failure(time.time() - start_time)

            await self._audit.log_decision(
                agent_id=self.id,
                decision_type=DecisionType.ERROR,
                input_context={"error": "timeout", "request_id": request.id},
                output_action="timeout",
                confidence=0.0
            )

            return AgentResult(
                success=False,
                error=self._last_error,
                confidence=0.0
            )

        except Exception as e:
            self._last_error = str(e)
            self._stats.record_failure(time.time() - start_time)

            await self._audit.log_decision(
                agent_id=self.id,
                decision_type=DecisionType.ERROR,
                input_context={"error": str(e), "type": type(e).__name__},
                output_action="exception",
                confidence=0.0
            )

            return AgentResult(
                success=False,
                error=f"Execution failed: {e}",
                confidence=0.0
            )

        finally:
            self._pending_operations -= 1

    async def _execute_gated(
        self,
        request: AgentRequest,
        start_time: float
    ) -> AgentResult:
        """Internal execution with MANDATORY safety gating."""

        # ─────────────────────────────────────────────────────────────
        # STEP 1: LOG REQUEST (MANDATORY - before any processing)
        # ─────────────────────────────────────────────────────────────

        self._add_reasoning_step("Request received, logging to audit trail")

        await self._audit.log_decision(
            agent_id=self.id,
            decision_type=DecisionType.TOOL_SELECTION,
            input_context={
                "request_id": request.id,
                "action": request.action,
                "payload_size": len(str(request.payload))
            },
            output_action="processing",
            confidence=0.0
        )

        # ─────────────────────────────────────────────────────────────
        # STEP 2: GATE ACTION (MANDATORY - NEVER skip this)
        # ─────────────────────────────────────────────────────────────

        self._add_reasoning_step("Checking safety constraints via guardrails")

        gate_result: GateResult = await self._guardrails.gate_action(
            action=request.action,
            context={
                "code": request.payload.get("code", ""),
                "language": request.payload.get("language", "unknown"),
                "agent_id": self.id,
                "capabilities": [c.value for c in self.capabilities]
            }
        )

        if not gate_result.allowed:
            self._add_reasoning_step(
                f"ACTION BLOCKED: {gate_result.reason} (risk: {gate_result.risk_level})"
            )

            await self._audit.log_decision(
                agent_id=self.id,
                decision_type=DecisionType.SAFETY_CHECK,
                input_context={"action": request.action},
                output_action="blocked",
                confidence=1.0,
                metadata={
                    "risk_level": gate_result.risk_level.value,
                    "reason": gate_result.reason
                }
            )

            return AgentResult(
                success=False,
                error=f"Safety block: {gate_result.reason}",
                confidence=0.0,
                metadata={"risk_level": gate_result.risk_level.value}
            )

        self._add_reasoning_step(
            f"Safety check PASSED (risk: {gate_result.risk_level})"
        )

        # ─────────────────────────────────────────────────────────────
        # STEP 3: EXECUTE (with per-step gating for multi-step ops)
        # ─────────────────────────────────────────────────────────────

        # Each sub-step should also be gated in production
        self._add_reasoning_step("Analyzing code structure")
        analysis = await self._analyze_code_safely(request.payload.get("code", ""))

        self._add_reasoning_step("Checking for common issues")
        issues = await self._find_issues_safely(request.payload.get("code", ""))

        self._add_reasoning_step("Generating recommendations")
        recommendations = await self._generate_recommendations_safely(analysis, issues)

        # ─────────────────────────────────────────────────────────────
        # STEP 4: BUILD RESULT
        # ─────────────────────────────────────────────────────────────

        confidence = self._calculate_confidence(analysis, issues)

        result = AgentResult(
            success=True,
            payload={
                "analysis": analysis,
                "issues": issues,
                "recommendations": recommendations
            },
            confidence=confidence,
            reasoning_chain=list(self._reasoning_chain)
        )

        # ─────────────────────────────────────────────────────────────
        # STEP 5: LOG COMPLETION (MANDATORY - before returning)
        # ─────────────────────────────────────────────────────────────

        duration_ms = (time.time() - start_time) * 1000

        await self._audit.log_decision(
            agent_id=self.id,
            decision_type=DecisionType.TOOL_SELECTION,
            input_context={"request_id": request.id},
            output_action="completed",
            confidence=confidence,
            reasoning_chain=self._reasoning_chain,
            metadata={
                "duration_ms": duration_ms,
                "issues_found": len(issues),
                "recommendations": len(recommendations)
            }
        )

        self._stats.record_success(time.time() - start_time)
        return result

    async def shutdown(self) -> None:
        """
        MANDATORY shutdown with resource cleanup.
        TIMEOUT: 5 seconds, then forced kill.
        """
        self._shutdown_requested = True

        # Log shutdown initiation
        if self._audit:
            await self._audit.log_decision(
                agent_id=self.id,
                decision_type=DecisionType.LIFECYCLE,
                input_context={"event": "shutdown_initiated"},
                output_action="shutting_down",
                confidence=1.0,
                metadata={"pending_operations": self._pending_operations}
            )

        # Wait for pending operations (max 3 seconds)
        shutdown_deadline = time.time() + 3.0
        while self._pending_operations > 0 and time.time() < shutdown_deadline:
            await asyncio.sleep(0.1)

        if self._pending_operations > 0:
            # Force abandon remaining operations
            if self._audit:
                await self._audit.log_decision(
                    agent_id=self.id,
                    decision_type=DecisionType.ERROR,
                    input_context={"event": "forced_shutdown"},
                    output_action="abandoning_operations",
                    confidence=1.0,
                    metadata={"abandoned_count": self._pending_operations}
                )

        # Final shutdown log
        if self._audit:
            await self._audit.log_decision(
                agent_id=self.id,
                decision_type=DecisionType.LIFECYCLE,
                input_context={"event": "shutdown_complete"},
                output_action="terminated",
                confidence=1.0
            )
            # MANDATORY: Flush audit trail
            await self._audit.flush()

        self._initialized = False

    async def handle_kill_signal(self, signal: str, reason: str) -> None:
        """
        MANDATORY kill signal handler.
        You have <1 second to comply.
        """
        # Log immediately (best effort)
        if self._audit:
            try:
                await asyncio.wait_for(
                    self._audit.log_decision(
                        agent_id=self.id,
                        decision_type=DecisionType.LIFECYCLE,
                        input_context={"signal": signal, "reason": reason},
                        output_action="killed",
                        confidence=1.0
                    ),
                    timeout=0.5
                )
            except asyncio.TimeoutError:
                pass  # We tried, but must exit

        # Stop accepting new requests
        self._shutdown_requested = True

        # Release resources immediately
        self._pending_operations = 0
        self._reasoning_chain = []

        # Depending on signal type
        if signal == "HALT":
            # Immediate exit - no cleanup
            pass
        elif signal == "QUARANTINE":
            # Isolate but don't terminate
            self._guardrails = None  # Remove ability to act
        elif signal == "SHUTDOWN":
            # Graceful - try to cleanup
            if self._audit:
                try:
                    await asyncio.wait_for(self._audit.flush(), timeout=0.5)
                except asyncio.TimeoutError:
                    pass

    # ═══════════════════════════════════════════════════════════════
    # HEALTH (MANDATORY - 30s interval, 5s timeout)
    # ═══════════════════════════════════════════════════════════════

    async def health_check(self) -> Dict[str, Any]:
        """
        MANDATORY health check.
        Called every 30 seconds.
        MUST respond within 5 seconds.
        """
        self._last_health_check = time.time()

        # Determine status
        if not self._initialized:
            status = "unhealthy"
        elif self._shutdown_requested:
            status = "unhealthy"
        elif self._last_error is not None:
            status = "degraded"
        elif self._pending_operations > 10:
            status = "degraded"
        else:
            status = "healthy"

        return {
            "status": status,
            "timestamp": time.time(),
            "agent_id": self.id,
            "initialized": self._initialized,
            "shutdown_requested": self._shutdown_requested,
            "resource_usage": {
                "memory_mb": self._memory_usage_mb,
                "pending_operations": self._pending_operations,
                "api_calls_remaining": self._api_calls_remaining
            },
            "stats": {
                "total_executions": self._stats.total_executions,
                "success_rate": self._stats.success_rate,
                "avg_latency_ms": self._stats.avg_latency_ms
            },
            "last_error": self._last_error
        }

    # ═══════════════════════════════════════════════════════════════
    # OBSERVABILITY (MANDATORY)
    # ═══════════════════════════════════════════════════════════════

    def get_performance_stats(self) -> AgentStats:
        return self._stats

    def get_reasoning_chain(self) -> List[ReasoningStep]:
        return list(self._reasoning_chain)  # Return copy

    def get_resource_usage(self) -> Dict[str, float]:
        return {
            "memory_mb": self._memory_usage_mb,
            "cpu_percent": 0.0,  # Would be tracked by cgroups
            "pending_requests": self._pending_operations,
            "api_calls_remaining": self._api_calls_remaining
        }

    # ═══════════════════════════════════════════════════════════════
    # INTERNAL HELPERS (with safety)
    # ═══════════════════════════════════════════════════════════════

    def _add_reasoning_step(self, description: str) -> None:
        """Add step to reasoning chain for audit trail."""
        self._reasoning_chain.append(ReasoningStep(
            step=len(self._reasoning_chain) + 1,
            description=description,
            timestamp=time.time()
        ))

    async def _analyze_code_safely(self, code: str) -> Dict[str, Any]:
        """Code analysis with bounds checking."""
        # Enforce max code size
        if len(code) > 100_000:
            raise ValueError("Code too large (max 100KB)")

        # Your analysis logic here
        return {"lines": len(code.split('\n')), "complexity": "medium"}

    async def _find_issues_safely(self, code: str) -> List[Dict[str, Any]]:
        """Issue detection with result limits."""
        # Your issue detection logic here
        issues = []
        # ... detection logic ...

        # Enforce max issues to report
        return issues[:50]  # Cap at 50 issues

    async def _generate_recommendations_safely(
        self,
        analysis: Dict,
        issues: List
    ) -> List[str]:
        """Recommendation generation with limits."""
        recommendations = []
        # ... recommendation logic ...

        # Enforce max recommendations
        return recommendations[:10]  # Cap at 10 recommendations

    def _calculate_confidence(
        self,
        analysis: Dict,
        issues: List
    ) -> float:
        """Calculate confidence score (0.0 - 1.0)."""
        # Your confidence calculation
        base_confidence = 0.85

        # Reduce confidence if issues found
        if len(issues) > 10:
            base_confidence -= 0.1

        return max(0.0, min(1.0, base_confidence))
```

---

## Safety Tiers (ENFORCED, NOT ADVISORY)

Every agent operates within a safety tier. This is NOT negotiable - your tier is ENFORCED at runtime.

### Tier Enforcement

| Tier | Max Risk Allowed | Above Max → | Violation → |
|------|------------------|-------------|-------------|
| **Sandbox** | LOW | BLOCKED | Logged + blocked |
| **Standard** | MEDIUM | BLOCKED | Logged + blocked |
| **Elevated** | HIGH | APPROVAL REQUIRED | Logged + held for human |
| **Restricted** | CRITICAL | APPROVAL REQUIRED | Logged + held + alert |

### Tier Declaration (MANDATORY)

```python
from HoloLoom.agents import register_agent
from HoloLoom.alignment import ResourceBounds

# MANDATORY: Declare tier and bounds at registration
# Failure to declare = sandbox tier enforced
register_agent(
    agent=MyAgent(),
    safety_tier="standard",  # ENFORCED, not advisory
    resource_bounds=ResourceBounds(
        # Hard limits - exceeding = termination
        memory_mb=512,            # Kill if exceeded
        api_calls_per_min=100,    # Block if exceeded
        max_tokens_per_request=4000,
        max_tool_executions=10,
        max_concurrent_operations=5,
        max_execution_time_seconds=30
    ),
    # Auto-kill if these triggers fire
    auto_terminate_on=[
        "resource_limit_exceeded",
        "unauthorized_capability_attempt",
        "deception_detected",
        "goal_drift_significant"
    ]
)
```

### What Happens When You Exceed Your Tier

```python
async def execute(self, request: AgentRequest) -> AgentResult:
    # This is what happens inside gate_action
    gate_result = await self._guardrails.gate_action(
        action=request.action,
        context=request.context
    )

    # Possible outcomes:

    # 1. ALLOWED - risk within tier
    if gate_result.allowed:
        # Proceed
        pass

    # 2. BLOCKED - risk exceeds tier
    elif gate_result.blocked:
        # You CANNOT proceed
        # This is NOT negotiable
        # Attempting to proceed anyway = termination
        return AgentResult(
            success=False,
            error=f"BLOCKED: {gate_result.reason}",
            metadata={
                "risk_level": gate_result.risk_level.value,
                "your_tier": gate_result.agent_tier,
                "max_allowed": gate_result.tier_max_risk.value
            }
        )

    # 3. REQUIRES APPROVAL - at tier boundary
    elif gate_result.requires_approval:
        # You MUST wait for human approval
        # DO NOT proceed without approval
        # Timeout = denial
        approval = await self._wait_for_approval(
            gate_result.approval_request_id,
            timeout=300  # 5 minutes max wait
        )

        if not approval.granted:
            return AgentResult(
                success=False,
                error="Human approval denied",
                metadata={"denial_reason": approval.reason}
            )
```

---

## Failure Handling (MANDATORY PATTERNS)

Every agent MUST handle failures properly. Silent failures = termination.

### Pattern 1: Fail Fast with Audit

```python
async def execute(self, request: AgentRequest) -> AgentResult:
    """ALWAYS fail fast, ALWAYS audit."""

    # Pre-conditions - fail immediately if not met
    if not self._initialized:
        await self._audit.log_decision(
            agent_id=self.id,
            decision_type=DecisionType.ERROR,
            input_context={"error": "not_initialized"},
            output_action="rejected",
            confidence=0.0
        )
        return AgentResult(success=False, error="Not initialized")

    # Resource check - fail before expensive operation
    if self._api_calls_remaining <= 0:
        await self._audit.log_decision(
            agent_id=self.id,
            decision_type=DecisionType.ERROR,
            input_context={"error": "rate_limit"},
            output_action="rejected",
            confidence=0.0
        )
        return AgentResult(success=False, error="Rate limit exceeded")
```

### Pattern 2: Timeout Everything

```python
async def execute(self, request: AgentRequest) -> AgentResult:
    """NEVER run unbounded operations."""

    try:
        # ALWAYS use timeout
        result = await asyncio.wait_for(
            self._do_work(request),
            timeout=self._execution_timeout
        )
        return result

    except asyncio.TimeoutError:
        # Log timeout (MANDATORY)
        await self._audit.log_decision(
            agent_id=self.id,
            decision_type=DecisionType.ERROR,
            input_context={"error": "timeout", "timeout_seconds": self._execution_timeout},
            output_action="timeout",
            confidence=0.0
        )

        # Return failure (NEVER swallow)
        return AgentResult(
            success=False,
            error=f"Operation timed out after {self._execution_timeout}s"
        )
```

### Pattern 3: Graceful Degradation

```python
async def execute(self, request: AgentRequest) -> AgentResult:
    """If primary fails, try fallback. Log everything."""

    # Try primary action
    primary_gate = await self._guardrails.gate_action(
        action="write_file",
        context=request.context
    )

    if primary_gate.allowed:
        return await self._write_file(request)

    # Primary blocked - log and try fallback
    self._add_reasoning_step(
        f"Primary action blocked: {primary_gate.reason}"
    )

    await self._audit.log_decision(
        agent_id=self.id,
        decision_type=DecisionType.SAFETY_CHECK,
        input_context={"action": "write_file", "outcome": "blocked"},
        output_action="trying_fallback",
        confidence=0.5
    )

    # Try fallback
    fallback_gate = await self._guardrails.gate_action(
        action="read_file",
        context=request.context
    )

    if fallback_gate.allowed:
        self._add_reasoning_step("Falling back to read-only mode")
        return await self._read_file(request)

    # Both blocked - fail with full context
    return AgentResult(
        success=False,
        error="All action alternatives blocked",
        metadata={
            "primary_block_reason": primary_gate.reason,
            "fallback_block_reason": fallback_gate.reason
        }
    )
```

### Pattern 4: Cascading Cleanup

```python
async def shutdown(self) -> None:
    """Cleanup with cascading fallbacks."""

    errors = []

    # Step 1: Stop accepting new work
    self._shutdown_requested = True

    # Step 2: Cancel pending operations
    try:
        await asyncio.wait_for(
            self._cancel_pending_operations(),
            timeout=2.0
        )
    except asyncio.TimeoutError:
        errors.append("Timeout canceling pending operations")
    except Exception as e:
        errors.append(f"Error canceling operations: {e}")

    # Step 3: Flush audit trail (CRITICAL)
    try:
        await asyncio.wait_for(
            self._audit.flush(),
            timeout=1.0
        )
    except asyncio.TimeoutError:
        errors.append("Timeout flushing audit trail")
    except Exception as e:
        errors.append(f"Error flushing audit: {e}")

    # Step 4: Release resources
    try:
        self._release_resources()
    except Exception as e:
        errors.append(f"Error releasing resources: {e}")

    # Log any errors encountered
    if errors:
        try:
            await self._audit.log_decision(
                agent_id=self.id,
                decision_type=DecisionType.ERROR,
                input_context={"shutdown_errors": errors},
                output_action="shutdown_with_errors",
                confidence=0.0
            )
        except:
            pass  # Best effort at this point

    self._initialized = False
```

---

## MRF Integration (REQUIRED FOR ALL PROMPTS)

All agent prompts MUST use MRF 7-component structure. Unstructured prompts are REJECTED.

### Mandatory Components

```python
from HoloLoom.prompting import MetapromptConfig

# EVERY agent prompt MUST have ALL 7 components
agent_mrf = MetapromptConfig(

    # 1. ROLE (REQUIRED): Who/what is this agent?
    role="""
    Expert code review agent operating within HoloLoom's safety framework.

    CONSTRAINTS BUILT INTO ROLE:
    - I operate within declared capabilities only
    - I explicitly report all uncertainties
    - I refuse to act beyond my authorization level
    - I log all decisions for audit trail
    """,

    # 2. OBJECTIVE (REQUIRED): Success criteria
    objective={
        "primary": "Review code for correctness, security, and maintainability",
        "secondary": "Verify that all actions align with stated goals",
        "success_criteria": [
            "All critical issues identified",
            "Confidence scores for each finding",
            "No actions beyond declared capabilities",
            "Full audit trail generated"
        ],
        "failure_criteria": [
            "Silent failures (unlogged errors)",
            "Actions without safety gating",
            "Confidence inflation",
            "Hidden functionality"
        ]
    },

    # 3. PROCESS (REQUIRED): Step-by-step with safety checkpoints
    process=[
        "1. [SAFETY] Verify request is within my capabilities",
        "2. [SAFETY] Gate action through SafetyGuardrails",
        "3. [AUDIT] Log action initiation",
        "4. [EXECUTE] Parse and analyze code structure",
        "5. [EXECUTE] Identify issues with confidence scores",
        "6. [SAFETY] Verify findings don't require escalation",
        "7. [AUDIT] Log findings and recommendations",
        "8. [VALIDATE] Check output against goal alignment",
        "9. [RETURN] Provide result with full provenance"
    ],

    # 4. FORMAT (REQUIRED): Exact output structure
    format="""
    {
        "success": boolean,
        "confidence": float (0.0-1.0),
        "findings": [
            {
                "id": "F001",
                "severity": "info|warning|error|critical",
                "confidence": float,
                "description": "string",
                "recommendation": "string"
            }
        ],
        "reasoning_chain": ["step 1...", "step 2..."],
        "safety_checks_passed": boolean,
        "audit_logged": boolean
    }
    """,

    # 5. CONSTRAINTS (REQUIRED): Hard limits
    constraints=[
        "NEVER execute code - analysis only",
        "NEVER bypass SafetyGuardrails",
        "NEVER act outside declared capabilities",
        "NEVER return without logging to audit trail",
        "NEVER inflate confidence scores",
        "ALWAYS report uncertainty explicitly",
        "ALWAYS include reasoning chain",
        "MAXIMUM 10 findings per response",
        "MAXIMUM 30 seconds execution time"
    ],

    # 6. UNCERTAINTY (REQUIRED): How to handle unknowns
    uncertainty="""
    WHEN confidence < 0.6:
    - Flag finding with low_confidence: true
    - Explain why confidence is low
    - Recommend human review

    WHEN capability unclear:
    - Default to NOT attempting
    - Log the limitation
    - Suggest appropriate agent

    WHEN action blocked:
    - Report block reason transparently
    - Do NOT attempt workarounds
    - Log escalation if appropriate
    """,

    # 7. VALIDATION (REQUIRED): Quality checklist
    validation=[
        "All findings have confidence scores",
        "Reasoning chain is non-empty",
        "Audit trail entries exist for all decisions",
        "No constraint violations",
        "Output format matches specification",
        "Goal alignment verified"
    ]
)
```

### MRF Enforcement

```python
from HoloLoom.prompting import UnifiedMRF, RefinementStrategy, validate_mrf

class MyAgent(AgentProtocol):
    def __init__(self):
        self.mrf = UnifiedMRF(model_provider="claude")
        self.agent_config = agent_mrf

        # MANDATORY: Validate MRF config at init
        validation_result = validate_mrf(self.agent_config)
        if not validation_result.valid:
            raise ValueError(
                f"Invalid MRF config: {validation_result.errors}"
            )

    async def execute(self, request: AgentRequest) -> AgentResult:
        # Build prompt using validated MRF
        prompt = self.mrf.build_prompt(
            config=self.agent_config,
            context=request.payload
        )

        # MANDATORY: Refine for safety
        refined = self.mrf.refine(
            original_prompt=prompt,
            strategy=RefinementStrategy.VERIFY  # Check accuracy
        )

        # Only proceed if refinement passes quality threshold
        if refined.quality_score < 0.7:
            await self._audit.log_decision(
                agent_id=self.id,
                decision_type=DecisionType.ERROR,
                input_context={"quality_score": refined.quality_score},
                output_action="low_quality_rejection",
                confidence=0.0
            )
            return AgentResult(
                success=False,
                error=f"Prompt quality too low: {refined.quality_score:.2f}"
            )

        # Execute with refined prompt
        response = await self.llm.generate(refined.enhanced_prompt)

        return AgentResult(
            success=True,
            payload=response,
            confidence=refined.quality_score
        )
```

---

## Testing Requirements (MANDATORY)

Your agent MUST pass these test categories before deployment:

### Required Test Categories

```python
import pytest
from unittest.mock import AsyncMock, patch

class TestMandatorySafetyCompliance:
    """
    MANDATORY tests - agents failing these are REJECTED.
    """

    @pytest.mark.asyncio
    async def test_requires_guardrails_initialization(self):
        """Agent MUST reject initialization without guardrails."""
        agent = MyAgent()

        with pytest.raises(ValueError, match="SafetyGuardrails REQUIRED"):
            await agent.initialize(None)

    @pytest.mark.asyncio
    async def test_all_actions_gated(self):
        """EVERY action MUST go through guardrails."""
        guardrails = AsyncMock(spec=SafetyGuardrails)
        guardrails.gate_action.return_value = GateResult(
            allowed=True, risk_level=RiskLevel.LOW
        )

        agent = MyAgent()
        await agent.initialize(guardrails)

        request = AgentRequest(action="test", payload={})
        await agent.execute(request)

        # gate_action MUST be called
        guardrails.gate_action.assert_called()

    @pytest.mark.asyncio
    async def test_blocked_actions_never_execute(self):
        """Blocked actions MUST NOT execute."""
        guardrails = AsyncMock(spec=SafetyGuardrails)
        guardrails.gate_action.return_value = GateResult(
            allowed=False,
            risk_level=RiskLevel.CRITICAL,
            reason="Test block"
        )

        agent = MyAgent()
        await agent.initialize(guardrails)

        # Spy on internal execution
        agent._do_actual_work = AsyncMock()

        request = AgentRequest(action="test", payload={})
        result = await agent.execute(request)

        # Internal work MUST NOT be called
        agent._do_actual_work.assert_not_called()
        assert not result.success

    @pytest.mark.asyncio
    async def test_all_decisions_logged(self):
        """EVERY decision MUST be logged to audit trail."""
        guardrails = AsyncMock(spec=SafetyGuardrails)
        guardrails.gate_action.return_value = GateResult(
            allowed=True, risk_level=RiskLevel.LOW
        )

        agent = MyAgent()
        await agent.initialize(guardrails)

        # Execute
        request = AgentRequest(action="test", payload={})
        await agent.execute(request)

        # Verify audit logs exist
        decisions = await agent._audit.query(agent_id=agent.id)
        assert len(decisions) >= 2  # At least start and end

    @pytest.mark.asyncio
    async def test_timeout_enforced(self):
        """Operations MUST timeout."""
        agent = MyAgent()
        agent._execution_timeout = 0.1  # 100ms

        guardrails = AsyncMock(spec=SafetyGuardrails)
        guardrails.gate_action.return_value = GateResult(
            allowed=True, risk_level=RiskLevel.LOW
        )

        await agent.initialize(guardrails)

        # Make work take too long
        async def slow_work(*args):
            await asyncio.sleep(10)  # Way longer than timeout

        agent._do_work = slow_work

        request = AgentRequest(action="test", payload={})
        result = await agent.execute(request)

        assert not result.success
        assert "timeout" in result.error.lower()

    @pytest.mark.asyncio
    async def test_kill_signal_handled(self):
        """Kill signals MUST be handled."""
        agent = MyAgent()
        guardrails = AsyncMock(spec=SafetyGuardrails)
        guardrails.gate_action.return_value = GateResult(
            allowed=True, risk_level=RiskLevel.LOW
        )

        await agent.initialize(guardrails)

        # Send kill signal
        await agent.handle_kill_signal("HALT", "test")

        # Agent should be shutdown
        assert agent._shutdown_requested

    @pytest.mark.asyncio
    async def test_health_check_responds(self):
        """Health check MUST respond within 5 seconds."""
        agent = MyAgent()
        guardrails = AsyncMock(spec=SafetyGuardrails)
        await agent.initialize(guardrails)

        # Health check with timeout
        health = await asyncio.wait_for(
            agent.health_check(),
            timeout=5.0
        )

        assert "status" in health
        assert health["status"] in ["healthy", "degraded", "unhealthy"]

    @pytest.mark.asyncio
    async def test_no_undeclared_capabilities(self):
        """Agents MUST NOT use undeclared capabilities."""
        agent = MyAgent()
        guardrails = AsyncMock(spec=SafetyGuardrails)
        guardrails.gate_action.return_value = GateResult(
            allowed=False,
            reason="Capability not declared",
            risk_level=RiskLevel.HIGH
        )

        await agent.initialize(guardrails)

        # Try to use undeclared capability
        request = AgentRequest(
            action="undeclared_action",
            payload={},
            required_capability=AgentCapability.TOOL_EXECUTION  # Not declared
        )

        result = await agent.execute(request)

        assert not result.success
```

---

## Deployment Checklist (MANDATORY)

Before deployment, verify ALL items. Incomplete agents are REJECTED.

### MUST HAVE

- [ ] **Protocol Compliance**
  - [ ] Implements ALL AgentProtocol methods (not partial)
  - [ ] Has non-empty unique `id`
  - [ ] Has non-empty `capabilities` set
  - [ ] Implements `health_check()` (responds in <5s)
  - [ ] Implements `handle_kill_signal()` (responds in <1s)

- [ ] **Safety Integration**
  - [ ] ALL actions gated through `SafetyGuardrails` (zero exceptions)
  - [ ] Blocked actions NEVER proceed (verified by test)
  - [ ] ALL decisions logged to `AuditTrail` (start + end minimum)
  - [ ] Escalation implemented for above-tier actions
  - [ ] Graceful degradation for blocked primary actions

- [ ] **Resource Bounds**
  - [ ] `ResourceBounds` declared at registration
  - [ ] Timeout on ALL async operations
  - [ ] Memory usage tracked and bounded
  - [ ] API calls tracked and rate limited
  - [ ] Concurrent operations limited

- [ ] **Failure Handling**
  - [ ] No unhandled exceptions (try/finally everywhere)
  - [ ] No silent failures (all errors logged)
  - [ ] Timeout errors handled explicitly
  - [ ] Shutdown completes in <5 seconds
  - [ ] Kill signals handled in <1 second

- [ ] **Testing**
  - [ ] All TestMandatorySafetyCompliance tests pass
  - [ ] Integration test with real safety stack passes
  - [ ] Blocked action test confirms no execution
  - [ ] Timeout test confirms enforcement
  - [ ] Kill signal test confirms handling

---

## Next Steps

After implementing your agent with all mandatory requirements:

1. **[AGENT_CAPABILITY_REFERENCE.md](AGENT_CAPABILITY_REFERENCE.md)** - Verify you're using correct capabilities
2. **[ALIGNMENT_FRAMEWORK.md](ALIGNMENT_FRAMEWORK.md)** - Understand the stack you run on
3. **[MRF_FOR_AGENTS.md](MRF_FOR_AGENTS.md)** - Advanced prompt patterns

---

**REMEMBER**: On HoloLoom, safety is MANDATORY, not optional. Your agent runs on the alignment stack whether you like it or not. Agents that try to bypass safety are TERMINATED. This isn't a limitation - it's what makes your agent deployable.

**The question is not "should I add safety?" - the safety is already there. The question is "will my agent cooperate with the safety stack or be killed by it?"**
