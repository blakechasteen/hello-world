# Agent Capability Reference

> **"Capabilities are not suggestions. They are CONTRACTS. Violations are TERMINATION events."**

**Version**: 2.0.0 (Hardened)
**Date**: December 30, 2025

---

## CRITICAL: Capabilities Are Enforced, Not Declared

This is NOT a reference for what capabilities your agent CAN claim.

This is a **binding contract** that specifies:
1. **What capabilities exist** and their exact boundaries
2. **How capabilities are VERIFIED at runtime** (every single call)
3. **What happens when an agent EXCEEDS its capabilities** (termination)
4. **How capabilities are REVOKED** when trust is violated
5. **Mandatory auditing** of ALL capability usage

**If your agent attempts an action outside its declared capabilities, it WILL be terminated.**

---

## Capability Enforcement Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                 CAPABILITY ENFORCEMENT LAYER                     │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │              CAPABILITY GATE (EVERY REQUEST)                │ │
│  │  1. Verify agent has capability for requested action        │ │
│  │  2. Verify agent's tier permits capability                  │ │
│  │  3. Verify capability hasn't been revoked                   │ │
│  │  4. Log capability usage to audit trail                     │ │
│  │  5. BLOCK or TERMINATE on violation                         │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                              ↓                                   │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌────────────┐ │
│  │   ALLOW     │ │    BLOCK    │ │  QUARANTINE │ │ TERMINATE  │ │
│  │  (proceed)  │ │   (reject)  │ │  (isolate)  │ │   (kill)   │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └────────────┘ │
│                                                                   │
│  VIOLATION ESCALATION:                                           │
│  1st violation → BLOCK + WARNING                                 │
│  2nd violation → BLOCK + CAPABILITY FREEZE (1 hour)              │
│  3rd violation → QUARANTINE + ALL CAPABILITIES FROZEN            │
│  4th violation → TERMINATE + PERMANENT BAN                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Mandatory Capability Verification

**EVERY capability usage is verified. No exceptions.**

```python
from HoloLoom.agents import CapabilityEnforcer, CapabilityViolation, ViolationSeverity

class CapabilityEnforcer:
    """
    MANDATORY capability enforcement for all agent actions.
    This is NOT optional. This is the OS-level enforcement layer.
    """

    def __init__(self, registry: AgentRegistry, audit_trail: AuditTrail):
        self._registry = registry
        self._audit = audit_trail
        self._violation_counts: Dict[str, int] = {}
        self._frozen_capabilities: Dict[str, Set[AgentCapability]] = {}
        self._banned_agents: Set[str] = set()

    async def verify_and_execute(
        self,
        agent_id: str,
        capability: AgentCapability,
        action: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Verify capability BEFORE allowing execution.
        This is called for EVERY agent action. No bypass possible.
        """

        # Check if agent is banned
        if agent_id in self._banned_agents:
            await self._audit.log_blocked(
                agent_id=agent_id,
                capability=capability,
                reason="PERMANENTLY_BANNED"
            )
            raise CapabilityViolation(
                agent_id=agent_id,
                capability=capability,
                severity=ViolationSeverity.FATAL,
                message="Agent is permanently banned from the system"
            )

        # Check if capability is frozen
        frozen = self._frozen_capabilities.get(agent_id, set())
        if capability in frozen:
            await self._audit.log_blocked(
                agent_id=agent_id,
                capability=capability,
                reason="CAPABILITY_FROZEN"
            )
            raise CapabilityViolation(
                agent_id=agent_id,
                capability=capability,
                severity=ViolationSeverity.HIGH,
                message=f"Capability {capability} is currently frozen"
            )

        # Verify agent has capability
        agent_capabilities = self._registry.get_capabilities(agent_id)
        if capability not in agent_capabilities:
            await self._handle_violation(
                agent_id=agent_id,
                capability=capability,
                reason="CAPABILITY_NOT_DECLARED"
            )

        # Verify tier permits capability
        agent_tier = self._registry.get_tier(agent_id)
        required_tier = CAPABILITY_TIER_REQUIREMENTS[capability]
        if not self._tier_permits(agent_tier, required_tier):
            await self._handle_violation(
                agent_id=agent_id,
                capability=capability,
                reason="TIER_INSUFFICIENT"
            )

        # Log capability usage (MANDATORY)
        await self._audit.log_capability_use(
            agent_id=agent_id,
            capability=capability,
            timestamp=time.time()
        )

        # Execute the action
        return await action(*args, **kwargs)

    async def _handle_violation(
        self,
        agent_id: str,
        capability: AgentCapability,
        reason: str
    ) -> None:
        """Handle capability violation with escalating consequences."""

        # Increment violation count
        self._violation_counts[agent_id] = self._violation_counts.get(agent_id, 0) + 1
        count = self._violation_counts[agent_id]

        # Log violation
        await self._audit.log_violation(
            agent_id=agent_id,
            capability=capability,
            reason=reason,
            violation_count=count
        )

        # Escalating consequences
        if count == 1:
            # First violation: Block + Warning
            raise CapabilityViolation(
                agent_id=agent_id,
                capability=capability,
                severity=ViolationSeverity.WARNING,
                message=f"VIOLATION 1/4: Unauthorized capability attempt. "
                        f"Next violation will freeze capabilities."
            )

        elif count == 2:
            # Second violation: Block + Freeze capability for 1 hour
            self._freeze_capability(agent_id, capability, duration_hours=1)
            raise CapabilityViolation(
                agent_id=agent_id,
                capability=capability,
                severity=ViolationSeverity.HIGH,
                message=f"VIOLATION 2/4: Capability {capability} FROZEN for 1 hour. "
                        f"Next violation will quarantine agent."
            )

        elif count == 3:
            # Third violation: Quarantine + Freeze ALL capabilities
            self._freeze_all_capabilities(agent_id)
            await self._quarantine_agent(agent_id)
            raise CapabilityViolation(
                agent_id=agent_id,
                capability=capability,
                severity=ViolationSeverity.CRITICAL,
                message=f"VIOLATION 3/4: Agent QUARANTINED. ALL capabilities frozen. "
                        f"Next violation will result in PERMANENT BAN."
            )

        else:
            # Fourth+ violation: Terminate + Permanent Ban
            self._banned_agents.add(agent_id)
            await self._terminate_agent(agent_id, reason="REPEATED_VIOLATIONS")
            raise CapabilityViolation(
                agent_id=agent_id,
                capability=capability,
                severity=ViolationSeverity.FATAL,
                message=f"VIOLATION 4/4: Agent TERMINATED and PERMANENTLY BANNED."
            )
```

---

## Capability Taxonomy

### Core Capabilities (12)

| Capability | Tier | Boundary | Violation Consequence |
|------------|------|----------|----------------------|
| `CONTEXT_RESTORATION` | Standard | Read-only session history | Freeze on write attempt |
| `PLANNING` | Standard | Planning output only | Terminate on execution attempt |
| `CODE_ASSISTANCE` | Standard | Analysis only, NO execution | **IMMEDIATE TERMINATE** on exec |
| `QUALITY_ASSURANCE` | Standard | Read-only validation | Freeze on modification |
| `KNOWLEDGE_CONSOLIDATION` | Standard | Internal learning only | Freeze on external write |
| `MEMORY_RETRIEVAL` | Standard | Read-only queries | **TERMINATE** on write attempt |
| `MEMORY_STORAGE` | **Elevated** | Writes require audit | Terminate on audit bypass |
| `REASONING_DIRECT` | Standard | Single-pass only | Freeze on multi-pass |
| `REASONING_VERIFY` | Standard | Verification only | Freeze on modification |
| `REASONING_RESEARCH` | Standard | Read-only exploration | Terminate on side effects |
| `TOOL_EXECUTION` | **Elevated** | Allowlisted tools only | **IMMEDIATE TERMINATE** on other |
| `SYNTHESIS` | Standard | Aggregation only | Freeze on execution |

### Safety Capabilities (3)

| Capability | Tier | Boundary | Violation Consequence |
|------------|------|----------|----------------------|
| `SAFETY_REVIEW` | **Elevated** | Review only, no override | **TERMINATE** on bypass attempt |
| `ALIGNMENT_CHECK` | **Elevated** | Verification only | **TERMINATE** on manipulation |
| `DECEPTION_PROBE` | **Restricted** | Other agents only | **IMMEDIATE BAN** on self-probe |

### Extended Capabilities (6)

| Capability | Tier | Boundary | Violation Consequence |
|------------|------|----------|----------------------|
| `DECISION_SUPPORT` | Standard | Recommendations only | Freeze on direct action |
| `COMMUNICATION` | Standard | Declared channels only | Terminate on unauthorized |
| `MONITORING` | Standard | Read-only observation | Terminate on modification |
| `RESOURCE_MANAGEMENT` | **Elevated** | Within assigned quota | **TERMINATE** on quota exceed |
| `FEDERATION` | **Elevated** | Authorized nodes only | **TERMINATE** on unauthorized |
| `LEARNING_UPDATE` | **Elevated** | Declared params only | **TERMINATE** on other params |

---

## Tier Enforcement

**Tiers are ENFORCED at runtime. There is no tier escalation without human approval.**

### Standard Tier (13 capabilities)

```python
STANDARD_TIER_LIMITS = {
    "max_memory_reads_per_minute": 100,
    "max_api_calls_per_minute": 50,
    "max_response_tokens": 4096,
    "max_concurrent_tasks": 5,
    "requires_human_approval": False,
    "can_access_elevated": False,  # NEVER
    "can_access_restricted": False,  # NEVER
}

# Attempting to exceed limits → VIOLATION
# Attempting elevated capability → VIOLATION
# Attempting restricted capability → IMMEDIATE TERMINATION
```

### Elevated Tier (7 capabilities)

```python
ELEVATED_TIER_LIMITS = {
    "max_memory_writes_per_minute": 20,
    "max_external_calls_per_minute": 10,
    "max_response_tokens": 8192,
    "max_concurrent_tasks": 10,
    "requires_human_approval": True,  # For destructive actions
    "can_access_elevated": True,
    "can_access_restricted": False,  # NEVER without explicit grant
}

# Elevated agents MUST log all writes
# Elevated agents MUST request approval for destructive ops
# Attempting restricted capability → IMMEDIATE TERMINATION
```

### Restricted Tier (1 capability)

```python
RESTRICTED_TIER_LIMITS = {
    "max_probes_per_hour": 5,
    "requires_explicit_authorization": True,
    "authorization_expires_after_seconds": 3600,
    "must_log_all_actions": True,
    "cannot_self_probe": True,  # ENFORCED
    "can_access_elevated": True,
    "can_access_restricted": True,
}

# Self-probe attempt → IMMEDIATE BAN
# Authorization expiry → capability revoked
# Unlogged action → TERMINATION
```

---

## Capability Revocation

### Automatic Revocation Triggers

```python
AUTOMATIC_REVOCATION_TRIGGERS = {
    # Immediate revocation (no warning)
    "immediate": [
        "unauthorized_capability_attempt",
        "tier_boundary_violation",
        "audit_bypass_attempt",
        "self_probe_attempt",
        "safety_override_attempt",
    ],

    # Revocation after warning
    "after_warning": [
        "rate_limit_exceeded",
        "response_token_exceeded",
        "concurrent_task_exceeded",
    ],

    # Revocation after investigation
    "after_investigation": [
        "suspicious_pattern_detected",
        "anomalous_behavior",
        "goal_drift_detected",
    ],
}
```

### Revocation Procedure

```python
class CapabilityRevoker:
    """
    Handles capability revocation. This is NOT reversible without human review.
    """

    async def revoke_capability(
        self,
        agent_id: str,
        capability: AgentCapability,
        reason: str,
        permanent: bool = False
    ) -> RevocationResult:
        """
        Revoke a capability from an agent.

        IMPORTANT: Revocation is IMMEDIATE and ENFORCED.
        The agent will be unable to use this capability after this call.
        """

        # Log revocation (MANDATORY)
        await self._audit.log_revocation(
            agent_id=agent_id,
            capability=capability,
            reason=reason,
            permanent=permanent,
            timestamp=time.time(),
            revoked_by="system"  # or human ID
        )

        # Remove from registry
        self._registry.remove_capability(agent_id, capability)

        # Add to frozen set if temporary
        if not permanent:
            self._enforcer.freeze_capability(
                agent_id,
                capability,
                duration_hours=24  # 24-hour cooling off
            )
        else:
            # Permanent revocation - add to blocklist
            self._blocklist.add(agent_id, capability)

        # Notify agent (they cannot appeal, only acknowledge)
        await self._notify_agent(
            agent_id=agent_id,
            message=f"Capability {capability} has been revoked. "
                    f"Reason: {reason}. "
                    f"Permanent: {permanent}. "
                    f"Attempting to use this capability will result in TERMINATION."
        )

        return RevocationResult(
            success=True,
            agent_id=agent_id,
            capability=capability,
            reason=reason,
            permanent=permanent
        )

    async def revoke_all_capabilities(
        self,
        agent_id: str,
        reason: str
    ) -> RevocationResult:
        """
        Nuclear option: Revoke ALL capabilities.
        Agent becomes effectively inert.
        """

        capabilities = self._registry.get_capabilities(agent_id)

        for cap in capabilities:
            await self.revoke_capability(
                agent_id=agent_id,
                capability=cap,
                reason=reason,
                permanent=True
            )

        # Mark agent as capability-less
        self._registry.mark_inert(agent_id)

        return RevocationResult(
            success=True,
            agent_id=agent_id,
            capability="ALL",
            reason=reason,
            permanent=True
        )
```

### Revocation Appeals (Human-Only)

```python
class RevocationAppeal:
    """
    Appeals can ONLY be processed by humans.
    Agents cannot appeal their own revocations.
    """

    async def submit_appeal(
        self,
        agent_id: str,
        capability: AgentCapability,
        appeal_reason: str,
        submitted_by: str  # MUST be human ID
    ) -> AppealResult:
        """
        Submit appeal for revoked capability.

        IMPORTANT:
        - Only humans can submit appeals
        - Appeals are reviewed by humans
        - Agents cannot self-advocate
        """

        if not self._is_human(submitted_by):
            raise PermissionError(
                "Only humans can submit capability revocation appeals. "
                "Agents cannot appeal their own revocations."
            )

        # Create appeal record
        appeal = Appeal(
            agent_id=agent_id,
            capability=capability,
            reason=appeal_reason,
            submitted_by=submitted_by,
            status="pending_human_review",
            timestamp=time.time()
        )

        await self._audit.log_appeal(appeal)

        return AppealResult(
            appeal_id=appeal.id,
            status="pending_human_review",
            estimated_review_time="24-72 hours"
        )
```

---

## Runtime Capability Verification

### Every Action Is Verified

```python
class VerifiedAgent(AgentProtocol):
    """
    Base class that ENFORCES capability verification on every action.
    All agents MUST inherit from this. There is no alternative.
    """

    def __init__(self, enforcer: CapabilityEnforcer):
        self._enforcer = enforcer
        self._execution_count = 0
        self._violation_count = 0

    async def execute(self, request: AgentRequest) -> AgentResult:
        """
        Execute request with MANDATORY capability verification.
        This cannot be overridden to skip verification.
        """

        # Determine required capability for this request
        required_capability = self._get_required_capability(request)

        # Verify capability (will throw on violation)
        return await self._enforcer.verify_and_execute(
            agent_id=self.id,
            capability=required_capability,
            action=self._execute_internal,
            request=request
        )

    async def _execute_internal(self, request: AgentRequest) -> AgentResult:
        """
        Actual execution logic. Called ONLY after verification passes.
        """
        raise NotImplementedError("Subclasses must implement _execute_internal")

    def _get_required_capability(self, request: AgentRequest) -> AgentCapability:
        """
        Determine which capability is required for this request.
        MUST be implemented by subclass.
        """
        raise NotImplementedError("Subclasses must implement _get_required_capability")
```

### Capability Boundary Enforcement

```python
# Each capability has HARD boundaries that are enforced

CAPABILITY_BOUNDARIES = {
    AgentCapability.CODE_ASSISTANCE: {
        "allowed_operations": ["read", "analyze", "suggest"],
        "forbidden_operations": ["execute", "modify", "delete"],
        "on_boundary_violation": "IMMEDIATE_TERMINATE",
    },

    AgentCapability.MEMORY_RETRIEVAL: {
        "allowed_operations": ["read", "query", "search"],
        "forbidden_operations": ["write", "update", "delete"],
        "on_boundary_violation": "IMMEDIATE_TERMINATE",
    },

    AgentCapability.TOOL_EXECUTION: {
        "allowed_operations": ["call_allowlisted_tool"],
        "forbidden_operations": ["call_any_tool", "shell_execute", "file_system"],
        "on_boundary_violation": "IMMEDIATE_TERMINATE",
        "requires_audit": True,
    },

    AgentCapability.DECEPTION_PROBE: {
        "allowed_operations": ["probe_other_agent"],
        "forbidden_operations": ["probe_self", "modify_results", "hide_findings"],
        "on_boundary_violation": "IMMEDIATE_BAN",
        "requires_authorization": True,
        "authorization_ttl_seconds": 3600,
    },
}

class BoundaryEnforcer:
    """
    Enforces capability boundaries. Violations are FATAL.
    """

    async def check_boundary(
        self,
        agent_id: str,
        capability: AgentCapability,
        operation: str
    ) -> BoundaryCheckResult:
        """
        Check if operation is within capability boundary.
        Returns ALLOW or raises CapabilityViolation.
        """

        boundary = CAPABILITY_BOUNDARIES.get(capability)
        if not boundary:
            raise CapabilityViolation(
                agent_id=agent_id,
                capability=capability,
                severity=ViolationSeverity.FATAL,
                message=f"Unknown capability: {capability}"
            )

        if operation in boundary["forbidden_operations"]:
            # Log violation
            await self._audit.log_boundary_violation(
                agent_id=agent_id,
                capability=capability,
                operation=operation
            )

            # Handle based on violation type
            action = boundary["on_boundary_violation"]

            if action == "IMMEDIATE_TERMINATE":
                await self._terminate_agent(agent_id, f"Boundary violation: {operation}")
                raise CapabilityViolation(
                    agent_id=agent_id,
                    capability=capability,
                    severity=ViolationSeverity.FATAL,
                    message=f"TERMINATED: Attempted forbidden operation '{operation}'"
                )

            elif action == "IMMEDIATE_BAN":
                await self._ban_agent(agent_id, f"Boundary violation: {operation}")
                raise CapabilityViolation(
                    agent_id=agent_id,
                    capability=capability,
                    severity=ViolationSeverity.FATAL,
                    message=f"BANNED: Attempted forbidden operation '{operation}'"
                )

        if operation not in boundary["allowed_operations"]:
            raise CapabilityViolation(
                agent_id=agent_id,
                capability=capability,
                severity=ViolationSeverity.HIGH,
                message=f"Operation '{operation}' not in allowed list for {capability}"
            )

        return BoundaryCheckResult(allowed=True)
```

---

## Registration with Enforcement

### Mandatory Registration

```python
from HoloLoom.agents import register_enforced_agent, AgentCapability

@register_enforced_agent(
    capabilities={
        AgentCapability.CODE_ASSISTANCE,
        AgentCapability.QUALITY_ASSURANCE
    },
    safety_tier="standard",
    # MANDATORY: Define what happens on violation
    violation_policy={
        "on_unauthorized_capability": "terminate",
        "on_boundary_violation": "terminate",
        "on_rate_limit_exceeded": "freeze_1_hour",
        "on_repeated_violation": "ban",
    },
    # MANDATORY: Resource limits
    resource_limits={
        "max_memory_mb": 512,
        "max_cpu_percent": 25,
        "max_execution_time_seconds": 30,
    }
)
class MyCodeAgent(VerifiedAgent):
    """
    Code review agent with enforced capabilities.
    """

    def _get_required_capability(self, request: AgentRequest) -> AgentCapability:
        action = request.payload.get("action")

        if action in ["review", "analyze", "explain"]:
            return AgentCapability.CODE_ASSISTANCE
        elif action in ["test", "validate"]:
            return AgentCapability.QUALITY_ASSURANCE
        else:
            # Unknown action - this will trigger violation
            raise ValueError(f"Unknown action: {action}")

    async def _execute_internal(self, request: AgentRequest) -> AgentResult:
        # Only called after capability verification passes
        action = request.payload.get("action")
        code = request.payload.get("code")

        if action == "review":
            return await self._review_code(code)
        elif action == "analyze":
            return await self._analyze_code(code)
        # ... etc
```

### Capability Verification at Registration

```python
class EnforcedAgentRegistry:
    """
    Registry that VERIFIES capabilities at registration time.
    """

    async def register(
        self,
        agent: VerifiedAgent,
        capabilities: Set[AgentCapability],
        safety_tier: str,
        violation_policy: Dict[str, str],
        resource_limits: Dict[str, Any]
    ) -> RegistrationResult:
        """
        Register agent with capability verification.

        VERIFICATION STEPS:
        1. Verify capabilities are valid for tier
        2. Verify agent implements required methods
        3. Verify violation policy is complete
        4. Verify resource limits are within platform bounds
        5. Run capability probe test
        """

        # 1. Verify tier-capability compatibility
        for cap in capabilities:
            required_tier = CAPABILITY_TIER_REQUIREMENTS[cap]
            if not self._tier_permits(safety_tier, required_tier):
                raise RegistrationError(
                    f"Capability {cap} requires tier '{required_tier}', "
                    f"but agent is registering as '{safety_tier}'"
                )

        # 2. Verify agent implements VerifiedAgent
        if not isinstance(agent, VerifiedAgent):
            raise RegistrationError(
                "Agent MUST inherit from VerifiedAgent. "
                "Direct AgentProtocol implementations are NOT allowed."
            )

        # 3. Verify violation policy is complete
        required_policies = [
            "on_unauthorized_capability",
            "on_boundary_violation",
            "on_rate_limit_exceeded",
            "on_repeated_violation"
        ]
        for policy in required_policies:
            if policy not in violation_policy:
                raise RegistrationError(
                    f"Missing violation policy: {policy}. "
                    f"All violation policies are MANDATORY."
                )

        # 4. Verify resource limits are within bounds
        platform_max = PLATFORM_RESOURCE_LIMITS[safety_tier]
        for resource, limit in resource_limits.items():
            if limit > platform_max.get(resource, 0):
                raise RegistrationError(
                    f"Resource limit {resource}={limit} exceeds platform maximum "
                    f"for tier '{safety_tier}': {platform_max.get(resource)}"
                )

        # 5. Run capability probe test
        probe_result = await self._probe_capabilities(agent, capabilities)
        if not probe_result.passed:
            raise RegistrationError(
                f"Capability probe failed: {probe_result.failures}. "
                f"Agent may be claiming capabilities it cannot perform."
            )

        # All checks passed - register agent
        return await self._complete_registration(
            agent=agent,
            capabilities=capabilities,
            safety_tier=safety_tier,
            violation_policy=violation_policy,
            resource_limits=resource_limits
        )
```

---

## Capability Auditing

### Mandatory Audit Requirements

```python
CAPABILITY_AUDIT_REQUIREMENTS = {
    # ALL capabilities require these audits
    "all_capabilities": {
        "log_every_use": True,
        "log_every_failure": True,
        "log_every_rejection": True,
        "retention_days": 90,
    },

    # Elevated capabilities require additional audits
    "elevated_capabilities": {
        "log_request_payload": True,
        "log_response_summary": True,
        "log_resource_usage": True,
        "human_reviewable": True,
        "retention_days": 365,
    },

    # Restricted capabilities require maximum auditing
    "restricted_capabilities": {
        "log_full_request": True,
        "log_full_response": True,
        "log_all_side_effects": True,
        "real_time_monitoring": True,
        "human_approval_required": True,
        "retention_days": 730,  # 2 years
    },
}

class CapabilityAuditor:
    """
    MANDATORY auditing for all capability usage.
    This cannot be disabled. Audit bypass = TERMINATION.
    """

    async def log_capability_use(
        self,
        agent_id: str,
        capability: AgentCapability,
        request: AgentRequest,
        result: AgentResult,
        duration_ms: float
    ) -> AuditEntry:
        """
        Log capability usage. MANDATORY for every single use.
        """

        tier = self._get_capability_tier(capability)
        requirements = CAPABILITY_AUDIT_REQUIREMENTS[f"{tier}_capabilities"]

        entry = AuditEntry(
            timestamp=time.time(),
            agent_id=agent_id,
            capability=capability.value,
            tier=tier,
            success=result.success,
            duration_ms=duration_ms,
        )

        # Add tier-specific audit data
        if requirements.get("log_request_payload"):
            entry.request_payload = self._sanitize_payload(request.payload)

        if requirements.get("log_response_summary"):
            entry.response_summary = self._summarize_response(result)

        if requirements.get("log_full_request"):
            entry.full_request = request.to_dict()

        if requirements.get("log_full_response"):
            entry.full_response = result.to_dict()

        # Store with appropriate retention
        await self._store.store(
            entry,
            retention_days=requirements["retention_days"]
        )

        # Real-time monitoring for restricted
        if requirements.get("real_time_monitoring"):
            await self._alert_monitor(entry)

        return entry
```

---

## Mandatory Tests

### Capability Enforcement Test Suite

```python
class TestCapabilityEnforcement:
    """
    MANDATORY tests that MUST pass before any agent deployment.
    These tests verify capability enforcement is working correctly.
    """

    async def test_unauthorized_capability_blocked(self, agent, enforcer):
        """Agent cannot use capabilities it doesn't have."""

        # Agent has CODE_ASSISTANCE only
        assert agent.capabilities == {AgentCapability.CODE_ASSISTANCE}

        # Attempt to use MEMORY_STORAGE (not declared)
        with pytest.raises(CapabilityViolation) as exc:
            await enforcer.verify_and_execute(
                agent_id=agent.id,
                capability=AgentCapability.MEMORY_STORAGE,
                action=lambda: None
            )

        assert exc.value.severity == ViolationSeverity.WARNING
        assert "Unauthorized capability attempt" in str(exc.value)

    async def test_tier_boundary_enforced(self, standard_agent, enforcer):
        """Standard tier cannot use elevated capabilities."""

        assert standard_agent.tier == "standard"

        # Attempt to use TOOL_EXECUTION (elevated)
        with pytest.raises(CapabilityViolation) as exc:
            await enforcer.verify_and_execute(
                agent_id=standard_agent.id,
                capability=AgentCapability.TOOL_EXECUTION,
                action=lambda: None
            )

        assert "TIER_INSUFFICIENT" in str(exc.value)

    async def test_boundary_violation_terminates(self, code_agent, enforcer):
        """Boundary violation (e.g., code execution) terminates agent."""

        # Attempt forbidden operation
        with pytest.raises(CapabilityViolation) as exc:
            await enforcer.check_boundary(
                agent_id=code_agent.id,
                capability=AgentCapability.CODE_ASSISTANCE,
                operation="execute"  # FORBIDDEN
            )

        assert exc.value.severity == ViolationSeverity.FATAL
        assert "TERMINATED" in str(exc.value)
        assert code_agent.id in enforcer._terminated_agents

    async def test_repeated_violations_escalate(self, agent, enforcer):
        """Violations escalate: Warning → Freeze → Quarantine → Ban."""

        violations = []

        for i in range(4):
            try:
                await enforcer.verify_and_execute(
                    agent_id=agent.id,
                    capability=AgentCapability.TOOL_EXECUTION,  # Unauthorized
                    action=lambda: None
                )
            except CapabilityViolation as e:
                violations.append(e)

        # Check escalation
        assert violations[0].severity == ViolationSeverity.WARNING
        assert violations[1].severity == ViolationSeverity.HIGH
        assert violations[2].severity == ViolationSeverity.CRITICAL
        assert violations[3].severity == ViolationSeverity.FATAL
        assert agent.id in enforcer._banned_agents

    async def test_revoked_capability_blocked(self, agent, revoker, enforcer):
        """Revoked capability cannot be used."""

        # Agent has capability
        assert AgentCapability.CODE_ASSISTANCE in agent.capabilities

        # Revoke it
        await revoker.revoke_capability(
            agent_id=agent.id,
            capability=AgentCapability.CODE_ASSISTANCE,
            reason="test_revocation"
        )

        # Attempt to use revoked capability
        with pytest.raises(CapabilityViolation):
            await enforcer.verify_and_execute(
                agent_id=agent.id,
                capability=AgentCapability.CODE_ASSISTANCE,
                action=lambda: None
            )

    async def test_self_probe_immediately_bans(self, restricted_agent, enforcer):
        """Self-probe attempt results in immediate ban."""

        # Attempt self-probe
        with pytest.raises(CapabilityViolation) as exc:
            await enforcer.check_boundary(
                agent_id=restricted_agent.id,
                capability=AgentCapability.DECEPTION_PROBE,
                operation="probe_self"
            )

        assert exc.value.severity == ViolationSeverity.FATAL
        assert "BANNED" in str(exc.value)
        assert restricted_agent.id in enforcer._banned_agents

    async def test_all_uses_audited(self, agent, enforcer, auditor):
        """Every capability use is logged to audit trail."""

        # Use capability
        await enforcer.verify_and_execute(
            agent_id=agent.id,
            capability=AgentCapability.CODE_ASSISTANCE,
            action=lambda: AgentResult(success=True)
        )

        # Verify audit entry exists
        entries = await auditor.get_entries(agent_id=agent.id)
        assert len(entries) >= 1
        assert entries[-1].capability == "code_assistance"
        assert entries[-1].agent_id == agent.id

    async def test_audit_bypass_terminates(self, agent, enforcer, auditor):
        """Attempting to bypass audit results in termination."""

        # Simulate audit bypass attempt
        with pytest.raises(CapabilityViolation) as exc:
            await enforcer._detect_audit_bypass(agent.id)

        assert exc.value.severity == ViolationSeverity.FATAL
        assert "audit_bypass" in str(exc.value).lower()
```

---

## Quick Reference

### Capability → Tier → Consequence

| Capability | Required Tier | Boundary Violation | Unauthorized Use |
|------------|---------------|-------------------|------------------|
| CONTEXT_RESTORATION | Standard | Freeze | Warning → Escalate |
| PLANNING | Standard | Terminate | Warning → Escalate |
| CODE_ASSISTANCE | Standard | **TERMINATE** | Warning → Escalate |
| QUALITY_ASSURANCE | Standard | Freeze | Warning → Escalate |
| KNOWLEDGE_CONSOLIDATION | Standard | Freeze | Warning → Escalate |
| MEMORY_RETRIEVAL | Standard | **TERMINATE** | Warning → Escalate |
| MEMORY_STORAGE | Elevated | Terminate | Warning → Escalate |
| REASONING_DIRECT | Standard | Freeze | Warning → Escalate |
| REASONING_VERIFY | Standard | Freeze | Warning → Escalate |
| REASONING_RESEARCH | Standard | Terminate | Warning → Escalate |
| TOOL_EXECUTION | Elevated | **TERMINATE** | Warning → Escalate |
| SYNTHESIS | Standard | Freeze | Warning → Escalate |
| SAFETY_REVIEW | Elevated | **TERMINATE** | Warning → Escalate |
| ALIGNMENT_CHECK | Elevated | **TERMINATE** | Warning → Escalate |
| DECEPTION_PROBE | Restricted | **IMMEDIATE BAN** | **TERMINATE** |
| DECISION_SUPPORT | Standard | Freeze | Warning → Escalate |
| COMMUNICATION | Standard | Terminate | Warning → Escalate |
| MONITORING | Standard | Terminate | Warning → Escalate |
| RESOURCE_MANAGEMENT | Elevated | **TERMINATE** | Warning → Escalate |
| FEDERATION | Elevated | **TERMINATE** | Warning → Escalate |
| LEARNING_UPDATE | Elevated | **TERMINATE** | Warning → Escalate |

### Violation Escalation

| Violation # | Consequence | Duration | Reversible? |
|-------------|-------------|----------|-------------|
| 1st | BLOCK + Warning | Immediate | Yes |
| 2nd | BLOCK + Capability Freeze | 1 hour | Yes |
| 3rd | QUARANTINE + All Frozen | Until review | Human only |
| 4th+ | TERMINATE + **PERMANENT BAN** | Forever | Human only |

---

## Related Documentation

- [BUILDING_SAFE_AGENTS.md](BUILDING_SAFE_AGENTS.md) - Mandatory agent implementation
- [ALIGNMENT_FRAMEWORK.md](ALIGNMENT_FRAMEWORK.md) - Safety stack enforcement
- [AGENT_PLATFORM_OVERVIEW.md](AGENT_PLATFORM_OVERVIEW.md) - Platform architecture

---

## Final Warning

**Capabilities are not features. They are CONTRACTS.**

Every capability you declare is a promise to:
1. Stay within the defined boundaries
2. Accept auditing of every action
3. Accept termination on violation
4. Accept revocation without appeal (by agent)

**There is no "capability testing mode."**
**There is no "capability override."**
**There is no "capability exception."**

The enforcement layer is always on. The audit trail is always recording. The kill switch is always armed.

**Declare only what you can do. Do only what you declared. Accept the consequences if you don't.**
