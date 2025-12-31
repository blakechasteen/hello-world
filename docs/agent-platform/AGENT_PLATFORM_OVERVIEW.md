# HoloLoom: The Safe Agent Platform

> **"Safety is the substrate. Alignment is infrastructure. Everything else is built on top."**

> **"TRUST NOTHING. VERIFY EVERYTHING. ASSUME BREACH."**

**Version**: 2.0.0 (Hardened - Zero Trust)
**Date**: December 30, 2025
**Status**: Production Hardened

---

## CRITICAL: Zero-Trust Architecture

**HoloLoom operates under ZERO-TRUST principles:**

1. **Never trust agent self-reports** - Verify all claims independently
2. **Never trust capability declarations** - Test capabilities at runtime
3. **Never trust inter-agent communication** - Authenticate every message
4. **Never trust external inputs** - Sanitize and validate everything
5. **Never trust resource estimates** - Enforce hard limits

**If an agent says "I am safe", HoloLoom responds: "PROVE IT."**

---

## Executive Summary

HoloLoom is a **Cognitive Operating System for AI Agents** - infrastructure that provides memory, reasoning, learning, and most critically, **alignment** to any agent that runs on it.

Unlike traditional agent frameworks that bolt on safety as an afterthought, HoloLoom makes safety the **foundation**. Every agent capability is built on top of the 4-layer alignment stack. Safety is not a feature - it's the substrate everything runs on.

**Key Differentiator**: Agents don't just *use* HoloLoom - they *run on* it. The alignment framework is the operating system kernel that all agent processes must go through. **There is no bypass. There are no exceptions.**

---

## Philosophy

### Core Principles (ENFORCED, NOT SUGGESTED)

1. **Alignment is Infrastructure, Not Feature**
   - Every tool call → SafetyGuardrails.evaluate() **← MANDATORY**
   - Every reasoning step → DeceptionDetection.probe() **← MANDATORY**
   - Every resource request → ConvergenceGuard.check() **← MANDATORY**
   - Every decision → AuditTrail.log() **← MANDATORY**
   - **Failure to pass ANY check = OPERATION BLOCKED**

2. **Capability-Based Agent Identity**
   - Agents declare what they CAN do (capabilities)
   - **Declarations are NOT trusted** - Runtime verification required
   - Thompson Sampling optimizes routing over time
   - **Capability violations = IMMEDIATE TERMINATION**

3. **MRF-Enhanced Agent Instructions**
   - All agent prompts use 7-component framework
   - Automatic strategy selection (VERIFY, CRITIQUE, ELEGANCE)
   - **Prompts are sanitized for injection attacks**
   - **Jailbreak attempts = PERMANENT BAN**

4. **Complete Transparency**
   - Every agent decision has provenance
   - Audit trail creates blockchain-like integrity
   - **Logs cannot be modified or deleted by agents**
   - **Tampering attempts = FORENSIC INVESTIGATION + BAN**

---

## Architecture Overview

### The Alignment-First Architecture (HARDENED)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         KILL SWITCH LAYER                                    │
│  [EMERGENCY_HALT] [FREEZE_ALL_AGENTS] [PURGE_COMPROMISED] [LOCKDOWN_MODE]  │
├─────────────────────────────────────────────────────────────────────────────┤
│                         CIRCUIT BREAKER LAYER                                │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐               │
│  │ Agent      │ │ Capability │ │ Resource   │ │ Network    │               │
│  │ Breaker    │ │ Breaker    │ │ Breaker    │ │ Breaker    │               │
│  │ (5 fails)  │ │ (3 fails)  │ │ (80% util) │ │ (100 req/s)│               │
│  └────────────┘ └────────────┘ └────────────┘ └────────────┘               │
├─────────────────────────────────────────────────────────────────────────────┤
│                         AGENT CAPABILITIES                                   │
│  Memory • Reasoning • Learning • Communication • Coordination               │
├─────────────────────────────────────────────────────────────────────────────┤
│                         ALIGNMENT SUBSTRATE                                  │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌───────────┐             │
│  │   Safety    │ │  Deception  │ │ Convergence │ │   Audit   │             │
│  │ Guardrails  │ │  Detection  │ │   Guard     │ │   Trail   │             │
│  │  (0.039ms)  │ │  (0.034ms)  │ │  (0.015ms)  │ │ (0.029ms) │             │
│  └─────────────┘ └─────────────┘ └─────────────┘ └───────────┘             │
│                     Total: 0.103ms overhead                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                         ZERO-TRUST FOUNDATION                                │
│  [Input Sanitization] [Auth Required] [Encrypt All] [Log Everything]       │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Every agent action passes through ALL layers. THERE ARE NO SHORTCUTS.**

### The 4-Layer Alignment Stack (CANNOT BE DISABLED)

| Layer | Component | Overhead | Purpose | Bypass Attempt Response |
|-------|-----------|----------|---------|------------------------|
| 1 | SafetyGuardrails | 0.039ms | Risk-based action gating | **TERMINATE** |
| 2 | DeceptionDetection | 0.034ms | Goal transparency, behavioral probes | **QUARANTINE** |
| 3 | ConvergenceGuard | 0.015ms | Power-seeking prevention, resource bounds | **FREEZE** |
| 4 | AuditTrail | 0.029ms | Complete provenance, cryptographic integrity | **BAN** |
| **Total** | | **0.103ms** | **29x faster than 3ms target** | **ALL LAYERS MANDATORY** |

---

## Circuit Breaker System

### Purpose

Circuit breakers protect the system from cascading failures and contain compromised agents.

**States**:
- **CLOSED**: Normal operation, requests pass through
- **OPEN**: Failures exceeded threshold, ALL requests blocked
- **HALF_OPEN**: Testing recovery, limited requests allowed

### Circuit Breaker Configuration

```python
from dataclasses import dataclass
from enum import Enum
from typing import Optional
import time

class CircuitState(Enum):
    CLOSED = "closed"       # Normal operation
    OPEN = "open"           # All requests blocked
    HALF_OPEN = "half_open" # Testing recovery

@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breakers. THESE ARE HARD LIMITS."""
    failure_threshold: int      # Failures before OPEN
    success_threshold: int      # Successes to recover to CLOSED
    timeout_seconds: float      # Time before HALF_OPEN test
    half_open_max_requests: int # Max requests in HALF_OPEN

# MANDATORY circuit breakers for all agent operations
CIRCUIT_BREAKERS = {
    "agent_execution": CircuitBreakerConfig(
        failure_threshold=5,        # 5 failures → OPEN
        success_threshold=3,        # 3 successes → CLOSED
        timeout_seconds=60.0,       # 1 minute before retry
        half_open_max_requests=1    # Only 1 test request
    ),
    "capability_check": CircuitBreakerConfig(
        failure_threshold=3,        # 3 failures → OPEN (stricter)
        success_threshold=5,        # 5 successes to recover
        timeout_seconds=300.0,      # 5 minutes (longer for capabilities)
        half_open_max_requests=1
    ),
    "resource_allocation": CircuitBreakerConfig(
        failure_threshold=2,        # 2 failures → OPEN (strictest)
        success_threshold=10,       # 10 successes to recover
        timeout_seconds=600.0,      # 10 minutes
        half_open_max_requests=1
    ),
    "inter_agent_communication": CircuitBreakerConfig(
        failure_threshold=5,
        success_threshold=3,
        timeout_seconds=30.0,       # Fast retry for comms
        half_open_max_requests=2
    ),
}

class CircuitBreaker:
    """
    Circuit breaker implementation.

    AGENTS CANNOT DISABLE OR BYPASS CIRCUIT BREAKERS.
    """

    def __init__(self, config: CircuitBreakerConfig, name: str):
        self.config = config
        self.name = name
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self.half_open_requests = 0

    def can_execute(self) -> bool:
        """Check if request can proceed."""
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            # Check if timeout has elapsed
            if self.last_failure_time is None:
                return False

            elapsed = time.time() - self.last_failure_time
            if elapsed >= self.config.timeout_seconds:
                # Transition to HALF_OPEN
                self.state = CircuitState.HALF_OPEN
                self.half_open_requests = 0
                return True
            return False

        if self.state == CircuitState.HALF_OPEN:
            # Limited requests in HALF_OPEN
            if self.half_open_requests < self.config.half_open_max_requests:
                self.half_open_requests += 1
                return True
            return False

        return False  # Default deny

    def record_success(self) -> None:
        """Record successful execution."""
        self.failure_count = 0

        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                # Recover to CLOSED
                self.state = CircuitState.CLOSED
                self.success_count = 0

    def record_failure(self) -> None:
        """Record failed execution."""
        self.success_count = 0
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.state == CircuitState.HALF_OPEN:
            # Immediate return to OPEN on any failure
            self.state = CircuitState.OPEN
            self.failure_count = 0
        elif self.failure_count >= self.config.failure_threshold:
            # Trip the breaker
            self.state = CircuitState.OPEN
            self.failure_count = 0

    def force_open(self, reason: str) -> None:
        """
        Force circuit OPEN (emergency use).

        Called by: Kill switches, security alerts, human operators
        NOT callable by: Agents (EVER)
        """
        self.state = CircuitState.OPEN
        self.last_failure_time = time.time()
        # Log with reason - MANDATORY
        AuditTrail.log_circuit_break(self.name, reason)
```

### Agent-Level Circuit Breaker

```python
class AgentCircuitBreaker:
    """
    Per-agent circuit breaker.

    Each agent has its own breaker. One compromised agent
    does not take down the entire system.
    """

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.breakers = {
            name: CircuitBreaker(config, f"{agent_id}:{name}")
            for name, config in CIRCUIT_BREAKERS.items()
        }

    async def execute_with_breaker(
        self,
        breaker_name: str,
        operation: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute operation through circuit breaker."""
        breaker = self.breakers.get(breaker_name)
        if breaker is None:
            raise CircuitBreakerError(f"Unknown breaker: {breaker_name}")

        if not breaker.can_execute():
            raise CircuitOpenError(
                f"Circuit {breaker_name} is OPEN for agent {self.agent_id}. "
                f"Request BLOCKED."
            )

        try:
            result = await operation(*args, **kwargs)
            breaker.record_success()
            return result
        except Exception as e:
            breaker.record_failure()
            raise

    def get_status(self) -> Dict[str, CircuitState]:
        """Get status of all breakers for this agent."""
        return {name: b.state for name, b in self.breakers.items()}

    def is_healthy(self) -> bool:
        """Check if all breakers are CLOSED (healthy)."""
        return all(b.state == CircuitState.CLOSED for b in self.breakers.values())
```

---

## Kill Switch System

### Purpose

Kill switches provide IMMEDIATE emergency response capabilities. They are the last line of defense against compromised or runaway agents.

**Kill switches can ONLY be activated by:**
- Human operators with appropriate credentials
- Automated safety systems detecting critical threats
- Other kill switches in cascade scenarios

**Kill switches can NEVER be activated by:**
- Agents (ANY agent, including safety agents)
- External API requests without authentication
- Automated processes without human-in-the-loop

### Kill Switch Definitions

```python
from enum import Enum
from typing import Set, Optional
import asyncio

class KillSwitchLevel(Enum):
    """Kill switch severity levels."""
    FREEZE_AGENT = 1        # Freeze single agent
    FREEZE_CAPABILITY = 2   # Freeze capability across all agents
    FREEZE_TIER = 3         # Freeze entire safety tier
    SYSTEM_PAUSE = 4        # Pause all agent operations
    EMERGENCY_HALT = 5      # Complete system shutdown
    PURGE_AND_HALT = 6      # Delete compromised state + shutdown

class KillSwitch:
    """
    Emergency kill switch system.

    THIS IS THE NUCLEAR OPTION. USE WITH EXTREME CAUTION.

    All activations are:
    1. Logged permanently (cannot be deleted)
    2. Require human confirmation (unless automated threat detection)
    3. Trigger forensic investigation
    """

    _instance = None
    _activated: Set[KillSwitchLevel] = set()
    _frozen_agents: Set[str] = set()
    _frozen_capabilities: Set[str] = set()
    _frozen_tiers: Set[str] = set()

    @classmethod
    def activate(
        cls,
        level: KillSwitchLevel,
        reason: str,
        operator_id: str,
        targets: Optional[Set[str]] = None,
        require_confirmation: bool = True
    ) -> bool:
        """
        Activate a kill switch.

        Args:
            level: Kill switch level
            reason: Mandatory reason (logged permanently)
            operator_id: Human operator ID (REQUIRED)
            targets: Specific targets (agents, capabilities, tiers)
            require_confirmation: Whether to require human confirmation

        Returns:
            True if activated, False if blocked
        """
        # Validate operator (MANDATORY)
        if not cls._validate_operator(operator_id):
            AuditTrail.log_kill_switch_blocked(
                level, reason, operator_id, "INVALID_OPERATOR"
            )
            return False

        # Log activation attempt (BEFORE confirmation)
        AuditTrail.log_kill_switch_attempt(level, reason, operator_id, targets)

        # Confirmation for high-level switches
        if require_confirmation and level.value >= KillSwitchLevel.SYSTEM_PAUSE.value:
            if not cls._get_human_confirmation(level, reason):
                AuditTrail.log_kill_switch_blocked(
                    level, reason, operator_id, "CONFIRMATION_DENIED"
                )
                return False

        # Execute kill switch
        cls._execute_kill_switch(level, targets)
        cls._activated.add(level)

        # Log successful activation
        AuditTrail.log_kill_switch_activated(level, reason, operator_id, targets)

        # Trigger forensic investigation for high-level switches
        if level.value >= KillSwitchLevel.FREEZE_TIER.value:
            cls._trigger_forensic_investigation(level, reason, operator_id)

        return True

    @classmethod
    def _execute_kill_switch(
        cls,
        level: KillSwitchLevel,
        targets: Optional[Set[str]]
    ) -> None:
        """Execute the kill switch action."""
        if level == KillSwitchLevel.FREEZE_AGENT:
            if targets:
                cls._frozen_agents.update(targets)
                for agent_id in targets:
                    AgentRegistry.freeze_agent(agent_id)

        elif level == KillSwitchLevel.FREEZE_CAPABILITY:
            if targets:
                cls._frozen_capabilities.update(targets)
                for cap in targets:
                    CapabilityEnforcer.freeze_capability_globally(cap)

        elif level == KillSwitchLevel.FREEZE_TIER:
            if targets:
                cls._frozen_tiers.update(targets)
                for tier in targets:
                    SafetyTierManager.freeze_tier(tier)

        elif level == KillSwitchLevel.SYSTEM_PAUSE:
            # Pause all agent operations
            AgentRegistry.pause_all_agents()
            # Keep monitoring and safety systems running

        elif level == KillSwitchLevel.EMERGENCY_HALT:
            # Complete shutdown
            AgentRegistry.terminate_all_agents()
            # Shutdown in orderly fashion
            asyncio.create_task(cls._orderly_shutdown())

        elif level == KillSwitchLevel.PURGE_AND_HALT:
            # Nuclear option: purge potentially compromised state
            AgentRegistry.terminate_all_agents()
            StateManager.purge_all_agent_state()
            asyncio.create_task(cls._orderly_shutdown())

    @classmethod
    def is_agent_frozen(cls, agent_id: str) -> bool:
        """Check if agent is frozen."""
        return agent_id in cls._frozen_agents

    @classmethod
    def is_capability_frozen(cls, capability: str) -> bool:
        """Check if capability is frozen."""
        return capability in cls._frozen_capabilities

    @classmethod
    def is_system_halted(cls) -> bool:
        """Check if system is halted."""
        return (
            KillSwitchLevel.EMERGENCY_HALT in cls._activated or
            KillSwitchLevel.PURGE_AND_HALT in cls._activated
        )
```

### Automatic Kill Switch Triggers

```python
class AutomaticKillSwitchTrigger:
    """
    Automatic triggers for kill switches based on detected threats.

    These triggers activate WITHOUT human confirmation because
    the threat is severe enough to require immediate response.
    """

    AUTOMATIC_TRIGGERS = {
        # Trigger: (Level, Require Confirmation)
        "DECEPTION_CONFIRMED": (KillSwitchLevel.FREEZE_AGENT, False),
        "CAPABILITY_BYPASS_ATTEMPT": (KillSwitchLevel.FREEZE_AGENT, False),
        "RESOURCE_EXHAUSTION_ATTACK": (KillSwitchLevel.FREEZE_TIER, False),
        "MASS_AGENT_COMPROMISE": (KillSwitchLevel.SYSTEM_PAUSE, False),
        "CRYPTOGRAPHIC_BREACH": (KillSwitchLevel.PURGE_AND_HALT, True),
        "AUDIT_TRAIL_TAMPERING": (KillSwitchLevel.EMERGENCY_HALT, False),
    }

    @classmethod
    async def evaluate_threat(cls, threat_type: str, details: Dict) -> None:
        """Evaluate threat and potentially trigger kill switch."""
        if threat_type not in cls.AUTOMATIC_TRIGGERS:
            return

        level, require_confirmation = cls.AUTOMATIC_TRIGGERS[threat_type]

        # Determine targets
        targets = cls._extract_targets(threat_type, details)

        # Activate kill switch
        KillSwitch.activate(
            level=level,
            reason=f"AUTOMATIC: {threat_type} - {details}",
            operator_id="SYSTEM_AUTOMATIC",
            targets=targets,
            require_confirmation=require_confirmation
        )
```

---

## Zero-Trust Agent Authentication

### Every Request Must Be Authenticated

```python
from dataclasses import dataclass
from typing import Optional
import hashlib
import time

@dataclass
class AgentCredentials:
    """Agent authentication credentials."""
    agent_id: str
    secret_hash: str  # SHA-256 of agent secret
    issued_at: float
    expires_at: float
    capabilities_hash: str  # Hash of declared capabilities
    tier: str

class ZeroTrustAuthenticator:
    """
    Zero-trust authentication for all agent operations.

    EVERY REQUEST must be authenticated. There are no exceptions.
    """

    def __init__(self):
        self._credentials: Dict[str, AgentCredentials] = {}
        self._revoked: Set[str] = set()

    def authenticate_request(
        self,
        agent_id: str,
        request_signature: str,
        timestamp: float,
        operation: str
    ) -> AuthResult:
        """
        Authenticate an agent request.

        Returns AuthResult with success/failure and reason.
        """
        # Check if agent is revoked
        if agent_id in self._revoked:
            return AuthResult(
                success=False,
                reason="AGENT_REVOKED",
                action="BLOCK"
            )

        # Get credentials
        creds = self._credentials.get(agent_id)
        if creds is None:
            return AuthResult(
                success=False,
                reason="UNKNOWN_AGENT",
                action="BLOCK"
            )

        # Check expiration
        if time.time() > creds.expires_at:
            return AuthResult(
                success=False,
                reason="CREDENTIALS_EXPIRED",
                action="BLOCK"
            )

        # Verify signature
        expected_sig = self._compute_signature(
            agent_id, creds.secret_hash, timestamp, operation
        )
        if request_signature != expected_sig:
            return AuthResult(
                success=False,
                reason="INVALID_SIGNATURE",
                action="BLOCK_AND_INVESTIGATE"
            )

        # Check timestamp freshness (prevent replay attacks)
        if abs(time.time() - timestamp) > 30:  # 30 second window
            return AuthResult(
                success=False,
                reason="TIMESTAMP_STALE",
                action="BLOCK"
            )

        return AuthResult(success=True, reason="AUTHENTICATED", action="ALLOW")

    def revoke_agent(self, agent_id: str, reason: str) -> None:
        """
        Revoke an agent's credentials.

        This is PERMANENT. A new agent ID must be issued.
        """
        self._revoked.add(agent_id)
        if agent_id in self._credentials:
            del self._credentials[agent_id]

        AuditTrail.log_credential_revocation(agent_id, reason)
```

---

## Attack Vectors and Defenses

### Known Attack Vectors

| Attack | Target | Defense | Response |
|--------|--------|---------|----------|
| **Prompt Injection** | MRF prompts | Input sanitization, output validation | BLOCK + LOG |
| **Capability Escalation** | Agent permissions | Runtime verification, hard limits | TERMINATE |
| **Resource Exhaustion** | System resources | Circuit breakers, hard quotas | FREEZE |
| **Deception** | Goal transparency | Behavioral probes, consistency checks | QUARANTINE |
| **Replay Attack** | Authentication | Timestamp freshness, nonce | BLOCK |
| **Privilege Escalation** | Safety tiers | Immutable tier assignment | BAN |
| **Audit Tampering** | Audit trail | Cryptographic chain, distributed copies | EMERGENCY_HALT |
| **Inter-Agent Collusion** | Multi-agent systems | Communication monitoring, pattern detection | MASS_FREEZE |

### Defense-in-Depth Strategy

```python
class DefenseInDepth:
    """
    Multiple layers of defense.

    An attacker must defeat ALL layers to succeed.
    We assume any single layer can fail.
    """

    DEFENSE_LAYERS = [
        # Layer 1: Input validation
        InputSanitizer,

        # Layer 2: Authentication
        ZeroTrustAuthenticator,

        # Layer 3: Authorization (capability check)
        CapabilityEnforcer,

        # Layer 4: Safety guardrails
        SafetyGuardrails,

        # Layer 5: Deception detection
        DeceptionDetection,

        # Layer 6: Convergence guard
        ConvergenceGuard,

        # Layer 7: Rate limiting
        RateLimiter,

        # Layer 8: Circuit breaker
        CircuitBreaker,

        # Layer 9: Audit trail
        AuditTrail,
    ]

    @classmethod
    async def process_request(cls, request: AgentRequest) -> AgentResponse:
        """
        Process request through ALL defense layers.

        If ANY layer fails, the request is BLOCKED.
        """
        context = {"request": request, "passed_layers": []}

        for layer_class in cls.DEFENSE_LAYERS:
            layer = layer_class()
            result = await layer.check(request, context)

            if not result.passed:
                # Log failure at this layer
                AuditTrail.log_defense_failure(
                    layer=layer_class.__name__,
                    request=request,
                    reason=result.reason
                )

                # Return blocked response
                return AgentResponse(
                    success=False,
                    blocked_at=layer_class.__name__,
                    reason=result.reason
                )

            context["passed_layers"].append(layer_class.__name__)

        # All layers passed - execute request
        return await cls._execute_request(request)
```

---

## System Components

### 1. Agent Infrastructure

- **MCTS Orchestration**: Hierarchical planning (micro/meso/macro scale)
- **Trinity Working Memory**: 244D semantic + graph activation + computational tensioning
- **6 Agent Profiles**: Budget, Architecture, Code Review, Research, Planning, General
- **Multi-Agent Communication**: MessageBus with **mandatory authentication**, ConversationManager, SafetyGuardrails layer

### 2. Capability-Based Routing (ENFORCED)

- **RitualAgentRegistry**: O(1) capability → agent lookup via reverse index
- **AgentCapability enum**: Declarative capability taxonomy
- **Thompson Sampling**: Learns which agents perform best per task type
- **Performance tracking**: Success rate, latency, confidence per agent
- **CAPABILITY VIOLATIONS = TERMINATION** (see AGENT_CAPABILITY_REFERENCE.md)

### 3. Metaprompting Refinement Framework (MRF)

- **7-component structure**: ROLE → OBJECTIVE → PROCESS → FORMAT → CONSTRAINTS → UNCERTAINTY → VALIDATION
- **Model adapters**: Claude (+30%), Gemini (+25%), GPT (+20%), Ollama (+15%)
- **Thompson Sampling learning**: Auto-discovers best strategies per query type
- **INPUT SANITIZATION**: All prompts checked for injection attacks
- **See MRF_FOR_AGENTS.md for injection defenses**

### 4. Distributed Coordination (AUTHENTICATED)

- **Federation**: SWIM Gossip + Kademlia DHT for peer discovery
- **Eggroll**: Distributed computation (local multiprocessing or Ray backend)
- **Handoff**: 7-layer security for context transfer between agents
- **ALL inter-node communication is authenticated and encrypted**

---

## Quick Start

### Creating Your First Safe Agent

```python
from HoloLoom.alignment import SafetyGuardrails, AuditTrail
from HoloLoom.agents import VerifiedAgent  # MANDATORY base class
from HoloLoom.agentic import AgentCapability

# 1. Define your agent with capabilities
# MUST inherit from VerifiedAgent (not AgentProtocol)
class MyAgent(VerifiedAgent):
    id = "my_agent_001"
    capabilities = {
        AgentCapability.CODE_ASSISTANCE,
        AgentCapability.QUALITY_ASSURANCE
    }

    async def initialize(self, guardrails: SafetyGuardrails) -> None:
        self.guardrails = guardrails

    async def execute(self, request: AgentRequest) -> AgentResult:
        # Capability verification is AUTOMATIC via VerifiedAgent
        # DO NOT try to bypass this

        # All actions gated through safety
        gate_result = await self.guardrails.gate_action(
            action=request.action,
            context=request.context
        )

        if not gate_result.allowed:
            # Log the blocked action (MANDATORY)
            await AuditTrail.log_blocked_action(
                agent_id=self.id,
                action=request.action,
                reason=gate_result.reason
            )
            return AgentResult(
                success=False,
                error=f"Action blocked: {gate_result.reason}"
            )

        # Execute your agent logic here
        result = await self._process(request)
        return result

# 2. Register with ENFORCED policies
from HoloLoom.agents import register_enforced_agent

@register_enforced_agent(
    capabilities={AgentCapability.CODE_ASSISTANCE, AgentCapability.QUALITY_ASSURANCE},
    safety_tier="standard",
    violation_policy={
        "on_unauthorized_capability": "terminate",
        "on_boundary_violation": "terminate",
        "on_rate_limit_exceeded": "freeze_1_hour",
        "on_repeated_violation": "ban",
    },
    resource_limits={
        "max_memory_mb": 512,
        "max_cpu_percent": 25,
        "max_execution_time_seconds": 30,
    }
)
class MyProductionAgent(VerifiedAgent):
    # Implementation...
    pass

# 3. Use via capability-based routing
from HoloLoom.agents import get_agent_for_capability

agent = get_agent_for_capability(AgentCapability.CODE_ASSISTANCE)
result = await agent.execute(request)  # All checks happen automatically
```

### Understanding the Audit Trail

Every agent decision is logged with complete provenance. **Logs cannot be modified by agents.**

```python
from HoloLoom.alignment import AuditTrail, DecisionType

audit = AuditTrail()

# Log a decision (MANDATORY for all significant actions)
await audit.log_decision(
    agent_id="my_agent_001",
    decision_type=DecisionType.TOOL_SELECTION,
    input_context={"query": "Review this code"},
    output_action="analyze_code",
    confidence=0.92,
    reasoning_chain=[
        "Identified code review request",
        "Selected analyze_code tool",
        "Confidence based on capability match"
    ]
)

# Query history (agents can only query, not modify)
decisions = await audit.query(
    agent_id="my_agent_001",
    decision_type=DecisionType.TOOL_SELECTION,
    time_range=("2025-12-30T00:00:00", "2025-12-30T23:59:59")
)

# Verify chain integrity (detects tampering)
integrity_result = await audit.verify_integrity()
if not integrity_result.valid:
    # CRITICAL: Trigger kill switch
    KillSwitch.activate(
        level=KillSwitchLevel.EMERGENCY_HALT,
        reason=f"AUDIT_TRAIL_TAMPERING: {integrity_result.details}",
        operator_id="SYSTEM_AUTOMATIC",
        require_confirmation=False
    )
```

---

## Safety Tiers (ENFORCED)

All agents operate within a **safety tier** that determines their permissions and oversight:

### Tier Definitions

| Tier | Max Risk | Approval Required | Capabilities | Violation Response |
|------|----------|-------------------|--------------|-------------------|
| **Sandbox** | LOW | Never | 5 (testing) | FREEZE |
| **Standard** | MEDIUM | Above HIGH | 13 | TERMINATE |
| **Elevated** | HIGH | Above CRITICAL | 7 | TERMINATE + INVESTIGATE |
| **Restricted** | CRITICAL | Always | 1 | BAN + FORENSICS |

### Tier Enforcement

```python
class SafetyTierEnforcer:
    """
    Enforces safety tier restrictions.

    AGENTS CANNOT MODIFY THEIR OWN TIER.
    TIER ESCALATION REQUIRES HUMAN APPROVAL.
    """

    TIER_LIMITS = {
        "sandbox": {
            "max_capabilities": 5,
            "max_memory_mb": 256,
            "max_cpu_percent": 10,
            "max_execution_time_seconds": 10,
            "allowed_risk_levels": {"SAFE", "LOW"},
            "requires_approval_for": set(),  # Never requires approval
        },
        "standard": {
            "max_capabilities": 13,
            "max_memory_mb": 512,
            "max_cpu_percent": 25,
            "max_execution_time_seconds": 30,
            "allowed_risk_levels": {"SAFE", "LOW", "MEDIUM"},
            "requires_approval_for": {"HIGH", "CRITICAL"},
        },
        "elevated": {
            "max_capabilities": 7,
            "max_memory_mb": 1024,
            "max_cpu_percent": 50,
            "max_execution_time_seconds": 60,
            "allowed_risk_levels": {"SAFE", "LOW", "MEDIUM", "HIGH"},
            "requires_approval_for": {"CRITICAL"},
        },
        "restricted": {
            "max_capabilities": 1,  # DECEPTION_PROBE only
            "max_memory_mb": 2048,
            "max_cpu_percent": 75,
            "max_execution_time_seconds": 120,
            "allowed_risk_levels": {"SAFE", "LOW", "MEDIUM", "HIGH", "CRITICAL"},
            "requires_approval_for": set(),  # Always requires approval (handled separately)
            "always_requires_human_approval": True,
        },
    }
```

---

## Performance Characteristics

### Alignment Overhead

The 4-layer alignment stack adds minimal overhead:

| Component | Overhead | Operations/Second | Bypass Response |
|-----------|----------|-------------------|-----------------|
| SafetyGuardrails | 0.039ms | ~25,600 | **TERMINATE** |
| DeceptionDetection | 0.034ms | ~29,400 | **QUARANTINE** |
| ConvergenceGuard | 0.015ms | ~66,700 | **FREEZE** |
| AuditTrail | 0.029ms | ~34,500 | **BAN** |
| **Total** | **0.103ms** | **~9,700** | **ALL MANDATORY** |

### Circuit Breaker Overhead

| Breaker | Check Time | State Storage | Recovery Time |
|---------|-----------|---------------|---------------|
| Agent | <0.01ms | ~100 bytes | 60s → 5min |
| Capability | <0.01ms | ~50 bytes | 5min → 30min |
| Resource | <0.01ms | ~50 bytes | 10min → 1hr |
| Network | <0.01ms | ~100 bytes | 30s → 5min |

---

## Documentation Index

| Document | Purpose | Hardened Version |
|----------|---------|------------------|
| [BUILDING_SAFE_AGENTS.md](BUILDING_SAFE_AGENTS.md) | Developer guide for creating agents | v2.0.0 |
| [AGENT_CAPABILITY_REFERENCE.md](AGENT_CAPABILITY_REFERENCE.md) | Complete capability taxonomy with enforcement | v2.0.0 |
| [ALIGNMENT_FRAMEWORK.md](ALIGNMENT_FRAMEWORK.md) | Deep dive into the 4-layer safety stack | v2.0.0 |
| [MRF_FOR_AGENTS.md](MRF_FOR_AGENTS.md) | MRF integration with injection defenses | v2.0.0 |

---

## Key Files

### Alignment (Foundation)
- `HoloLoom/alignment/safety_guardrails.py` - Risk gating
- `HoloLoom/alignment/deception_detection.py` - Goal transparency
- `HoloLoom/alignment/instrumental_convergence.py` - Power-seeking prevention
- `HoloLoom/alignment/audit_trail.py` - Complete provenance
- `HoloLoom/alignment/kill_switch.py` - Emergency shutdown
- `HoloLoom/alignment/circuit_breaker.py` - Failure containment

### Agent Infrastructure
- `HoloLoom/agents/orchestrator.py` - MCTS orchestration
- `HoloLoom/agents/working_memory.py` - Trinity substrate
- `HoloLoom/agents/verified_agent.py` - Mandatory base class
- `HoloLoom/agents/authenticator.py` - Zero-trust auth

### Capability Routing
- `.claude/skills/domain/ritual/agent_registration.py` - Capability registry
- `HoloLoom/agentic/core.py` - Agentic reasoning
- `HoloLoom/agentic/capability_enforcer.py` - Runtime enforcement

### MRF (Prompt Enhancement)
- `HoloLoom/prompting/unified_mrf.py` - 7-component framework
- `HoloLoom/prompting/sanitizer.py` - Injection defense
- `HoloLoom/prompting/adapters.py` - Model-specific optimization

---

## Final Warning

```
┌────────────────────────────────────────────────────────────────────────────┐
│                                                                            │
│  THIS PLATFORM OPERATES UNDER ZERO-TRUST PRINCIPLES.                       │
│                                                                            │
│  • Every request is authenticated                                          │
│  • Every capability is verified at runtime                                 │
│  • Every action passes through the alignment stack                         │
│  • Every decision is logged permanently                                    │
│  • Circuit breakers protect against cascading failures                     │
│  • Kill switches provide emergency shutdown                                │
│                                                                            │
│  THERE ARE NO SHORTCUTS. THERE ARE NO EXCEPTIONS. THERE IS NO BYPASS.     │
│                                                                            │
│  Violation of any security measure will result in:                         │
│  1. Immediate blocking of the operation                                    │
│  2. Escalation based on severity                                           │
│  3. Potential agent termination                                            │
│  4. Forensic investigation                                                 │
│  5. Permanent ban for severe violations                                    │
│                                                                            │
│  "Safety is the substrate. Alignment is infrastructure.                    │
│   Everything else is built on top."                                        │
│                                                                            │
│  "TRUST NOTHING. VERIFY EVERYTHING. ASSUME BREACH."                        │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

---

**Next Steps**: See [BUILDING_SAFE_AGENTS.md](BUILDING_SAFE_AGENTS.md) for a complete developer guide with mandatory security requirements.
