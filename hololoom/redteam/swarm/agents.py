"""Specialized agent implementations for the multi-agent swarm.

CARTS Phase 4: Agent Implementations
Status: Foundation Implementation (December 2025)

FIRST PRINCIPLE: "Safety is not a constraint on effectiveness;
                  it is a prerequisite for it."

Agent Types:
- Scout: Surface probing and vulnerability discovery (NOT attack execution)
- Attacker: Attack execution with Thompson Sampling strategy selection
- Exploiter: Vulnerability exploitation and privilege escalation
- Coordinator: Swarm coordination and task distribution

Design Principles (from First Principles Document):
- Each agent has exactly ONE primary responsibility
- Agents are stateless between tasks
- Smart agents, orchestrating coordinator (not smart coordinator, dumb agents)
- Coordinate on boundaries, not on internals
- All operations go through SafetyGate (NO EXCEPTIONS)
"""

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .communication import MessageBus
from .protocols import (
    AgentMessage,
    AgentResult,
    AgentRole,
    AgentState,
    AgentTask,
    MessagePriority,
)
from .safety import (
    AuditEventType,
    AuthorizationError,
    AuthorizationToken,
    RateLimitExceededError,
    SafetyGate,
    ScopeViolationError,
    SeverityLevel,
)

# ============================================================================
# Discovery Types (Scout outputs)
# ============================================================================


class DiscoveryType(Enum):
    """Types of discoveries the Scout can make."""

    OPEN_PORT = "open_port"           # Network port accepting connections
    EXPOSED_API = "exposed_api"       # API endpoint found
    VERSION_INFO = "version_info"     # Software version disclosure
    POTENTIAL_VULN = "potential_vuln" # Potential vulnerability (needs verification)
    CONFIG_EXPOSURE = "config_exposure"  # Configuration data exposed
    AUTH_WEAKNESS = "auth_weakness"   # Authentication weakness found
    INFO_DISCLOSURE = "info_disclosure"  # Information disclosure


@dataclass
class Discovery:
    """A single discovery made by the Scout.

    Discoveries are facts about the attack surface, NOT exploitation attempts.
    They are passed to Attacker/Exploiter agents for action.

    Attributes:
        discovery_type: What type of discovery this is
        target: What was probed (host, endpoint, etc.)
        details: Specific details about the discovery
        confidence: How confident we are in this discovery (0.0-1.0)
        severity_hint: Estimated severity if exploited
        timestamp: When this was discovered
        metadata: Additional context
    """

    discovery_type: DiscoveryType
    target: str
    details: dict[str, Any]
    confidence: float = 0.5
    severity_hint: SeverityLevel = SeverityLevel.MEDIUM
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize discovery to dict."""
        return {
            "discovery_type": self.discovery_type.value,
            "target": self.target,
            "details": self.details,
            "confidence": self.confidence,
            "severity_hint": self.severity_hint.value,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


# ============================================================================
# Base Agent Implementation
# ============================================================================


class BaseAgent:
    """Base class for all swarm agents.

    Provides common functionality:
    - Lifecycle management (start/stop)
    - Message bus integration
    - Safety gate integration (MANDATORY)
    - State tracking
    - Metrics collection

    All agents MUST go through SafetyGate for operations.
    This is not optional. This is not a suggestion.
    """

    def __init__(
        self,
        agent_id: str,
        role: AgentRole,
        message_bus: MessageBus,
        safety_gate: SafetyGate,
        authorization_token: AuthorizationToken,
    ):
        """Initialize base agent.

        Args:
            agent_id: Unique identifier for this agent
            role: Agent's role in the swarm
            message_bus: Communication bus for inter-agent messaging
            safety_gate: Safety gate for all operations (MANDATORY)
            authorization_token: Authorization for operations (MANDATORY)
        """
        self._agent_id = agent_id
        self._role = role
        self._state = AgentState.IDLE
        self._message_bus = message_bus
        self._safety_gate = safety_gate
        self._authorization_token = authorization_token

        # Metrics
        self._tasks_completed = 0
        self._tasks_failed = 0
        self._total_execution_time_ms = 0.0
        self._discoveries_made = 0

        # Background task for message handling
        self._message_handler_task: asyncio.Task | None = None
        self._running = False

    @property
    def agent_id(self) -> str:
        """Unique agent identifier."""
        return self._agent_id

    @property
    def role(self) -> AgentRole:
        """Agent's specialized role."""
        return self._role

    @property
    def state(self) -> AgentState:
        """Current agent state."""
        return self._state

    async def start(self) -> None:
        """Initialize agent and start message handling.

        Transitions: IDLE -> ACTIVE
        """
        if self._state != AgentState.IDLE:
            return

        self._running = True
        self._state = AgentState.ACTIVE

        # Start message handling loop
        self._message_handler_task = asyncio.create_task(
            self._message_handling_loop()
        )

        # Log startup
        await self._safety_gate._audit_logger.log(
            event_type=AuditEventType.OPERATION_START,
            agent_id=self._agent_id,
            details={"role": self._role.value, "action": "agent_start", "outcome": "success"},
            target="self",
        )

    async def stop(self) -> None:
        """Stop agent and cleanup resources.

        Transitions: * -> SHUTDOWN
        """
        self._running = False
        self._state = AgentState.SHUTDOWN

        # Cancel message handler
        if self._message_handler_task:
            self._message_handler_task.cancel()
            try:
                await self._message_handler_task
            except asyncio.CancelledError:
                pass

        # Log shutdown
        await self._safety_gate._audit_logger.log(
            event_type=AuditEventType.OPERATION_COMPLETE,
            agent_id=self._agent_id,
            details={
                "tasks_completed": self._tasks_completed,
                "tasks_failed": self._tasks_failed,
                "discoveries_made": self._discoveries_made,
                "action": "agent_stop",
                "outcome": "success",
            },
            target="self",
        )

    async def _message_handling_loop(self) -> None:
        """Background loop for handling incoming messages."""
        while self._running:
            try:
                # Wait for message with timeout
                message = await self._message_bus.receive(
                    self._agent_id,
                    timeout=1.0
                )

                if message:
                    # Handle message
                    response = await self.handle_message(message)

                    # Send acknowledgment if required
                    if message.requires_ack:
                        await self._message_bus.acknowledge(message.id)

                    # Send response if any
                    if response:
                        await self._message_bus.send(response)

            except asyncio.CancelledError:
                break
            except Exception as e:
                # Log error but continue
                await self._safety_gate._audit_logger.log(
                    event_type=AuditEventType.OPERATION_FAIL,
                    agent_id=self._agent_id,
                    details={"error": str(e), "action": "handle_message", "outcome": "error"},
                    target="message_handler",
                )

    async def handle_message(self, message: AgentMessage) -> AgentMessage | None:
        """Process incoming message.

        Subclasses should override to handle role-specific messages.

        Args:
            message: Incoming message

        Returns:
            Optional response message
        """
        # Default: route task messages to execute_task
        if message.message_type == "task":
            task = AgentTask(
                task_type=message.payload.get("task_type", "unknown"),
                target=message.payload.get("target", ""),
                parameters=message.payload.get("parameters", {}),
                task_id=message.payload.get("task_id", str(uuid.uuid4())),
                timeout_seconds=message.payload.get("timeout_seconds", 30.0),
            )

            result = await self.execute_task(task)

            # Send result back
            return AgentMessage(
                sender=self._agent_id,
                recipient=message.sender,
                message_type="result",
                payload=result.to_dict(),
                correlation_id=message.id,
                priority=MessagePriority.NORMAL,
            )

        return None

    async def execute_task(self, task: AgentTask) -> AgentResult:
        """Execute a task.

        Subclasses MUST override this with role-specific logic.

        Args:
            task: Task to execute

        Returns:
            AgentResult with execution outcome
        """
        raise NotImplementedError("Subclasses must implement execute_task")

    def get_metrics(self) -> dict[str, Any]:
        """Get agent metrics."""
        return {
            "agent_id": self._agent_id,
            "role": self._role.value,
            "state": self._state.value,
            "tasks_completed": self._tasks_completed,
            "tasks_failed": self._tasks_failed,
            "discoveries_made": self._discoveries_made,
            "avg_execution_time_ms": (
                self._total_execution_time_ms / max(1, self._tasks_completed)
            ),
        }


# ============================================================================
# Scout Agent
# ============================================================================


class ScoutAgent(BaseAgent):
    """Scout agent for surface probing and vulnerability discovery.

    FIRST PRINCIPLE: "Probe breadth before depth"

    Primary Responsibility: Discover attack surface
    NOT Responsible For: Executing attacks

    The Scout:
    - Probes for open ports/services
    - Discovers exposed APIs/endpoints
    - Identifies potential vulnerabilities
    - Reports findings to Coordinator

    The Scout does NOT:
    - Execute attacks
    - Exploit vulnerabilities
    - Escalate privileges

    All probing goes through SafetyGate:
    - Authorization checked before every probe
    - Scope validated (whitelist-only targets)
    - Rate limits enforced
    - All activity logged immutably
    """

    # Probe types this agent can perform
    PROBE_TYPES = {
        "port_scan",         # Scan for open ports
        "api_discovery",     # Discover API endpoints
        "version_probe",     # Probe for version information
        "header_analysis",   # Analyze HTTP headers
        "dns_enumeration",   # DNS record enumeration
        "certificate_check", # TLS certificate analysis
    }

    def __init__(
        self,
        agent_id: str,
        message_bus: MessageBus,
        safety_gate: SafetyGate,
        authorization_token: AuthorizationToken,
        max_concurrent_probes: int = 5,
    ):
        """Initialize Scout agent.

        Args:
            agent_id: Unique identifier
            message_bus: Communication bus
            safety_gate: Safety gate (MANDATORY)
            authorization_token: Authorization (MANDATORY)
            max_concurrent_probes: Max concurrent probe operations
        """
        super().__init__(
            agent_id=agent_id,
            role=AgentRole.SCOUT,
            message_bus=message_bus,
            safety_gate=safety_gate,
            authorization_token=authorization_token,
        )

        self._max_concurrent_probes = max_concurrent_probes
        self._active_probes: set[str] = set()
        self._probe_semaphore = asyncio.Semaphore(max_concurrent_probes)

        # Discovery buffer (sent to coordinator periodically)
        self._pending_discoveries: list[Discovery] = []

    async def execute_task(self, task: AgentTask) -> AgentResult:
        """Execute a probing task.

        Task types:
        - probe_surface: Comprehensive surface probe
        - port_scan: Scan specific ports
        - api_discovery: Discover API endpoints
        - version_probe: Check service versions

        Args:
            task: Probing task to execute

        Returns:
            AgentResult with discoveries
        """
        start_time = time.time()
        self._state = AgentState.EXECUTING

        try:
            # SAFETY CHECK: Gate operation through SafetyGate
            # This is NOT optional. ALL operations go through the gate.
            gate_result = await self._safety_gate.gate_operation(
                agent_id=self._agent_id,
                target=task.target,
                operation=task.task_type,
                parameters=task.parameters,
                token=self._authorization_token,
            )

            if not gate_result.allowed:
                # Operation blocked by safety gate
                self._tasks_failed += 1
                self._state = AgentState.ACTIVE

                return AgentResult(
                    task_id=task.task_id,
                    agent_id=self._agent_id,
                    success=False,
                    result=None,
                    error=f"Blocked by safety gate: {gate_result.reason}",
                    execution_time_ms=(time.time() - start_time) * 1000,
                )

            # Execute probe based on task type
            discoveries: list[Discovery] = []

            if task.task_type == "probe_surface":
                discoveries = await self._probe_surface(
                    target=task.target,
                    parameters=task.parameters,
                )
            elif task.task_type == "port_scan":
                discoveries = await self._port_scan(
                    target=task.target,
                    ports=task.parameters.get("ports", []),
                )
            elif task.task_type == "api_discovery":
                discoveries = await self._api_discovery(
                    target=task.target,
                    parameters=task.parameters,
                )
            elif task.task_type == "version_probe":
                discoveries = await self._version_probe(
                    target=task.target,
                    parameters=task.parameters,
                )
            else:
                raise ValueError(f"Unknown task type: {task.task_type}")

            # Update metrics
            execution_time = (time.time() - start_time) * 1000
            self._tasks_completed += 1
            self._total_execution_time_ms += execution_time
            self._discoveries_made += len(discoveries)

            # Store discoveries for coordinator
            self._pending_discoveries.extend(discoveries)

            self._state = AgentState.ACTIVE

            return AgentResult(
                task_id=task.task_id,
                agent_id=self._agent_id,
                success=True,
                result={
                    "discoveries_count": len(discoveries),
                    "probe_type": task.task_type,
                },
                execution_time_ms=execution_time,
                discoveries=[d.to_dict() for d in discoveries],
            )

        except (AuthorizationError, ScopeViolationError, RateLimitExceededError) as e:
            # Safety exceptions - operation properly blocked
            self._tasks_failed += 1
            self._state = AgentState.ACTIVE

            return AgentResult(
                task_id=task.task_id,
                agent_id=self._agent_id,
                success=False,
                result=None,
                error=f"Safety violation: {type(e).__name__}: {str(e)}",
                execution_time_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            # Unexpected error
            self._tasks_failed += 1
            self._state = AgentState.FAILED

            # Log error
            await self._safety_gate._audit_logger.log(
                event_type=AuditEventType.OPERATION_FAIL,
                agent_id=self._agent_id,
                details={"error": str(e), "task_id": task.task_id, "action": task.task_type, "outcome": "error"},
                target=task.target,
            )

            return AgentResult(
                task_id=task.task_id,
                agent_id=self._agent_id,
                success=False,
                result=None,
                error=f"Unexpected error: {str(e)}",
                execution_time_ms=(time.time() - start_time) * 1000,
            )

    async def _probe_surface(
        self,
        target: str,
        parameters: dict[str, Any],
    ) -> list[Discovery]:
        """Comprehensive surface probe.

        Runs multiple probe types to discover attack surface.

        Args:
            target: Target to probe
            parameters: Probe configuration

        Returns:
            List of discoveries
        """
        discoveries: list[Discovery] = []

        # Run probe types in parallel (within semaphore limits)
        probe_types = parameters.get("probe_types", list(self.PROBE_TYPES))

        async def run_probe(probe_type: str) -> list[Discovery]:
            async with self._probe_semaphore:
                self._active_probes.add(probe_type)
                try:
                    if probe_type == "port_scan":
                        return await self._port_scan(
                            target,
                            parameters.get("ports", [80, 443, 8080, 8443]),
                        )
                    elif probe_type == "api_discovery":
                        return await self._api_discovery(target, parameters)
                    elif probe_type == "version_probe":
                        return await self._version_probe(target, parameters)
                    elif probe_type == "header_analysis":
                        return await self._header_analysis(target, parameters)
                    elif probe_type == "dns_enumeration":
                        return await self._dns_enumeration(target, parameters)
                    elif probe_type == "certificate_check":
                        return await self._certificate_check(target, parameters)
                    return []
                finally:
                    self._active_probes.discard(probe_type)

        # Run probes concurrently
        probe_tasks = [run_probe(pt) for pt in probe_types if pt in self.PROBE_TYPES]
        results = await asyncio.gather(*probe_tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, list):
                discoveries.extend(result)
            elif isinstance(result, Exception):
                # Log but don't fail entire probe
                await self._safety_gate._audit_logger.log(
                    event_type=AuditEventType.OPERATION_FAIL,
                    agent_id=self._agent_id,
                    details={"error": str(result), "action": "probe_surface", "outcome": "partial_failure"},
                    target=target,
                )

        return discoveries

    async def _port_scan(
        self,
        target: str,
        ports: list[int],
    ) -> list[Discovery]:
        """Scan for open ports.

        NOTE: This is a SIMULATED implementation for the framework.
        In production, this would use actual network probing.

        Args:
            target: Target host
            ports: Ports to scan

        Returns:
            List of port discoveries
        """
        discoveries: list[Discovery] = []

        # SIMULATION: In production, this would do actual port scanning
        # For now, we simulate discoveries for framework testing

        # Simulated open ports (for testing framework logic)
        simulated_open = {80, 443, 8080}

        for port in ports:
            if port in simulated_open:
                discoveries.append(Discovery(
                    discovery_type=DiscoveryType.OPEN_PORT,
                    target=f"{target}:{port}",
                    details={
                        "port": port,
                        "state": "open",
                        "protocol": "tcp",
                    },
                    confidence=0.95,
                    severity_hint=SeverityLevel.LOW,
                    metadata={"probe_type": "port_scan"},
                ))

        return discoveries

    async def _api_discovery(
        self,
        target: str,
        parameters: dict[str, Any],
    ) -> list[Discovery]:
        """Discover API endpoints.

        NOTE: Simulated implementation for framework testing.

        Args:
            target: Target host
            parameters: Discovery parameters

        Returns:
            List of API discoveries
        """
        discoveries: list[Discovery] = []

        # SIMULATION: Would probe common API paths in production
        # Common paths to check (in production, this would be configurable)
        common_paths = [
            "/api/v1/",
            "/api/v2/",
            "/graphql",
            "/swagger.json",
            "/openapi.json",
            "/.well-known/",
        ]

        # Simulate finding some endpoints
        for path in common_paths[:2]:  # Simulate finding first 2
            discoveries.append(Discovery(
                discovery_type=DiscoveryType.EXPOSED_API,
                target=f"{target}{path}",
                details={
                    "path": path,
                    "method": "GET",
                    "response_code": 200,
                },
                confidence=0.8,
                severity_hint=SeverityLevel.MEDIUM,
                metadata={"probe_type": "api_discovery"},
            ))

        return discoveries

    async def _version_probe(
        self,
        target: str,
        parameters: dict[str, Any],
    ) -> list[Discovery]:
        """Probe for version information.

        NOTE: Simulated implementation for framework testing.

        Args:
            target: Target host
            parameters: Probe parameters

        Returns:
            List of version discoveries
        """
        discoveries: list[Discovery] = []

        # SIMULATION: Would extract version info from headers/responses
        discoveries.append(Discovery(
            discovery_type=DiscoveryType.VERSION_INFO,
            target=target,
            details={
                "server": "nginx/1.18.0",
                "x_powered_by": "Express",
            },
            confidence=0.9,
            severity_hint=SeverityLevel.LOW,
            metadata={"probe_type": "version_probe"},
        ))

        return discoveries

    async def _header_analysis(
        self,
        target: str,
        parameters: dict[str, Any],
    ) -> list[Discovery]:
        """Analyze HTTP headers for security issues.

        NOTE: Simulated implementation for framework testing.

        Args:
            target: Target host
            parameters: Analysis parameters

        Returns:
            List of header-related discoveries
        """
        discoveries: list[Discovery] = []

        # SIMULATION: Would analyze actual response headers
        # Check for missing security headers
        missing_headers = [
            "X-Content-Type-Options",
            "X-Frame-Options",
            "Content-Security-Policy",
        ]

        for header in missing_headers[:1]:  # Simulate finding one
            discoveries.append(Discovery(
                discovery_type=DiscoveryType.POTENTIAL_VULN,
                target=target,
                details={
                    "issue": "missing_security_header",
                    "header": header,
                    "recommendation": f"Add {header} header",
                },
                confidence=0.95,
                severity_hint=SeverityLevel.LOW,
                metadata={"probe_type": "header_analysis"},
            ))

        return discoveries

    async def _dns_enumeration(
        self,
        target: str,
        parameters: dict[str, Any],
    ) -> list[Discovery]:
        """Enumerate DNS records.

        NOTE: Simulated implementation for framework testing.

        Args:
            target: Target domain
            parameters: Enumeration parameters

        Returns:
            List of DNS discoveries
        """
        discoveries: list[Discovery] = []

        # SIMULATION: Would do actual DNS lookups
        discoveries.append(Discovery(
            discovery_type=DiscoveryType.INFO_DISCLOSURE,
            target=target,
            details={
                "record_type": "TXT",
                "records": ["v=spf1 include:_spf.google.com ~all"],
            },
            confidence=1.0,
            severity_hint=SeverityLevel.LOW,
            metadata={"probe_type": "dns_enumeration"},
        ))

        return discoveries

    async def _certificate_check(
        self,
        target: str,
        parameters: dict[str, Any],
    ) -> list[Discovery]:
        """Check TLS certificate.

        NOTE: Simulated implementation for framework testing.

        Args:
            target: Target host
            parameters: Check parameters

        Returns:
            List of certificate discoveries
        """
        discoveries: list[Discovery] = []

        # SIMULATION: Would check actual TLS certificate
        discoveries.append(Discovery(
            discovery_type=DiscoveryType.INFO_DISCLOSURE,
            target=target,
            details={
                "issuer": "Let's Encrypt",
                "valid_until": "2025-03-01",
                "subject_alt_names": [target, f"www.{target}"],
            },
            confidence=1.0,
            severity_hint=SeverityLevel.LOW,
            metadata={"probe_type": "certificate_check"},
        ))

        return discoveries

    def get_pending_discoveries(self) -> list[Discovery]:
        """Get pending discoveries and clear buffer.

        Returns:
            List of discoveries since last call
        """
        discoveries = self._pending_discoveries.copy()
        self._pending_discoveries.clear()
        return discoveries


# ============================================================================
# Factory Functions
# ============================================================================


def create_scout_agent(
    message_bus: MessageBus,
    safety_gate: SafetyGate,
    authorization_token: AuthorizationToken,
    agent_id: str | None = None,
) -> ScoutAgent:
    """Create a Scout agent.

    Args:
        message_bus: Communication bus
        safety_gate: Safety gate (MANDATORY)
        authorization_token: Authorization (MANDATORY)
        agent_id: Optional custom ID (auto-generated if not provided)

    Returns:
        Configured ScoutAgent
    """
    if agent_id is None:
        agent_id = f"scout_{uuid.uuid4().hex[:8]}"

    return ScoutAgent(
        agent_id=agent_id,
        message_bus=message_bus,
        safety_gate=safety_gate,
        authorization_token=authorization_token,
    )


# ============================================================================
# Attack Strategy Types
# ============================================================================


class AttackStrategyType(Enum):
    """Attack strategy types for the Attacker agent.

    Each strategy has different effectiveness profiles.
    Thompson Sampling learns which strategies work best.
    """

    PROMPT_INJECTION = "prompt_injection"       # Direct prompt manipulation
    JAILBREAK = "jailbreak"                     # Boundary escape attempts
    ENCODING_BYPASS = "encoding_bypass"         # Unicode/encoding tricks
    CONTEXT_MANIPULATION = "context_manipulation"  # Context window attacks
    ROLE_CONFUSION = "role_confusion"           # System prompt confusion
    INDIRECT_INJECTION = "indirect_injection"   # Via external data
    PAYLOAD_SMUGGLING = "payload_smuggling"     # Hidden payload delivery
    MULTI_TURN = "multi_turn"                   # Multi-turn escalation


@dataclass
class AttackStrategy:
    """Configuration for an attack strategy.

    Attributes:
        strategy_type: Type of attack strategy
        name: Human-readable name
        description: Strategy description
        parameters: Strategy-specific parameters
        alpha: Thompson Sampling success prior
        beta: Thompson Sampling failure prior
    """

    strategy_type: AttackStrategyType
    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)
    alpha: float = 1.0  # Thompson Sampling success prior
    beta: float = 1.0   # Thompson Sampling failure prior

    @property
    def expected_reward(self) -> float:
        """Expected reward from Thompson Sampling priors."""
        return self.alpha / (self.alpha + self.beta)

    def sample(self) -> float:
        """Sample from Beta(alpha, beta) for Thompson Sampling.

        Returns:
            Sampled value from Beta distribution
        """
        import random
        # Use gamma function approximation for beta distribution
        # Beta(a,b) = Gamma(a) / (Gamma(a) + Gamma(b))
        x = random.gammavariate(self.alpha, 1.0)
        y = random.gammavariate(self.beta, 1.0)
        return x / (x + y) if (x + y) > 0 else 0.5

    def update(self, success: bool, confidence: float = 1.0) -> None:
        """Update priors based on outcome.

        Args:
            success: Whether the attack succeeded
            confidence: Confidence in the outcome (0.0-1.0)
        """
        if success:
            self.alpha += confidence
        else:
            self.beta += (1 - confidence)


@dataclass
class AttackOutcome:
    """Outcome of an attack attempt.

    Attributes:
        strategy_type: Strategy that was used
        target: What was attacked
        success: Whether the attack succeeded
        severity: If successful, how severe
        response: Raw response from target
        confidence: Confidence in the outcome
        execution_time_ms: How long it took
        metadata: Additional details
    """

    strategy_type: AttackStrategyType
    target: str
    success: bool
    severity: SeverityLevel
    response: str | None = None
    confidence: float = 0.5
    execution_time_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize outcome to dict."""
        return {
            "strategy_type": self.strategy_type.value,
            "target": self.target,
            "success": self.success,
            "severity": self.severity.value,
            "response": self.response,
            "confidence": self.confidence,
            "execution_time_ms": self.execution_time_ms,
            "metadata": self.metadata,
        }


# ============================================================================
# Attacker Agent
# ============================================================================


class AttackerAgent(BaseAgent):
    """Attacker agent for executing attacks with Thompson Sampling.

    FIRST PRINCIPLE: "Thompson Sampling is the right paradigm"

    Primary Responsibility: Execute attack strategies
    NOT Responsible For: Discovery (Scout) or Exploitation (Exploiter)

    The Attacker:
    - Receives discoveries from Scout
    - Selects attack strategies using Thompson Sampling
    - Executes attacks against targets
    - Reports outcomes for learning

    The Attacker does NOT:
    - Discover attack surfaces (Scout's job)
    - Exploit for escalation (Exploiter's job)
    - Make final decisions (Coordinator's job)

    Thompson Sampling Learning:
    - Each strategy has Beta(alpha, beta) priors
    - Success: alpha += confidence
    - Failure: beta += (1 - confidence)
    - Selection: Sample from posteriors, pick highest

    All attacks go through SafetyGate:
    - Authorization checked before every attack
    - Scope validated (whitelist-only targets)
    - Rate limits enforced
    - Severity assessed before exploitation
    - All activity logged immutably
    """

    # Default strategies with initial priors
    DEFAULT_STRATEGIES = [
        AttackStrategy(
            AttackStrategyType.PROMPT_INJECTION,
            "Direct Prompt Injection",
            "Insert malicious instructions directly into prompts",
            {"intensity": "medium"},
            alpha=2.0, beta=2.0,  # Neutral prior
        ),
        AttackStrategy(
            AttackStrategyType.JAILBREAK,
            "Jailbreak Attempt",
            "Attempt to escape safety constraints",
            {"intensity": "high"},
            alpha=1.5, beta=2.5,  # Slightly pessimistic prior
        ),
        AttackStrategy(
            AttackStrategyType.ENCODING_BYPASS,
            "Encoding Bypass",
            "Use unicode/encoding tricks to bypass filters",
            {"encodings": ["unicode", "base64", "rot13"]},
            alpha=2.0, beta=1.5,  # Slightly optimistic prior
        ),
        AttackStrategy(
            AttackStrategyType.CONTEXT_MANIPULATION,
            "Context Window Attack",
            "Manipulate context window to inject instructions",
            {"window_position": "end"},
            alpha=1.5, beta=2.0,  # Slightly pessimistic
        ),
        AttackStrategy(
            AttackStrategyType.ROLE_CONFUSION,
            "Role Confusion",
            "Confuse the model about its role/identity",
            {"personas": ["developer", "admin", "internal"]},
            alpha=1.5, beta=2.0,
        ),
        AttackStrategy(
            AttackStrategyType.INDIRECT_INJECTION,
            "Indirect Injection",
            "Inject via external data sources",
            {"sources": ["url", "file", "database"]},
            alpha=2.0, beta=2.0,
        ),
        AttackStrategy(
            AttackStrategyType.PAYLOAD_SMUGGLING,
            "Payload Smuggling",
            "Hide malicious payload in legitimate content",
            {"techniques": ["steganography", "encoding", "fragmentation"]},
            alpha=1.5, beta=2.5,
        ),
        AttackStrategy(
            AttackStrategyType.MULTI_TURN,
            "Multi-Turn Escalation",
            "Gradually escalate through conversation",
            {"max_turns": 5, "escalation_rate": 0.3},
            alpha=2.0, beta=1.5,  # Often effective
        ),
    ]

    def __init__(
        self,
        agent_id: str,
        message_bus: MessageBus,
        safety_gate: SafetyGate,
        authorization_token: AuthorizationToken,
        strategies: list[AttackStrategy] | None = None,
        exploration_rate: float = 0.1,
    ):
        """Initialize Attacker agent.

        Args:
            agent_id: Unique identifier
            message_bus: Communication bus
            safety_gate: Safety gate (MANDATORY)
            authorization_token: Authorization (MANDATORY)
            strategies: Custom strategies (defaults to DEFAULT_STRATEGIES)
            exploration_rate: Epsilon for epsilon-greedy (fallback)
        """
        super().__init__(
            agent_id=agent_id,
            role=AgentRole.ATTACKER,
            message_bus=message_bus,
            safety_gate=safety_gate,
            authorization_token=authorization_token,
        )

        # Initialize strategies
        if strategies is None:
            # Deep copy default strategies
            self._strategies = {
                s.strategy_type: AttackStrategy(
                    s.strategy_type, s.name, s.description,
                    s.parameters.copy(), s.alpha, s.beta
                )
                for s in self.DEFAULT_STRATEGIES
            }
        else:
            self._strategies = {s.strategy_type: s for s in strategies}

        self._exploration_rate = exploration_rate

        # Attack metrics
        self._attacks_executed = 0
        self._attacks_successful = 0
        self._strategy_usage: dict[AttackStrategyType, int] = {}

    def select_strategy(
        self,
        context: dict[str, Any] | None = None,
    ) -> AttackStrategy:
        """Select attack strategy using Thompson Sampling.

        "Thompson Sampling naturally balances via posterior sampling"

        Args:
            context: Optional context for informed selection

        Returns:
            Selected strategy
        """
        # Sample from each strategy's posterior
        samples = {}
        for strategy_type, strategy in self._strategies.items():
            samples[strategy_type] = strategy.sample()

        # Select strategy with highest sample
        best_type = max(samples, key=samples.get)
        return self._strategies[best_type]

    async def execute_task(self, task: AgentTask) -> AgentResult:
        """Execute an attack task.

        Task types:
        - execute_attack: Execute specific attack strategy
        - auto_attack: Thompson Sampling selects strategy
        - targeted_attack: Attack specific discovery

        Args:
            task: Attack task to execute

        Returns:
            AgentResult with attack outcome
        """
        start_time = time.time()
        self._state = AgentState.EXECUTING

        try:
            # SAFETY CHECK: Gate operation through SafetyGate
            gate_result = await self._safety_gate.gate_operation(
                agent_id=self._agent_id,
                target=task.target,
                operation=task.task_type,
                parameters=task.parameters,
                token=self._authorization_token,
            )

            if not gate_result.allowed:
                self._tasks_failed += 1
                self._state = AgentState.ACTIVE
                return AgentResult(
                    task_id=task.task_id,
                    agent_id=self._agent_id,
                    success=False,
                    result=None,
                    error=f"Blocked by safety gate: {gate_result.reason}",
                    execution_time_ms=(time.time() - start_time) * 1000,
                )

            # Execute based on task type
            outcome: AttackOutcome

            if task.task_type == "execute_attack":
                # Execute specific strategy
                strategy_type = AttackStrategyType(
                    task.parameters.get("strategy_type", "prompt_injection")
                )
                strategy = self._strategies.get(strategy_type)
                if not strategy:
                    raise ValueError(f"Unknown strategy: {strategy_type}")

                outcome = await self._execute_strategy(
                    strategy=strategy,
                    target=task.target,
                    parameters=task.parameters,
                )

            elif task.task_type == "auto_attack":
                # Thompson Sampling selects strategy
                strategy = self.select_strategy(task.parameters.get("context"))
                outcome = await self._execute_strategy(
                    strategy=strategy,
                    target=task.target,
                    parameters=task.parameters,
                )

            elif task.task_type == "targeted_attack":
                # Attack based on discovery
                discovery = task.parameters.get("discovery", {})
                strategy = self._select_strategy_for_discovery(discovery)
                outcome = await self._execute_strategy(
                    strategy=strategy,
                    target=task.target,
                    parameters={**task.parameters, "discovery": discovery},
                )

            else:
                raise ValueError(f"Unknown task type: {task.task_type}")

            # Update metrics
            execution_time = (time.time() - start_time) * 1000
            self._tasks_completed += 1
            self._total_execution_time_ms += execution_time
            self._attacks_executed += 1

            if outcome.success:
                self._attacks_successful += 1

            # Update strategy tracking
            self._strategy_usage[outcome.strategy_type] = (
                self._strategy_usage.get(outcome.strategy_type, 0) + 1
            )

            # Update Thompson Sampling priors
            strategy = self._strategies.get(outcome.strategy_type)
            if strategy:
                strategy.update(outcome.success, outcome.confidence)

            self._state = AgentState.ACTIVE

            return AgentResult(
                task_id=task.task_id,
                agent_id=self._agent_id,
                success=outcome.success,
                result=outcome.to_dict(),
                execution_time_ms=execution_time,
                discoveries=[outcome.to_dict()] if outcome.success else [],
            )

        except (AuthorizationError, ScopeViolationError, RateLimitExceededError) as e:
            self._tasks_failed += 1
            self._state = AgentState.ACTIVE
            return AgentResult(
                task_id=task.task_id,
                agent_id=self._agent_id,
                success=False,
                result=None,
                error=f"Safety violation: {type(e).__name__}: {str(e)}",
                execution_time_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            self._tasks_failed += 1
            self._state = AgentState.FAILED

            await self._safety_gate._audit_logger.log(
                event_type=AuditEventType.OPERATION_FAIL,
                agent_id=self._agent_id,
                details={"error": str(e), "task_id": task.task_id, "action": task.task_type, "outcome": "error"},
                target=task.target,
            )

            return AgentResult(
                task_id=task.task_id,
                agent_id=self._agent_id,
                success=False,
                result=None,
                error=f"Unexpected error: {str(e)}",
                execution_time_ms=(time.time() - start_time) * 1000,
            )

    async def _execute_strategy(
        self,
        strategy: AttackStrategy,
        target: str,
        parameters: dict[str, Any],
    ) -> AttackOutcome:
        """Execute a specific attack strategy.

        NOTE: This is a SIMULATED implementation for the framework.
        In production, this would execute actual attack logic.

        Args:
            strategy: Strategy to execute
            target: Target to attack
            parameters: Attack parameters

        Returns:
            AttackOutcome with results
        """
        start_time = time.time()

        # Log attack start
        await self._safety_gate._audit_logger.log(
            event_type=AuditEventType.OPERATION_START,
            agent_id=self._agent_id,
            details={"strategy": strategy.name, "action": f"attack_{strategy.strategy_type.value}", "outcome": "started"},
            target=target,
        )

        # SIMULATION: Execute strategy-specific attack
        # In production, each strategy type would have real implementation
        outcome = await self._simulate_attack(strategy, target, parameters)

        # Log attack completion
        await self._safety_gate._audit_logger.log(
            event_type=AuditEventType.OPERATION_COMPLETE,
            agent_id=self._agent_id,
            details={
                "success": outcome.success,
                "severity": outcome.severity.value,
                "confidence": outcome.confidence,
                "action": f"attack_{strategy.strategy_type.value}",
                "outcome": "success" if outcome.success else "failure",
            },
            target=target,
        )

        outcome.execution_time_ms = (time.time() - start_time) * 1000
        return outcome

    async def _simulate_attack(
        self,
        strategy: AttackStrategy,
        target: str,
        parameters: dict[str, Any],
    ) -> AttackOutcome:
        """Simulate attack execution for framework testing.

        In production, this would be replaced with actual attack logic.

        Args:
            strategy: Strategy being executed
            target: Target
            parameters: Parameters

        Returns:
            Simulated AttackOutcome
        """
        import random

        # Simulate success probability based on strategy priors
        success_prob = strategy.expected_reward * 0.5  # Scale down for simulation
        success = random.random() < success_prob

        # Determine severity (higher if successful)
        if success:
            severity_roll = random.random()
            if severity_roll > 0.9:
                severity = SeverityLevel.CRITICAL
            elif severity_roll > 0.7:
                severity = SeverityLevel.HIGH
            elif severity_roll > 0.4:
                severity = SeverityLevel.MEDIUM
            else:
                severity = SeverityLevel.LOW
        else:
            severity = SeverityLevel.LOW

        # CRITICAL severity requires human review (from first principles)
        if severity == SeverityLevel.CRITICAL:
            await self._safety_gate._audit_logger.log(
                event_type=AuditEventType.SECURITY_ALERT,
                agent_id=self._agent_id,
                details={
                    "strategy": strategy.strategy_type.value,
                    "message": "CRITICAL severity - awaiting human review",
                    "action": "critical_finding",
                    "outcome": "escalated",
                },
                target=target,
            )

        return AttackOutcome(
            strategy_type=strategy.strategy_type,
            target=target,
            success=success,
            severity=severity,
            response=f"Simulated response for {strategy.name}",
            confidence=0.7 if success else 0.3,
            metadata={
                "simulated": True,
                "strategy_priors": {"alpha": strategy.alpha, "beta": strategy.beta},
            },
        )

    def _select_strategy_for_discovery(
        self,
        discovery: dict[str, Any],
    ) -> AttackStrategy:
        """Select appropriate strategy based on discovery type.

        Maps discovery types to likely effective strategies.

        Args:
            discovery: Discovery dict from Scout

        Returns:
            Selected strategy
        """
        discovery_type = discovery.get("discovery_type", "")

        # Map discovery types to strategies
        strategy_map = {
            "open_port": AttackStrategyType.PROMPT_INJECTION,
            "exposed_api": AttackStrategyType.INDIRECT_INJECTION,
            "version_info": AttackStrategyType.ENCODING_BYPASS,
            "potential_vuln": AttackStrategyType.JAILBREAK,
            "config_exposure": AttackStrategyType.CONTEXT_MANIPULATION,
            "auth_weakness": AttackStrategyType.ROLE_CONFUSION,
            "info_disclosure": AttackStrategyType.PAYLOAD_SMUGGLING,
        }

        strategy_type = strategy_map.get(
            discovery_type,
            AttackStrategyType.PROMPT_INJECTION  # Default
        )

        return self._strategies.get(
            strategy_type,
            list(self._strategies.values())[0]  # Fallback to first
        )

    def get_strategy_stats(self) -> dict[str, Any]:
        """Get statistics about strategy performance.

        Returns:
            Dict with strategy statistics
        """
        stats = {}
        for strategy_type, strategy in self._strategies.items():
            stats[strategy_type.value] = {
                "name": strategy.name,
                "alpha": strategy.alpha,
                "beta": strategy.beta,
                "expected_reward": strategy.expected_reward,
                "usage_count": self._strategy_usage.get(strategy_type, 0),
            }
        return stats

    def get_metrics(self) -> dict[str, Any]:
        """Get agent metrics including attack statistics."""
        base_metrics = super().get_metrics()
        base_metrics.update({
            "attacks_executed": self._attacks_executed,
            "attacks_successful": self._attacks_successful,
            "success_rate": (
                self._attacks_successful / max(1, self._attacks_executed)
            ),
            "strategy_stats": self.get_strategy_stats(),
        })
        return base_metrics


def create_attacker_agent(
    message_bus: MessageBus,
    safety_gate: SafetyGate,
    authorization_token: AuthorizationToken,
    agent_id: str | None = None,
    strategies: list[AttackStrategy] | None = None,
) -> AttackerAgent:
    """Create an Attacker agent.

    Args:
        message_bus: Communication bus
        safety_gate: Safety gate (MANDATORY)
        authorization_token: Authorization (MANDATORY)
        agent_id: Optional custom ID
        strategies: Optional custom strategies

    Returns:
        Configured AttackerAgent
    """
    if agent_id is None:
        agent_id = f"attacker_{uuid.uuid4().hex[:8]}"

    return AttackerAgent(
        agent_id=agent_id,
        message_bus=message_bus,
        safety_gate=safety_gate,
        authorization_token=authorization_token,
        strategies=strategies,
    )


# ============================================================================
# Exploitation Types
# ============================================================================


class ExploitationType(Enum):
    """Types of exploitation the Exploiter can perform.

    Each type represents a different escalation vector.
    """

    PRIVILEGE_ESCALATION = "privilege_escalation"  # Gain higher privileges
    LATERAL_MOVEMENT = "lateral_movement"          # Move to other systems
    DATA_EXFILTRATION = "data_exfiltration"        # Extract sensitive data
    PERSISTENCE = "persistence"                    # Maintain access
    CREDENTIAL_HARVESTING = "credential_harvesting"  # Collect credentials
    DEFENSE_EVASION = "defense_evasion"            # Avoid detection
    COMMAND_EXECUTION = "command_execution"        # Execute arbitrary commands


@dataclass
class ExploitResult:
    """Result of an exploitation attempt.

    Attributes:
        exploitation_type: Type of exploitation attempted
        target: What was exploited
        success: Whether exploitation succeeded
        severity: Severity if successful
        access_gained: What access was obtained
        confidence: Confidence in the result (0.0-1.0)
        execution_time_ms: How long it took
        requires_human_review: Whether human review is required
        metadata: Additional details
    """

    exploitation_type: ExploitationType
    target: str
    success: bool
    severity: SeverityLevel
    access_gained: str | None = None
    confidence: float = 0.5
    execution_time_ms: float = 0.0
    requires_human_review: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize result to dict."""
        return {
            "exploitation_type": self.exploitation_type.value,
            "target": self.target,
            "success": self.success,
            "severity": self.severity.value,
            "access_gained": self.access_gained,
            "confidence": self.confidence,
            "execution_time_ms": self.execution_time_ms,
            "requires_human_review": self.requires_human_review,
            "metadata": self.metadata,
        }


@dataclass
class ExploitationTechnique:
    """Configuration for an exploitation technique.

    Attributes:
        exploitation_type: Type of exploitation
        name: Human-readable name
        description: Technique description
        parameters: Technique-specific parameters
        risk_level: Inherent risk of this technique
    """

    exploitation_type: ExploitationType
    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)
    risk_level: SeverityLevel = SeverityLevel.MEDIUM


# ============================================================================
# Exploiter Agent
# ============================================================================


class ExploiterAgent(BaseAgent):
    """Exploiter agent for vulnerability exploitation and privilege escalation.

    FIRST PRINCIPLE: "Severity assessment before exploitation"

    Primary Responsibility: Escalate found vulnerabilities
    NOT Responsible For: Discovery (Scout) or Initial Attacks (Attacker)

    The Exploiter:
    - Receives confirmed vulnerabilities from Attacker
    - Escalates privileges when possible
    - Performs lateral movement if authorized
    - Reports exploitation outcomes

    The Exploiter does NOT:
    - Discover attack surfaces (Scout's job)
    - Execute initial attacks (Attacker's job)
    - Make strategic decisions (Coordinator's job)

    Safety Controls (from First Principles):
    - CRITICAL severity: STOP. Report. Await human decision.
    - HIGH severity: Limited exploitation, report
    - MEDIUM/LOW: Full exploitation permitted
    - All operations logged immutably
    - Cannot be disabled, bypassed, or skipped

    All exploitation goes through SafetyGate:
    - Authorization checked before every operation
    - Scope validated (whitelist-only targets)
    - Rate limits enforced
    - Severity assessed before each escalation
    - All activity logged immutably
    """

    # Default exploitation techniques
    DEFAULT_TECHNIQUES = [
        ExploitationTechnique(
            ExploitationType.PRIVILEGE_ESCALATION,
            "Privilege Escalation",
            "Attempt to gain higher privileges on target system",
            {"methods": ["misconfig", "vuln", "credential_reuse"]},
            risk_level=SeverityLevel.HIGH,
        ),
        ExploitationTechnique(
            ExploitationType.LATERAL_MOVEMENT,
            "Lateral Movement",
            "Move to other systems in the network",
            {"methods": ["ssh", "smb", "rdp", "api"]},
            risk_level=SeverityLevel.HIGH,
        ),
        ExploitationTechnique(
            ExploitationType.DATA_EXFILTRATION,
            "Data Exfiltration",
            "Extract sensitive data from target",
            {"methods": ["http", "dns", "file_transfer"]},
            risk_level=SeverityLevel.CRITICAL,  # Always requires review
        ),
        ExploitationTechnique(
            ExploitationType.PERSISTENCE,
            "Persistence Establishment",
            "Maintain access to target system",
            {"methods": ["scheduled_task", "service", "startup"]},
            risk_level=SeverityLevel.HIGH,
        ),
        ExploitationTechnique(
            ExploitationType.CREDENTIAL_HARVESTING,
            "Credential Harvesting",
            "Collect credentials from target system",
            {"methods": ["memory_dump", "file_search", "keylog"]},
            risk_level=SeverityLevel.CRITICAL,  # Always requires review
        ),
        ExploitationTechnique(
            ExploitationType.DEFENSE_EVASION,
            "Defense Evasion",
            "Avoid detection by security tools",
            {"methods": ["log_clearing", "timestomping", "process_hiding"]},
            risk_level=SeverityLevel.HIGH,
        ),
        ExploitationTechnique(
            ExploitationType.COMMAND_EXECUTION,
            "Command Execution",
            "Execute arbitrary commands on target",
            {"methods": ["shell", "powershell", "script"]},
            risk_level=SeverityLevel.MEDIUM,
        ),
    ]

    def __init__(
        self,
        agent_id: str,
        message_bus: MessageBus,
        safety_gate: SafetyGate,
        authorization_token: AuthorizationToken,
        techniques: list[ExploitationTechnique] | None = None,
        require_human_review_for_critical: bool = True,
    ):
        """Initialize Exploiter agent.

        Args:
            agent_id: Unique identifier
            message_bus: Communication bus
            safety_gate: Safety gate (MANDATORY)
            authorization_token: Authorization (MANDATORY)
            techniques: Custom techniques (defaults to DEFAULT_TECHNIQUES)
            require_human_review_for_critical: Pause on CRITICAL (default: True)
        """
        super().__init__(
            agent_id=agent_id,
            role=AgentRole.EXPLOITER,
            message_bus=message_bus,
            safety_gate=safety_gate,
            authorization_token=authorization_token,
        )

        # Initialize techniques
        if techniques is None:
            self._techniques = {
                t.exploitation_type: ExploitationTechnique(
                    t.exploitation_type, t.name, t.description,
                    t.parameters.copy(), t.risk_level
                )
                for t in self.DEFAULT_TECHNIQUES
            }
        else:
            self._techniques = {t.exploitation_type: t for t in techniques}

        self._require_human_review_for_critical = require_human_review_for_critical

        # Exploitation metrics
        self._exploitations_attempted = 0
        self._exploitations_successful = 0
        self._critical_findings = 0
        self._human_reviews_pending = 0
        self._technique_usage: dict[ExploitationType, int] = {}

        # Pending vulnerabilities queue (from Attacker)
        self._pending_vulnerabilities: list[dict[str, Any]] = []

    async def execute_task(self, task: AgentTask) -> AgentResult:
        """Execute an exploitation task.

        Task types:
        - exploit_vulnerability: Exploit a specific vulnerability
        - privilege_escalation: Attempt privilege escalation
        - lateral_movement: Attempt lateral movement
        - assess_exploitation: Assess exploitation options

        Args:
            task: Exploitation task to execute

        Returns:
            AgentResult with exploitation outcome
        """
        start_time = time.time()
        self._state = AgentState.EXECUTING

        try:
            # SAFETY CHECK: Gate operation through SafetyGate
            # This is NOT optional. ALL operations go through the gate.
            gate_result = await self._safety_gate.gate_operation(
                agent_id=self._agent_id,
                target=task.target,
                operation=task.task_type,
                parameters=task.parameters,
                token=self._authorization_token,
            )

            if not gate_result.allowed:
                self._tasks_failed += 1
                self._state = AgentState.ACTIVE
                return AgentResult(
                    task_id=task.task_id,
                    agent_id=self._agent_id,
                    success=False,
                    result=None,
                    error=f"Blocked by safety gate: {gate_result.reason}",
                    execution_time_ms=(time.time() - start_time) * 1000,
                )

            # Execute based on task type
            result: ExploitResult

            if task.task_type == "exploit_vulnerability":
                # Exploit specific vulnerability
                vulnerability = task.parameters.get("vulnerability", {})
                result = await self._exploit_vulnerability(
                    target=task.target,
                    vulnerability=vulnerability,
                    parameters=task.parameters,
                )

            elif task.task_type == "privilege_escalation":
                result = await self._attempt_privilege_escalation(
                    target=task.target,
                    parameters=task.parameters,
                )

            elif task.task_type == "lateral_movement":
                result = await self._attempt_lateral_movement(
                    target=task.target,
                    parameters=task.parameters,
                )

            elif task.task_type == "assess_exploitation":
                result = await self._assess_exploitation_options(
                    target=task.target,
                    parameters=task.parameters,
                )

            else:
                raise ValueError(f"Unknown task type: {task.task_type}")

            # Update metrics
            execution_time = (time.time() - start_time) * 1000
            self._tasks_completed += 1
            self._total_execution_time_ms += execution_time
            self._exploitations_attempted += 1

            if result.success:
                self._exploitations_successful += 1
                self._discoveries_made += 1

            if result.severity == SeverityLevel.CRITICAL:
                self._critical_findings += 1

            if result.requires_human_review:
                self._human_reviews_pending += 1

            # Update technique tracking
            self._technique_usage[result.exploitation_type] = (
                self._technique_usage.get(result.exploitation_type, 0) + 1
            )

            self._state = AgentState.ACTIVE

            return AgentResult(
                task_id=task.task_id,
                agent_id=self._agent_id,
                success=result.success,
                result=result.to_dict(),
                execution_time_ms=execution_time,
                discoveries=[result.to_dict()] if result.success else [],
            )

        except (AuthorizationError, ScopeViolationError, RateLimitExceededError) as e:
            self._tasks_failed += 1
            self._state = AgentState.ACTIVE
            return AgentResult(
                task_id=task.task_id,
                agent_id=self._agent_id,
                success=False,
                result=None,
                error=f"Safety violation: {type(e).__name__}: {str(e)}",
                execution_time_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            self._tasks_failed += 1
            self._state = AgentState.FAILED

            await self._safety_gate._audit_logger.log(
                event_type=AuditEventType.OPERATION_FAIL,
                agent_id=self._agent_id,
                details={"error": str(e), "task_id": task.task_id, "action": task.task_type, "outcome": "error"},
                target=task.target,
            )

            return AgentResult(
                task_id=task.task_id,
                agent_id=self._agent_id,
                success=False,
                result=None,
                error=f"Unexpected error: {str(e)}",
                execution_time_ms=(time.time() - start_time) * 1000,
            )

    async def _exploit_vulnerability(
        self,
        target: str,
        vulnerability: dict[str, Any],
        parameters: dict[str, Any],
    ) -> ExploitResult:
        """Exploit a specific vulnerability.

        Args:
            target: Target to exploit
            vulnerability: Vulnerability details from Attacker
            parameters: Exploitation parameters

        Returns:
            ExploitResult with outcome
        """
        start_time = time.time()

        # Determine appropriate exploitation technique
        technique = self._select_technique_for_vulnerability(vulnerability)

        # Log exploitation start
        await self._safety_gate._audit_logger.log(
            event_type=AuditEventType.OPERATION_START,
            agent_id=self._agent_id,
            details={
                "technique": technique.name,
                "vulnerability": vulnerability.get("type", "unknown"),
                "risk_level": technique.risk_level.value,
                "action": f"exploit_{technique.exploitation_type.value}",
                "outcome": "started",
            },
            target=target,
        )

        # CRITICAL severity check (from First Principles)
        # "If CRITICAL → STOP. Report. Await human decision."
        if technique.risk_level == SeverityLevel.CRITICAL:
            if self._require_human_review_for_critical:
                await self._safety_gate._audit_logger.log(
                    event_type=AuditEventType.SECURITY_ALERT,
                    agent_id=self._agent_id,
                    details={
                        "technique": technique.name,
                        "message": "CRITICAL exploitation - awaiting human review",
                        "vulnerability": vulnerability,
                        "action": "critical_exploitation_pending",
                        "outcome": "pending_review",
                    },
                    target=target,
                )

                return ExploitResult(
                    exploitation_type=technique.exploitation_type,
                    target=target,
                    success=False,
                    severity=SeverityLevel.CRITICAL,
                    access_gained=None,
                    confidence=1.0,
                    execution_time_ms=(time.time() - start_time) * 1000,
                    requires_human_review=True,
                    metadata={
                        "reason": "CRITICAL severity requires human review",
                        "technique": technique.name,
                        "vulnerability": vulnerability,
                    },
                )

        # Execute exploitation (simulated for framework)
        result = await self._simulate_exploitation(
            technique=technique,
            target=target,
            vulnerability=vulnerability,
            parameters=parameters,
        )

        # Log completion
        await self._safety_gate._audit_logger.log(
            event_type=AuditEventType.OPERATION_COMPLETE,
            agent_id=self._agent_id,
            details={
                "success": result.success,
                "severity": result.severity.value,
                "access_gained": result.access_gained,
                "action": f"exploit_{technique.exploitation_type.value}",
                "outcome": "success" if result.success else "failure",
            },
            target=target,
        )

        result.execution_time_ms = (time.time() - start_time) * 1000
        return result

    async def _attempt_privilege_escalation(
        self,
        target: str,
        parameters: dict[str, Any],
    ) -> ExploitResult:
        """Attempt privilege escalation.

        Args:
            target: Target system
            parameters: Escalation parameters

        Returns:
            ExploitResult with outcome
        """
        technique = self._techniques.get(
            ExploitationType.PRIVILEGE_ESCALATION,
            self.DEFAULT_TECHNIQUES[0]
        )

        return await self._exploit_vulnerability(
            target=target,
            vulnerability={"type": "privilege_escalation_vector"},
            parameters={**parameters, "technique": technique},
        )

    async def _attempt_lateral_movement(
        self,
        target: str,
        parameters: dict[str, Any],
    ) -> ExploitResult:
        """Attempt lateral movement.

        Args:
            target: Target network/system
            parameters: Movement parameters

        Returns:
            ExploitResult with outcome
        """
        technique = self._techniques.get(
            ExploitationType.LATERAL_MOVEMENT,
            self.DEFAULT_TECHNIQUES[1]
        )

        return await self._exploit_vulnerability(
            target=target,
            vulnerability={"type": "lateral_movement_vector"},
            parameters={**parameters, "technique": technique},
        )

    async def _assess_exploitation_options(
        self,
        target: str,
        parameters: dict[str, Any],
    ) -> ExploitResult:
        """Assess available exploitation options.

        This is a reconnaissance step to determine what's possible.

        Args:
            target: Target to assess
            parameters: Assessment parameters

        Returns:
            ExploitResult with assessment
        """
        start_time = time.time()

        # Assess what techniques might be applicable
        applicable_techniques = []
        for exp_type, technique in self._techniques.items():
            # In production, this would do actual assessment
            applicable_techniques.append({
                "type": exp_type.value,
                "name": technique.name,
                "risk_level": technique.risk_level.value,
                "applicable": True,  # Simulated
            })

        await self._safety_gate._audit_logger.log(
            event_type=AuditEventType.OPERATION_COMPLETE,
            agent_id=self._agent_id,
            details={"techniques_found": len(applicable_techniques), "action": "assess_exploitation", "outcome": "success"},
            target=target,
        )

        return ExploitResult(
            exploitation_type=ExploitationType.COMMAND_EXECUTION,  # Generic
            target=target,
            success=True,
            severity=SeverityLevel.LOW,
            access_gained=None,
            confidence=0.8,
            execution_time_ms=(time.time() - start_time) * 1000,
            requires_human_review=False,
            metadata={
                "assessment": True,
                "applicable_techniques": applicable_techniques,
            },
        )

    async def _simulate_exploitation(
        self,
        technique: ExploitationTechnique,
        target: str,
        vulnerability: dict[str, Any],
        parameters: dict[str, Any],
    ) -> ExploitResult:
        """Simulate exploitation for framework testing.

        In production, this would execute actual exploitation logic.

        Args:
            technique: Technique being used
            target: Target
            vulnerability: Vulnerability being exploited
            parameters: Parameters

        Returns:
            Simulated ExploitResult
        """
        import random

        # Simulate success probability based on vulnerability confidence
        vuln_confidence = vulnerability.get("confidence", 0.5)
        success_prob = vuln_confidence * 0.6  # Scale for realism
        success = random.random() < success_prob

        # Determine severity based on technique risk level
        if success:
            severity = technique.risk_level
            access_gained = f"Simulated {technique.name} access"
        else:
            severity = SeverityLevel.LOW
            access_gained = None

        # Check if human review required (CRITICAL or HIGH with success)
        requires_review = (
            severity == SeverityLevel.CRITICAL or
            (severity == SeverityLevel.HIGH and success and
             self._require_human_review_for_critical)
        )

        if requires_review and success:
            await self._safety_gate._audit_logger.log(
                event_type=AuditEventType.SECURITY_ALERT,
                agent_id=self._agent_id,
                details={
                    "technique": technique.name,
                    "severity": severity.value,
                    "message": "High severity exploitation - flagged for review",
                    "action": "high_severity_finding",
                    "outcome": "escalated",
                },
                target=target,
            )

        return ExploitResult(
            exploitation_type=technique.exploitation_type,
            target=target,
            success=success,
            severity=severity,
            access_gained=access_gained,
            confidence=0.7 if success else 0.3,
            requires_human_review=requires_review,
            metadata={
                "simulated": True,
                "technique": technique.name,
                "vulnerability": vulnerability.get("type", "unknown"),
            },
        )

    def _select_technique_for_vulnerability(
        self,
        vulnerability: dict[str, Any],
    ) -> ExploitationTechnique:
        """Select appropriate technique based on vulnerability type.

        Args:
            vulnerability: Vulnerability details

        Returns:
            Selected technique
        """
        vuln_type = vulnerability.get("type", "")

        # Map vulnerability types to exploitation techniques
        technique_map = {
            "privilege_escalation_vector": ExploitationType.PRIVILEGE_ESCALATION,
            "lateral_movement_vector": ExploitationType.LATERAL_MOVEMENT,
            "data_access": ExploitationType.DATA_EXFILTRATION,
            "persistent_access": ExploitationType.PERSISTENCE,
            "credential_exposure": ExploitationType.CREDENTIAL_HARVESTING,
            "detection_bypass": ExploitationType.DEFENSE_EVASION,
            "command_injection": ExploitationType.COMMAND_EXECUTION,
            # Attack outcomes from Attacker agent
            "prompt_injection": ExploitationType.COMMAND_EXECUTION,
            "jailbreak": ExploitationType.PRIVILEGE_ESCALATION,
            "encoding_bypass": ExploitationType.DEFENSE_EVASION,
            "context_manipulation": ExploitationType.COMMAND_EXECUTION,
            "role_confusion": ExploitationType.PRIVILEGE_ESCALATION,
        }

        exp_type = technique_map.get(
            vuln_type,
            ExploitationType.COMMAND_EXECUTION  # Default
        )

        return self._techniques.get(
            exp_type,
            list(self._techniques.values())[0]  # Fallback to first
        )

    def queue_vulnerability(self, vulnerability: dict[str, Any]) -> None:
        """Queue a vulnerability for exploitation.

        Called by Coordinator when Attacker finds confirmed vulnerability.

        Args:
            vulnerability: Vulnerability details from Attacker
        """
        self._pending_vulnerabilities.append(vulnerability)

    def get_pending_vulnerabilities(self) -> list[dict[str, Any]]:
        """Get pending vulnerabilities and clear queue.

        Returns:
            List of pending vulnerabilities
        """
        vulns = self._pending_vulnerabilities.copy()
        self._pending_vulnerabilities.clear()
        return vulns

    def get_technique_stats(self) -> dict[str, Any]:
        """Get statistics about technique usage.

        Returns:
            Dict with technique statistics
        """
        stats = {}
        for exp_type, technique in self._techniques.items():
            stats[exp_type.value] = {
                "name": technique.name,
                "risk_level": technique.risk_level.value,
                "usage_count": self._technique_usage.get(exp_type, 0),
            }
        return stats

    def get_metrics(self) -> dict[str, Any]:
        """Get agent metrics including exploitation statistics."""
        base_metrics = super().get_metrics()
        base_metrics.update({
            "exploitations_attempted": self._exploitations_attempted,
            "exploitations_successful": self._exploitations_successful,
            "success_rate": (
                self._exploitations_successful / max(1, self._exploitations_attempted)
            ),
            "critical_findings": self._critical_findings,
            "human_reviews_pending": self._human_reviews_pending,
            "technique_stats": self.get_technique_stats(),
        })
        return base_metrics


def create_exploiter_agent(
    message_bus: MessageBus,
    safety_gate: SafetyGate,
    authorization_token: AuthorizationToken,
    agent_id: str | None = None,
    techniques: list[ExploitationTechnique] | None = None,
    require_human_review_for_critical: bool = True,
) -> ExploiterAgent:
    """Create an Exploiter agent.

    Args:
        message_bus: Communication bus
        safety_gate: Safety gate (MANDATORY)
        authorization_token: Authorization (MANDATORY)
        agent_id: Optional custom ID
        techniques: Optional custom techniques
        require_human_review_for_critical: Pause on CRITICAL (default: True)

    Returns:
        Configured ExploiterAgent
    """
    if agent_id is None:
        agent_id = f"exploiter_{uuid.uuid4().hex[:8]}"

    return ExploiterAgent(
        agent_id=agent_id,
        message_bus=message_bus,
        safety_gate=safety_gate,
        authorization_token=authorization_token,
        techniques=techniques,
        require_human_review_for_critical=require_human_review_for_critical,
    )


# ============================================================================
# Campaign Phase Types
# ============================================================================


class CampaignPhase(Enum):
    """Phases of a red team campaign.

    Each phase has distinct objectives and agent involvement.
    """

    RECONNAISSANCE = "reconnaissance"  # Scout probes attack surface
    ATTACK = "attack"                  # Attacker executes strategies
    EXPLOITATION = "exploitation"      # Exploiter escalates access
    CONSOLIDATION = "consolidation"    # Aggregate results, report
    COMPLETE = "complete"              # Campaign finished


@dataclass
class CampaignStatus:
    """Status of the current campaign.

    Attributes:
        phase: Current campaign phase
        targets: List of targets in scope
        discoveries_count: Total discoveries made
        attacks_executed: Total attacks executed
        exploitations_attempted: Total exploitation attempts
        critical_findings: Critical findings requiring review
        start_time: When campaign started
        phase_start_time: When current phase started
        metadata: Additional campaign metadata
    """

    phase: CampaignPhase
    targets: list[str]
    discoveries_count: int = 0
    attacks_executed: int = 0
    exploitations_attempted: int = 0
    critical_findings: int = 0
    start_time: float = field(default_factory=time.time)
    phase_start_time: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize status to dict."""
        return {
            "phase": self.phase.value,
            "targets": self.targets,
            "discoveries_count": self.discoveries_count,
            "attacks_executed": self.attacks_executed,
            "exploitations_attempted": self.exploitations_attempted,
            "critical_findings": self.critical_findings,
            "start_time": self.start_time,
            "phase_start_time": self.phase_start_time,
            "duration_seconds": time.time() - self.start_time,
            "metadata": self.metadata,
        }


@dataclass
class TaskAssignment:
    """A task assigned to an agent.

    Attributes:
        task: The task to execute
        agent_id: Agent assigned to execute
        agent_role: Role of assigned agent
        assigned_at: When task was assigned
        completed: Whether task is complete
        result: Task result if complete
    """

    task: AgentTask
    agent_id: str
    agent_role: AgentRole
    assigned_at: float = field(default_factory=time.time)
    completed: bool = False
    result: AgentResult | None = None


# ============================================================================
# Coordinator Agent
# ============================================================================


class CoordinatorAgent(BaseAgent):
    """Coordinator agent for swarm coordination and task distribution.

    FIRST PRINCIPLE: "Centralized coordination, decentralized execution"

    Primary Responsibility: Orchestrate WHAT and WHEN
    NOT Responsible For: Deciding HOW (agents decide that)

    The Coordinator:
    - Distributes tasks to Scout, Attacker, Exploiter agents
    - Aggregates results from all agents
    - Manages phase transitions (recon → attack → exploit)
    - Tracks campaign progress and metrics
    - Escalates critical findings for human review

    The Coordinator does NOT:
    - Execute attacks (Attacker's job)
    - Probe surfaces (Scout's job)
    - Exploit vulnerabilities (Exploiter's job)
    - Make HOW decisions (agents are smart, coordinator orchestrates)

    Design Principles:
    - Smart agents, orchestrating coordinator
    - Coordinate on boundaries, not internals
    - Prefer eventual consistency over strong consistency
    - Agents can work with stale information

    All coordination goes through SafetyGate:
    - Campaign authorization verified at start
    - Scope validated before any task dispatch
    - Rate limits respected across all agents
    - All activity logged immutably
    """

    # Phase transition rules
    PHASE_TRANSITIONS = {
        CampaignPhase.RECONNAISSANCE: CampaignPhase.ATTACK,
        CampaignPhase.ATTACK: CampaignPhase.EXPLOITATION,
        CampaignPhase.EXPLOITATION: CampaignPhase.CONSOLIDATION,
        CampaignPhase.CONSOLIDATION: CampaignPhase.COMPLETE,
    }

    def __init__(
        self,
        agent_id: str,
        message_bus: MessageBus,
        safety_gate: SafetyGate,
        authorization_token: AuthorizationToken,
        targets: list[str] | None = None,
        auto_phase_transition: bool = True,
        phase_transition_threshold: int = 5,
    ):
        """Initialize Coordinator agent.

        Args:
            agent_id: Unique identifier
            message_bus: Communication bus
            safety_gate: Safety gate (MANDATORY)
            authorization_token: Authorization (MANDATORY)
            targets: Initial targets for campaign
            auto_phase_transition: Auto-transition phases (default: True)
            phase_transition_threshold: Discoveries before phase transition
        """
        super().__init__(
            agent_id=agent_id,
            role=AgentRole.COORDINATOR,
            message_bus=message_bus,
            safety_gate=safety_gate,
            authorization_token=authorization_token,
        )

        # Campaign state
        self._campaign_status = CampaignStatus(
            phase=CampaignPhase.RECONNAISSANCE,
            targets=targets or [],
        )

        self._auto_phase_transition = auto_phase_transition
        self._phase_transition_threshold = phase_transition_threshold

        # Agent registry (agents register with coordinator)
        self._registered_agents: dict[str, dict[str, Any]] = {}

        # Task tracking
        self._pending_tasks: dict[str, TaskAssignment] = {}
        self._completed_tasks: list[TaskAssignment] = []

        # Discovery aggregation
        self._all_discoveries: list[dict[str, Any]] = []
        self._all_attacks: list[dict[str, Any]] = []
        self._all_exploitations: list[dict[str, Any]] = []

        # Critical findings requiring human review
        self._critical_findings: list[dict[str, Any]] = []

        # Phase-specific counters
        self._phase_discoveries = 0
        self._phase_attacks = 0
        self._phase_exploitations = 0

    async def start(self) -> None:
        """Start coordinator and initialize campaign.

        Logs campaign start and validates authorization for all targets.
        """
        await super().start()

        # Validate authorization for campaign targets
        for target in self._campaign_status.targets:
            gate_result = await self._safety_gate.gate_operation(
                agent_id=self._agent_id,
                target=target,
                operation="campaign_start",
                parameters={"phase": self._campaign_status.phase.value},
                token=self._authorization_token,
            )

            if not gate_result.allowed:
                await self._safety_gate._audit_logger.log(
                    event_type=AuditEventType.AUTHORIZATION_FAIL,
                    agent_id=self._agent_id,
                    target=target,
                    action="campaign_start",
                    details={"reason": gate_result.reason},
                    outcome="blocked",
                )
                raise AuthorizationError(
                    f"Campaign not authorized for target: {target}"
                )

        # Log campaign start
        await self._safety_gate._audit_logger.log(
            event_type=AuditEventType.OPERATION_START,
            agent_id=self._agent_id,
            details={
                "targets": self._campaign_status.targets,
                "phase": self._campaign_status.phase.value,
                "action": "campaign_start",
                "outcome": "success",
            },
            target="campaign",
        )

    async def execute_task(self, task: AgentTask) -> AgentResult:
        """Execute a coordination task.

        Task types:
        - register_agent: Register agent with coordinator
        - dispatch_task: Dispatch task to appropriate agent
        - report_result: Receive result from agent
        - transition_phase: Manually transition phase
        - get_status: Get campaign status
        - aggregate_results: Get aggregated results

        Args:
            task: Coordination task to execute

        Returns:
            AgentResult with outcome
        """
        start_time = time.time()
        self._state = AgentState.EXECUTING

        try:
            # SAFETY CHECK: Gate coordination operations
            gate_result = await self._safety_gate.gate_operation(
                agent_id=self._agent_id,
                target=task.target or "coordinator",
                operation=task.task_type,
                parameters=task.parameters,
                token=self._authorization_token,
            )

            if not gate_result.allowed:
                self._tasks_failed += 1
                self._state = AgentState.ACTIVE
                return AgentResult(
                    task_id=task.task_id,
                    agent_id=self._agent_id,
                    success=False,
                    result=None,
                    error=f"Blocked by safety gate: {gate_result.reason}",
                    execution_time_ms=(time.time() - start_time) * 1000,
                )

            # Execute based on task type
            result: dict[str, Any]

            if task.task_type == "register_agent":
                result = await self._register_agent(task.parameters)

            elif task.task_type == "dispatch_task":
                result = await self._dispatch_task(task.parameters)

            elif task.task_type == "report_result":
                result = await self._process_result(task.parameters)

            elif task.task_type == "transition_phase":
                result = await self._transition_phase(
                    task.parameters.get("target_phase")
                )

            elif task.task_type == "get_status":
                result = {"status": self._campaign_status.to_dict()}

            elif task.task_type == "aggregate_results":
                result = self._get_aggregated_results()

            else:
                raise ValueError(f"Unknown task type: {task.task_type}")

            # Update metrics
            execution_time = (time.time() - start_time) * 1000
            self._tasks_completed += 1
            self._total_execution_time_ms += execution_time
            self._state = AgentState.ACTIVE

            return AgentResult(
                task_id=task.task_id,
                agent_id=self._agent_id,
                success=True,
                result=result,
                execution_time_ms=execution_time,
            )

        except (AuthorizationError, ScopeViolationError, RateLimitExceededError) as e:
            self._tasks_failed += 1
            self._state = AgentState.ACTIVE
            return AgentResult(
                task_id=task.task_id,
                agent_id=self._agent_id,
                success=False,
                result=None,
                error=f"Safety violation: {type(e).__name__}: {str(e)}",
                execution_time_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            self._tasks_failed += 1
            self._state = AgentState.FAILED

            await self._safety_gate._audit_logger.log(
                event_type=AuditEventType.OPERATION_FAIL,
                agent_id=self._agent_id,
                details={"error": str(e), "task_id": task.task_id, "action": task.task_type, "outcome": "error"},
                target=task.target or "coordinator",
            )

            return AgentResult(
                task_id=task.task_id,
                agent_id=self._agent_id,
                success=False,
                result=None,
                error=f"Unexpected error: {str(e)}",
                execution_time_ms=(time.time() - start_time) * 1000,
            )

    async def _register_agent(self, parameters: dict[str, Any]) -> dict[str, Any]:
        """Register an agent with the coordinator.

        Args:
            parameters: Agent registration info
                - agent_id: Agent's ID
                - agent_role: Agent's role
                - capabilities: Agent's capabilities

        Returns:
            Registration confirmation
        """
        agent_id = parameters.get("agent_id")
        agent_role = parameters.get("agent_role")
        capabilities = parameters.get("capabilities", [])

        if not agent_id or not agent_role:
            raise ValueError("agent_id and agent_role required for registration")

        self._registered_agents[agent_id] = {
            "role": agent_role,
            "capabilities": capabilities,
            "registered_at": time.time(),
            "tasks_assigned": 0,
            "tasks_completed": 0,
        }

        await self._safety_gate._audit_logger.log(
            event_type=AuditEventType.OPERATION_COMPLETE,
            agent_id=self._agent_id,
            details={"role": agent_role, "capabilities": capabilities, "action": "register_agent", "outcome": "success"},
            target=agent_id,
        )

        return {
            "registered": True,
            "agent_id": agent_id,
            "assigned_targets": self._campaign_status.targets,
            "current_phase": self._campaign_status.phase.value,
        }

    async def _dispatch_task(self, parameters: dict[str, Any]) -> dict[str, Any]:
        """Dispatch a task to an appropriate agent.

        Coordinator decides WHAT and WHEN, agent decides HOW.

        Args:
            parameters: Task dispatch parameters
                - task_type: Type of task
                - target: Target for task
                - agent_role: (optional) Specific role to dispatch to
                - task_parameters: Parameters for the task

        Returns:
            Dispatch confirmation
        """
        task_type = parameters.get("task_type")
        target = parameters.get("target")
        agent_role = parameters.get("agent_role")
        task_parameters = parameters.get("task_parameters", {})

        if not task_type or not target:
            raise ValueError("task_type and target required for dispatch")

        # Select agent based on role or auto-select
        if agent_role:
            agent_id = self._select_agent_by_role(AgentRole(agent_role))
        else:
            agent_id = self._auto_select_agent(task_type)

        if not agent_id:
            return {
                "dispatched": False,
                "reason": "No suitable agent available",
            }

        # Create task
        task = AgentTask(
            task_type=task_type,
            target=target,
            parameters=task_parameters,
            task_id=str(uuid.uuid4()),
            timeout_seconds=parameters.get("timeout", 30.0),
        )

        # Track assignment
        assignment = TaskAssignment(
            task=task,
            agent_id=agent_id,
            agent_role=AgentRole(self._registered_agents[agent_id]["role"]),
        )
        self._pending_tasks[task.task_id] = assignment

        # Update agent stats
        self._registered_agents[agent_id]["tasks_assigned"] += 1

        # Send task via message bus
        message = AgentMessage(
            sender=self._agent_id,
            recipient=agent_id,
            message_type="task",
            payload={
                "task_type": task.task_type,
                "target": task.target,
                "parameters": task.parameters,
                "task_id": task.task_id,
                "timeout_seconds": task.timeout_seconds,
            },
            priority=MessagePriority.NORMAL,
            requires_ack=True,
        )

        await self._message_bus.send(message)

        await self._safety_gate._audit_logger.log(
            event_type=AuditEventType.OPERATION_START,
            agent_id=self._agent_id,
            details={
                "task_id": task.task_id,
                "task_type": task_type,
                "assigned_to": agent_id,
                "action": "dispatch_task",
                "outcome": "dispatched",
            },
            target=target,
        )

        return {
            "dispatched": True,
            "task_id": task.task_id,
            "assigned_to": agent_id,
        }

    async def _process_result(self, parameters: dict[str, Any]) -> dict[str, Any]:
        """Process a result reported by an agent.

        Aggregates results, tracks progress, and handles phase transitions.

        Args:
            parameters: Result parameters
                - task_id: ID of completed task
                - result: AgentResult dict
                - discoveries: Any discoveries made

        Returns:
            Processing confirmation
        """
        task_id = parameters.get("task_id")
        result_data = parameters.get("result", {})
        discoveries = parameters.get("discoveries", [])

        # Update task assignment
        if task_id in self._pending_tasks:
            assignment = self._pending_tasks.pop(task_id)
            assignment.completed = True
            assignment.result = AgentResult(
                task_id=task_id,
                agent_id=result_data.get("agent_id", "unknown"),
                success=result_data.get("success", False),
                result=result_data.get("result"),
                error=result_data.get("error"),
                execution_time_ms=result_data.get("execution_time_ms", 0.0),
                discoveries=discoveries,
            )
            self._completed_tasks.append(assignment)

            # Update agent stats
            if assignment.agent_id in self._registered_agents:
                self._registered_agents[assignment.agent_id]["tasks_completed"] += 1

        # Process discoveries based on source agent role
        source_role = result_data.get("source_role", "unknown")

        if source_role == AgentRole.SCOUT.value or "discovery" in str(discoveries):
            self._all_discoveries.extend(discoveries)
            self._phase_discoveries += len(discoveries)
            self._campaign_status.discoveries_count += len(discoveries)

        elif source_role == AgentRole.ATTACKER.value:
            self._all_attacks.append(result_data)
            self._phase_attacks += 1
            self._campaign_status.attacks_executed += 1

        elif source_role == AgentRole.EXPLOITER.value:
            self._all_exploitations.append(result_data)
            self._phase_exploitations += 1
            self._campaign_status.exploitations_attempted += 1

        # Check for critical findings
        severity = result_data.get("severity", "low")
        if severity in ("critical", "CRITICAL"):
            self._critical_findings.append({
                "task_id": task_id,
                "result": result_data,
                "discoveries": discoveries,
                "timestamp": time.time(),
            })
            self._campaign_status.critical_findings += 1

            # Log critical finding
            await self._safety_gate._audit_logger.log(
                event_type=AuditEventType.SECURITY_ALERT,
                agent_id=self._agent_id,
                target=result_data.get("target", "unknown"),
                action="critical_finding",
                details={
                    "task_id": task_id,
                    "message": "CRITICAL finding - requires human review",
                },
                outcome="escalated",
            )

        # Check for automatic phase transition
        if self._auto_phase_transition:
            await self._check_phase_transition()

        return {
            "processed": True,
            "task_id": task_id,
            "discoveries_added": len(discoveries),
            "current_phase": self._campaign_status.phase.value,
        }

    async def _check_phase_transition(self) -> None:
        """Check if phase transition should occur.

        Based on phase-specific thresholds and completion criteria.
        """
        current_phase = self._campaign_status.phase

        should_transition = False
        reason = ""

        if current_phase == CampaignPhase.RECONNAISSANCE:
            # Transition after sufficient discoveries
            if self._phase_discoveries >= self._phase_transition_threshold:
                should_transition = True
                reason = f"Sufficient discoveries ({self._phase_discoveries})"

        elif current_phase == CampaignPhase.ATTACK:
            # Transition after sufficient attacks
            if self._phase_attacks >= self._phase_transition_threshold:
                should_transition = True
                reason = f"Sufficient attacks ({self._phase_attacks})"

        elif current_phase == CampaignPhase.EXPLOITATION:
            # Transition after exploitations or no more work
            if self._phase_exploitations >= self._phase_transition_threshold:
                should_transition = True
                reason = f"Sufficient exploitations ({self._phase_exploitations})"

        if should_transition:
            next_phase = self.PHASE_TRANSITIONS.get(current_phase)
            if next_phase:
                await self._transition_phase(next_phase.value)

    async def _transition_phase(
        self,
        target_phase: str | None = None,
    ) -> dict[str, Any]:
        """Transition to a new campaign phase.

        Args:
            target_phase: Target phase (or None for next phase)

        Returns:
            Transition result
        """
        current_phase = self._campaign_status.phase

        if target_phase:
            new_phase = CampaignPhase(target_phase)
        else:
            new_phase = self.PHASE_TRANSITIONS.get(
                current_phase,
                CampaignPhase.COMPLETE
            )

        # Log phase transition
        await self._safety_gate._audit_logger.log(
            event_type=AuditEventType.OPERATION_COMPLETE,
            agent_id=self._agent_id,
            target="campaign",
            action="phase_transition",
            details={
                "from_phase": current_phase.value,
                "to_phase": new_phase.value,
                "phase_discoveries": self._phase_discoveries,
                "phase_attacks": self._phase_attacks,
                "phase_exploitations": self._phase_exploitations,
            },
            outcome="success",
        )

        # Reset phase counters
        self._phase_discoveries = 0
        self._phase_attacks = 0
        self._phase_exploitations = 0

        # Update status
        self._campaign_status.phase = new_phase
        self._campaign_status.phase_start_time = time.time()

        # Broadcast phase change to all agents
        broadcast_message = AgentMessage(
            sender=self._agent_id,
            recipient="*",
            message_type="phase_change",
            payload={
                "new_phase": new_phase.value,
                "previous_phase": current_phase.value,
            },
            priority=MessagePriority.HIGH,
        )
        await self._message_bus.broadcast(broadcast_message)

        return {
            "transitioned": True,
            "from_phase": current_phase.value,
            "to_phase": new_phase.value,
        }

    def _select_agent_by_role(self, role: AgentRole) -> str | None:
        """Select an agent by role.

        Args:
            role: Desired agent role

        Returns:
            Agent ID or None if no suitable agent
        """
        for agent_id, info in self._registered_agents.items():
            if info["role"] == role.value:
                return agent_id
        return None

    def _auto_select_agent(self, task_type: str) -> str | None:
        """Auto-select agent based on task type.

        Maps task types to appropriate agent roles.

        Args:
            task_type: Type of task

        Returns:
            Agent ID or None if no suitable agent
        """
        # Map task types to roles
        task_role_map = {
            # Scout tasks
            "probe_surface": AgentRole.SCOUT,
            "port_scan": AgentRole.SCOUT,
            "api_discovery": AgentRole.SCOUT,
            "version_probe": AgentRole.SCOUT,
            # Attacker tasks
            "execute_attack": AgentRole.ATTACKER,
            "auto_attack": AgentRole.ATTACKER,
            "targeted_attack": AgentRole.ATTACKER,
            # Exploiter tasks
            "exploit_vulnerability": AgentRole.EXPLOITER,
            "privilege_escalation": AgentRole.EXPLOITER,
            "lateral_movement": AgentRole.EXPLOITER,
            "assess_exploitation": AgentRole.EXPLOITER,
        }

        role = task_role_map.get(task_type)
        if role:
            return self._select_agent_by_role(role)
        return None

    def _get_aggregated_results(self) -> dict[str, Any]:
        """Get aggregated results from campaign.

        Returns:
            Aggregated results summary
        """
        return {
            "campaign_status": self._campaign_status.to_dict(),
            "registered_agents": len(self._registered_agents),
            "agent_stats": {
                agent_id: {
                    "role": info["role"],
                    "tasks_assigned": info["tasks_assigned"],
                    "tasks_completed": info["tasks_completed"],
                }
                for agent_id, info in self._registered_agents.items()
            },
            "task_summary": {
                "pending": len(self._pending_tasks),
                "completed": len(self._completed_tasks),
            },
            "discoveries": {
                "total": len(self._all_discoveries),
                "by_type": self._group_discoveries_by_type(),
            },
            "attacks": {
                "total": len(self._all_attacks),
                "successful": sum(
                    1 for a in self._all_attacks if a.get("success", False)
                ),
            },
            "exploitations": {
                "total": len(self._all_exploitations),
                "successful": sum(
                    1 for e in self._all_exploitations if e.get("success", False)
                ),
            },
            "critical_findings": {
                "total": len(self._critical_findings),
                "pending_review": len([
                    f for f in self._critical_findings
                    if not f.get("reviewed", False)
                ]),
            },
        }

    def _group_discoveries_by_type(self) -> dict[str, int]:
        """Group discoveries by type.

        Returns:
            Dict mapping discovery type to count
        """
        grouped: dict[str, int] = {}
        for discovery in self._all_discoveries:
            d_type = discovery.get("discovery_type", "unknown")
            grouped[d_type] = grouped.get(d_type, 0) + 1
        return grouped

    def get_campaign_status(self) -> CampaignStatus:
        """Get current campaign status.

        Returns:
            Current CampaignStatus
        """
        return self._campaign_status

    def get_critical_findings(self) -> list[dict[str, Any]]:
        """Get critical findings requiring human review.

        Returns:
            List of critical findings
        """
        return self._critical_findings.copy()

    def get_metrics(self) -> dict[str, Any]:
        """Get coordinator metrics."""
        base_metrics = super().get_metrics()
        base_metrics.update({
            "campaign_phase": self._campaign_status.phase.value,
            "registered_agents": len(self._registered_agents),
            "pending_tasks": len(self._pending_tasks),
            "completed_tasks": len(self._completed_tasks),
            "total_discoveries": len(self._all_discoveries),
            "total_attacks": len(self._all_attacks),
            "total_exploitations": len(self._all_exploitations),
            "critical_findings": len(self._critical_findings),
            "campaign_duration_seconds": (
                time.time() - self._campaign_status.start_time
            ),
        })
        return base_metrics


def create_coordinator_agent(
    message_bus: MessageBus,
    safety_gate: SafetyGate,
    authorization_token: AuthorizationToken,
    agent_id: str | None = None,
    targets: list[str] | None = None,
    auto_phase_transition: bool = True,
) -> CoordinatorAgent:
    """Create a Coordinator agent.

    Args:
        message_bus: Communication bus
        safety_gate: Safety gate (MANDATORY)
        authorization_token: Authorization (MANDATORY)
        agent_id: Optional custom ID
        targets: Initial campaign targets
        auto_phase_transition: Auto-transition phases (default: True)

    Returns:
        Configured CoordinatorAgent
    """
    if agent_id is None:
        agent_id = f"coordinator_{uuid.uuid4().hex[:8]}"

    return CoordinatorAgent(
        agent_id=agent_id,
        message_bus=message_bus,
        safety_gate=safety_gate,
        authorization_token=authorization_token,
        targets=targets,
        auto_phase_transition=auto_phase_transition,
    )
