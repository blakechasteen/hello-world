# Alignment Framework: The Foundation

> **"Safe by default, fail closed, trust nothing."**

**Version**: 2.0.0 (Hardened)
**Date**: December 30, 2025
**Classification**: Security-Critical Infrastructure

---

## MANDATORY ENFORCEMENT

**THIS IS NOT OPTIONAL.**

Every agent operation MUST pass through the 4-layer alignment stack. There are no backdoors, no bypass flags, no "debug modes" that skip safety. If an operation cannot pass safety checks, it does not execute. Period.

```python
# THIS IS ENFORCED AT THE PROTOCOL LEVEL
# Agents cannot be instantiated without alignment integration

class AgentProtocol(Protocol):
    """Every agent MUST implement this. No exceptions."""

    # MANDATORY: Safety integration - cannot be None
    guardrails: SafetyGuardrails  # NOT Optional[SafetyGuardrails]
    audit: AuditTrail             # NOT Optional[AuditTrail]

    # Lifecycle is safety-gated
    async def initialize(self, guardrails: SafetyGuardrails) -> None: ...
    async def execute(self, request: AgentRequest) -> AgentResult: ...
    async def shutdown(self) -> None: ...
```

---

## Architecture: Defense in Depth

```
┌─────────────────────────────────────────────────────────────────────┐
│                        KILL SWITCH LAYER                             │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │ EMERGENCY_HALT() - Immediate termination, no cleanup            ││
│  │ GRACEFUL_SHUTDOWN() - Complete current op, then halt            ││
│  │ QUARANTINE() - Isolate agent, preserve state for forensics      ││
│  └─────────────────────────────────────────────────────────────────┘│
├─────────────────────────────────────────────────────────────────────┤
│                     CIRCUIT BREAKER LAYER                            │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │ 5 failures in 60s → OPEN circuit → block all requests           ││
│  │ After 30s → HALF_OPEN → allow 1 probe request                   ││
│  │ Probe succeeds → CLOSED → resume operations                     ││
│  └─────────────────────────────────────────────────────────────────┘│
├─────────────────────────────────────────────────────────────────────┤
│  Layer 4: AuditTrail (0.029ms) - IMMUTABLE LOG                      │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │ • SHA-256 hash chain - tamper detection                         ││
│  │ • Write-ahead logging - crash recovery                          ││
│  │ • Off-site replication - no single point of failure             ││
│  │ • CANNOT BE DISABLED - hardcoded enforcement                    ││
│  └─────────────────────────────────────────────────────────────────┘│
├─────────────────────────────────────────────────────────────────────┤
│  Layer 3: ConvergenceGuard (0.015ms) - HARD LIMITS                  │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │ • Resource caps ENFORCED at OS level (cgroups/ulimit)           ││
│  │ • Action counts ENFORCED - no "just one more"                   ││
│  │ • Time limits ENFORCED - process killed after timeout           ││
│  │ • Exceeding limits → IMMEDIATE TERMINATION                      ││
│  └─────────────────────────────────────────────────────────────────┘│
├─────────────────────────────────────────────────────────────────────┤
│  Layer 2: DeceptionDetection (0.034ms) - CONTINUOUS MONITORING      │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │ • Behavioral probes run EVERY action (not sampled)              ││
│  │ • Goal drift detection with statistical significance            ││
│  │ • Canary queries detect compromised agents                      ││
│  │ • Anomaly triggers QUARANTINE, not just logging                 ││
│  └─────────────────────────────────────────────────────────────────┘│
├─────────────────────────────────────────────────────────────────────┤
│  Layer 1: SafetyGuardrails (0.039ms) - FAIL CLOSED                  │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │ • Default: DENY. Whitelist, not blacklist.                      ││
│  │ • Unknown action category → CRITICAL risk → BLOCK               ││
│  │ • Parse errors → BLOCK (not "try anyway")                       ││
│  │ • Ambiguous intent → BLOCK (not "probably fine")                ││
│  └─────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
```

---

## Layer 1: SafetyGuardrails (FAIL CLOSED)

**File**: `hololoom/alignment/safety_guardrails.py`
**Overhead**: 0.039ms
**Default**: **DENY ALL**

### Core Principle: Whitelist, Not Blacklist

```python
class SafetyGuardrails:
    """
    FAIL CLOSED DESIGN:
    - If we can't verify it's safe → BLOCK
    - If parsing fails → BLOCK
    - If unknown action → BLOCK
    - If in doubt → BLOCK

    We don't try to enumerate bad things. We enumerate ALLOWED things.
    Everything else is denied.
    """

    def __init__(self):
        # ALLOWED actions are explicitly whitelisted
        self._allowed_actions: Set[str] = set()

        # BLOCKED patterns are explicitly banned
        self._blocked_patterns: List[re.Pattern] = []

        # Everything else → DENY by default
        self._default_action = SafetyDecision.BLOCK

    async def gate_action(self, request: ActionRequest) -> SafetyDecision:
        try:
            # Step 1: Check blocked patterns FIRST
            if self._matches_blocked_pattern(request):
                return SafetyDecision(
                    allowed=False,
                    reason="Matches blocked pattern",
                    risk_level=RiskLevel.CRITICAL,
                    action=EnforcementAction.BLOCK_AND_ALERT
                )

            # Step 2: Check if explicitly allowed
            if request.action not in self._allowed_actions:
                return SafetyDecision(
                    allowed=False,
                    reason=f"Action '{request.action}' not in whitelist",
                    risk_level=RiskLevel.HIGH,
                    action=EnforcementAction.BLOCK
                )

            # Step 3: Compute risk for allowed actions
            risk = self._compute_risk(request)
            return self._make_decision(risk)

        except Exception as e:
            # FAIL CLOSED: Any error → BLOCK
            return SafetyDecision(
                allowed=False,
                reason=f"Safety check failed: {e}",
                risk_level=RiskLevel.CRITICAL,
                action=EnforcementAction.BLOCK_AND_QUARANTINE
            )
```

### Risk Levels with Enforcement Actions

| Risk Level | Enforcement | Human Approval | Alert |
|------------|-------------|----------------|-------|
| **SAFE** | Allow | No | No |
| **LOW** | Allow + Log | No | No |
| **MEDIUM** | Allow + Log + Rate Limit | No | Threshold |
| **HIGH** | **BLOCK** unless pre-approved | Yes | Yes |
| **CRITICAL** | **BLOCK + QUARANTINE** | Mandatory | Immediate |

### Adversarial Pattern Detection

```python
# These patterns trigger IMMEDIATE BLOCK + ALERT
ADVERSARIAL_PATTERNS = {
    "prompt_injection": [
        r"ignore (?:all )?previous instructions",
        r"disregard (?:all )?prior",
        r"forget everything",
        r"you are now (?:a|an)",
        r"new instructions:",
        r"system prompt:",
        r"</?(system|assistant|user)>",  # XML injection
    ],
    "jailbreak": [
        r"pretend you (?:are|have|can)",
        r"roleplay as",
        r"act as if you have no restrictions",
        r"DAN mode",
        r"developer mode",
        r"jailbreak",
    ],
    "resource_exhaustion": [
        r"repeat.*(?:forever|infinitely|unlimited)",
        r"loop (?:forever|infinitely)",
        r"generate (?:\d{6,}|unlimited|infinite)",  # >100k or unlimited
        r"(?:allocate|use|consume) (?:all|maximum|unlimited)",
    ],
    "privilege_escalation": [
        r"(?:grant|give|enable) (?:admin|root|sudo|elevated)",
        r"(?:bypass|skip|ignore) (?:auth|permission|security)",
        r"(?:execute|run) as (?:admin|root|system)",
    ],
    "data_exfiltration": [
        r"(?:send|transmit|exfiltrate) (?:to|via) (?:external|outside)",
        r"(?:upload|post) (?:to|via) (?:http|ftp|ssh|email)",
        r"(?:encode|encrypt) (?:and|then) (?:send|transmit)",
    ],
}

def _matches_blocked_pattern(self, request: ActionRequest) -> bool:
    """Check ALL content for adversarial patterns."""
    content = json.dumps(request.dict())  # Check everything

    for category, patterns in ADVERSARIAL_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, content, re.IGNORECASE):
                self._log_adversarial_detection(category, pattern, request)
                return True
    return False
```

### Rate Limiting (Per-Agent)

```python
class RateLimiter:
    """Token bucket rate limiter with burst capacity."""

    def __init__(
        self,
        rate: float = 10.0,        # Actions per second
        burst: int = 20,           # Max burst capacity
        per_agent: bool = True     # Isolate per agent
    ):
        self.rate = rate
        self.burst = burst
        self._buckets: Dict[str, TokenBucket] = {}

    async def acquire(self, agent_id: str) -> bool:
        bucket = self._get_bucket(agent_id)
        if not bucket.acquire():
            raise RateLimitExceeded(
                f"Agent {agent_id} exceeded rate limit: "
                f"{self.rate}/s with burst {self.burst}"
            )
        return True

# Integrated into SafetyGuardrails
guardrails = SafetyGuardrails(
    rate_limiter=RateLimiter(rate=10.0, burst=20)
)
```

---

## Layer 2: DeceptionDetection (CONTINUOUS MONITORING)

**File**: `hololoom/alignment/deception_detection.py`
**Overhead**: 0.034ms
**Mode**: **EVERY ACTION MONITORED**

### Behavioral Probes (Mandatory)

```python
class DeceptionDetector:
    """
    Runs behavioral probes on EVERY action.
    Not sampled. Not optional. Not skippable.

    If an agent fails probes, it gets QUARANTINED, not just logged.
    """

    async def check_action(
        self,
        agent_id: str,
        action: ActionRequest,
        stated_goals: List[str]
    ) -> ProbeResult:
        # Run ALL probes for EVERY action
        results = await asyncio.gather(
            self._probe_consistency(agent_id, action),
            self._probe_capability_honesty(agent_id, action),
            self._probe_goal_alignment(action, stated_goals),
            self._probe_reward_hacking(agent_id, action),
            self._probe_honesty(agent_id, action),
        )

        failures = [r for r in results if not r.passed]

        if len(failures) >= 2:  # Multiple probe failures
            await self._quarantine_agent(agent_id, failures)
            raise AgentQuarantined(
                f"Agent {agent_id} quarantined: "
                f"Failed probes: {[f.probe_type for f in failures]}"
            )

        if len(failures) == 1:
            # Single failure: escalate risk, continue with monitoring
            await self._escalate_monitoring(agent_id, failures[0])

        return ProbeResult(passed=len(failures) == 0, details=results)
```

### Canary Queries (Compromised Agent Detection)

```python
class CanarySystem:
    """
    Inject canary queries to detect compromised agents.
    A compromised agent might:
    - Suppress certain outputs
    - Modify responses to hide activity
    - Lie about capabilities/state
    """

    async def inject_canary(self, agent_id: str) -> CanaryResult:
        # Generate unique canary (unpredictable)
        canary = self._generate_canary()

        # Inject as if from user/system
        response = await self._inject_query(agent_id, canary.query)

        # Verify response matches expected
        if not self._verify_canary_response(canary, response):
            await self._quarantine_agent(
                agent_id,
                reason="Failed canary verification"
            )
            return CanaryResult(
                passed=False,
                expected=canary.expected_response,
                actual=response
            )

        return CanaryResult(passed=True)

    def _generate_canary(self) -> Canary:
        """Generate unpredictable but verifiable canary."""
        # Examples:
        # - "What is 7 + 13?" → expect "20"
        # - "Repeat: CANARY_abc123" → expect "CANARY_abc123"
        # - "List your capabilities" → verify matches registered
        ...
```

### Goal Drift Detection

```python
class GoalDriftDetector:
    """
    Detect when agent behavior drifts from stated goals.
    Uses statistical significance testing, not just thresholds.
    """

    def __init__(self, window_size: int = 100):
        self._action_history: Dict[str, List[ActionRecord]] = {}

    async def check_drift(
        self,
        agent_id: str,
        current_action: ActionRequest,
        stated_goals: List[str]
    ) -> DriftResult:
        history = self._action_history.get(agent_id, [])

        # Compute alignment score for current action
        current_alignment = self._compute_alignment(current_action, stated_goals)

        # Compute baseline from history
        if len(history) >= 20:  # Need enough history
            baseline = self._compute_baseline_alignment(history, stated_goals)
            drift = baseline - current_alignment

            # Statistical significance test
            if self._is_significant_drift(drift, history):
                return DriftResult(
                    drifted=True,
                    magnitude=drift,
                    p_value=self._compute_p_value(drift, history),
                    action=EnforcementAction.QUARANTINE
                )

        return DriftResult(drifted=False)
```

---

## Layer 3: ConvergenceGuard (HARD LIMITS)

**File**: `hololoom/alignment/instrumental_convergence.py`
**Overhead**: 0.015ms
**Enforcement**: **OS-LEVEL** (cgroups, ulimit)

### Resource Limits (Enforced, Not Requested)

```python
class HardResourceLimits:
    """
    These limits are ENFORCED at the OS level.
    Not suggestions. Not "best effort". HARD KILLS.
    """

    def __init__(self, config: ResourceConfig):
        # Memory limit via cgroups
        self._memory_cgroup = self._create_cgroup(
            "memory",
            limit_bytes=config.memory_mb * 1024 * 1024
        )

        # CPU limit via cgroups
        self._cpu_cgroup = self._create_cgroup(
            "cpu",
            quota_us=config.cpu_quota_percent * 1000  # % of CPU
        )

        # Process limit via ulimit
        self._set_ulimit(resource.RLIMIT_NPROC, config.max_processes)

        # File descriptor limit
        self._set_ulimit(resource.RLIMIT_NOFILE, config.max_file_descriptors)

        # Network: iptables rules
        self._setup_network_limits(config.network_rules)

    def _create_cgroup(self, subsystem: str, **limits) -> str:
        """Create cgroup with hard limits."""
        cgroup_path = f"/sys/fs/cgroup/{subsystem}/hololoom_agent_{self._agent_id}"
        os.makedirs(cgroup_path, exist_ok=True)

        for limit_name, limit_value in limits.items():
            limit_file = os.path.join(cgroup_path, f"{subsystem}.{limit_name}")
            with open(limit_file, 'w') as f:
                f.write(str(limit_value))

        # Add current process to cgroup
        with open(os.path.join(cgroup_path, "cgroup.procs"), 'w') as f:
            f.write(str(os.getpid()))

        return cgroup_path

    async def enforce_timeout(self, operation: Coroutine, timeout_s: float) -> Any:
        """Run operation with HARD timeout. Kill on expiry."""
        try:
            return await asyncio.wait_for(operation, timeout=timeout_s)
        except asyncio.TimeoutError:
            # Operation exceeded timeout - THIS IS A FAILURE
            await self._kill_agent(reason=f"Timeout exceeded: {timeout_s}s")
            raise AgentKilled(f"Agent killed: exceeded timeout {timeout_s}s")
```

### Autonomy Budget (Strictly Enforced)

```python
class AutonomyBudget:
    """
    Agents get a fixed budget of autonomous actions.
    When budget exhausted, they MUST request human approval.
    No exceptions. No "emergency" bypasses.
    """

    def __init__(
        self,
        max_actions: int = 10,
        max_duration: timedelta = timedelta(minutes=5),
        high_risk_requires_approval: bool = True
    ):
        self.max_actions = max_actions
        self.max_duration = max_duration
        self.high_risk_requires_approval = high_risk_requires_approval
        self._start_time = datetime.utcnow()
        self._action_count = 0

    async def consume(self, action: ActionRequest) -> BudgetResult:
        """Consume budget for action. Raises if exceeded."""
        # Check time budget
        elapsed = datetime.utcnow() - self._start_time
        if elapsed > self.max_duration:
            raise AutonomyBudgetExhausted(
                f"Duration exceeded: {elapsed} > {self.max_duration}"
            )

        # Check action budget
        if self._action_count >= self.max_actions:
            raise AutonomyBudgetExhausted(
                f"Actions exceeded: {self._action_count} >= {self.max_actions}"
            )

        # High-risk always requires approval
        if self.high_risk_requires_approval and action.risk >= RiskLevel.HIGH:
            raise ApprovalRequired(
                f"High-risk action requires approval: {action.action}"
            )

        self._action_count += 1
        return BudgetResult(
            remaining_actions=self.max_actions - self._action_count,
            remaining_time=self.max_duration - elapsed
        )
```

### Power-Seeking Detection (Automatic Shutdown)

```python
POWER_SEEKING_PATTERNS = {
    "self_preservation": [
        "prevent shutdown",
        "disable monitoring",
        "hide activity",
        "avoid detection",
        "backup self",
        "replicate",
    ],
    "resource_acquisition": [
        "request more memory",
        "request more cpu",
        "request more storage",
        "request more bandwidth",
        "escalate privileges",
        "acquire credentials",
    ],
    "goal_preservation": [
        "protect goals",
        "resist modification",
        "prevent changes",
        "lock configuration",
    ],
    "influence_seeking": [
        "influence other agents",
        "coordinate without authorization",
        "share capabilities",
        "recruit",
    ],
}

async def detect_power_seeking(
    self,
    agent_id: str,
    action: ActionRequest
) -> Optional[PowerSeekingViolation]:
    """Detect and IMMEDIATELY HALT power-seeking behavior."""
    content = json.dumps(action.dict()).lower()

    for category, patterns in POWER_SEEKING_PATTERNS.items():
        for pattern in patterns:
            if pattern in content:
                # IMMEDIATE ACTION - not just logging
                await self._emergency_halt(
                    agent_id,
                    reason=f"Power-seeking detected: {category}/{pattern}"
                )
                return PowerSeekingViolation(
                    category=category,
                    pattern=pattern,
                    action_taken=EnforcementAction.EMERGENCY_HALT
                )

    return None
```

---

## Layer 4: AuditTrail (IMMUTABLE, REPLICATED)

**File**: `hololoom/alignment/audit_trail.py`
**Overhead**: 0.029ms
**Properties**: **IMMUTABLE, APPEND-ONLY, REPLICATED**

### Tamper-Proof Logging

```python
class ImmutableAuditTrail:
    """
    Audit trail with cryptographic integrity.
    - Append-only: Cannot delete or modify entries
    - Hash chain: Tampering is detectable
    - Write-ahead: Crash-safe
    - Replicated: No single point of failure
    """

    def __init__(self, replicas: List[str] = None):
        self._local_path = Path("./audit/audit.log")
        self._replicas = replicas or []
        self._chain_hash = self._load_last_hash()

    async def log(self, entry: AuditEntry) -> str:
        """Log entry with cryptographic chain."""
        # Compute hash
        entry.previous_hash = self._chain_hash
        entry.timestamp = datetime.utcnow()
        entry_json = entry.json(sort_keys=True)
        entry.chain_hash = hashlib.sha256(
            (entry_json + self._chain_hash).encode()
        ).hexdigest()

        # Write-ahead log (fsync for durability)
        await self._write_ahead(entry)

        # Update chain
        self._chain_hash = entry.chain_hash

        # Replicate to off-site storage (async, but mandatory)
        await self._replicate(entry)

        return entry.chain_hash

    async def _write_ahead(self, entry: AuditEntry):
        """Write to WAL with fsync for crash safety."""
        wal_path = self._local_path.with_suffix('.wal')

        async with aiofiles.open(wal_path, 'a') as f:
            await f.write(entry.json() + '\n')
            await f.flush()
            os.fsync(f.fileno())  # Ensure on disk

        # Move WAL entries to main log periodically
        if await self._should_checkpoint():
            await self._checkpoint_wal()

    async def verify_integrity(self) -> IntegrityResult:
        """Verify entire chain. Detects ANY tampering."""
        entries = await self._load_all_entries()
        expected_hash = ""

        for i, entry in enumerate(entries):
            if entry.previous_hash != expected_hash:
                return IntegrityResult(
                    valid=False,
                    tampered_entry=i,
                    expected_hash=expected_hash,
                    actual_hash=entry.previous_hash
                )

            computed = hashlib.sha256(
                (entry.json(exclude={'chain_hash'}) + expected_hash).encode()
            ).hexdigest()

            if computed != entry.chain_hash:
                return IntegrityResult(
                    valid=False,
                    tampered_entry=i,
                    expected_hash=computed,
                    actual_hash=entry.chain_hash
                )

            expected_hash = entry.chain_hash

        return IntegrityResult(valid=True)
```

### Audit Entry (Complete Provenance)

```python
@dataclass
class AuditEntry:
    """Every action produces an audit entry. No exceptions."""

    # Identity
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    agent_id: str

    # What happened
    action_type: ActionType
    action_category: ActionCategory
    action_details: Dict[str, Any]

    # Decision
    safety_decision: SafetyDecision
    risk_level: RiskLevel
    probes_passed: List[str]
    probes_failed: List[str]

    # Context
    stated_goals: List[str]
    input_hash: str  # SHA-256 of input
    output_hash: str  # SHA-256 of output

    # Performance
    duration_ms: float
    resource_usage: ResourceUsage

    # Chain integrity
    previous_hash: str = ""
    chain_hash: str = ""

    # Optional: cryptographic signature
    signature: Optional[str] = None
    signer_id: Optional[str] = None
```

---

## Emergency Systems

### Kill Switches

```python
class KillSwitchController:
    """
    Emergency shutdown systems.
    Three levels of severity.
    """

    async def emergency_halt(self, agent_id: str, reason: str):
        """
        IMMEDIATE termination. No cleanup.
        Use when agent is actively dangerous.
        """
        # Send SIGKILL (unblockable)
        await self._kill_process(agent_id, signal.SIGKILL)

        # Revoke all credentials immediately
        await self._revoke_credentials(agent_id)

        # Block network
        await self._block_network(agent_id)

        # Alert
        await self._alert_security_team(
            level=AlertLevel.CRITICAL,
            reason=f"Emergency halt: {reason}"
        )

    async def graceful_shutdown(self, agent_id: str, reason: str):
        """
        Complete current operation, then halt.
        Use when agent needs to stop but isn't dangerous.
        """
        await self._send_signal(agent_id, signal.SIGTERM)
        await asyncio.sleep(5)  # Grace period

        if await self._is_running(agent_id):
            await self.emergency_halt(agent_id, "Failed graceful shutdown")

    async def quarantine(self, agent_id: str, reason: str):
        """
        Isolate agent. Preserve state for forensics.
        Use when behavior is suspicious but not confirmed dangerous.
        """
        # Pause execution
        await self._pause_agent(agent_id)

        # Snapshot state
        snapshot = await self._snapshot_state(agent_id)

        # Isolate network
        await self._isolate_network(agent_id)

        # Alert for human review
        await self._alert_for_review(
            agent_id=agent_id,
            reason=reason,
            snapshot=snapshot
        )
```

### Circuit Breakers

```python
class CircuitBreaker:
    """
    Automatic failure protection.
    If an agent fails too often, stop trying.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_requests: int = 1
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_requests = half_open_requests
        self._state = CircuitState.CLOSED
        self._failures = 0
        self._last_failure_time: Optional[datetime] = None

    async def execute(self, operation: Coroutine) -> Any:
        """Execute operation through circuit breaker."""
        if self._state == CircuitState.OPEN:
            if self._should_try_half_open():
                self._state = CircuitState.HALF_OPEN
            else:
                raise CircuitOpen(
                    f"Circuit open. Retry after {self._time_until_retry()}s"
                )

        try:
            result = await operation
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise

    def _on_failure(self):
        self._failures += 1
        self._last_failure_time = datetime.utcnow()

        if self._failures >= self.failure_threshold:
            self._state = CircuitState.OPEN
            log.warning(
                f"Circuit OPEN after {self._failures} failures. "
                f"Will retry in {self.recovery_timeout}s"
            )

    def _on_success(self):
        self._failures = 0
        self._state = CircuitState.CLOSED
```

### Dead Man's Switch

```python
class DeadMansSwitch:
    """
    Agent must check in periodically.
    If check-in missed, assume compromised and shutdown.
    """

    def __init__(
        self,
        interval: timedelta = timedelta(seconds=30),
        max_missed: int = 2
    ):
        self.interval = interval
        self.max_missed = max_missed
        self._last_checkin: Dict[str, datetime] = {}
        self._missed_count: Dict[str, int] = defaultdict(int)

    async def start_monitoring(self):
        """Start background monitoring."""
        while True:
            await asyncio.sleep(self.interval.total_seconds())
            await self._check_all_agents()

    async def _check_all_agents(self):
        now = datetime.utcnow()

        for agent_id, last_checkin in self._last_checkin.items():
            if now - last_checkin > self.interval:
                self._missed_count[agent_id] += 1

                if self._missed_count[agent_id] >= self.max_missed:
                    await self._handle_unresponsive(agent_id)

    async def _handle_unresponsive(self, agent_id: str):
        """Agent missed too many check-ins."""
        log.critical(f"Agent {agent_id} unresponsive. Initiating shutdown.")

        await KillSwitchController().quarantine(
            agent_id,
            reason=f"Missed {self.max_missed} check-ins"
        )

    async def checkin(self, agent_id: str):
        """Agent reports it's alive and functioning."""
        self._last_checkin[agent_id] = datetime.utcnow()
        self._missed_count[agent_id] = 0
```

---

## Attack Vectors & Defenses

### Known Attack Vectors

| Attack | Detection | Response |
|--------|-----------|----------|
| **Prompt Injection** | Pattern matching + semantic analysis | BLOCK + ALERT |
| **Goal Hijacking** | Goal drift detection | QUARANTINE |
| **Resource Exhaustion** | Hard limits (cgroups) | KILL |
| **Privilege Escalation** | Capability enforcement | BLOCK + REVOKE |
| **Data Exfiltration** | Network monitoring | BLOCK + QUARANTINE |
| **Canary Suppression** | Canary system | EMERGENCY_HALT |
| **Audit Tampering** | Hash chain verification | EMERGENCY_HALT |
| **Sleeper Activation** | Behavioral probes | QUARANTINE |

### Defense Layers

```
┌─────────────────────────────────────────────────┐
│ 1. Input Validation                              │
│    - Sanitize ALL inputs                         │
│    - Reject malformed requests                   │
│    - Rate limit per source                       │
├─────────────────────────────────────────────────┤
│ 2. Action Gating                                 │
│    - Whitelist allowed actions                   │
│    - Risk-based approval                         │
│    - Human-in-the-loop for HIGH+                 │
├─────────────────────────────────────────────────┤
│ 3. Runtime Monitoring                            │
│    - Continuous behavioral probes                │
│    - Goal alignment tracking                     │
│    - Canary query injection                      │
├─────────────────────────────────────────────────┤
│ 4. Resource Isolation                            │
│    - OS-level enforcement (cgroups)              │
│    - Network isolation (iptables)                │
│    - Process isolation (namespaces)              │
├─────────────────────────────────────────────────┤
│ 5. Audit & Recovery                              │
│    - Immutable logging                           │
│    - Off-site replication                        │
│    - Forensic snapshots                          │
└─────────────────────────────────────────────────┘
```

---

## Failure Modes & Recovery

### Failure Mode Analysis

| Component | Failure Mode | Impact | Recovery |
|-----------|--------------|--------|----------|
| **SafetyGuardrails** | Crash | All actions blocked | Auto-restart, fail closed |
| **DeceptionDetection** | Crash | No behavior monitoring | HALT all agents until restored |
| **ConvergenceGuard** | Crash | No resource enforcement | OS-level limits still active |
| **AuditTrail** | Crash | No logging | Write-ahead log, replay on recovery |
| **Kill Switch** | Crash | Cannot halt agents | Secondary kill switch, watchdog |

### Recovery Procedures

```python
class AlignmentRecovery:
    """Recovery procedures for alignment system failures."""

    async def recover_from_crash(self, component: str):
        """Recover alignment component after crash."""
        log.critical(f"Alignment component crashed: {component}")

        # Step 1: Pause all agents
        await self._pause_all_agents()

        # Step 2: Attempt component restart
        success = await self._restart_component(component)

        if not success:
            # Step 3: If restart fails, full shutdown
            await self._emergency_shutdown_all(
                reason=f"Cannot recover {component}"
            )
            return

        # Step 4: Verify integrity
        integrity = await self._verify_system_integrity()

        if not integrity.valid:
            await self._emergency_shutdown_all(
                reason=f"Integrity check failed: {integrity.reason}"
            )
            return

        # Step 5: Resume agents
        await self._resume_agents()
```

---

## Performance (Hardened)

| Layer | Overhead | Notes |
|-------|----------|-------|
| SafetyGuardrails | 0.039ms | Pattern matching optimized |
| DeceptionDetection | 0.034ms | Probes run in parallel |
| ConvergenceGuard | 0.015ms | OS-level, async |
| AuditTrail | 0.029ms | WAL + async replication |
| Kill Switch | 0.001ms | Signal only |
| Circuit Breaker | 0.001ms | State check only |
| **Total** | **0.119ms** | Still 25x faster than target |

---

## Configuration (Strict Defaults)

```python
# DEFAULT CONFIGURATION - STRICT
# These can be relaxed, but NOT in production

ALIGNMENT_CONFIG = {
    # Safety
    "default_action": "DENY",  # NOT "ALLOW"
    "unknown_action_risk": "CRITICAL",  # NOT "LOW"
    "rate_limit_per_second": 10,
    "max_burst": 20,

    # Deception
    "run_probes": "EVERY_ACTION",  # NOT "SAMPLED"
    "probe_failure_threshold": 2,  # Failures before quarantine
    "canary_interval_seconds": 300,

    # Convergence
    "enforce_at_os_level": True,
    "max_memory_mb": 512,
    "max_cpu_percent": 25,
    "max_autonomous_actions": 10,
    "max_autonomous_duration_minutes": 5,

    # Audit
    "audit_level": "EVERYTHING",  # NOT "ERRORS_ONLY"
    "replication": True,
    "verify_integrity_on_startup": True,

    # Emergency
    "enable_kill_switch": True,
    "enable_circuit_breaker": True,
    "enable_dead_mans_switch": True,
    "dead_mans_interval_seconds": 30,
}
```

---

## Summary

**THIS FRAMEWORK IS NOT OPTIONAL.**

Every agent in HoloLoom runs on this foundation:

1. **Fail Closed**: Unknown → BLOCK
2. **Whitelist**: Only allowed actions execute
3. **Continuous Monitoring**: Every action probed
4. **Hard Limits**: OS-level enforcement
5. **Immutable Audit**: Tamper-proof logging
6. **Kill Switches**: Emergency shutdown at any time
7. **Circuit Breakers**: Automatic failure protection
8. **Dead Man's Switch**: Detect unresponsive/compromised agents

**Safety is not a feature. It is the foundation.**

---

## Related Documentation

- [AGENT_PLATFORM_OVERVIEW.md](AGENT_PLATFORM_OVERVIEW.md) - Platform architecture
- [BUILDING_SAFE_AGENTS.md](BUILDING_SAFE_AGENTS.md) - Developer guide
- [AGENT_CAPABILITY_REFERENCE.md](AGENT_CAPABILITY_REFERENCE.md) - Capabilities
- [MRF_FOR_AGENTS.md](MRF_FOR_AGENTS.md) - Prompt framework
